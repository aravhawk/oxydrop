"""Multi-task losses with per-head label masks (a head contributes only on windows where its label exists).

* hr:   soft-label cross-entropy over 141 bins (40..180 bpm, Gaussian target, sd `sigma` bins), masked by has_hr and 40 <= hr <= 180.
* spo2: Gaussian negative log-likelihood (log-variance clamped to [-3, 3]) + Huber (delta 2 points), masked by has_spo2 and a per-row weight
        (BUT PPG has one spot SpO2 per person, not time aligned, so it carries a lower weight).
* low:  BCE (optionally focal) for SpO2 < 90 with a positive-class weight, masked by has_low and a per-row weight.
* sqi:  BCE; target = 1 if the (detached) HR estimate is within 5 bpm of the reference. Masked by has_hr.
* aux:  Huber on the OpenOx ABG-anchored SaO2 head (contact oximeter, training only).
* adv:  gradient-reversal domain classifier on the embedding (optional).
"""
import torch
import torch.nn.functional as F
from torch import nn

from .model import HR_LO, HR_HI, HR_BINS, hr_from_logits

LOGVAR_MIN, LOGVAR_MAX = -3.0, 3.0


def hr_soft_ce(logits, hr, mask, sigma=1.5):
    ok = mask & torch.isfinite(hr) & (hr >= HR_LO) & (hr <= HR_HI)
    if not ok.any():
        return logits.sum() * 0.0
    bins = torch.arange(HR_BINS, device=logits.device, dtype=logits.dtype)[None]
    tgt = torch.exp(-0.5 * ((bins - (hr[ok, None] - HR_LO)) / sigma) ** 2)
    tgt = tgt / tgt.sum(1, keepdim=True)
    return -(tgt * F.log_softmax(logits[ok], -1)).sum(1).mean()


def spo2_loss(pred, logvar, y, mask, w, huber_w=0.5):
    ok = mask & torch.isfinite(y)
    if not ok.any():
        return pred.sum() * 0.0
    lv = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    d = pred - torch.nan_to_num(y, nan=0.0)
    nll = 0.5 * (lv + d ** 2 * torch.exp(-lv))
    hub = F.huber_loss(pred, torch.nan_to_num(y, nan=0.0), reduction='none', delta=2.0)
    per = (nll + huber_w * hub) * w
    return (per * ok).sum() / (w * ok).sum().clamp_min(1e-6)


def low_loss(logit, y, mask, w, pos_weight=1.0, focal=0.0):
    ok = mask & torch.isfinite(y)
    if not ok.any():
        return logit.sum() * 0.0
    yy = torch.nan_to_num(y, nan=0.0)
    bce = F.binary_cross_entropy_with_logits(logit, yy, reduction='none')
    cw = torch.where(yy > 0.5, torch.as_tensor(pos_weight, device=logit.device, dtype=logit.dtype), torch.ones_like(logit))
    if focal > 0:
        p = torch.sigmoid(logit)
        pt = torch.where(yy > 0.5, p, 1 - p)
        bce = bce * (1 - pt) ** focal
    per = bce * cw * w
    return (per * ok).sum() / (w * ok).sum().clamp_min(1e-6)


def sqi_loss(logit, hr_logits, hr, mask, tol=5.0):
    ok = mask & torch.isfinite(hr)
    if not ok.any():
        return logit.sum() * 0.0
    with torch.no_grad():
        est = hr_from_logits(hr_logits.float(), 'expect')
        tgt = ((est - torch.nan_to_num(hr, nan=0.0)).abs() <= tol).float()
    return F.binary_cross_entropy_with_logits(logit[ok], tgt[ok])


def aux_loss(pred, y, mask):
    ok = mask & torch.isfinite(y)
    if not ok.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[ok], y[ok], delta=2.0)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AdvHead(nn.Module):
    """Domain-adversarial head: predicts the dataset id from the embedding through a gradient-reversal layer (training only, never exported)."""

    def __init__(self, emb, n_ds, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU(), nn.Linear(hidden, n_ds))

    def forward(self, emb, lam):
        return self.net(_GRL.apply(emb, lam))
