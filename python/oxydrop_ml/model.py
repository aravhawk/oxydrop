"""OxyNet: small multi-task 1D temporal CNN (ResNet-1D / TCN style) for OxyDrop.

One class builds both the on-device student (< 300k parameters) and the distillation teacher (1M to 3M).
Only ops that app/js/cnn.js implements are used: Conv1d (dilated, strided, same or causal padding), BatchNorm1d
(folded into the conv at export), ReLU, residual add, global average+max pool, Linear, a per-domain embedding add.
The exact tensor layout and op list are in python/oxydrop_ml/MODEL_SPEC.md.

Input  x      [B, 10, 300] float32 (see CH_NAMES; 3 waveform "slots" + 7 auxiliary channels)
       domain [B] int64     0 = phone camera (R,G,B), 1 = contact oximeter (red, IR in slots 0, 1)
       side   [B, 9]        baseline-relative scalar physics features (optional, zeros if omitted)
       emb_base [B, E]      the person's baseline window embedding (optional, zeros if omitted)
Output dict: hr_logits [B,141] (40..180 bpm), spo2 [B] (%), spo2_logvar [B], lowox_logit [B], sqi_logit [B], emb [B,E]
       and, only when with_aux=True, aux_spo2 [B] (contact-oximeter head; training only, never exported).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
from torch import nn
import torch.nn.functional as F

T_WIN = 300
FS = 30
HR_LO, HR_HI = 40, 180
HR_BINS = HR_HI - HR_LO + 1  # 141

# Canonical channel layout. It IS python/oxydrop_ml/features.CHANNELS / SIDE (FEAT-003); a test asserts they agree when features.py is present.
# Waveform "slots" 0..2 go through the channel-shared stem. Phone: ac_R, ac_G, ac_B. Contact oximeter: red -> slot 0, infrared -> slot 1
# (the data builders store contact PPG as [red, IR, 0] and mask the third wavelength), so slot 2 is empty for domain 1.
SLOT_NAMES = ['ac_R', 'ac_G', 'ac_B']
AUX_NAMES = ['dt', 'ror_RG_rel', 'ror_RB_rel', 'beat_ok', 'dc_R_rel', 'dc_G_rel', 'dc_B_rel']
CH_NAMES = SLOT_NAMES + AUX_NAMES
N_SLOTS, N_AUX, N_CH = len(SLOT_NAMES), len(AUX_NAMES), len(CH_NAMES)
SIDE_NAMES = ['d_ror_RG', 'd_ror_RB', 'd_dc_R', 'd_dc_G', 'd_dc_B', 'd_lac_R', 'd_lac_G', 'd_lac_B', 'beat_frac']
SIDE_DIM = len(SIDE_NAMES)
DOMAINS = ['phone_rgb', 'contact_red_ir']      # same indices as oxydrop_ml.DOMAIN (0 phone, 1 contact)
DOMAIN_SLOT_MASK = [[1, 1, 1], [1, 1, 0]]      # which waveform slots exist per domain
SPO2_CENTER, SPO2_SCALE = 93.0, 5.0  # spo2 = center + scale * raw head output


@dataclass
class OxyCfg:
    in_ch: int = N_CH
    n_slots: int = N_SLOTS
    stem: str = 'shared'          # 'shared' (channel-shared slot conv + domain embedding) or 'joint' (plain Conv1d on all channels)
    slot_width: int = 8
    slot_k: int = 5
    stem_width: int = 24
    stem_k: int = 5
    widths: tuple = (24, 40, 56, 80)
    depths: tuple = (2, 2, 2, 2)  # residual blocks per stage; stages 1.. start with a stride-2 block
    k: int = 3
    dil_cycle: int = 2            # block j in a stage uses dilation 2**(j % dil_cycle)
    emb: int = 64
    side_dim: int = SIDE_DIM
    n_domains: int = len(DOMAINS)
    hr_hidden: int = 64
    dropout: float = 0.0
    causal: bool = False          # left-only padding
    domain_slot_mask: list = field(default_factory=lambda: [list(r) for r in DOMAIN_SLOT_MASK])

    def to_dict(self):
        return asdict(self)


def student_cfg(**kw) -> OxyCfg:
    return OxyCfg(**kw)


def teacher_cfg(**kw) -> OxyCfg:
    base = dict(slot_width=16, stem_width=64, widths=(64, 128, 192, 256), depths=(2, 3, 3, 2), emb=128, hr_hidden=128, dropout=0.1)
    base.update(kw)
    return OxyCfg(**base)


def pads(k: int, d: int, causal: bool):
    total = d * (k - 1)
    return (total, 0) if causal else (total // 2, total - total // 2)


class Conv(nn.Module):
    """Conv1d with explicit (left, right) zero padding; optional BatchNorm. Padding is applied in forward, so the same
    numbers hold in JS (cnn.js zero-pads identically) and in ONNX (Pad op)."""

    def __init__(self, ci, co, k=3, stride=1, dilation=1, causal=False, bn=True, relu=True, pad=None):
        super().__init__()
        self.pad = tuple(pad) if pad is not None else pads(k, dilation, causal)
        self.conv = nn.Conv1d(ci, co, k, stride=stride, dilation=dilation, padding=0, bias=not bn)
        self.bn = nn.BatchNorm1d(co) if bn else None
        self.relu = relu
        self.stride, self.dilation, self.k = stride, dilation, k

    def forward(self, x, relu=None):
        x = F.pad(x, self.pad)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x) if (self.relu if relu is None else relu) else x


class Block(nn.Module):
    """y = relu( conv2(relu(conv1(x))) + skip(x) ), BN after each conv. skip is identity or 1x1 (strided) conv + BN."""

    def __init__(self, ci, co, k, stride, dilation, causal):
        super().__init__()
        self.c1 = Conv(ci, co, k, stride, dilation, causal, bn=True, relu=True)
        self.c2 = Conv(co, co, k, 1, dilation, causal, bn=True, relu=False)
        self.proj = Conv(ci, co, 1, stride, 1, causal, bn=True, relu=False, pad=(0, 0)) if (ci != co or stride != 1) else None

    def forward(self, x):
        y = self.c2(self.c1(x))
        s = x if self.proj is None else self.proj(x)
        return F.relu(y + s)


class OxyNet(nn.Module):
    def __init__(self, cfg: OxyCfg | None = None):
        super().__init__()
        self.cfg = cfg = cfg or OxyCfg()
        assert cfg.stem in ('shared', 'joint')
        if cfg.stem == 'shared':
            self.slot_conv = nn.Conv1d(1, cfg.slot_width, cfg.slot_k, padding=0)  # one set of weights for every waveform slot
            self.slot_pad = pads(cfg.slot_k, 1, cfg.causal)
            stem_in = cfg.n_slots * cfg.slot_width + (cfg.in_ch - cfg.n_slots)
            self.register_buffer('domain_slot_mask', torch.tensor(cfg.domain_slot_mask, dtype=torch.float32))
            self.domain_emb = nn.Embedding(cfg.n_domains, cfg.stem_width)
            nn.init.normal_(self.domain_emb.weight, std=0.1)
        else:
            stem_in = cfg.in_ch
        self.stem = Conv(stem_in, cfg.stem_width, cfg.stem_k, causal=cfg.causal, bn=True, relu=(cfg.stem == 'joint'))
        blocks, ci = [], cfg.stem_width
        for si, (w, n) in enumerate(zip(cfg.widths, cfg.depths)):
            for j in range(n):
                blocks.append(Block(ci, w, cfg.k, 2 if (si > 0 and j == 0) else 1, 2 ** (j % cfg.dil_cycle), cfg.causal))
                ci = w
        self.blocks = nn.ModuleList(blocks)
        self.feat_dim = 2 * ci  # global average + global max pool
        self.drop = nn.Dropout(cfg.dropout)
        E = cfg.emb
        self.fc_emb = nn.Linear(self.feat_dim, E)
        self.hr1 = nn.Linear(E, cfg.hr_hidden)
        self.hr2 = nn.Linear(cfg.hr_hidden, HR_BINS)
        self.spo2 = nn.Linear(E + cfg.side_dim, 2)          # mean, log-variance
        self.lowox = nn.Linear(E + cfg.side_dim, 1)        # on (emb - baseline emb, side)
        self.sqi = nn.Linear(E, 1)
        self.aux_spo2 = nn.Linear(E, 1)                     # contact-oximeter SpO2, training only

    def encode(self, x, domain=None):
        cfg = self.cfg
        if cfg.stem == 'shared':
            B, _, T = x.shape
            slots = x[:, :cfg.n_slots].reshape(B * cfg.n_slots, 1, T)
            h = F.relu(self.slot_conv(F.pad(slots, self.slot_pad)))
            h = h.reshape(B, cfg.n_slots, cfg.slot_width, T)
            if domain is None:
                domain = torch.zeros(B, dtype=torch.long, device=x.device)
            m = self.domain_slot_mask[domain]                      # [B, n_slots]
            h = (h * m[:, :, None, None]).reshape(B, cfg.n_slots * cfg.slot_width, T)
            z = torch.cat([h, x[:, cfg.n_slots:]], dim=1)
            z = self.stem(z, relu=False) + self.domain_emb(domain)[:, :, None]
            z = F.relu(z)
        else:
            z = self.stem(x)
        for b in self.blocks:
            z = b(z)
        return torch.cat([z.mean(-1), z.amax(-1)], dim=1)

    def forward(self, x, domain=None, side=None, emb_base=None, with_aux=False):
        B = x.shape[0]
        p = self.encode(x, domain)
        emb = F.relu(self.fc_emb(self.drop(p)))
        if side is None:
            side = x.new_zeros(B, self.cfg.side_dim)
        if emb_base is None:
            emb_base = x.new_zeros(B, self.cfg.emb)
        hr = self.hr2(self.drop(F.relu(self.hr1(emb))))
        so = self.spo2(torch.cat([emb, side], 1))
        lo = self.lowox(torch.cat([emb - emb_base, side], 1))
        out = {'hr_logits': hr, 'spo2': SPO2_CENTER + SPO2_SCALE * so[:, 0], 'spo2_logvar': so[:, 1],
               'lowox_logit': lo[:, 0], 'sqi_logit': self.sqi(emb)[:, 0], 'emb': emb}
        if with_aux:
            out['aux_spo2'] = SPO2_CENTER + SPO2_SCALE * self.aux_spo2(emb)[:, 0]
        return out


OUT_NAMES = ['hr_logits', 'spo2', 'spo2_logvar', 'lowox_logit', 'sqi_logit', 'emb']


class ExportWrapper(nn.Module):
    """Tuple-output wrapper for ONNX (inputs: x, domain, side, emb_base). Aux head is dropped."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, domain, side, emb_base):
        o = self.net(x, domain, side, emb_base)
        return tuple(o[k] for k in OUT_NAMES)


def count_params(m: nn.Module, include_aux=True) -> int:
    n = sum(p.numel() for p in m.parameters())
    if not include_aux and hasattr(m, 'aux_spo2'):
        n -= sum(p.numel() for p in m.aux_spo2.parameters())
    return n


def hr_from_logits(logits, mode='expect', window=7):
    """Decode HR (bpm). 'argmax', or 'expect' = expectation over a +-window bin neighbourhood of the argmax (robust to doubling tails)."""
    p = torch.softmax(logits, -1)
    am = p.argmax(-1)
    if mode == 'argmax':
        return (am + HR_LO).float()
    bins = torch.arange(HR_BINS, device=logits.device)
    w = ((bins[None] - am[:, None]).abs() <= window).float() * p
    return (w * (bins[None] + HR_LO)).sum(-1) / w.sum(-1).clamp_min(1e-9)
