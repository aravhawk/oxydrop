"""Chain teacher for the residual hybrid (`R`) and the distilled student (`D`) of FEAT-008 (see amendments/AMENDMENT_2_hybrid.md).

The teacher is the v1 logistic chain (two class-balanced ridge-logistic models on the dHR feature and the 19 calibration-relative FUSION features, shipped C).
Its logit is an out-of-person PRIOR for every Hoffman row:
  * a training person g gets the chain fit on the other training people (never on its own labels);
  * a held-out person (outer or inner) gets the chain fit on all training people.
Families (`install` swaps the loss and the validation prediction of `train.fit`, nothing in train.py is edited):
  P  plain CNN, no change.
  R  residual: low-oxygen logit = prior + net output (the network learns a correction). MTHS low-oxygen terms are off (no prior there).
  D  distillation: P plus `lam * BCE(net logit, sigmoid(prior))` on Hoffman training rows.
The exported network is the same graph for all families; for R the app adds the chain logit to the network output.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import losses as LS
from . import train as T

C_HR, C_FUS = 0.01, 0.1                 # shipped C of the v1 chain (baselines/chain_v1.SHIPPED)
LAM_DISTILL = 0.5
FAMILIES = ('P', 'R', 'D')


# ------------------------------------------------------------------------------------------------ the chain
class Chain:
    """v1 chain: mean of the probabilities of two logistic regressions; `logit` is the logit of that mean (what `stack_run.chain_logit` returns)."""

    def __init__(self, X, y, c_hr=C_HR, c_fus=C_FUS):
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        self.m = []
        for cols, c in (([0], c_hr), (list(range(19)), c_fus)):
            sc = StandardScaler().fit(X[:, cols])
            lr = LogisticRegression(C=c, class_weight='balanced', max_iter=2000).fit(sc.transform(X[:, cols]), y)
            self.m.append((cols, sc, lr))

    def logit(self, X):
        X = np.asarray(X, float)
        p = sum(lr.predict_proba(sc.transform(X[:, cols]))[:, 1] for cols, sc, lr in self.m) / len(self.m)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def params(self):
        """Plain-number form for the JS export: per model feature indices, mean, scale, weights, bias, plus the combination rule."""
        out = []
        for cols, sc, lr in self.m:
            out.append(dict(cols=cols, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0])))
        return out


def chain_prior(X19, y, subject, evalmask, train_people, other_people):
    """Prior logits [n] for every row of the Hoffman table (X19 [n,19], y labels, subject ids, evalmask = rows the chain may be fit on).
    train_people: OOF (fit on the other train_people). other_people: fit on all train_people. Rows of anybody else stay 0."""
    prior = np.zeros(len(X19), np.float32)
    subject = np.asarray(subject)
    fit_rows = lambda ps: np.nonzero(np.isin(subject, ps) & evalmask)[0]
    for g in train_people:
        rest = [p for p in train_people if p != g]
        if not rest:
            continue
        r = fit_rows(rest)
        m = subject == g
        prior[m] = Chain(X19[r], y[r]).logit(X19[m])
    r = fit_rows(list(train_people))
    ch = Chain(X19[r], y[r])
    for p in other_people:
        m = subject == p
        prior[m] = ch.logit(X19[m])
    return prior


def prior_for_pool(pool, X19, train_people, other_people):
    """Prior over pool rows (zeros for non-Hoffman rows). X19 is indexed by the Hoffman shard-local row id (pool.local)."""
    prior = np.zeros(pool.N, np.float32)
    hrows = np.nonzero(pool.ds == 'hoffman')[0]
    loc = pool.local[hrows]
    sub = pool.subject[hrows]
    ev = pool.eval[hrows] & pool.has_low[hrows]
    y = pool.low[hrows].astype(int)
    pr = chain_prior(X19[loc], y, sub, ev, train_people, other_people)
    prior[hrows] = pr
    return prior


# ------------------------------------------------------------------------------------------------ loss and prediction patches
class _State:
    family = 'P'
    prior = None            # np.ndarray [pool.N]
    lam = LAM_DISTILL


STATE = _State()
_ORIG = dict(batch_loss=T.batch_loss, predict=T.predict)


def batch_loss_hybrid(net, pool, rows, x, s, dom, eb, hp, pos_w, adv=None, lam=0.0, ds_t=None, device='cpu'):
    """train.batch_loss with the family-specific low-oxygen term; identical to it for family P."""
    fam = STATE.family
    if fam == 'P':
        return _ORIG['batch_loss'](net, pool, rows, x, s, dom, eb, hp, pos_w, adv, lam, ds_t, device)
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    o = net(x, dom, s, eb, with_aux=True)
    m = lambda a: t(a, torch.bool)
    hr, spo2, low, sao2 = t(pool.hr[rows]), t(pool.spo2[rows]), t(pool.low[rows]), t(pool.sao2[rows])
    w_low = t(np.where(pool.is_ox[rows], hp.w_low_ox, 1.0))
    prior = t(STATE.prior[rows])
    is_hof = pool.ds[rows] == 'hoffman'
    L = {}
    L['hr'] = LS.hr_soft_ce(o['hr_logits'].float(), hr, m(pool.has_hr[rows]), hp.sigma)
    L['spo2'] = LS.spo2_loss(o['spo2'].float(), o['spo2_logvar'].float(), spo2, m(pool.has_spo2[rows]), t(pool.w_spo2[rows]))
    raw = o['lowox_logit'].float()
    has_low = pool.has_low[rows].copy()
    if fam == 'R':
        has_low &= pool.ds[rows] != 'mths'
        L['low'] = LS.low_loss(raw + prior * t(is_hof), low, m(has_low), w_low, pos_w, hp.focal)
    else:                                                  # D
        L['low'] = LS.low_loss(raw, low, m(has_low), w_low, pos_w, hp.focal)
        if is_hof.any():
            soft = torch.sigmoid(prior)
            bce = F.binary_cross_entropy_with_logits(raw, soft, reduction='none')
            L['distill'] = (bce * t(is_hof)).sum() / t(is_hof).sum().clamp_min(1.0)
    L['sqi'] = LS.sqi_loss(o['sqi_logit'].float(), o['hr_logits'].float(), hr, m(pool.has_hr[rows]))
    L['aux'] = LS.aux_loss(o['aux_spo2'].float(), sao2, m(pool.has_sao2[rows]))
    tot = hp.w_hr * L['hr'] + hp.w_spo2 * L['spo2'] + hp.w_low * L['low'] + hp.w_sqi * L['sqi'] + hp.w_aux * L['aux']
    if 'distill' in L:
        tot = tot + hp.w_low * STATE.lam * L['distill']
    return tot, L


def predict_hybrid(net, pool, rows, norm, hp, device='cpu', bs=512):
    """train.predict with the prior added to the low-oxygen logit for family R (used by the early-stopping score)."""
    out = _ORIG['predict'](net, pool, rows, norm, hp, device, bs)
    if STATE.family == 'R':
        rows = np.asarray(rows)
        p = np.clip(out['p_low'].astype(np.float64), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)) + STATE.prior[rows] * (pool.ds[rows] == 'hoffman')
        out['p_low'] = 1.0 / (1.0 + np.exp(-z))
    return out


@torch.no_grad()
def predict_raw(net, pool, rows, norm, hp, device='cpu', bs=512):
    """Raw network outputs of `rows` (eval mode, no augmentation, baseline embedding from the rows' own recording): low-oxygen logit (the correction for R),
    HR (expectation decode and argmax), SQI probability, SpO2."""
    net.eval()
    rows = np.asarray(rows)
    E = net.cfg.emb
    table = T.group_emb(net, pool, np.unique(pool.grp[rows]), norm, hp, device, E) if hp.base_cond else {}
    res = {k: [] for k in ('logit', 'hr', 'hr_argmax', 'p_sqi', 'spo2')}
    for a in range(0, len(rows), bs):
        r = rows[a:a + bs]
        x, s, dom = pool.inputs(r, hp=hp)
        x, s = norm.apply(x, s, dom)
        eb = T._emb_for_rows(pool, r, table, E, hp)
        o = net(torch.from_numpy(x).to(device), torch.from_numpy(dom).to(device), torch.from_numpy(s).to(device), eb.to(device))
        res['logit'].append(o['lowox_logit'].float().cpu().numpy())
        res['hr'].append(T.hr_from_logits(o['hr_logits'].float(), 'expect').cpu().numpy())
        res['hr_argmax'].append(T.hr_from_logits(o['hr_logits'].float(), 'argmax').cpu().numpy())
        res['p_sqi'].append(torch.sigmoid(o['sqi_logit'].float()).cpu().numpy())
        res['spo2'].append(o['spo2'].float().cpu().numpy())
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in res.items()}


@contextlib.contextmanager
def install(family, prior=None, lam=LAM_DISTILL):
    """Context manager: train.fit uses the family loss and the prior-aware validation prediction inside the block; the originals are restored on exit."""
    assert family in FAMILIES, family
    if family != 'P':
        assert prior is not None, 'family R/D needs the chain prior'
    STATE.family, STATE.prior, STATE.lam = family, prior, lam
    T.batch_loss, T.predict = batch_loss_hybrid, predict_hybrid
    try:
        yield
    finally:
        T.batch_loss, T.predict = _ORIG['batch_loss'], _ORIG['predict']
        STATE.family, STATE.prior = 'P', None
