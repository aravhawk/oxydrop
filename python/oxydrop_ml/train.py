"""Multi-task training of OxyNet on pooled windows from Hoffman, MTHS, BUT PPG and OpenOximetry.

    python -m oxydrop_ml.train --help
    python -m oxydrop_ml.train --smoke                      # tiny end-to-end run on the cached shards (a minute on a laptop CPU)
    python -m oxydrop_ml.train --final --out ckpt/student_v1.pt   # fit on every training subject (no outer test subject), see loso_run.py for LOSO

Design (all selections use training subjects only; see PREREGISTRATION.md):
  * `Pool` concatenates the shard metadata of the datasets and gathers [B, 450, 3] log-RGB buffers on demand; the model input is computed by the causal
    reference `features.windows_to_tensor` (augmentation acts on the raw buffers, then features are recomputed).
  * Domain-aware sampler: each batch draws datasets with fixed shares (hoffman .30, mths .20, butppg .20, openox .30); Hoffman rows are drawn 50/50
    hypoxic / not; OpenOx rows come from a fixed subsample of people that are not in the held-out OpenOx set (`openox_holdout`).
  * Per-head masks: a loss term only sees windows that have that label. BUT PPG SpO2 is one spot value per person (weight 0.3). OpenOx has no phone
    SpO2: its ABG-anchored SaO2 trains the auxiliary head, its HR and low-oxygen labels train the shared heads (contact domain flag).
  * Input standardisation is a per-DOMAIN scale (phone, contact) fit on the training rows only (scale only, no mean shift, one scalar for the AC slots),
    so it folds exactly into the exported weights (`fold_norm`). The spec asked for per-dataset scaling; per-domain keeps training and the exported net identical.
  * Baseline-relative head: `emb_base` of a window = mean embedding of its recording's baseline windows (first minute, label-free). During training the
    table is refreshed every `eb_every` steps with the current network and treated as a constant.
  * Early stopping on an INNER validation split of training subjects (1 Hoffman, 6 MTHS, 4 BUT people); the best-validation weights are kept.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import DATA_DIR, WIN, CTX, DOMAIN
from . import features as FT
from . import losses as LS
from . import metrics as M
from .augment import AugCfg, augment_batch
from .model import OxyNet, OxyCfg, hr_from_logits, count_params, N_CH, SIDE_DIM
from .windows import WindowSet

PHONE = ('hoffman', 'mths', 'butppg')
DS_ID = {'hoffman': 0, 'mths': 1, 'butppg': 2, 'openox': 3}
SPOT_W = 0.3                       # BUT PPG: one spot SpO2 per person, not time aligned
BASE_ROWS = 8                      # baseline windows per recording used for emb_base


# ----------------------------------------------------------------------------------------------------------------- hyper-parameters
@dataclass
class HP:
    steps: int = 600
    bs: int = 128
    lr: float = 2e-3
    wd: float = 1e-4
    warm: float = 0.05
    dropout: float = 0.1
    sigma: float = 1.5
    w_hr: float = 1.0
    w_spo2: float = 1.0
    w_low: float = 1.5
    w_sqi: float = 0.3
    w_aux: float = 0.5
    w_low_ox: float = 0.5
    focal: float = 0.0
    p_ds: dict = field(default_factory=lambda: {'hoffman': 0.30, 'mths': 0.20, 'butppg': 0.20, 'openox': 0.30})
    p_pos_hof: float = 0.5
    aug: bool = True
    p_aug: float = 0.85
    adv: float = 0.0               # gradient-reversal weight (0 = off)
    use_openox: bool = True        # OpenOx rows in supervised training (HR, low, aux SaO2)
    init: str | None = None        # SSL checkpoint path
    drop_ch: tuple = ()            # input channels zeroed (ablations)
    drop_side: tuple = ()
    base_cond: bool = True         # baseline-relative conditioning (emb_base); False also zeroes the baseline-relative inputs
    eval_every: int = 100
    patience: int = 3
    eb_every: int = 100
    ox_cap: int = 30000
    win: int = 10                  # input window in seconds (10 = shipped; 20 = ablation: two consecutive 10 s feature windows concatenated, BUT clips excluded)
    cfg: dict = field(default_factory=dict)   # OxyCfg overrides

    def to_dict(self):
        d = asdict(self)
        return d


def variant_hp(name: str, **over) -> HP:
    """Named ablation variants. The final recipe is HP() with the values chosen by the nested selection (see loso_run.py / results/ml/nested_selection.csv)."""
    hp = HP(**over)
    if name == 'full':
        pass
    elif name == 'no_ssl':                           # keep the init passed in: stage B from scratch (no SSL), see loso_run.INIT
        pass
    elif name == 'no_pretrain':                      # exploratory: neither SSL nor the OpenOx warm-up; fine-tune from random init
        hp.init = None
    elif name == 'no_physics':                       # drop the ratio-of-ratios channels (4, 5), beat_ok (6) and their side features
        hp.drop_ch = (4, 5, 6); hp.drop_side = (0, 1, 8)
    elif name == 'no_openox':                        # no OpenOx in supervised training; the SSL checkpoint is the one pretrained without OpenOx
        hp.use_openox = False
    elif name == 'no_aug':
        hp.aug = False
    elif name == 'no_basecond':                      # no baseline-relative inputs or head conditioning
        hp.base_cond = False
        hp.drop_ch = (4, 5, 7, 8, 9); hp.drop_side = (0, 1, 2, 3, 4, 5, 6, 7)
    elif name == 'hr_only':
        hp.w_spo2 = hp.w_low = hp.w_aux = 0.0
    elif name == 'adv':
        hp.adv = 0.3
    elif name == 'win20':
        hp.win = 20
    else:
        raise ValueError(name)
    return hp


# ----------------------------------------------------------------------------------------------------------------- data pool
def openox_holdout(meta: pd.DataFrame) -> list:
    """Deterministic held-out OpenOx people (about 25 %, stratified by Monk skin-tone group): never used in SSL pretraining nor in supervised training,
    so FEAT-007/008 can evaluate OpenOx-specific tasks (skin tone, SaO2) on people the student never saw. Same list for every run."""
    p = meta.groupby('subject').agg(monk=('monk', 'first'), lab=('has_hr', 'max')).reset_index()
    p['monk'] = p.monk.fillna(-1)
    p = p.sort_values(['monk', 'subject']).reset_index(drop=True)
    return sorted(p.subject[np.arange(len(p)) % 4 == 3].tolist())


class Pool:
    """Concatenated metadata of the shards plus on-demand window gathering."""

    def __init__(self, datasets=('hoffman', 'mths', 'butppg', 'openox'), root=None, ox_cap=30000, ox_thin=3, seed=0, with_ox_all=False):
        root = root or DATA_DIR
        self.ws, frames = {}, []
        for ds in datasets:
            ws = WindowSet(ds, os.path.join(root, ds))
            self.ws[ds] = ws
            m = ws.meta.copy()
            m['local'] = np.arange(len(m))
            if ds == 'openox':
                self.ox_holdout = openox_holdout(m)
                if not with_ox_all:
                    keep = ~m.subject.isin(self.ox_holdout) & (m.has_hr | m.has_sao2 | m.has_spo2)
                    keep &= (m.t_start.values // 2 % ox_thin == 0)
                    m = m[keep]
                    if len(m) > ox_cap:
                        m = m.sample(ox_cap, random_state=seed).sort_index()
            frames.append(m)
            ws.meta = ws.meta[['end_idx', 'rec_start']].copy()      # the gather only needs these two columns; frees the full OpenOx metadata (RAM)
        m = pd.concat(frames, ignore_index=True)
        self.meta = m
        N = len(m)
        self.N = N
        self.ds = m.ds.values
        self.dsid = np.array([DS_ID[d] for d in m.ds.values])
        self.subject = m.subject.values.astype(str)
        self.domain = m.domain.values.astype(np.int64)
        self.local = m.local.values
        f = lambda c: m[c].values.astype(np.float32)
        self.hr, self.has_hr = f('hr_ref'), m.has_hr.values.astype(bool)
        self.spo2, self.has_spo2 = f('spo2_ref'), m.has_spo2.values.astype(bool)
        self.low, self.has_low = f('low'), m.has_low.values.astype(bool)
        self.sao2, self.has_sao2 = f('sao2'), m.has_sao2.values.astype(bool)
        self.t = f('t_start')
        self.eval = m['eval'].values.astype(bool)
        self.in_base = m.in_baseline.values.astype(bool)
        self.base = m[['base_' + b for b in FT.BASE]].values.astype(np.float32)
        self.mask = m[['m0', 'm1', 'm2']].values.astype(bool)
        # BUT PPG: spot SpO2 only for SpO2; no time-aligned low-oxygen label from it
        is_but, is_ox, is_hof, is_mths = self.ds == 'butppg', self.ds == 'openox', self.ds == 'hoffman', self.ds == 'mths'
        self.has_low = self.has_low & ~is_but
        self.has_spo2 = self.has_spo2 & ~is_ox                 # OpenOx SpO2 trains only the auxiliary SaO2 head
        self.w_spo2 = np.where(is_but, SPOT_W, 1.0).astype(np.float32)
        self.is_ox = is_ox
        # baseline groups
        key = (m.ds.astype(str) + '|' + m.base_key.astype(str)).values
        codes, uniq = pd.factorize(key)
        self.grp = codes.astype(np.int64)
        self.n_grp = len(uniq)
        self.base_rows = {}
        order = np.argsort(self.grp, kind='stable')
        g_sorted = self.grp[order]
        bounds = np.searchsorted(g_sorted, np.arange(self.n_grp + 1))
        for g in range(self.n_grp):
            rows = order[bounds[g]:bounds[g + 1]]
            rows = rows[self.in_base[rows]]
            if len(rows):
                rows = rows[np.argsort(self.t[rows], kind='stable')]
                self.base_rows[g] = rows[np.unique(np.linspace(0, len(rows) - 1, min(BASE_ROWS, len(rows))).round().astype(int))]
        self.ds_rows = {d: np.nonzero(self.ds == d)[0] for d in datasets}

    # ---- window gathering
    def buffers(self, rows, L=CTX + WIN):
        """[B, L, 3] log-RGB buffers and [B, L] dt ending at each window's end. L = 450 is the shard convention (5 s warm-up + 10 s); longer buffers extend
        into the past with the same odd reflection about the first sample of the recording."""
        rows = np.asarray(rows)
        lrgb = np.empty((len(rows), L, 3), np.float32)
        dt = np.empty((len(rows), L), np.float32)
        for d in np.unique(self.ds[rows]):
            sel = np.nonzero(self.ds[rows] == d)[0]
            li = self.local[rows[sel]]
            ws = self.ws[d]
            if L == CTX + WIN:
                lrgb[sel] = ws.lrgb(li)
                dt[sel] = ws.dt(li)
            else:
                e = ws.meta.end_idx.values[li]; r = ws.meta.rec_start.values[li]
                ii = e[:, None] - L + np.arange(L)[None, :]
                below = ii < r[:, None]
                src = np.where(below, 2 * r[:, None] - ii, ii)
                v = np.asarray(ws.series[src], np.float32)
                first = np.asarray(ws.series[r], np.float32)
                lrgb[sel] = np.where(below[..., None], 2 * first[:, None, :] - v, v)
                dt[sel] = np.asarray(ws.dtv[np.where(below, r[:, None], ii)], np.float32)
        return lrgb, dt

    def inputs(self, rows, aug=False, rng=None, acfg: AugCfg | None = None, p_aug=1.0, hp: HP | None = None):
        """x [B, 10, 300], s [B, 9] float32 (numpy), domain [B] int64. `aug`: augment each row with probability p_aug."""
        rows = np.asarray(rows)
        w20 = hp is not None and hp.win == 20
        lrgb, dt = self.buffers(rows, CTX + 2 * WIN if w20 else CTX + WIN)
        base = self.base[rows].copy()
        mask = self.mask[rows]
        if aug:
            sel = np.nonzero(rng.random(len(rows)) < p_aug)[0]
            if len(sel):
                l2, d2, b2 = augment_batch(lrgb[sel], base[sel], mask[sel], rng, acfg)
                lrgb[sel], dt[sel], base[sel] = l2, d2, b2
        x, s, _ = FT.windows_to_tensor(lrgb[:, -(CTX + WIN):], dt[:, -(CTX + WIN):], base, mask)
        if w20:
            x0, _, _ = FT.windows_to_tensor(lrgb[:, :CTX + WIN], dt[:, :CTX + WIN], base, mask)
            x = np.concatenate([x0, x], axis=2)
        if hp is not None:
            x, s = apply_drop(x, s, hp)
        return x, s, self.domain[rows]

    def subjects_of(self, ds):
        return sorted(np.unique(self.subject[self.ds == ds]).tolist())


def apply_drop(x, s, hp: HP):
    if hp.drop_ch:
        x[:, list(hp.drop_ch)] = 0.0
    if hp.drop_side:
        s[:, list(hp.drop_side)] = 0.0
    return x, s


# ----------------------------------------------------------------------------------------------------------------- input scale
class Norm:
    """Per-domain scale-only standardisation, fit on training rows only. x_scale [2, 10], s_scale [2, 9] (index = domain)."""

    def __init__(self, x_scale=None, s_scale=None):
        self.x_scale = np.ones((2, N_CH), np.float32) if x_scale is None else np.asarray(x_scale, np.float32)
        self.s_scale = np.ones((2, SIDE_DIM), np.float32) if s_scale is None else np.asarray(s_scale, np.float32)

    @staticmethod
    def fit(pool: Pool, rows, hp: HP, rng, n=2500):
        nm = Norm()
        for dom in (0, 1):
            r = rows[pool.domain[rows] == dom]
            if len(r) < 50:
                continue
            r = rng.choice(r, min(n, len(r)), replace=False)
            x, s, _ = pool.inputs(r, hp=hp)
            sd = x.std(axis=(0, 2))
            slots = [c for c in range(3) if (pool.mask[r][:, c]).any()]
            ac = float(np.sqrt(np.mean(sd[slots] ** 2))) if slots else 1.0
            sc = np.ones(N_CH, np.float32)
            sc[0:3] = max(ac, 1e-3)
            for c in (4, 5, 7, 8, 9):
                sc[c] = max(float(sd[c]), 0.02)
            ss = np.maximum(s.std(0), 0.05).astype(np.float32)
            ss[8] = 1.0
            nm.x_scale[dom], nm.s_scale[dom] = sc, ss
        return nm

    def apply(self, x, s, domain):
        d = np.asarray(domain)
        return x / self.x_scale[d][:, :, None], s / self.s_scale[d]

    def to_dict(self):
        return dict(x_scale=self.x_scale.tolist(), s_scale=self.s_scale.tolist())


def fold_norm(net: OxyNet, norm: Norm, domain=0) -> OxyNet:
    """Fold the scale of `domain` into the weights so the net consumes raw features (exact: scale-only, zero padding)."""
    net = copy.deepcopy(net)
    xs, ss = norm.x_scale[domain].astype(np.float64), norm.s_scale[domain].astype(np.float64)
    cfg = net.cfg
    with torch.no_grad():
        if cfg.stem == 'shared':
            net.slot_conv.weight /= float(xs[0])
            off = cfg.n_slots * cfg.slot_width
            for j, c in enumerate(range(cfg.n_slots, cfg.in_ch)):
                net.stem.conv.weight[:, off + j] /= float(xs[c])
        else:
            for c in range(cfg.in_ch):
                net.stem.conv.weight[:, c] /= float(xs[c])
        E = cfg.emb
        for lin in (net.spo2, net.lowox):
            lin.weight[:, E:] /= torch.tensor(ss, dtype=lin.weight.dtype)[None]
    return net


# ----------------------------------------------------------------------------------------------------------------- model helpers
def make_net(hp: HP, device='cpu') -> OxyNet:
    cfg = OxyCfg(dropout=hp.dropout, **hp.cfg)
    net = OxyNet(cfg)
    if hp.init:
        ck = torch.load(hp.init, map_location='cpu', weights_only=False)
        st = ck['state']
        own = net.state_dict()
        ok = {k: v for k, v in st.items() if k in own and own[k].shape == v.shape}
        net.load_state_dict(ok, strict=False)
    return net.to(device)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def group_emb(net, pool: Pool, groups, norm: Norm, hp: HP, device, E, max_rows=BASE_ROWS):
    """Mean embedding of the baseline windows of each group in `groups` -> dict g -> tensor [E]. Eval mode, no augmentation."""
    net.eval()
    rows, own = [], []
    for g in groups:
        r = pool.base_rows.get(int(g))
        if r is not None:
            if len(r) > max_rows:
                r = r[np.unique(np.linspace(0, len(r) - 1, max_rows).round().astype(int))]
            rows.append(r); own.append(np.full(len(r), int(g)))
    out = {}
    if not rows:
        return out
    rows = np.concatenate(rows); own = np.concatenate(own)
    embs = []
    for a in range(0, len(rows), 512):
        r = rows[a:a + 512]
        x, s, dom = pool.inputs(r, hp=hp)
        x, s = norm.apply(x, s, dom)
        o = net(torch.from_numpy(x).to(device), torch.from_numpy(dom).to(device), torch.from_numpy(s).to(device), None)
        embs.append(o['emb'].float().cpu())
    embs = torch.cat(embs)
    for g in np.unique(own):
        out[int(g)] = embs[own == g].mean(0)
    return out


def _emb_for_rows(pool, rows, table, E, hp):
    eb = torch.zeros(len(rows), E)
    if not hp.base_cond:
        return eb
    for i, g in enumerate(pool.grp[rows]):
        v = table.get(int(g))
        if v is not None:
            eb[i] = v
    return eb


@torch.no_grad()
def predict(net, pool: Pool, rows, norm: Norm, hp: HP, device='cpu', bs=512):
    """Deterministic forward (eval mode, no augmentation) of `rows`; the baseline embedding comes from the rows' own recording baseline windows."""
    net.eval()
    rows = np.asarray(rows)
    E = net.cfg.emb
    table = group_emb(net, pool, np.unique(pool.grp[rows]), norm, hp, device, E) if hp.base_cond else {}
    res = {k: [] for k in ('hr', 'hr_argmax', 'spo2', 'logvar', 'p_low', 'p_sqi', 'aux')}
    for a in range(0, len(rows), bs):
        r = rows[a:a + bs]
        x, s, dom = pool.inputs(r, hp=hp)
        x, s = norm.apply(x, s, dom)
        eb = _emb_for_rows(pool, r, table, E, hp)
        o = net(torch.from_numpy(x).to(device), torch.from_numpy(dom).to(device), torch.from_numpy(s).to(device), eb.to(device), with_aux=True)
        res['hr'].append(hr_from_logits(o['hr_logits'].float(), 'expect').cpu().numpy())
        res['hr_argmax'].append(hr_from_logits(o['hr_logits'].float(), 'argmax').cpu().numpy())
        res['spo2'].append(o['spo2'].float().cpu().numpy()); res['logvar'].append(o['spo2_logvar'].float().cpu().numpy())
        res['p_low'].append(torch.sigmoid(o['lowox_logit'].float()).cpu().numpy()); res['p_sqi'].append(torch.sigmoid(o['sqi_logit'].float()).cpu().numpy())
        res['aux'].append(o['aux_spo2'].float().cpu().numpy())
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in res.items()}


# ----------------------------------------------------------------------------------------------------------------- sampling and loss
class Sampler:
    def __init__(self, pool: Pool, rows, hp: HP, rng):
        self.pool, self.hp, self.rng = pool, hp, rng
        self.by = {}
        for d in pool.ds_rows:
            r = rows[pool.ds[rows] == d]
            if (d == 'openox' and not hp.use_openox) or (d == 'butppg' and hp.win == 20):
                r = r[:0]
            if len(r):
                self.by[d] = r
        w = {d: hp.p_ds.get(d, 0.0) for d in self.by}
        z = sum(w.values())
        self.p = {d: w[d] / z for d in w}
        self.names = list(self.by)
        self.probs = np.array([self.p[d] for d in self.names])
        if 'hoffman' in self.by:
            h = self.by['hoffman']
            self.hof_pos = h[(pool.low[h] > 0.5) & pool.has_low[h]]
            self.hof_neg = h[(pool.low[h] <= 0.5) | ~pool.has_low[h]]

    def draw(self, bs):
        n = self.rng.multinomial(bs, self.probs)
        out = []
        for d, k in zip(self.names, n):
            if k == 0:
                continue
            if d == 'hoffman' and len(self.hof_pos) and len(self.hof_neg):
                kp = int(round(k * self.hp.p_pos_hof))
                out.append(self.rng.choice(self.hof_pos, kp)); out.append(self.rng.choice(self.hof_neg, k - kp))
            else:
                out.append(self.rng.choice(self.by[d], k))
        return np.concatenate(out)

    def pos_weight(self):
        r = np.concatenate([self.by[d] for d in self.names if d != 'openox'] or [np.zeros(0, int)])
        r = r[self.pool.has_low[r]]
        if not len(r):
            return 1.0
        # expected positive rate under the sampler: hoffman is drawn p_pos_hof positive; the others by their label rate
        pr, tot = 0.0, 0.0
        for d in self.names:
            if d == 'openox':
                continue
            q = self.p[d]
            rr = self.by[d][self.pool.has_low[self.by[d]]]
            if not len(rr):
                continue
            rate = self.hp.p_pos_hof if d == 'hoffman' else float(self.pool.low[rr].mean())
            pr += q * rate; tot += q
        rate = max(pr / max(tot, 1e-9), 1e-3)
        return float(np.clip((1 - rate) / rate, 0.5, 8.0))


def batch_loss(net, pool: Pool, rows, x, s, dom, eb, hp: HP, pos_w, adv=None, lam=0.0, ds_t=None, device='cpu'):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    o = net(x, dom, s, eb, with_aux=True)
    m = lambda a: t(a, torch.bool)
    hr, spo2, low, sao2 = t(pool.hr[rows]), t(pool.spo2[rows]), t(pool.low[rows]), t(pool.sao2[rows])
    w_low = t(np.where(pool.is_ox[rows], hp.w_low_ox, 1.0))
    L = {}
    L['hr'] = LS.hr_soft_ce(o['hr_logits'].float(), hr, m(pool.has_hr[rows]), hp.sigma)
    L['spo2'] = LS.spo2_loss(o['spo2'].float(), o['spo2_logvar'].float(), spo2, m(pool.has_spo2[rows]), t(pool.w_spo2[rows]))
    L['low'] = LS.low_loss(o['lowox_logit'].float(), low, m(pool.has_low[rows]), w_low, pos_w, hp.focal)
    L['sqi'] = LS.sqi_loss(o['sqi_logit'].float(), o['hr_logits'].float(), hr, m(pool.has_hr[rows]))
    L['aux'] = LS.aux_loss(o['aux_spo2'].float(), sao2, m(pool.has_sao2[rows]))
    tot = hp.w_hr * L['hr'] + hp.w_spo2 * L['spo2'] + hp.w_low * L['low'] + hp.w_sqi * L['sqi'] + hp.w_aux * L['aux']
    if adv is not None and hp.adv > 0:
        logit = adv(o['emb'].float(), lam)
        L['adv'] = torch.nn.functional.cross_entropy(logit, ds_t)
        tot = tot + hp.adv * L['adv']
    return tot, L


# ----------------------------------------------------------------------------------------------------------------- fit
@dataclass
class FitResult:
    net: OxyNet
    norm: Norm
    hp: HP
    history: list
    best_step: int
    train_rows: np.ndarray
    seconds: float


def fit(pool: Pool, train_rows, val_rows, hp: HP, seed=0, device='cpu', log=None) -> FitResult:
    t0 = time.time()
    seed_all(seed)
    rng = np.random.default_rng(seed)
    train_rows = np.asarray(train_rows)
    if hp.win == 20:                                   # BUT clips are 10 s long: no 20 s window exists
        train_rows = train_rows[pool.ds[train_rows] != 'butppg']
        val_rows = None if val_rows is None else np.asarray(val_rows)[pool.ds[np.asarray(val_rows)] != 'butppg']
    net = make_net(hp, device)
    norm = Norm.fit(pool, train_rows, hp, rng)
    sam = Sampler(pool, train_rows, hp, rng)
    pos_w = sam.pos_weight()
    adv = LS.AdvHead(net.cfg.emb, len(DS_ID)).to(device) if hp.adv > 0 else None
    params = list(net.parameters()) + (list(adv.parameters()) if adv is not None else [])
    opt = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.wd)
    warm = max(1, int(hp.warm * hp.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, hp.steps - warm))) * 0.98 + 0.02)
    acfg = AugCfg()
    use_amp = device != 'cpu'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    E = net.cfg.emb
    train_groups = np.unique(pool.grp[train_rows])
    table = {}
    hist, best, best_state, best_step, bad = [], np.inf, None, 0, 0
    val_rows = np.asarray(val_rows) if val_rows is not None else np.zeros(0, int)
    if len(val_rows) > 800:
        val_rows = np.random.default_rng(1).choice(val_rows, 800, replace=False)
    for step in range(hp.steps):
        if hp.base_cond and (step in (min(25, hp.steps - 1), min(60, hp.steps - 1)) or (step > 0 and step % hp.eb_every == 0)):
            table = group_emb(net, pool, train_groups, norm, hp, device, E, max_rows=4)
        net.train()
        rows = sam.draw(hp.bs)
        x, s, dom = pool.inputs(rows, aug=hp.aug, rng=rng, acfg=acfg, p_aug=hp.p_aug, hp=hp)
        x, s = norm.apply(x, s, dom)
        eb = _emb_for_rows(pool, rows, table, E, hp)
        xt, st, dt_, eb = torch.from_numpy(x).to(device), torch.from_numpy(s).to(device), torch.from_numpy(dom).to(device), eb.to(device)
        ds_t = torch.from_numpy(pool.dsid[rows]).to(device)
        lam = hp.adv * min(1.0, step / max(1, 0.3 * hp.steps)) if adv is not None else 0.0
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            loss, L = batch_loss(net, pool, rows, xt, st, dt_, eb, hp, pos_w, adv, lam, ds_t, device)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % hp.eval_every == 0 or step == hp.steps - 1:
            rec = dict(step=step + 1, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in L.items()})
            if len(val_rows):
                rec['val'] = val_score(net, pool, val_rows, norm, hp, device, pos_w)
                if rec['val'] < best - 1e-4:
                    best, best_state, best_step, bad = rec['val'], copy.deepcopy(net.state_dict()), step + 1, 0
                else:
                    bad += 1
            hist.append(rec)
            if log:
                log(f"  step {step + 1}/{hp.steps} loss {rec['loss']:.3f} hr {rec['hr']:.3f} spo2 {rec['spo2']:.3f} low {rec['low']:.3f}" + (f" val {rec['val']:.3f}" if 'val' in rec else ''))
            if len(val_rows) and bad >= hp.patience and step + 1 >= 0.5 * hp.steps:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        best_step = hp.steps
    return FitResult(net, norm, hp, hist, best_step, train_rows, time.time() - t0)


@torch.no_grad()
def val_score(net, pool, rows, norm, hp, device, pos_w):
    """Inner-validation loss (lower is better): HR CE + SpO2 Huber (scaled) + low-oxygen BCE, unweighted by the training task weights, no augmentation."""
    out = predict(net, pool, rows, norm, hp, device)
    # recompute heads' logits is expensive; use the decoded outputs instead: HR within-5 miss rate, SpO2 MAE/5, low-oxygen Brier
    s = 0.0
    k = 0
    h = pool.has_hr[rows]
    if h.any():
        s += float((np.abs(out['hr'][h] - pool.hr[rows][h]) > 5).mean()); k += 1
    sp = pool.has_spo2[rows]
    if sp.any() and hp.w_spo2 > 0:
        s += float(np.abs(out['spo2'][sp] - pool.spo2[rows][sp]).mean() / 5.0); k += 1
    lo = pool.has_low[rows]
    if lo.any() and hp.w_low > 0:
        y = pool.low[rows][lo]
        s += float(np.mean((out['p_low'][lo] - y) ** 2) * 2.0); k += 1
    return s / max(k, 1)


# ----------------------------------------------------------------------------------------------------------------- checkpoints and fold metrics
def save_ckpt(path, fr: FitResult, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'cfg': fr.net.cfg.to_dict(), 'state': fr.net.state_dict(), 'norm': fr.norm.to_dict(), 'hp': fr.hp.to_dict(), 'best_step': fr.best_step, **(extra or {})}, path)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def choose_val_subjects(pool: Pool, exclude, seed, n=dict(hoffman=1, mths=6, butppg=4)):
    """Inner validation people drawn from the training subjects only (seeded)."""
    rng = np.random.default_rng(seed)
    val = []
    for d, k in n.items():
        subs = [s for s in pool.subjects_of(d) if s not in exclude]
        if len(subs) > k:
            val += rng.choice(subs, k, replace=False).tolist()
    return val


def fold_rows(pool: Pool, test_subjects, seed, val=True):
    ts = set(test_subjects)
    in_test = np.isin(pool.subject, list(ts))
    vs = choose_val_subjects(pool, ts, seed) if val else []
    in_val = np.isin(pool.subject, vs)
    tr = np.nonzero(~in_test & ~in_val)[0]
    va = np.nonzero(in_val)[0]
    return tr, va, vs


def train_scores(pool, fr: FitResult, device='cpu', cap=1500, seed=0):
    """Same three metrics on the model's own TRAINING rows (the train-vs-test gap). P1: time-matched balanced accuracy on training Hoffman people;
    P2: HR within 5 bpm (all labelled windows) on training MTHS+BUT rows; P3: SpO2 MAE on training MTHS rows."""
    rng = np.random.default_rng(seed)
    tr = fr.train_rows
    out = {}
    h = tr[(pool.ds[tr] == 'hoffman') & pool.eval[tr] & pool.has_low[tr]]
    if len(h):
        h = rng.choice(h, min(cap, len(h)), replace=False)
        o = predict(fr.net, pool, h, fr.norm, fr.hp, device)
        tm = M.time_matched(pool.t[h], pool.low[h], (o['p_low'] > 0.5).astype(int), o['p_low'])
        out['p1'] = tm['tm_bacc']
    q = tr[np.isin(pool.ds[tr], ('mths',) if fr.hp.win == 20 else ('mths', 'butppg')) & pool.has_hr[tr]]
    if len(q):
        q = rng.choice(q, min(cap, len(q)), replace=False)
        o = predict(fr.net, pool, q, fr.norm, fr.hp, device)
        out['p2'] = M.hr_metrics(o['hr'], pool.hr[q])['within5_full']
    q = tr[(pool.ds[tr] == 'mths') & pool.has_spo2[tr] & pool.eval[tr]]          # same t >= 30 s windows as the held-out P3 score
    if len(q):
        q = rng.choice(q, min(cap, len(q)), replace=False)
        o = predict(fr.net, pool, q, fr.norm, fr.hp, device)
        out['p3'] = M.spo2_metrics(o['spo2'], pool.spo2[q])['mae']
        out['p3_eval'] = True
    return out


def scores_frame(pool, rows, out, extra=None):
    d = pd.DataFrame(dict(ds=pool.ds[rows], subject=pool.subject[rows], rec=pool.meta.rec.values[rows], t_start=pool.t[rows], eval=pool.eval[rows],
                          hr_ref=pool.hr[rows], spo2_ref=pool.spo2[rows], low=pool.low[rows], hr_pred=out['hr'], hr_argmax=out['hr_argmax'],
                          spo2_pred=out['spo2'], spo2_logvar=out['logvar'], p_low=out['p_low'], p_sqi=out['p_sqi']))
    for k, v in (extra or {}).items():
        d[k] = v
    return d


def stage_b(pool: Pool, steps=2000, init=None, seed=0, device='cpu', log=None, hp_over=None):
    """Supervised warm-up on OpenOx only (contact PPG): HR (170k labelled windows), low oxygen and the ABG-anchored SaO2 auxiliary head, from the SSL checkpoint
    (or from scratch for the no-SSL ablation). OpenOx people are never outer test subjects of P1/P2/P3 and the held-out OpenOx people are not in the pool,
    so one shared checkpoint is legitimate for every fold. Returns a FitResult."""
    kw = dict(steps=steps, init=init, p_ds={'openox': 1.0}, w_spo2=0.0, w_low_ox=1.0, w_low=1.0, lr=2e-3, eval_every=max(100, steps // 8))
    kw.update(hp_over or {})
    hp = HP(**kw)
    rows = np.nonzero(pool.is_ox)[0]
    assert not set(pool.subject[rows]) & set(pool.ox_holdout)
    return fit(pool, rows, None, hp, seed, device, log=log)


# ----------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description='Train OxyNet (multi-task) on pooled windows. See the module docstring; LOSO runs are in loso_run.py.')
    ap.add_argument('--smoke', action='store_true', help='tiny end-to-end run (few steps, small OpenOx subsample)')
    ap.add_argument('--stageb', action='store_true', help='OpenOx supervised warm-up (stage B) from --init (or scratch), saved to --out')
    ap.add_argument('--final', action='store_true', help='fit on every training subject (no outer test subject) and save a checkpoint')
    ap.add_argument('--variant', default='full')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default=None, help='SSL checkpoint (ssl_pretrain.py)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--data', default=None, help='shard root (default OXY_CACHE/data)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hp', default=None, help='JSON dict of HP overrides')
    ap.add_argument('--recipe', default=None, help='JSON file with the nested-selection recipe (HP overrides)')
    a = ap.parse_args(argv)
    over = json.load(open(a.recipe)) if a.recipe else {}
    over.update(json.loads(a.hp) if a.hp else {})
    if a.steps:
        over['steps'] = a.steps
    if a.init:
        over['init'] = a.init
    if a.smoke:
        over.setdefault('steps', 30); over.setdefault('eval_every', 10); over.setdefault('eb_every', 10); over.setdefault('bs', 64)
    hp = variant_hp(a.variant, **over)
    pool = Pool(root=a.data, ox_cap=3000 if a.smoke else hp.ox_cap, seed=a.seed)
    if a.stageb:
        fr = stage_b(pool, a.steps or 2000, a.init, a.seed, a.device, log=lambda s: print(s, flush=True))
        print('stage B done', f'{fr.seconds:.0f}s', 'saved', a.out, save_ckpt(a.out, fr, {'stage': 'B', 'ssl_init': a.init}))
        return
    print('pool rows', pool.N, {d: len(r) for d, r in pool.ds_rows.items()}, flush=True)
    test = ['hof_100001'] if (a.smoke or not a.final) else []
    tr, va, vs = fold_rows(pool, test, a.seed)
    fr = fit(pool, tr, va, hp, a.seed, a.device, log=lambda s: print(s, flush=True))
    print('params', count_params(fr.net), 'best_step', fr.best_step, f'{fr.seconds:.0f}s', flush=True)
    if test:
        rows = np.nonzero(np.isin(pool.subject, test) & pool.eval)[0]
        o = predict(fr.net, pool, rows, fr.norm, fr.hp, a.device)
        tm = M.time_matched(pool.t[rows], pool.low[rows], (o['p_low'] > 0.5).astype(int), o['p_low'])
        print('held-out', test, 'TM bacc', tm['tm_bacc'], 'HR within5', M.hr_metrics(o['hr'], pool.hr[rows])['within5_full'], 'SpO2 MAE', M.spo2_metrics(o['spo2'], pool.spo2[rows])['mae'])
    if a.out and a.final:
        # the exported student consumes RAW features: fold the phone-domain input scale into the weights (exact, see fold_norm and its test)
        folded = fold_norm(fr.net, fr.norm, 0)
        class _F:
            pass
        ff = _F(); ff.net, ff.norm, ff.hp, ff.best_step = folded, fr.norm, fr.hp, fr.best_step
        print('saved', a.out, save_ckpt(a.out, ff, {'folded_norm_domain': 0, 'val_subjects': vs, 'params_shipped': count_params(folded, include_aux=False), 'seed': a.seed}))
    elif a.out:
        print('saved', a.out, save_ckpt(a.out, fr))


if __name__ == '__main__':
    main()
