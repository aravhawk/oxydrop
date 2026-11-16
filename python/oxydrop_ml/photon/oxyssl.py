"""OxySSL (round-2 candidate R2C, FEAT-010): the SSL-pretrained OxyNet (ml_cache/ckpt/stageB_full.pt = SimCLR pretraining ssl_full.pt + OpenOx
supervised warm-up) fine-tuned leave-one-person-out on Hoffman with its own multi-task heads (hypoxia, SpO2 + log-variance, HR, quality/abstain)
and a distant-flash augmentation. The deployed model is ONE OxyNet (same architecture as app/js/cnn.js runs); see oxyssl_export.py.

    cd python && python -m oxydrop_ml.photon.oxyssl prep                    # encoder features of every recording version (clean, aug copies, L1..L5, X1)
    cd python && python -m oxydrop_ml.photon.oxyssl fit --seeds 0 --nulls 3 # nested LOSO fits, score files
    cd python && python -m oxydrop_ml.photon.oxyssl ensemble --seeds 0 1 2  # oxyssl__* = ensemble of the seeds on disk
    cd python && python -m oxydrop_ml.photon.oxyssl init                    # diagnostic oxyssl_init__*: pretrained net, no Hoffman training

Fine-tuning scope (compute disclosure). The Daytona free tier was full when this ran (two other sandboxes held 10/10 GiB memory and 8/10 vCPU), so
everything ran on the shared local box under a 600 MB / 2-thread cap. The convolutional encoder (OxyNet.encode: shared slot conv, stem, 8 residual
blocks, global avg+max pool, 160 features) is therefore FROZEN at its pretrained weights (eval-mode BatchNorm); everything after it is trained
end-to-end per fold: fc_emb (160 -> 64), the HR head (64 -> 64 -> 141), the SpO2 head (mean, log-variance), the hypoxia head on
(emb - emb_base, side) and the quality head. All heads start from the pretrained weights. The exported network is the full OxyNet.

Inputs: features.windows_to_tensor of each 10 s window (causal, relative to the label-free first-minute personal baseline, recomputed on the
degraded/augmented recording itself, never on clean data), domain 0 (phone). emb_base = mean embedding of the recording's first 8 baseline windows.

Protocol (ROUND2.md):
  * LOSO by person (6 outer folds, both hands together). Per outer fold, INNER leave-one-person-out over the 5 training people selects the learning
    rate (LR_GRID) and the number of epochs (EPOCH_GRID checkpoints of one run) on the inner out-of-fold objective
    0.5 * TM bal. acc.(clean inner OOF) + 0.5 * TM bal. acc.(augmented copies of the inner held-out people), and the abstain threshold on
    sigmoid(quality head): the inner-OOF quantile q in Q_GRID (coverage >= 70 %) with the highest answered bal. acc. (ties: higher coverage).
    The held-out person is never used for any choice. pred = score > 0.5 (balanced class weight in training).
  * Training data: clean training people + K_AUG distant-flash copies per training recording (flashgap.sample_params / optics_gain / flash_augment:
    the FlashGap continuous physics prior, own code, no stress.degrade call) drawn from an OxySSL-only random stream
    SeedSequence([AUG_TAG, aug_seed, rec_idx, k]) with AUG_TAG = 'OXSS' >= every evaluation seed; X1 code is never imported for training.
  * Test: the held-out person's recordings degraded by stress.degrade(level, seed=0) / degrade_x('X1', 0) (whole recording), not refit per level.
  * Quality head target: whether the hypoxia head's current prediction of that training window is correct (detached), so abstention is the
    network's own output.
  * Null: labels (low, SpO2, HR jointly) permuted globally across the training windows of each fold (3 draws), same nested pipeline.
  * Pretraining corpora are disjoint from Hoffman (assert_pretrain_disjoint).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from torch import nn

from .. import CACHE, DATA_DIR
from .. import features as FE
from .. import metrics as M
from ..model import OxyNet, OxyCfg, HR_LO, HR_BINS, SPO2_CENTER, SPO2_SCALE

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
CK_INIT = Path(os.environ.get('OXYSSL_INIT', os.path.join(CACHE, 'ckpt', 'stageB_full.pt')))
CACHE_DIR = Path(os.environ.get('OXYSSL_CACHE', os.path.join(CACHE, 'photon_oxyssl')))
NAME = 'oxyssl'

AUG_TAG = 0x4F585353            # 'OXSS'
EVAL_SEED_MAX = 1_000_000
assert AUG_TAG >= EVAL_SEED_MAX
K_AUG = 3
AUG_SEED = 0
LEVELS = [0, 1, 2, 3, 4, 5]
EVAL = LEVELS + ['X1']
LR_GRID = (3e-3, 1e-3)
EPOCH_GRID = (2, 4, 8, 16)
Q_GRID = (0.0, 0.1, 0.2, 0.3)
BS = 512
WD = 1e-4
BASE_ROWS = 8
W_LOW, W_SPO2, W_HR, W_SQI = 1.0, 0.5, 0.3, 0.3


def vname(v):
    return v if isinstance(v, str) else f'L{v}'


def set_threads(n=2):
    torch.set_num_threads(n)


# ------------------------------------------------------------------------------------------------ data
def load_init():
    ck = torch.load(CK_INIT, map_location='cpu', weights_only=False)
    cfg = {k: (tuple(v) if k in ('widths', 'depths') else v) for k, v in ck['cfg'].items()}
    net = OxyNet(OxyCfg(**cfg))
    net.load_state_dict(ck['state'])
    net.eval()
    return net, ck


def assert_pretrain_disjoint(ck=None):
    """The SSL corpus (data/pretrain) and the stage-B pool (OpenOx only) contain no Hoffman person."""
    hof = set(pd.read_parquet(os.path.join(DATA_DIR, 'hoffman', 'meta.parquet'), columns=['subject']).subject.astype(str))
    pm = pd.read_parquet(os.path.join(DATA_DIR, 'pretrain', 'meta.parquet'))
    cols = [c for c in ('subject', 'ds', 'source') if c in pm.columns]
    for c in cols:
        vals = set(pm[c].astype(str))
        assert not (vals & hof), f'pretraining {c} overlaps Hoffman people'
        assert not any('hof' in v.lower() or 'hoffman' in v.lower() for v in vals), f'pretraining {c} names a Hoffman source'
    if ck is None:
        ck = torch.load(CK_INIT, map_location='cpu', weights_only=False)
    assert ck.get('stage') == 'B' and ck['hp']['p_ds'] == {'openox': 1.0}, 'stage B must be OpenOx-only'
    om = pd.read_parquet(os.path.join(DATA_DIR, 'openox', 'meta.parquet'), columns=['subject'])
    assert not (set(om.subject.astype(str)) & hof)
    return dict(pretrain_rows=len(pm), pretrain_cols=cols, hoffman_people=len(hof))


def shard():
    d = os.path.join(DATA_DIR, 'hoffman')
    meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    recs = pd.read_parquet(os.path.join(d, 'recs.parquet'))
    n = json.load(open(os.path.join(d, 'info.json')))['n_samples']
    series = np.memmap(os.path.join(d, 'series.f32'), dtype=np.float32, mode='r', shape=(n, 3))
    dtv = np.memmap(os.path.join(d, 'dt.f16'), dtype=np.float16, mode='r', shape=(n,))
    return meta, recs, series, dtv


def aug_rng(aug_seed, rec_idx, k):
    return np.random.default_rng(np.random.SeedSequence([AUG_TAG, int(aug_seed), int(rec_idx), int(k)]))


def aug_draw(aug_seed, rec_idx, k):
    from . import flashgap as FG
    rng = aug_rng(aug_seed, rec_idx, k)
    p = FG.sample_params(rng)
    a, m = FG.optics_gain(p)
    p['a_c'] = [float(v) for v in a]; p['m_c'] = [float(v) for v in m]
    return p, rng


def assert_aug_disjoint(params):
    from . import stress as S
    lv = {(float(S.D_RHO_MM[l]), float(S.AMBIENT[l])) for l in S.LEVELS if l != 0}
    for p in params:
        assert (float(p['d_rho_mm']), float(p['leak'])) not in lv
    return True


def version_rgb(rgb, version, rec_idx):
    """Linear RGB of one recording version: 'clean', 'aug<k>' (training prior), 'L<l>' / 'X1' (evaluation stress)."""
    if version == 'clean' or version == 'L0':
        return rgb, None
    if version.startswith('aug'):
        from . import flashgap as FG
        p, rng = aug_draw(AUG_SEED, rec_idx, int(version[3:]))
        return FG.flash_augment(rgb, p, rng), p
    from . import stress as S
    if version == 'X1':
        return S.degrade_x(rgb, 'X1', 0), None
    return S.degrade(rgb, int(version[1:]), 0), None


def gather(lrgb, dt, ends):
    ii = ends[:, None] - 450 + np.arange(450)[None]
    below = ii < 0
    src = np.where(below, -ii, ii)
    v = lrgb[src]
    v = np.where(below[..., None], 2 * lrgb[0][None, None] - v, v)
    d = dt[np.where(below, 0, ii)]
    return v, d


@torch.no_grad()
def encode_rec(net, lrgb, dt, ends, in_base, chunk=256):
    buf, dtb = gather(lrgb, dt, ends)
    _, _, absf = FE.windows_to_tensor(buf, dtb, None, None)
    base = FE.baseline_from_abs(absf[in_base])
    x, s, _ = FE.windows_to_tensor(buf, dtb, np.repeat(base[None], len(ends), 0), None)
    P = []
    for a in range(0, len(x), chunk):
        P.append(net.encode(torch.from_numpy(x[a:a + chunk])).numpy())
    return np.concatenate(P).astype(np.float32), s.astype(np.float32), bool(np.isfinite(base[:8]).all())


def versions():
    return ['clean'] + [f'aug{k}' for k in range(K_AUG)] + [vname(l) for l in EVAL if l != 0]


def prep(vers=None, log=print):
    set_threads(2)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    net, ck = load_init()
    log('disjoint:', assert_pretrain_disjoint(ck))
    meta, recs, series, dtv = shard()
    params = []
    for v in (vers or versions()):
        out = CACHE_DIR / f'{v}.npz'
        if out.exists():
            continue
        t0 = time.time()
        P = np.zeros((len(meta), 160), np.float32); S_ = np.zeros((len(meta), 9), np.float32); ok = np.zeros(len(recs), bool)
        for ri, r in recs.iterrows():
            rows = np.nonzero(meta.rec.values == r.rec)[0]
            off, ln = int(r.offset), int(r.length)
            assert (meta.rec_start.values[rows] == off).all() and (meta.end_idx.values[rows] <= off + ln).all()
            rgb = np.expm1(np.asarray(series[off:off + ln], np.float64))
            rgbv, p = version_rgb(rgb, v, ri)
            if p is not None:
                params.append(p)
            lrgb = FE.log_rgb(rgbv)
            dt = np.asarray(dtv[off:off + ln], np.float64)
            ends = meta.end_idx.values[rows] - off
            ib = meta.in_baseline.values[rows].astype(bool)
            P[rows], S_[rows], ok[ri] = encode_rec(net, lrgb, dt, ends, ib)
        np.savez(out, P=P, S=S_, base_ok=ok)
        log(f'prep {v}: {time.time() - t0:.0f}s, baseline ok {ok.sum()}/{len(ok)}, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB')
    if params:
        assert_aug_disjoint(params)
        json.dump(params, open(CACHE_DIR / f'aug_params_seed{AUG_SEED}.json', 'w'))


def load_bank():
    d = {v: np.load(CACHE_DIR / f'{v}.npz') for v in versions()}
    return {v: (z['P'], z['S']) for v, z in d.items()}


# ------------------------------------------------------------------------------------------------ the trainable part of OxyNet
class Top(nn.Module):
    """Everything of OxyNet after encode(): initialised from the pretrained net."""

    def __init__(self, net: OxyNet):
        super().__init__()
        import copy
        self.fc_emb = copy.deepcopy(net.fc_emb); self.hr1 = copy.deepcopy(net.hr1); self.hr2 = copy.deepcopy(net.hr2)
        self.spo2 = copy.deepcopy(net.spo2); self.lowox = copy.deepcopy(net.lowox); self.sqi = copy.deepcopy(net.sqi)
        self.drop = nn.Dropout(net.cfg.dropout)

    def emb(self, p):
        return TF.relu(self.fc_emb(self.drop(p)))

    def forward(self, p, s, eb):
        e = self.emb(p)
        hr = self.hr2(self.drop(TF.relu(self.hr1(e))))
        so = self.spo2(torch.cat([e, s], 1))
        lo = self.lowox(torch.cat([e - eb, s], 1))
        return dict(hr_logits=hr, spo2=SPO2_CENTER + SPO2_SCALE * so[:, 0], spo2_logvar=so[:, 1], lowox_logit=lo[:, 0], sqi_logit=self.sqi(e)[:, 0])


class Data:
    """Rows = (version, window). Holds P, S (side scaled), labels, group ids for emb_base."""

    def __init__(self, bank, meta, vers, rows, labels, s_scale):
        self.P = np.concatenate([bank[v][0][rows] for v in vers]); self.S = np.concatenate([bank[v][1][rows] for v in vers]) / s_scale
        rec = meta.rec.values
        self.grp_keys = [(v, r) for v in vers for r in pd.unique(rec[rows])]
        gid = {k: i for i, k in enumerate(self.grp_keys)}
        self.g = np.concatenate([[gid[(v, r)] for r in rec[rows]] for v in vers]).astype(np.int64)
        self.is_clean = np.concatenate([np.full(len(rows), v in ('clean', 'L0')) for v in vers])
        self.idx = np.concatenate([rows for _ in vers])
        L = labels
        self.y = L['low'][self.idx]; self.spo2 = L['spo2'][self.idx]; self.hsp = L['has_spo2'][self.idx]
        self.hr = L['hr'][self.idx]; self.hhr = L['has_hr'][self.idx]
        # baseline windows per group (label-free, first minute)
        GB = []
        inb = meta.in_baseline.values.astype(bool)
        for v, r in self.grp_keys:
            br = np.nonzero((rec == r) & inb)[0][:BASE_ROWS]
            GB.append(bank[v][0][br])
        self.GB = torch.from_numpy(np.stack(GB).astype(np.float32))           # [G, 8, 160]
        self.Pt = torch.from_numpy(self.P); self.St = torch.from_numpy(self.S.astype(np.float32))

    def eb(self, top, g):
        G = self.GB.shape
        e = top.emb(self.GB.reshape(-1, G[-1])).reshape(G[0], G[1], -1).mean(1)
        return e[torch.as_tensor(g)]


def labels_of(meta, perm=None):
    L = dict(low=meta.low.values.astype(np.float32), spo2=meta.spo2_ref.values.astype(np.float32), has_spo2=meta.has_spo2.values.astype(bool),
             hr=meta.hr_ref.values.astype(np.float32), has_hr=meta.has_hr.values.astype(bool))
    if perm is not None:                                  # perm: dict src_rows -> permuted rows (null control)
        src, dst = perm
        for k in L:
            v = L[k].copy(); v[src] = L[k][dst]; L[k] = v
    return L


def hr_target(hr):
    b = np.clip(np.round(hr - HR_LO), 0, HR_BINS - 1)
    ar = np.arange(HR_BINS)[None]
    t = np.exp(-0.5 * ((ar - b[:, None]) / 1.5) ** 2)
    return t / t.sum(1, keepdims=True)


def train(net, D: Data, lr, epochs_ckpt, seed, on_ckpt=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    top = Top(net)
    opt = torch.optim.AdamW(top.parameters(), lr=lr, weight_decay=WD)
    n = len(D.y)
    npos = max(D.y.sum(), 1.0); pw = torch.tensor((n - npos) / npos, dtype=torch.float32)
    E = max(epochs_ckpt)
    steps_ep = int(np.ceil(n / BS)); total = E * steps_ep; step = 0
    y_all = torch.from_numpy(D.y); sp_all = torch.from_numpy(D.spo2); hsp_all = torch.from_numpy(D.hsp)
    hhr_all = torch.from_numpy(D.hhr); hrt_all = torch.from_numpy(hr_target(np.nan_to_num(D.hr, nan=80.0)).astype(np.float32))
    for ep in range(1, E + 1):
        top.train()
        order = rng.permutation(n)
        for a in range(0, n, BS):
            b = torch.from_numpy(order[a:a + BS])
            for gr in opt.param_groups:
                gr['lr'] = lr * 0.5 * (1 + np.cos(np.pi * step / total))
            o = top(D.Pt[b], D.St[b], D.eb(top, D.g[b]))
            y = y_all[b]
            l_low = TF.binary_cross_entropy_with_logits(o['lowox_logit'], y, pos_weight=pw)
            m = hsp_all[b]
            l_sp = (0.5 * (torch.exp(-o['spo2_logvar'][m]) * ((sp_all[b][m] - o['spo2'][m]) / SPO2_SCALE) ** 2 + o['spo2_logvar'][m])).mean() if m.any() else 0.0
            mh = hhr_all[b]
            l_hr = -(hrt_all[b][mh] * TF.log_softmax(o['hr_logits'][mh], -1)).sum(-1).mean() if mh.any() else 0.0
            corr = ((o['lowox_logit'].detach() > 0).float() == y).float()
            l_sqi = TF.binary_cross_entropy_with_logits(o['sqi_logit'], corr)
            loss = W_LOW * l_low + W_SPO2 * l_sp + W_HR * l_hr + W_SQI * l_sqi
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if ep in epochs_ckpt and on_ckpt is not None:
            on_ckpt(ep, top)
    return top


@torch.no_grad()
def predict(top, D: Data):
    top.eval()
    o = top(D.Pt, D.St, D.eb(top, D.g))
    return dict(score=torch.sigmoid(o['lowox_logit']).numpy(), conf=torch.sigmoid(o['sqi_logit']).numpy(), spo2=o['spo2'].numpy(),
                logvar=o['spo2_logvar'].numpy())


def tm(t, y, pred):
    return M.time_matched(t, y, pred)['tm_bacc']


def bacc(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    if y.min() == y.max():
        return np.nan
    return 0.5 * (((pred == 1) & (y == 1)).sum() / (y == 1).sum() + ((pred == 0) & (y == 0)).sum() / (y == 0).sum())


# ------------------------------------------------------------------------------------------------ one outer fold
def fit_fold(net, bank, meta, test_sid, seed, labels, train_vers, log=print):
    subj = meta.subject.values
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    tr_people = sorted(set(subj[lab]) - {test_sid})
    tr_rows = np.nonzero(lab & (subj != test_sid))[0]
    t_all = meta.t_start.values.astype(float)
    aug_vers = [v for v in train_vers if v != 'clean']
    # inner LOSO over the training people
    oof = {}                                               # (lr, ep) -> dict(version -> arrays over inner OOF rows)
    for ip in tr_people:
        itr = tr_rows[subj[tr_rows] != ip]; ite = tr_rows[subj[tr_rows] == ip]
        s_scale = side_scale(bank, itr)
        Dtr = Data(bank, meta, train_vers, itr, labels, s_scale)
        Dte = Data(bank, meta, ['clean'] + aug_vers, ite, labels, s_scale)
        for lr in LR_GRID:
            def keep(ep, top, lr=lr):
                pr = predict(top, Dte)
                oof.setdefault((lr, ep), []).append(dict(pr, y=Dte.y, t=t_all[Dte.idx], clean=Dte.is_clean))
            train(net, Dtr, lr, EPOCH_GRID, seed, keep)
    res = []
    for key, parts in oof.items():
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        c = d['clean']; pred = d['score'] > 0.5
        obj = 0.5 * tm(d['t'][c], d['y'][c], pred[c]) + 0.5 * tm(d['t'][~c], d['y'][~c], pred[~c]) if (~c).any() else tm(d['t'], d['y'], pred)
        res.append((key, obj, d))
    res.sort(key=lambda r: (-np.round(r[1], 4), r[0][1], -r[0][0]))
    (lr, ep), obj, d = res[0]
    pred = d['score'] > 0.5
    best_q = None
    for q in Q_GRID:
        thr = float(np.quantile(d['conf'], q)) if q > 0 else -np.inf
        ans = d['conf'] >= thr
        ba = bacc(d['y'][ans], pred[ans])
        cand = (np.round(ba, 4), ans.mean())
        if best_q is None or cand > best_q[0]:
            best_q = (cand, q, thr)
    _, q, thr = best_q
    sel = dict(test=test_sid, lr=lr, epochs=ep, inner_obj=float(obj), q=q, thr=float(thr), inner_answered_bacc=float(best_q[0][0]),
               inner_cov=float(best_q[0][1]), grid={f'{k[0]:g}/{k[1]}': round(float(o), 4) for k, o, _ in res})
    # final fit on all 5 training people
    s_scale = side_scale(bank, tr_rows)
    Dtr = Data(bank, meta, train_vers, tr_rows, labels, s_scale)
    top = train(net, Dtr, lr, (ep,), seed)
    te_rows = np.nonzero(lab & (subj == test_sid))[0]
    out = {}
    for lv in EVAL:
        v = 'clean' if lv == 0 else vname(lv)
        Dte = Data(bank, meta, [v], te_rows, labels_of(meta), s_scale)
        out[vname(lv)] = dict(rows=te_rows, **predict(top, Dte))
    log(f'  fold {test_sid}: lr {lr:g} ep {ep} inner {obj:.3f} q {q} thr {thr:.3f}')
    return sel, out, dict(top=top, s_scale=s_scale)


def side_scale(bank, rows):
    s = bank['clean'][1][rows]
    sc = np.maximum(s.std(0), 0.05).astype(np.float32); sc[8] = 1.0
    return sc


def score_frame(meta, rows, score, pred, answer, conf, spo2):
    from .evalkit import eval_windows
    ref = eval_windows()
    m = meta.iloc[rows]
    d = pd.DataFrame(dict(rec=m.rec.str.replace('hof_', '', regex=False).values, t=m.t_start.astype(int).values, low=m.low.astype(int).values,
                          score=score, pred=pred.astype(int), answer=answer.astype(bool), conf=conf, spo2_hat=spo2))
    out = ref.merge(d, on=['rec', 't'], how='left', validate='one_to_one')
    assert out.score.notna().all(), 'every evaluated window must be scored'
    assert (out.y.values == out.low.values).all()
    return out[['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer', 'conf', 'spo2_hat']]


def run_variant(name, seed=0, perm_draw=None, aug=True, log=print, write=True):
    """All 6 outer folds of one variant. Returns (selections, frames by level name, raw fold outputs)."""
    set_threads(2)
    net, _ = load_init()
    bank = load_bank()
    meta = shard()[0]
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    people = sorted(set(meta.subject.values[lab]))
    train_vers = ['clean'] + ([f'aug{k}' for k in range(K_AUG)] if aug else [])
    sels, raw = [], {}
    t0 = time.time()
    for sid in people:
        if perm_draw is not None:          # global permutation of the labels across the training windows of this fold
            src = np.nonzero(lab & (meta.subject.values != sid))[0]
            rng = np.random.default_rng(np.random.SeedSequence([AUG_TAG, 7, int(perm_draw), people.index(sid)]))
            labels = labels_of(meta, (src, rng.permutation(src)))
        else:
            labels = labels_of(meta)
        sel, out, _ = fit_fold(net, bank, meta, sid, seed, labels, train_vers, log)
        sels.append(sel)
        for k, v in out.items():
            raw.setdefault(k, []).append(dict(v, thr=sel['thr']))
    frames = {}
    for k, parts in raw.items():
        rows = np.concatenate([p['rows'] for p in parts]); sc = np.concatenate([p['score'] for p in parts])
        cf = np.concatenate([p['conf'] for p in parts]); sp = np.concatenate([p['spo2'] for p in parts])
        ans = np.concatenate([p['conf'] >= p['thr'] for p in parts])
        frames[k] = score_frame(meta, rows, sc, sc > 0.5, ans, cf, sp)
    if write:
        SCORES.mkdir(parents=True, exist_ok=True)
        RAW = RES / 'oxyssl_raw'; RAW.mkdir(parents=True, exist_ok=True)
        for k, df in frames.items():
            if perm_draw is not None and k != 'L0':
                continue
            df.to_csv(SCORES / f'{name}__{k}.csv', index=False, float_format='%.6g')
        np.save(RAW / f'{name}_thr.npy', np.array([s['thr'] for s in sels]))
        json.dump(sels, open(RAW / f'{name}_selection.json', 'w'), indent=1)
    l0 = frames['L0']
    log(f'{name}: L0 TM {M.time_matched(l0.t.values, l0.y.values, l0.pred.values)["tm_bacc"]:.3f} ({time.time() - t0:.0f}s)')
    return sels, frames


def init_only(name=f'{NAME}_init', log=print):
    """Diagnostic: the pretrained network with NO Hoffman training (stage-B heads as they are), side scale 1, pred = sigmoid(lowox) > 0.5,
    answer = True everywhere. Shows how much hypoxia signal the OpenOx/SSL pretraining alone carries (it explains a permutation null above 0.5)."""
    set_threads(2)
    net, _ = load_init()
    bank = load_bank()
    meta = shard()[0]
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    rows = np.nonzero(lab)[0]
    top = Top(net)
    for lv in EVAL:
        v = 'clean' if lv == 0 else vname(lv)
        pr = predict(top, Data(bank, meta, [v], rows, labels_of(meta), np.ones(9, np.float32)))
        df = score_frame(meta, rows, pr['score'], pr['score'] > 0.5, np.ones(len(rows), bool), np.abs(pr['score'] - 0.5), pr['spo2'])
        df.to_csv(SCORES / f'{name}__{vname(lv)}.csv', index=False, float_format='%.6g')
        log(f'{name} {vname(lv)}: TM {M.time_matched(df.t.values, df.y.values, df.pred.values)["tm_bacc"]:.3f}')


def run(level: int = 0, seed: int = 0) -> pd.DataFrame:
    """Contract entry point: out-of-fold score frame of seed `seed` at `level` (int or 'X1'); fits all folds."""
    _, frames = run_variant(f'{NAME}_s{seed}', seed=seed, write=False)
    return frames[vname(level)]


def ensemble(seeds, name=NAME, log=print):
    """oxyssl__<level>.csv = mean score / mean confidence of the per-seed files; answer = mean conf >= mean of the per-seed nested thresholds
    of that fold (each seed's threshold was chosen on its own inner OOF)."""
    meta = shard()[0]
    r1 = meta.drop_duplicates('rec')
    sid_of = pd.Series(r1.subject.values, index=r1.rec.str.replace('hof_', '', regex=False).values)
    RAW = RES / 'oxyssl_raw'
    thr = {s: dict(zip([x['test'] for x in json.load(open(RAW / f'{name}_s{s}_selection.json'))],
                       [x['thr'] for x in json.load(open(RAW / f'{name}_s{s}_selection.json'))])) for s in seeds}
    for lv in EVAL:
        k = vname(lv)
        fs = [pd.read_csv(SCORES / f'{name}_s{s}__{k}.csv') for s in seeds]
        o = fs[0].copy()
        o['score'] = np.mean([f.score.values for f in fs], 0); o['conf'] = np.mean([f.conf.values for f in fs], 0)
        o['spo2_hat'] = np.mean([f.spo2_hat.values for f in fs], 0)
        subj = o.rec.map(sid_of).values
        t_mean = np.array([np.mean([thr[s][x] for s in seeds]) for x in subj])
        o['pred'] = (o.score > 0.5).astype(int); o['answer'] = o.conf.values >= t_mean
        o.to_csv(SCORES / f'{name}__{k}.csv', index=False, float_format='%.6g')
        log(f'ensemble {k}: TM {M.time_matched(o.t.values, o.y.values, o.pred.values)["tm_bacc"]:.3f} answered {o.answer.mean():.2f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('prep'); p.add_argument('--versions', nargs='*')
    p = sp.add_parser('fit'); p.add_argument('--seeds', nargs='*', type=int, default=[0]); p.add_argument('--nulls', type=int, default=0)
    p.add_argument('--noaug', action='store_true'); p.add_argument('--null-first', action='store_true')
    p = sp.add_parser('ensemble'); p.add_argument('--seeds', nargs='*', type=int, default=[0])
    p = sp.add_parser('init')
    a = ap.parse_args(argv)
    log = lambda *s: print(*s, flush=True)
    if a.cmd == 'prep':
        prep(a.versions, log)
    elif a.cmd == 'fit':
        for s in a.seeds:
            run_variant(f'{NAME}_s{s}', seed=s, log=log)
        for d in range(a.nulls):
            run_variant(f'{NAME}_null' if d == 0 else f'{NAME}_d{d}_null', seed=0, perm_draw=d, log=log)
        if a.noaug:
            run_variant(f'{NAME}_noaug', seed=0, aug=False, log=log)
    elif a.cmd == 'ensemble':
        ensemble(a.seeds, log=log)
    elif a.cmd == 'init':
        init_only(log=log)


if __name__ == '__main__':
    main()
