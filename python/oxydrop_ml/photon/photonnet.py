"""Round-2 candidate R2A, PhotonNet: ONE jointly trained network with the differentiable PhotonTwin renderer inside it.

    cd python && python -m oxydrop_ml.photon.photonnet               # photonnet (3 seeds), nulls (3 global-permutation draws), nophys ablation
    cd python && python -m oxydrop_ml.photon.photonnet --only main   # just the main model
    cd python && python -m oxydrop_ml.photon.photonnet --quick       # smoke run (1 outer fold, few steps), writes nothing

INPUT (per 10 s window, causal, relative to the recording's own label-free first minute)
  x [39] = the FlashGap window table (flashgap.COLS): chain v1 deltas (19, oxydrop.chain.deltas: window minus first-minute median, causal forward fill),
           absolute quality features (15: pulse SNR, log pulse index, log DC, clipping, quantisation, autocorrelation, beat fraction) and the
           baseline-relative respiratory group (5). All are computed from the per-frame ROI-mean RGB only (flashgap.rec_features), no labels.
  The features of training recordings, of their augmented copies and of the stress-degraded held-out recordings are read from FlashGap's feature cache
  (same code path; see `load_data`).

THE NETWORK (`PhotonNet`, 4,937 trainable parameters: the MLP heads + 4 learned physics scalars + 3 tissue-light fractions + 12 noise parameters)
  trunk     h = tanh(W2 tanh(W1 x_std + b1) + b2)                                  48 units, winsorised + standardised inputs (training rows of the fit)
  latents   z = P_SD * 2 tanh(W_z h)  = (d_sat, d_ln_fb, d_ln_alpha, d_ln_rho)       physical change of the finger vs the first minute
  physio    e = tanh(W_e h) [8]  -> r = w_r . e  (SpO2 points)                      learned physiological embedding (pulse rate / HRV / breathing cues)
  physics   y_hat = PhotonTwin(z | sat_b, rho0)  (twin_inverse.Decoder: Farrell diffusion, band-integrated camera channels, learned spectral scalars and
            tissue-light fractions eta_c) = predicted d ln DC_R,G,B and d ln|AC|_R,G,B; reconstruction loss = Gaussian NLL of the OBSERVED six optical
            changes (x columns dc_c, pi_c) with heteroscedastic variance exp(2a) + exp(2b) exp(-snr_c)
  SpO2 head mu = 100 (sat_b + d_sat) + r ;  sigma = softplus(w_s . h + b_s) + 0.5    (SpO2 points; Gaussian NLL vs the reference SpO2)
  P(hypoxic) score = Phi((90 - mu) / sigma)                                         (class-balanced BCE vs SpO2 < 90)
  abstention conf = 1 - min(1, sigma / |mu - 90|) ; answer = conf >= c*             (the model's own predictive uncertainty; c* nested)
  loss = W_REC * recon + NLL(SpO2) + W_BCE * BCE + W_PRIOR * |z / P_SD|^2 + W_R * r^2
  `photonnet_nophys`: the same network and loss with the physics layer removed (W_REC = 0 and the decoder never called; d_sat stays a free latent).
  sat_b = mean first-minute reference SpO2 of a TRAINING recording; for the held-out person sat_b = the mean over the training recordings (no test labels).
  rho0 = stress.RHO0 (3 mm) for every recording; geometry changes are absorbed by d_ln_rho.
AUGMENTATION PRIOR (declared): FlashGap's distant-flash prior (flashgap.AUG_PRIOR, own RNG namespace AUG_TAG = 'FGAP', never a stress (level, seed) pair,
  never X1; flashgap.assert_disjoint is called). Model seed s trains on the clean training people plus their 4 augmented copies of aug seed s
  (weight 0.25 each, so the 4 copies weigh as much as the clean recording).
PROTOCOL
  Outer: leave one person out (both hands). Inner: 5 person-grouped folds over the training people. Per outer fold and model seed, the training snapshot
  (SNAPS) is chosen by the pooled inner out-of-fold TM bal. acc. on the clean inner validation people; c* by baselines.common.choose_band's rule
  (twin_inverse.choose_conf) on the pooled inner OOF windows of the clean AND augmented copies of the inner validation people (training-side data only).
  The held-out person is never in a fit (asserted). Trained once per fold, scored at every stress level and X1 with the same weights.
  Ensemble (photonnet): mean mu, mixture sigma sqrt(mean sigma^2 + var mu) over the 3 seeds, score = mean of the seed scores, c* = mean of the seeds' c*.
  Null control (photonnet_null): whole pipeline (inner selection included, seed 0) with the training labels permuted globally across the outer fold's
  training windows (the same permutation applied to the augmented copies; training sat_b replaced by the training mean); 3 draws, the file holds draw 0
  and every draw is written as photonnet_null_d<k>__L0.csv; the log reports the mean. `--only null_nophys` runs the same null for the ablation
  (photonnet_nophys_null). Remote runs: PHOTONNET_BUNDLE=<npz from make_bundle()> PHOTONNET_WORKERS=<n>.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import optics as OP
from . import flashgap as FG
from . import stress as ST
from . import twin_inverse as TI

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
LOG = RES / 'photonnet_log.md'
CHOICES = RES / 'photonnet_choices.csv'

TEST_CONDS = {lv: ('L', lv, TI.STRESS_SEED) for lv in ST.LEVELS if lv != 0}
TEST_CONDS['X1'] = ('X', 'X1', TI.STRESS_SEED)
LEVEL_SET = list(ST.LEVELS) + ['X1']
SEEDS = (0, 1, 2)
N_NULL = 3
K_AUG = 4
AUG_W = 0.25
# FlashGap's cache directory carries a crc of stress.py. The only change to stress.py since that cache was written (git diff 3448e0b b48e14b) is a
# docstring edit, so the cached degraded features are those of the current stress.degrade; we read them instead of recomputing.
STRESS_TAG = 'ba26b85a'

I_DC = [16, 17, 18]                       # flashgap.COLS: dc_R, dc_G, dc_B (chain deltas)
I_PI = [5, 6, 7]                          # pi_R, pi_G, pi_B
I_SNR = [len(FG.FUSION) + i for i in range(3)]   # QUAL snr_R, snr_G, snr_B (absolute log pulse SNR)

HID, EMB = 48, 8
OPT_CFG = dict(learn_bands=False, lam_max=720.0, n_quad=13)    # 25 nm quadrature for speed (twin_inverse uses 12.5 nm)
BATCH, N_REC_ROWS, LR, WD = 512, 192, 4e-3, 1e-4
STEPS, SNAPS = 600, (150, 300, 600)
W_REC, W_BCE, W_PRIOR, W_R = 1.0, 1.0, 1e-2, 1e-3
WINSOR = 0.005
N_WORKERS = int(os.environ.get('PHOTONNET_WORKERS', 1))                          # ONE local process (shared box, planner 18:42 CT); >1 = forked pool of 1-thread workers
THREADS = 2


def _f64(x):
    return torch.as_tensor(np.asarray(x, np.float64), dtype=torch.float64)


# ================================================================================================ data
_DATA = {}


def load_data(log=print):
    """meta (5,644 evaluated windows), X per condition [n, 39], first-minute sat_b per recording. Cached in-process (forked workers inherit it)."""
    if _DATA:
        return _DATA
    if os.environ.get('PHOTONNET_BUNDLE'):                # remote run (Daytona): the same tables, packed by make_bundle()
        z = np.load(os.environ['PHOTONNET_BUNDLE'], allow_pickle=True)    # our own bundle (rec names are an object array)
        mt = pd.DataFrame({c: z['mt_' + c] for c in ('rec', 't', 'sid', 'y_spo2', 'y', 'ri', 'sat_b')})
        mt['sid'] = mt.sid.astype(str); mt['rec'] = mt.rec.astype(str)
        X = {_cond_of(k[2:]): z[k] for k in z.files if k.startswith('X_')}
        FG.assert_disjoint([tuple(int(v) for v in k) for k in z['aug_keys']], [(lv, TI.STRESS_SEED) for lv in ST.LEVELS] + [('X1', TI.STRESS_SEED)])
        _DATA.update(mt=mt, X=X)
        return _DATA
    FG.stress_hash = lambda: STRESS_TAG
    conds = ['clean'] + [('aug', s, k) for s in SEEDS for k in range(K_AUG)] + list(TEST_CONDS.values())
    X, params = FG.compute_features(conds, jobs=2, log=log)
    FG.assert_disjoint(list(FG.aug_key(s, ri, k) for (s, ri, k) in params), [(lv, TI.STRESS_SEED) for lv in ST.LEVELS] + [('X1', TI.STRESS_SEED)],
                       list(params.values()))
    mt = FG.meta().copy()
    mt['sid'] = mt.sid.astype(str)
    recs = FG.recordings()
    satb = {}
    for r in recs:
        v = [r['y'][s] for s in range(0, TI.CAL_S - TI.WIN_S + 1, 2) if s in r['y']]
        satb[r['rec']] = float(np.mean(v)) / 100.0 if v else np.nan
    mt['sat_b'] = mt.rec.map(satb)
    _DATA.update(mt=mt, X=X)
    return _DATA


def _cond_of(name):
    if name == 'clean':
        return 'clean'
    p = name.split('_')
    if p[0] == 'aug':
        return ('aug', int(p[1]), int(p[2]))
    return ('L', int(p[1]), TI.STRESS_SEED) if p[0] == 'L' else ('X', p[1], TI.STRESS_SEED)


def _cond_key(c):
    return 'clean' if c == 'clean' else '_'.join(str(v) for v in c[:-1]) if c[0] != 'aug' else f'aug_{c[1]}_{c[2]}'


def make_bundle(path):
    """Pack meta + the feature tables of every condition into one npz (numpy/pandas only, no torch) for a remote run."""
    FG.stress_hash = lambda: STRESS_TAG
    conds = ['clean'] + [('aug', s, k) for s in SEEDS for k in range(K_AUG)] + list(TEST_CONDS.values())
    mt = FG.meta().copy()
    pos = {(ri, t): i for i, (ri, t) in enumerate(zip(mt.ri.values, mt.t.values))}
    out, keys = {}, []
    for c in conds:
        X = np.full((len(mt), len(FG.COLS)), np.nan)
        for ri in range(int(mt.ri.max()) + 1):
            z = np.load(FG._cache_path(c, ri), allow_pickle=False)
            for ti, xi in zip(z['t'], z['X']):
                j = pos.get((ri, int(ti)))
                if j is not None:
                    X[j] = xi
            if c != 'clean' and c[0] == 'aug':
                keys.append(FG.aug_key(c[1], ri, c[2]))
        assert np.isfinite(X[:, :len(FG.FUSION)]).all()
        out['X_' + _cond_key(c)] = X
    recs = FG.recordings()
    satb = {}
    for r in recs:
        v = [r['y'][s] for s in range(0, TI.CAL_S - TI.WIN_S + 1, 2) if s in r['y']]
        satb[r['rec']] = float(np.mean(v)) / 100.0 if v else np.nan
    mt['sat_b'] = mt.rec.map(satb)
    for c in ('rec', 't', 'sid', 'y_spo2', 'y', 'ri', 'sat_b'):
        out['mt_' + c] = mt[c].values.astype(str) if c in ('rec', 'sid') else mt[c].values
    out['aug_keys'] = np.array(keys, np.int64)
    np.savez_compressed(path, **out)
    return path


# ================================================================================================ model
class Decoder(TI.Decoder):
    """twin_inverse.Decoder with this module's renderer quadrature."""

    def __init__(self):
        nn.Module.__init__(self)
        self.twin = OP.PhotonTwin(OP.OpticsCfg(learn_scales=True, **OPT_CFG))
        self.eta_raw = nn.Parameter(torch.full((3,), 2.0, dtype=torch.float64))


class PhotonNet(nn.Module):
    def __init__(self, n_in, phys=True):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh()).double()
        self.z_head = nn.Linear(HID, 4).double()
        self.e_head = nn.Linear(HID, EMB).double()
        self.r_head = nn.Linear(EMB, 1).double()
        self.s_head = nn.Linear(HID, 1).double()
        with torch.no_grad():
            self.z_head.weight.mul_(0.1); self.z_head.bias.zero_()
            self.r_head.weight.mul_(0.1); self.r_head.bias.zero_()
            self.s_head.bias.fill_(1.5)
        self.phys = phys
        self.dec = Decoder() if phys else None
        self.a = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.b = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.register_buffer('psd', _f64(TI.P_SD))

    def forward(self, x, sat_b):
        h = self.trunk(x)
        z = 2.0 * self.psd * torch.tanh(self.z_head(h))
        r = self.r_head(torch.tanh(self.e_head(h)))[:, 0]
        mu = 100.0 * (sat_b + z[:, 0]) + r
        sigma = nn.functional.softplus(self.s_head(h)[:, 0]) + 0.5
        return dict(z=z, r=r, mu=mu, sigma=sigma)

    def recon_nll(self, z, sat_b, y6, snr):
        rho = torch.full_like(sat_b, ST.RHO0)
        yhat = self.dec(z, sat_b, rho)
        e = torch.exp(-snr)
        var = torch.exp(2 * self.a)[None] + torch.exp(2 * self.b)[None] * torch.cat([e, e], 1)
        return 0.5 * (((y6 - yhat) ** 2) / var + torch.log(var)).sum(1).mean()


def n_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class Prep:
    """Winsorisation bounds and scaler of the network inputs, fitted on the training rows of one fit (clean + augmented)."""

    def __init__(self, X):
        self.lo, self.hi = np.nanquantile(X, WINSOR, 0), np.nanquantile(X, 1 - WINSOR, 0)
        Xc = np.clip(np.nan_to_num(X, nan=0.0), self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-6

    def x(self, X):
        return _f64((np.clip(np.nan_to_num(X, nan=0.0), self.lo, self.hi) - self.mu) / self.sd)


def _rows(D, sids, conds):
    mt = D['mt']
    m = mt.sid.isin(sids).values
    return [(c, np.where(m)[0]) for c in conds]


def _stack(D, parts, ylab, satb):
    X = np.concatenate([D['X'][c][i] for c, i in parts])
    y = np.concatenate([ylab[i] for c, i in parts])
    sb = np.concatenate([satb[i] for c, i in parts])
    w = np.concatenate([np.full(len(i), 1.0 if c == 'clean' else AUG_W) for c, i in parts])
    return X, y, sb, w


def fit(D, tr_sids, held_out, seed, phys, ylab, satb, steps=STEPS, snaps=SNAPS):
    """Train one PhotonNet on the clean + augmented windows of tr_sids. Returns (prep, net, [(step, state)])."""
    assert not set(map(str, held_out)) & set(map(str, tr_sids)), 'held-out person among the training people'
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 7919)
    conds = ['clean'] + [('aug', seed % len(SEEDS), k) for k in range(K_AUG)]
    X, y, sb, w = _stack(D, _rows(D, tr_sids, conds), ylab, satb)
    ok = np.isfinite(y)
    X, y, sb, w = X[ok], y[ok], sb[ok], w[ok]
    prep = Prep(X)
    xt, yt, sbt, wt = prep.x(X), _f64(y), _f64(sb), _f64(w)
    y6 = _f64(np.nan_to_num(X[:, I_DC + I_PI])); snr = _f64(np.nan_to_num(X[:, I_SNR], nan=-5.0))
    yb = (y < 90).astype(float)
    wpos = w.sum() / (2 * max((w * yb).sum(), 1e-9)); wneg = w.sum() / (2 * max((w * (1 - yb)).sum(), 1e-9))
    ybt = _f64(yb)
    net = PhotonNet(X.shape[1], phys)
    nn_p = [p for n_, p in net.named_parameters() if n_.split('.')[0] in ('trunk', 'z_head', 'e_head', 'r_head', 's_head')]
    ph_p = [p for n_, p in net.named_parameters() if p.requires_grad and n_.split('.')[0] not in ('trunk', 'z_head', 'e_head', 'r_head', 's_head')]
    opt = torch.optim.Adam([dict(params=nn_p, weight_decay=WD), dict(params=ph_p)], lr=LR)
    prob = w / w.sum()
    out = []
    for step in range(1, steps + 1):
        bi = torch.as_tensor(rng.choice(len(y), BATCH, p=prob))
        o = net(xt[bi], sbt[bi])
        nll = (0.5 * ((yt[bi] - o['mu']) / o['sigma']) ** 2 + torch.log(o['sigma'])).mean()
        p = torch.special.ndtr((90.0 - o['mu']) / o['sigma']).clamp(1e-6, 1 - 1e-6)
        yy = ybt[bi]
        bce = -(wpos * yy * torch.log(p) + wneg * (1 - yy) * torch.log(1 - p)).mean()
        loss = nll + W_BCE * bce + W_PRIOR * ((o['z'] / net.psd) ** 2).sum(1).mean() + W_R * (o['r'] ** 2).mean()
        if phys:
            k = N_REC_ROWS
            loss = loss + W_REC * net.recon_nll(o['z'][:k], sbt[bi][:k], y6[bi][:k], snr[bi][:k])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step in snaps:
            out.append((step, {k_: v.detach().clone() for k_, v in net.state_dict().items()}))
    return prep, net, out


@torch.no_grad()
def predict(prep, net, state, X, sat_b):
    net.load_state_dict(state)
    o = net(prep.x(X), _f64(sat_b))
    return dict(mu=o['mu'].numpy(), sigma=o['sigma'].numpy(), d_sat=100 * o['z'][:, 0].numpy(), r=o['r'].numpy())


# ================================================================================================ nested LOSO
def _labels(D, null_draw, so):
    """Training-side labels (SpO2) and sat_b per meta row. null_draw k >= 0: global permutation across the outer fold's training windows."""
    mt = D['mt']
    ylab = mt.y_spo2.values.astype(float).copy()
    satb = mt.sat_b.values.astype(float).copy()
    tr = (mt.sid != str(so)).values
    satb_tr_mean = float(np.nanmean(mt.loc[tr].groupby('rec').sat_b.first().values))
    if null_draw is not None and null_draw >= 0:
        rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(str(so).encode()), 1000 + int(null_draw)]))
        idx = np.where(tr)[0]
        ylab[idx] = rng.permutation(ylab[idx])
        satb[tr] = satb_tr_mean
    satb[~tr] = satb_tr_mean
    return ylab, satb, satb_tr_mean


def _job(args):
    kind, so, si, seed, phys, null_draw, steps, snaps = args
    torch.set_num_threads(THREADS if N_WORKERS == 1 else 1)
    D = _DATA
    sids = sorted(D['mt'].sid.unique())
    ylab, satb, satb_mean = _labels(D, null_draw, so)
    tr = [s for s in sids if s != so]
    mt = D['mt']
    if kind == 'inner':
        tr_in = [s for s in tr if s != si]
        satb_in = satb.copy()
        satb_in[(mt.sid == str(si)).values] = float(np.nanmean(mt[mt.sid.isin(tr_in)].groupby('rec').sat_b.first().values)) \
            if null_draw is None or null_draw < 0 else satb_mean
        prep, net, st = fit(D, tr_in, [si, so], seed, phys, ylab, satb_in, steps, snaps)
        iv = np.where((mt.sid == str(si)).values)[0]
        res = {}
        for step, s in st:
            res[step] = {c: predict(prep, net, s, D['X'][c][iv], satb_in[iv]) for c in ['clean'] + [('aug', seed % 3, k) for k in range(K_AUG)]}
        return kind, (so, si, seed), dict(idx=iv, ylab=ylab[iv], pred=res)
    prep, net, st = fit(D, tr, [so], seed, phys, ylab, satb, steps, snaps)
    te = np.where((mt.sid == str(so)).values)[0]
    res = {}
    for step, s in st:
        res[step] = {lv: predict(prep, net, s, D['X']['clean' if lv == 0 else TEST_CONDS[lv]][te], satb[te]) for lv in LEVEL_SET}
    rep = net.dec.twin.param_report() if phys else {}
    rep = {k: (v if not hasattr(v, 'tolist') else v.tolist()) for k, v in rep.items() if k not in ('band_centre', 'band_fwhm')}
    if phys:
        rep['eta'] = [float(v) for v in net.dec.eta().detach()]
    rep['n_params'] = n_params(net)
    return kind, (so, seed), dict(idx=te, pred=res, rep=rep)


def conf_of(mu, sigma):
    return 1.0 - np.minimum(1.0, sigma / np.maximum(np.abs(mu - 90.0), 1e-6))


def score_of(mu, sigma):
    from scipy.special import ndtr
    return ndtr((90.0 - mu) / sigma)


def run_variant(tag, phys=True, seeds=SEEDS, null_draw=None, quick=False, log=print):
    """Nested LOSO for one variant. Returns (dict level -> ensemble frame, dict seed -> level-0 frame, choice log rows, reports)."""
    D = load_data(log)
    mt = D['mt']
    sids = sorted(mt.sid.unique())
    outer_s = sids[:1] if quick else sids
    steps, snaps = (40, (20, 40)) if quick else (STEPS, SNAPS)
    jobs = [('inner', so, si, sd, phys, null_draw, steps, snaps) for so in outer_s for sd in seeds for si in sids if si != so]
    jobs += [('outer', so, None, sd, phys, null_draw, steps, snaps) for so in outer_s for sd in seeds]
    t0 = time.time()
    inner, outer = {}, {}
    def _collect(it):
        for i, (kind, key, r) in enumerate(it):
            (inner if kind == 'inner' else outer)[key] = r
            if (i + 1) % 12 == 0:
                log(f'    {tag}: {i + 1}/{len(jobs)} fits, {time.time() - t0:.0f} s')
    if N_WORKERS == 1:
        _collect(map(_job, jobs))
    else:
        with mp.get_context('fork').Pool(N_WORKERS) as pool:
            _collect(pool.imap_unordered(_job, jobs, chunksize=1))
    log(f'  {tag}: {len(jobs)} fits in {time.time() - t0:.0f} s')
    choice, rows = {}, []
    for so in outer_s:
        for sd in seeds:
            P = [inner[(so, si, sd)] for si in sids if si != so]
            idx = np.concatenate([p['idx'] for p in P]); yv = np.concatenate([p['ylab'] for p in P])
            t = mt.t.values[idx]
            best = None
            for step in snaps:
                mu = np.concatenate([p['pred'][step]['clean']['mu'] for p in P])
                ok = np.isfinite(yv)
                v = TI._tm(t[ok], yv[ok], (mu[ok] < 90).astype(int))
                v = -1.0 if not np.isfinite(v) else v
                if best is None or v > best[0] + 1e-12:
                    best = (v, step)
            v, step = best
            conds = list(P[0]['pred'][step].keys())
            mu = np.concatenate([np.concatenate([p['pred'][step][c]['mu'] for p in P]) for c in conds])
            sg = np.concatenate([np.concatenate([p['pred'][step][c]['sigma'] for p in P]) for c in conds])
            yy = np.concatenate([yv for _ in conds]); ok = np.isfinite(yy)
            c_star = TI.choose_conf(conf_of(mu[ok], sg[ok]), yy[ok], (mu[ok] < 90).astype(int))
            choice[(so, sd)] = dict(step=step, c_star=c_star, inner_tm=v)
            rows.append(dict(model=tag, held_out=so, seed=sd, steps=step, c_star=c_star, inner_tm=round(v, 4)))
    ens, per_seed, reps = {}, {sd: [] for sd in seeds}, {}
    for lv in LEVEL_SET:
        parts = []
        for so in outer_s:
            Ps = [outer[(so, sd)]['pred'][choice[(so, sd)]['step']][lv] for sd in seeds]
            te = outer[(so, seeds[0])]['idx']
            mu_s = np.stack([p['mu'] for p in Ps]); sg_s = np.stack([p['sigma'] for p in Ps])
            mu = mu_s.mean(0); sg = np.sqrt((sg_s ** 2).mean(0) + mu_s.var(0))
            score = np.mean([score_of(m, s) for m, s in zip(mu_s, sg_s)], 0)
            c_star = float(np.mean([choice[(so, sd)]['c_star'] for sd in seeds]))
            parts.append(_frame(mt.iloc[te], mu, sg, score, c_star))
            if lv == 0:
                for sd, p in zip(seeds, Ps):
                    per_seed[sd].append(_frame(mt.iloc[te], p['mu'], p['sigma'], score_of(p['mu'], p['sigma']), choice[(so, sd)]['c_star']))
        ens[lv] = pd.concat(parts, ignore_index=True)
    for so in outer_s:
        reps[so] = outer[(so, seeds[0])]['rep']
    per_seed = {sd: pd.concat(v, ignore_index=True) for sd, v in per_seed.items()}
    return ens, per_seed, rows, reps


def _frame(m, mu, sigma, score, c_star):
    conf = conf_of(mu, sigma)
    return pd.DataFrame(dict(rec=m.rec.values, t=m.t.values.astype(int), sid=m.sid.values, y_spo2=m.y_spo2.values, y=(m.y_spo2.values < 90).astype(int),
                             score=np.round(score, 6), pred=(score > 0.5).astype(int), answer=conf >= c_star, conf=np.round(conf, 6),
                             spo2_hat=np.round(mu, 4), spo2_lo=np.round(mu - 1.96 * sigma, 4), spo2_hi=np.round(mu + 1.96 * sigma, 4)))


def run(level: int = 0, seed: int = 0) -> pd.DataFrame:
    """CONTRACT entry point: out-of-fold frame of the single seed `seed` at `level` (trains the nested pipeline for that seed)."""
    ens, _, _, _ = run_variant('photonnet', seeds=(seed,))
    return ens[level]


def _write(name, lv, df):
    SCORES.mkdir(parents=True, exist_ok=True)
    suf = f'L{lv}' if not isinstance(lv, str) else lv
    df.to_csv(SCORES / f'{name}__{suf}.csv', index=False)


def _quick_tm(df):
    return TI._tm(df.t.values, df.y_spo2.values, df.pred.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='main,null,nophys')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(THREADS)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)
    load_data(log)
    rows_all = []
    jpath = RES / 'photonnet_run.json'
    state = json.loads(jpath.read_text()) if jpath.exists() and not a.quick else {}
    for what in a.only.split(','):
        t0 = time.time()
        if what == 'main':
            ens, per_seed, rows, reps = run_variant('photonnet', quick=a.quick, log=log)
            rows_all += rows
            if not a.quick:
                for lv, df in ens.items():
                    _write('photonnet', lv, df)
                for sd, df in per_seed.items():
                    _write(f'photonnet_seed{sd}', 0, df)
                state['main'] = dict(tm={str(lv): _quick_tm(df) for lv, df in ens.items()}, cov={str(lv): float(df.answer.mean()) for lv, df in ens.items()},
                                     seed_tm={str(sd): _quick_tm(df) for sd, df in per_seed.items()}, reps={str(k): v for k, v in reps.items()})
            log(f'photonnet: ' + ', '.join(f'{lv}: TM {_quick_tm(df):.3f} cov {df.answer.mean():.2f}' for lv, df in ens.items()))
            log('  seeds L0: ' + ', '.join(f'{_quick_tm(df):.3f}' for df in per_seed.values()))
        elif what == 'nophys':
            ens, per_seed, rows, reps = run_variant('photonnet_nophys', phys=False, quick=a.quick, log=log)
            rows_all += rows
            if not a.quick:
                for lv, df in ens.items():
                    _write('photonnet_nophys', lv, df)
                state['nophys'] = dict(tm={str(lv): _quick_tm(df) for lv, df in ens.items()})
            log(f'photonnet_nophys: ' + ', '.join(f'{lv}: TM {_quick_tm(df):.3f}' for lv, df in ens.items()))
        elif what in ('null', 'null_nophys'):
            phys = what == 'null'
            base = 'photonnet_null' if phys else 'photonnet_nophys_null'
            tms = []
            for k in range(1 if a.quick else N_NULL):
                ens, _, rows, _ = run_variant(f'{base}_d{k}', phys=phys, seeds=(0,), null_draw=k, quick=a.quick, log=log)
                rows_all += rows
                tm = _quick_tm(ens[0]); tms.append(tm)
                if not a.quick:
                    _write(f'{base}_d{k}', 0, ens[0])
                    if k == 0:
                        _write(base, 0, ens[0])
                log(f'{base} draw {k}: TM {tm:.3f}')
            state[what] = dict(tm_draws=tms, mean=float(np.mean(tms)))
            log(f'{base} mean over {len(tms)} draws: {np.mean(tms):.3f}')
        log(f'{what} done in {time.time() - t0:.0f} s')
        if not a.quick:                                    # persist after every variant (a later variant may be cut by the timeline)
            old = pd.read_csv(CHOICES) if CHOICES.exists() else pd.DataFrame()
            new = pd.DataFrame(rows_all)
            if len(old):
                old = old[~old.model.isin(new.model.unique())]
            pd.concat([old, new], ignore_index=True).to_csv(CHOICES, index=False)
            jpath.write_text(json.dumps(state, indent=1, default=float))
            with open(RES / 'photonnet_run.log', 'a') as f:
                f.write('\n'.join(lines) + '\n')
            lines.clear()


if __name__ == '__main__':
    main()
