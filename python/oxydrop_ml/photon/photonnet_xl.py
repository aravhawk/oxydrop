"""Round-3 candidate R3-PHOTONNET-XL: a larger PhotonNet whose PhotonTwin physics layer is a TRAINING-ONLY decoder.

    cd python && python -m oxydrop_ml.photon.photonnet_xl --part main --seeds 0          # nested sweep + outer fits for one seed (remote)
    cd python && python -m oxydrop_ml.photon.photonnet_xl --part null --draws 0,1,2      # plain global-permutation null (whole pipeline, seed 0)
    cd python && python -m oxydrop_ml.photon.photonnet_xl --part nullscr --draws 0,1,2   # null with the physics targets scrambled as well
    cd python && python -m oxydrop_ml.photon.photonnet_xl --part combine                 # local, light: 3-seed ensemble -> photonnet_xl__L*.csv
    cd python && python -m oxydrop_ml.photon.photonnet_xl --quick                        # smoke run (1 outer fold, few steps), writes nothing

Imports photonnet.py (data bundle, labels, Prep, PhotonTwin Decoder); photonnet.py itself is unchanged.

INFERENCE NETWORK (plain ops on the 39 FlashGap / distill window features, flashgap.COLS, the same features app/js/distill_features.js computes):
  x_std = (clip(x, lo, hi) - m) / s                          training-row winsorisation + scaler (photonnet.Prep)
  h = tanh(W2 tanh(W1 x_std + b1) + b2)                      encoder 39 -> 96 -> 96
  z = 2 P_SD tanh(W_z h)                                     4 physical latents (d_sat, d_ln_fb, d_ln_alpha, d_ln_rho) vs the first minute
  e = tanh(W_e h) [16]                                       physiological embedding, learned jointly with everything else
  mu    = 100 sat_b + beta * 100 d_sat + w_r . e + b_r       SpO2 head (points); beta learned; sat_b = training-recording mean (constant at deployment)
  sigma = softplus(w_s . h + b_s) + 0.5
  logit = w_y . [h, e] + b_y + 1.6 (90 - mu) / sigma         hypoxia head (score = sigmoid(logit), pred = score > 0.5)
  abst  = sigmoid(w_a . [h, e] + b_a)                        abstain head = P(hyp head correct); answer = abst >= c*, c* nested
TRAINING-ONLY PHYSICS DECODER: PhotonTwin(z | sat_b, rho0) renders d ln DC_R,G,B and d ln |AC|_R,G,B; reconstruction loss = heteroscedastic Gaussian NLL
  of the six observed optical changes (photonnet.PhotonNet.recon_nll), weight W_REC. The decoder is dropped after training.
PHYSICS-PRIOR STRENGTH W_PHYS: Gaussian prior W_PHYS (beta - 1)^2 tying the SpO2 head's use of the physical latent d_sat to its physical meaning
  (beta = 1: one unit of d_sat is 100 SpO2 points). beta starts at 0, so with W_PHYS = 0 the label data alone decide how much d_sat counts.
LOSS = NLL(SpO2 | mu, sigma) + balanced BCE(logit, y) + 0.5 BCE(abst, 1[hyp correct], detached) + W_REC recon + W_PHYS (beta - 1)^2
       + 1e-2 |z / P_SD|^2 + 1e-3 r^2 ; Adam lr 3e-3, wd 1e-4 on the network weights, batch 512 (128 rows through the renderer), snapshots SNAPS.
SWEEP (nested, inner person-grouped CV of the 5 training people only): GRID = W_REC x W_PHYS, snapshot in SNAPS; criterion
  0.5 TM(clean inner OOF) + 0.5 mean TM(augmented inner OOF copies). c* by twin_inverse.choose_conf on the pooled inner OOF windows (clean + augmented).
AUGMENTATION: flashgap's distant-flash prior (AUG_TAG 'FGAP'), 4 copies per training recording of aug seed = model seed, weight 0.25 each (as photonnet).
NULLS (both run the whole nested pipeline with seed 0, 3 draws):
  photonnet_xl_null      plain: training labels (and sat_b) permuted globally across the outer fold's training windows (photonnet._labels).
  photonnet_xl_nullscr   additionally the physics decoder's targets (the six observed optical changes and their SNR) are permuted across the training
                         windows with an independent permutation, so the physics layer cannot supply a label-free colour -> saturation map either.
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

from . import photonnet as PN
from . import twin_inverse as TI

RES, SCORES = PN.RES, PN.SCORES
RAW = RES / 'photonnet_xl_raw'
LOG = RES / 'photonnet_xl_log.md'
NAME = 'photonnet_xl'
LEVEL_SET = PN.LEVEL_SET
SEEDS = (0, 1, 2)
HID, EMB = 96, 16
LR, WD, BATCH, N_REC_ROWS = 3e-3, 1e-4, 512, 128
STEPS, SNAPS = 400, (200, 400)
GRID = [dict(w_rec=wr, w_phys=wp) for wr in (0.3, 3.0) for wp in (0.0, 1.0)]
W_PRIOR, W_R, W_ABS = 1e-2, 1e-3, 0.5
N_WORKERS = int(os.environ.get('PHOTONNET_WORKERS', 1))
_f64 = PN._f64


class PhotonNetXL(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh()).double()
        self.z_head = nn.Linear(HID, 4).double()
        self.e_head = nn.Linear(HID, EMB).double()
        self.r_head = nn.Linear(EMB, 1).double()
        self.s_head = nn.Linear(HID, 1).double()
        self.y_head = nn.Linear(HID + EMB, 1).double()
        self.a_head = nn.Linear(HID + EMB, 1).double()
        self.beta = nn.Parameter(torch.zeros((), dtype=torch.float64))
        with torch.no_grad():
            for m in (self.z_head, self.r_head, self.y_head):
                m.weight.mul_(0.1); m.bias.zero_()
            self.s_head.bias.fill_(1.5)
        self.dec = PN.Decoder()                              # training-only
        self.a = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.b = nn.Parameter(torch.full((6,), np.log(0.03), dtype=torch.float64))
        self.register_buffer('psd', _f64(TI.P_SD))

    def forward(self, x, sat_b):
        h = self.trunk(x)
        z = 2.0 * self.psd * torch.tanh(self.z_head(h))
        e = torch.tanh(self.e_head(h))
        r = self.r_head(e)[:, 0]
        mu = 100.0 * sat_b + self.beta * 100.0 * z[:, 0] + r
        sigma = nn.functional.softplus(self.s_head(h)[:, 0]) + 0.5
        he = torch.cat([h, e], 1)
        logit = self.y_head(he)[:, 0] + 1.6 * (90.0 - mu) / sigma
        alog = self.a_head(he.detach())[:, 0]
        return dict(z=z, r=r, mu=mu, sigma=sigma, logit=logit, alog=alog)

    recon_nll = PN.PhotonNet.recon_nll


INFER_PARTS = ('trunk', 'z_head', 'e_head', 'r_head', 's_head', 'y_head', 'a_head', 'beta')


def n_infer_params(net):
    return sum(p.numel() for n_, p in net.named_parameters() if n_.split('.')[0] in INFER_PARTS)


def fit(D, tr_sids, held_out, seed, cfg, ylab, satb, scr_perm=None, steps=STEPS, snaps=SNAPS):
    assert not set(map(str, held_out)) & set(map(str, tr_sids)), 'held-out person among the training people'
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 7919)
    conds = ['clean'] + [('aug', seed % len(SEEDS), k) for k in range(PN.K_AUG)]
    parts = PN._rows(D, tr_sids, conds)
    X, y, sb, w = PN._stack(D, parts, ylab, satb)
    ok = np.isfinite(y)
    X, y, sb, w = X[ok], y[ok], sb[ok], w[ok]
    prep = PN.Prep(X)
    xt, yt, sbt, wt = prep.x(X), _f64(y), _f64(sb), _f64(w)
    Y6 = np.nan_to_num(X[:, PN.I_DC + PN.I_PI]); SNR = np.nan_to_num(X[:, PN.I_SNR], nan=-5.0)
    if scr_perm is not None:                                 # physics-scrambled null: decoder targets permuted across training windows
        p = np.random.default_rng(scr_perm).permutation(len(y))
        Y6, SNR = Y6[p], SNR[p]
    y6, snr = _f64(Y6), _f64(SNR)
    yb = (y < 90).astype(float)
    wpos = w.sum() / (2 * max((w * yb).sum(), 1e-9)); wneg = w.sum() / (2 * max((w * (1 - yb)).sum(), 1e-9))
    ybt = _f64(yb)
    net = PhotonNetXL(X.shape[1])
    nn_p = [p for n_, p in net.named_parameters() if n_.split('.')[0] in INFER_PARTS and n_ != 'beta']
    ph_p = [p for n_, p in net.named_parameters() if p.requires_grad and (n_.split('.')[0] not in INFER_PARTS or n_ == 'beta')]
    opt = torch.optim.Adam([dict(params=nn_p, weight_decay=WD), dict(params=ph_p)], lr=LR)
    prob = w / w.sum()
    out = []
    for step in range(1, steps + 1):
        bi = torch.as_tensor(rng.choice(len(y), BATCH, p=prob))
        o = net(xt[bi], sbt[bi])
        nll = (0.5 * ((yt[bi] - o['mu']) / o['sigma']) ** 2 + torch.log(o['sigma'])).mean()
        yy = ybt[bi]
        lg = o['logit']
        bce = (wpos * yy * nn.functional.softplus(-lg) + wneg * (1 - yy) * nn.functional.softplus(lg)).mean()
        corr = ((lg.detach() > 0).double() == yy).double()
        abce = nn.functional.binary_cross_entropy_with_logits(o['alog'], corr)
        loss = nll + bce + W_ABS * abce + W_PRIOR * ((o['z'] / net.psd) ** 2).sum(1).mean() + W_R * (o['r'] ** 2).mean() \
            + cfg['w_phys'] * (net.beta - 1.0) ** 2
        k = N_REC_ROWS
        loss = loss + cfg['w_rec'] * net.recon_nll(o['z'][:k], sbt[bi][:k], y6[bi][:k], snr[bi][:k])
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
    return dict(mu=o['mu'].numpy(), sigma=o['sigma'].numpy(), score=torch.sigmoid(o['logit']).numpy(), abst=torch.sigmoid(o['alog']).numpy())


def _job(args):
    kind, so, si, seed, ci, null_draw, scr, steps, snaps = args
    torch.set_num_threads(1 if N_WORKERS > 1 else 2)
    D = PN._DATA
    mt = D['mt']
    sids = sorted(mt.sid.unique())
    ylab, satb, satb_mean = PN._labels(D, null_draw, so)
    scr_perm = None if not scr else [zlib.crc32(str(so).encode()), 5000 + int(null_draw), 0 if si is None else zlib.crc32(str(si).encode())]
    tr = [s for s in sids if s != so]
    cfg = GRID[ci]
    if kind == 'inner':
        tr_in = [s for s in tr if s != si]
        satb_in = satb.copy()
        satb_in[(mt.sid == str(si)).values] = float(np.nanmean(mt[mt.sid.isin(tr_in)].groupby('rec').sat_b.first().values)) \
            if null_draw is None or null_draw < 0 else satb_mean
        prep, net, st = fit(D, tr_in, [si, so], seed, cfg, ylab, satb_in, scr_perm, steps, snaps)
        iv = np.where((mt.sid == str(si)).values)[0]
        res = {step: {c: predict(prep, net, s, D['X'][c][iv], satb_in[iv]) for c in ['clean'] + [('aug', seed % 3, k) for k in range(PN.K_AUG)]}
               for step, s in st}
        return kind, (so, si, seed, ci), dict(idx=iv, ylab=ylab[iv], pred=res)
    prep, net, st = fit(D, tr, [so], seed, cfg, ylab, satb, scr_perm, steps, snaps)
    te = np.where((mt.sid == str(so)).values)[0]
    res = {step: {lv: predict(prep, net, s, D['X']['clean' if lv == 0 else PN.TEST_CONDS[lv]][te], satb[te]) for lv in LEVEL_SET} for step, s in st}
    beta = {step: float(s['beta']) for step, s in st}
    return kind, (so, seed, ci), dict(idx=te, pred=res, beta=beta, n_infer=n_infer_params(net), n_all=PN.n_params(net))


def _crit(P, step, mt):
    idx = np.concatenate([p['idx'] for p in P]); yv = np.concatenate([p['ylab'] for p in P]); t = mt.t.values[idx]
    ok = np.isfinite(yv)
    vals = []
    for c in P[0]['pred'][step]:
        sc = np.concatenate([p['pred'][step][c]['score'] for p in P])
        v = TI._tm(t[ok], yv[ok], (sc[ok] > 0.5).astype(int))
        vals.append(-1.0 if not np.isfinite(v) else v)
    return 0.5 * vals[0] + 0.5 * float(np.mean(vals[1:])), vals[0]


def run_variant(tag, seeds, null_draw=None, scr=False, quick=False, log=print):
    D = PN.load_data(log)
    mt = D['mt']
    sids = sorted(mt.sid.unique())
    outer_s = sids[:1] if quick else sids
    steps, snaps = (30, (15, 30)) if quick else (STEPS, SNAPS)
    grid_i = range(1 if quick else len(GRID))
    jobs = [('inner', so, si, sd, ci, null_draw, scr, steps, snaps) for so in outer_s for sd in seeds for ci in grid_i for si in sids if si != so]
    jobs += [('outer', so, None, sd, ci, null_draw, scr, steps, snaps) for so in outer_s for sd in seeds for ci in grid_i]
    t0 = time.time()
    inner, outer = {}, {}
    def _collect(it):
        for i, (kind, key, r) in enumerate(it):
            (inner if kind == 'inner' else outer)[key] = r
            if (i + 1) % 24 == 0:
                log(f'    {tag}: {i + 1}/{len(jobs)} fits, {time.time() - t0:.0f} s')
    if N_WORKERS == 1:
        _collect(map(_job, jobs))
    else:
        with mp.get_context('fork').Pool(N_WORKERS) as pool:
            _collect(pool.imap_unordered(_job, jobs, chunksize=1))
    log(f'  {tag}: {len(jobs)} fits in {time.time() - t0:.0f} s')
    rows, frames = [], {sd: {lv: [] for lv in LEVEL_SET} for sd in seeds}
    for so in outer_s:
        for sd in seeds:
            best = None
            for ci in grid_i:
                P = [inner[(so, si, sd, ci)] for si in sids if si != so]
                for step in snaps:
                    v, vclean = _crit(P, step, mt)
                    if best is None or v > best[0] + 1e-12:
                        best = (v, ci, step, vclean)
            v, ci, step, vclean = best
            P = [inner[(so, si, sd, ci)] for si in sids if si != so]
            yv = np.concatenate([p['ylab'] for p in P])
            conds = list(P[0]['pred'][step].keys())
            ab = np.concatenate([np.concatenate([p['pred'][step][c]['abst'] for p in P]) for c in conds])
            sc = np.concatenate([np.concatenate([p['pred'][step][c]['score'] for p in P]) for c in conds])
            yy = np.concatenate([yv for _ in conds]); ok = np.isfinite(yy)
            c_star = TI.choose_conf(ab[ok], yy[ok], (sc[ok] > 0.5).astype(int))
            O = outer[(so, sd, ci)]
            rows.append(dict(model=tag, held_out=so, seed=sd, w_rec=GRID[ci]['w_rec'], w_phys=GRID[ci]['w_phys'], steps=step, c_star=c_star,
                             inner_crit=round(v, 4), inner_tm_clean=round(vclean, 4), beta=round(O['beta'][step], 4), n_infer=O['n_infer'], n_all=O['n_all']))
            te = O['idx']
            for lv in LEVEL_SET:
                p = O['pred'][step][lv]
                frames[sd][lv].append(_frame(mt.iloc[te], p['mu'], p['sigma'], p['score'], p['abst'], c_star))
    frames = {sd: {lv: pd.concat(v, ignore_index=True) for lv, v in d.items()} for sd, d in frames.items()}
    return frames, rows


def _frame(m, mu, sigma, score, abst, c_star):
    return pd.DataFrame(dict(rec=m.rec.values, t=m.t.values.astype(int), sid=m.sid.values, y_spo2=m.y_spo2.values, y=(m.y_spo2.values < 90).astype(int),
                             score=np.round(score, 6), pred=(score > 0.5).astype(int), answer=abst >= c_star, conf=np.round(abst, 6),
                             spo2_hat=np.round(mu, 4), spo2_lo=np.round(mu - 1.96 * sigma, 4), spo2_hi=np.round(mu + 1.96 * sigma, 4),
                             sigma=np.round(sigma, 4), c_star=c_star))


def _suf(lv):
    return f'L{lv}' if not isinstance(lv, str) else lv


def _tmq(df):
    return TI._tm(df.t.values, df.y_spo2.values, df.pred.values)


def combine(log=print):
    """Light local step: 3-seed ensemble of the per-seed raw frames -> contract score files (mean score, mean SpO2, mean abstain prob vs mean c*)."""
    SCORES.mkdir(parents=True, exist_ok=True)
    for lv in LEVEL_SET:
        fs = [pd.read_csv(RAW / f'{NAME}_seed{sd}__{_suf(lv)}.csv') for sd in SEEDS]
        b = fs[0]
        for f in fs[1:]:
            assert (f.rec.values == b.rec.values).all() and (f.t.values == b.t.values).all()
        score = np.mean([f.score.values for f in fs], 0)
        mu = np.mean([f.spo2_hat.values for f in fs], 0)
        sg = np.sqrt(np.mean([f.sigma.values ** 2 for f in fs], 0) + np.var([f.spo2_hat.values for f in fs], 0))
        ab = np.mean([f.conf.values for f in fs], 0)
        cs = np.mean([f.c_star.values for f in fs], 0)
        df = pd.DataFrame(dict(rec=b.rec, t=b.t, sid=b.sid, y_spo2=b.y_spo2, y=b.y, score=np.round(score, 6), pred=(score > 0.5).astype(int),
                               answer=ab >= cs, conf=np.round(ab, 6), spo2_hat=np.round(mu, 4), spo2_lo=np.round(mu - 1.96 * sg, 4),
                               spo2_hi=np.round(mu + 1.96 * sg, 4)))
        df.to_csv(SCORES / f'{NAME}__{_suf(lv)}.csv', index=False)
        log(f'{NAME} {_suf(lv)}: TM {_tmq(df):.3f} cov {df.answer.mean():.2f}')
    for sd in SEEDS:
        f = pd.read_csv(RAW / f'{NAME}_seed{sd}__L0.csv')
        f.drop(columns=['sigma', 'c_star']).to_csv(SCORES / f'{NAME}_seed{sd}__L0.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='main')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--draws', default='0,1,2')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.part == 'combine':
        combine()
        return
    RAW.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)
    PN.load_data(log)
    t0 = time.time()
    if a.part == 'main':
        seeds = tuple(int(s) for s in a.seeds.split(','))
        frames, rows = run_variant(NAME, seeds, quick=a.quick, log=log)
        for sd, d in frames.items():
            log(f'{NAME} seed {sd}: ' + ', '.join(f'{lv}: {_tmq(df):.3f}/{df.answer.mean():.2f}' for lv, df in d.items()))
            if not a.quick:
                for lv, df in d.items():
                    df.to_csv(RAW / f'{NAME}_seed{sd}__{_suf(lv)}.csv', index=False)
        tag = 'main_s' + a.seeds.replace(',', '')
    else:
        scr = a.part == 'nullscr'
        base = f'{NAME}_nullscr' if scr else f'{NAME}_null'
        rows = []
        for k in (int(d) for d in a.draws.split(',')):
            frames, r = run_variant(f'{base}_d{k}', (0,), null_draw=k, scr=scr, quick=a.quick, log=log)
            rows += r
            df = frames[0][0]
            log(f'{base} draw {k}: TM {_tmq(df):.3f}')
            if not a.quick:
                df.drop(columns=['sigma', 'c_star']).to_csv(RAW / f'{base}_d{k}__L0.csv', index=False)
        tag = a.part
    log(f'{a.part} done in {time.time() - t0:.0f} s')
    if not a.quick:
        pd.DataFrame(rows).to_csv(RAW / f'choices_{tag}.csv', index=False)
        (RAW / f'run_{tag}.log').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
