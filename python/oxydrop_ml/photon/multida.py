"""Round-3 candidate R3-MULTI, `multida`: ONE network trained jointly on Hoffman training people + MTHS + BUT PPG smartphone PPG, with domain adaptation.

    cd python && MULTIDA_PACK=... MULTIDA_AUX=... python -m oxydrop_ml.photon.multida --models multida_s0     # one model (all levels + X1)
    cd python && python -m oxydrop_ml.photon.multida --build-aux OUT.npz --mths DIR --butppg DIR            # aux feature pack (remote box)
    cd python && python -m oxydrop_ml.photon.multida --ensemble

FRONT END (identical to `distill`, so the browser path is app/js/distill_features.js + app/js/nn_runtime.js): the 39 causal per-window features of
FlashGap (`flashgap.COLS`, `flashgap.rec_features`), 10 s windows on a 2 s hop, baseline-relative to the recording's own label-free first minute.
Auxiliary datasets go through the same `flashgap.rec_features` code from their per-frame ROI-mean RGB (cached log(rgb + 1) series of
oxydrop_ml.windows, inverted with exp(s) - 1):
    MTHS (62 smartphone fingertip videos, 30 Hz RGB, per-second SpO2 / HR; CC BY-NC-ND 4.0, nothing committed): recordings are 26-79 s long, so the
         baseline is the first 30 s (CAL patched to 30 for MTHS only; the info.json of the cache states the same rule); windows t >= 30 s.
    BUT PPG v2.0 (smartphone finger/ear PPG, 10 s clips, per-clip HR; CC BY 4.0): the motion-free clips of one subject are concatenated in clip order
         into one pseudo-recording; the first 6 clips (60 s) are the baseline (as in the cache's info.json); a window is kept only when it coincides with
         one clip (t = 10 k, t >= 60 s). Trailing-30 s beat features span clip joins (disclosed).
    OpenOximetry was NOT used (Daytona capacity was full at launch; Kaggle may not see it).

NETWORK (deployed per outer fold): Hoffman Prep (flashgap.Prep, fitted on the fold's Hoffman training rows) -> 39 -> Linear(32) -> ReLU -> Linear(32)
    -> ReLU -> Linear(4) heads [hyp logit, (SpO2 - 90) / 5, abstain logit, dHR / 10]. The first three heads have the meaning and order of `distill`.
DOMAIN ADAPTATION (training only): per-domain normalisation (a separate Prep per domain, fitted on that domain's rows; Hoffman rows always use the Hoffman
    Prep, which is the only one deployed) + a domain-adversarial head (gradient reversal, coefficient GRL, 32 -> 16 -> ReLU -> 3 domains, domain-balanced
    cross-entropy) on the shared 32-unit trunk.
MULTI-TASK LOSS (per row; a row contributes only to the heads it has labels for):
    Hoffman (clean weight 1, flashgap augmented copies 0.25, class-balanced): the distill loss = BCE(hyp, y) + lam_kd * T^2 * BCE(hyp / T, teacher)
        + 0.3 Huber(spo2) + 0.5 BCE(abstain, 1[hyp correct]); teachers T_chain (clean rows) / T_dc (augmented rows) refit on the fold's training people
        exactly as distill_student.soft_targets.
    MTHS (weight w_aux): BCE(hyp, SpO2 < 90) (19 hypoxic windows), 0.3 Huber(spo2), 0.3 Huber(dHR), dHR = HR - median HR of baseline windows.
    BUT PPG (weight w_aux): 0.3 Huber(dHR).
    The abstain head is trained on Hoffman rows only.
NESTED SELECTION (per outer Hoffman fold; leave-one-inner-person-out on the 5 Hoffman training people; aux rows are in every inner and outer fit, they
    contain no Hoffman person): lam_kd in {2, 8} x w_aux in {0.3, 1} x epochs in {15, 30}; GRL = 0.1, weight decay 1e-4 fixed. Criterion and tau as
    distill (0.5 clean + 0.5 augmented inner OOF TM; flashgap.select_tau on the clean inner OOF abstain outputs). Ties: fewer epochs, less aux, less KD.
    The Hoffman held-out person is never seen. Grid values and GRL were fixed before any multida score, after distill's round-2 outer results
    (lam_kd hit 8 in 18/18 distill folds); disclosed (Amendment 8).
MODELS: multida_s0/s1/s2 (seeds = augmentation seeds 0-2), multida (3-seed ensemble, as distill), multida_hofonly (ablation: w_aux = 0, GRL = 0, seed 0),
    multida_null / multida_null_d1 / multida_null_d2 (seed-0 pipeline, Hoffman training labels + SpO2 permuted globally across training windows, aux
    labels permuted within dataset; teachers see the permuted labels). Score files multida_null__L0, multida_null_d1__L0, multida_null_d2__L0.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .. import splits as SP
from . import distill_student as DS
from . import flashgap as F

SCORES = F.SCORES
RES = F.RES
LEVELS = [0, 1, 2, 3, 4, 5, 'X1']
OUT_DIR = Path(os.environ.get('MULTIDA_OUT', str(Path(DS.CACHE_DIR).parent / 'photon_multida')))
HID, NOUT = 32, 4
GRL = 0.1
W_HR = 0.3
GRID_LAM = (2.0, 8.0)
GRID_AUX = (0.3, 1.0)
GRID_EP = (15, 30)
DOMAINS = ('hoffman', 'mths', 'butppg')
MODELS = {
    'multida_s0': dict(seed=0, aux=True, null=None),
    'multida_s1': dict(seed=1, aux=True, null=None),
    'multida_s2': dict(seed=2, aux=True, null=None),
    'multida_hofonly': dict(seed=0, aux=False, null=None),
    'multida_null': dict(seed=0, aux=True, null=0),
    'multida_null_d1': dict(seed=0, aux=True, null=1),
    'multida_null_d2': dict(seed=0, aux=True, null=2),
}


# ------------------------------------------------------------------------------------------------ auxiliary feature pack (MTHS + BUT PPG)
def _series(d):
    rd = (lambda f: pd.read_csv(os.path.join(d, f + '.csv'))) if os.path.exists(os.path.join(d, 'recs.csv')) else (lambda f: pd.read_parquet(os.path.join(d, f + '.parquet')))
    recs, meta = rd('recs'), rd('meta')
    n = int(recs.offset.iloc[-1] + recs.length.iloc[-1])
    s = np.fromfile(os.path.join(d, 'series.f32'), dtype=np.float32).reshape(n, 3)
    return recs, meta, s


def build_aux(out, mths_dir, but_dir, log=print):
    from oxydrop import chain as CH
    rows = []
    # MTHS: baseline = first 30 s of each recording
    recs, meta, s = _series(mths_dir)
    old = (CH.CAL, F.CAL)
    CH.CAL, F.CAL = 30, 30
    try:
        for _, r in recs.iterrows():
            rgb = np.exp(s[r.offset:r.offset + r.length].astype(np.float64)) - 1
            if len(rgb) < 45 * F.FS:
                continue
            t, X = F.rec_features(rgb)
            m = meta[meta.rec == r.rec].set_index('t_start')
            base_hr = np.nanmedian([m.hr_ref.get(ti, np.nan) for ti in t if ti + F.WIN <= 30])
            for ti, x in zip(t, X):
                if ti < 30 or ti not in m.index:
                    continue
                mr = m.loc[ti]
                rows.append((1, str(r.subject), float(mr.spo2_ref) if bool(mr.has_spo2) else np.nan, float(mr.hr_ref - base_hr) if bool(mr.has_hr) else np.nan, x))
    finally:
        CH.CAL, F.CAL = old
    log('mths windows', len(rows))
    # BUT PPG: motion-free clips of a subject concatenated in clip order, baseline = first 6 clips (60 s)
    recs, meta, s = _series(but_dir)
    hr_of = meta.groupby('rec').hr_ref.first()
    rc = recs[recs.motion.astype(int) == 0].assign(_c=lambda d: d['clip'].astype(int)).sort_values(['subject', '_c'])
    nb = len(rows)
    for subj, g in rc.groupby('subject'):
        g = g[g.length >= 10 * F.FS]
        if len(g) < 9:
            continue
        rgb = np.vstack([np.exp(s[o:o + 10 * F.FS].astype(np.float64)) - 1 for o in g.offset.values])
        t, X = F.rec_features(rgb)
        hrs = hr_of.reindex(g.rec.values).values.astype(float)
        bh = np.nanmedian(hrs[:6])
        for ti, x in zip(t, X):
            if ti < 60 or ti % 10:
                continue
            k = ti // 10
            if k < len(hrs) and np.isfinite(hrs[k]):
                rows.append((2, str(subj), np.nan, float(hrs[k] - bh), x))
    log('butppg windows', len(rows) - nb)
    dom = np.array([r[0] for r in rows]); subj = np.array([r[1] for r in rows]); sp = np.array([r[2] for r in rows]); dh = np.array([r[3] for r in rows])
    X = np.vstack([r[4] for r in rows]).astype(np.float32)
    np.savez_compressed(out, dom=dom, subj=subj, spo2=sp, dhr=dh, X=X)
    return out


_AUX = {}


def aux():
    if not _AUX:
        z = np.load(os.environ['MULTIDA_AUX'], allow_pickle=False)
        _AUX.update({k: z[k] for k in z.files})
    return _AUX


# ------------------------------------------------------------------------------------------------ network
class Net:
    """Deployed network of one fold (numpy): Hoffman Prep + MLP. Same weight names as distill_student.Student (export reuses that layout)."""

    def __init__(self, prep, W):
        self.prep, self.W = prep, W

    def forward(self, X):
        Z = self.prep.transform(X)
        h = np.maximum(Z @ self.W['w1'].T + self.W['b1'], 0)
        h = np.maximum(h @ self.W['w2'].T + self.W['b2'], 0)
        return h @ self.W['wo'].T + self.W['bo']

    def predict(self, X):
        o = self.forward(X)
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return sig(o[:, 0]), 90 + 5 * o[:, 1], sig(o[:, 2])


def _init(rng, fan_in, fan_out):
    b = 1 / np.sqrt(fan_in)                                   # torch.nn.Linear default (kaiming-uniform a = sqrt(5)) bound for W and b
    return rng.uniform(-b, b, (fan_out, fan_in)), rng.uniform(-b, b, fan_out)


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_net(H, A, lam, w_aux, grl, epochs, seed, checkpoints):
    """H: Hoffman training dict (X, y, w, q, spo2). A: aux dict (X, dom, spo2, dhr) or None. Returns {epoch: Net}.
    Plain numpy (manual backprop + Adam with L2 weight decay, torch.optim.Adam semantics), so the remote box needs no torch."""
    prep = F.Prep().fit(H['X'])
    Zs, dom = [prep.transform(H['X'])], [np.zeros(len(H['X']), int)]
    n_h = len(H['X'])
    if A is not None and w_aux > 0:
        for d in (1, 2):
            m = A['dom'] == d
            if m.any():
                Zs.append(F.Prep().fit(A['X'][m]).transform(A['X'][m])); dom.append(np.full(int(m.sum()), d))
        order = np.concatenate([np.nonzero(A['dom'] == d)[0] for d in (1, 2)])
        a_sp, a_dh = A['spo2'][order], A['dhr'][order]
    else:
        a_sp = a_dh = np.zeros(0)
    Z = np.vstack(Zs); D = np.concatenate(dom)
    n = len(Z); ish = (np.arange(n) < n_h).astype(float)
    y = np.concatenate([H['y'], np.where(np.isfinite(a_sp), (a_sp < 90).astype(float), 0.0)]).astype(float)
    has_y = np.concatenate([np.ones(n_h), np.isfinite(a_sp).astype(float)])           # MTHS has SpO2 (and the < 90 label); BUT PPG does not
    pos = float((H['w'] * (H['y'] == 1)).sum()); neg = float((H['w'] * (H['y'] == 0)).sum())
    pw = neg / max(pos, 1e-9)
    wrow = np.concatenate([H['w'], np.full(n - n_h, w_aux)])
    cw = np.where(y == 1, pw, 1.0) * wrow * has_y
    q = np.concatenate([np.clip(H['q'], 1e-4, 1 - 1e-4), np.full(n - n_h, 0.5)])
    Q = _sig(np.log(q / (1 - q)) / DS.T_KD)
    sp = (np.concatenate([np.nan_to_num(H['spo2'], nan=90.0), np.nan_to_num(a_sp, nan=90.0)]) - 90) / 5
    w_sp = wrow * np.concatenate([np.isfinite(H['spo2']), np.isfinite(a_sp)]).astype(float)
    dh = np.concatenate([np.zeros(n_h), np.nan_to_num(a_dh)]) / 10
    w_dh = np.concatenate([np.zeros(n_h), np.isfinite(a_dh).astype(float) * w_aux])
    dcount = np.bincount(D, minlength=3).astype(float)
    w_dom = (n / (len(dcount[dcount > 0]) * np.maximum(dcount, 1)))[D]
    use_dom = grl > 0 and len(Zs) > 1
    rng = np.random.default_rng([7, seed])
    P = {}
    P['w1'], P['b1'] = _init(rng, Z.shape[1], HID)
    P['w2'], P['b2'] = _init(rng, HID, HID)
    P['wo'], P['bo'] = _init(rng, HID, NOUT)
    P['wd1'], P['bd1'] = _init(rng, HID, 16)
    P['wd2'], P['bd2'] = _init(rng, 16, 3)
    keys = list(P) if use_dom else ['w1', 'b1', 'w2', 'b2', 'wo', 'bo']
    m1 = {k: np.zeros_like(P[k]) for k in keys}; m2 = {k: np.zeros_like(P[k]) for k in keys}
    b1_, b2_, eps, lr, wd = 0.9, 0.999, 1e-8, DS.LR, DS.WD
    onehot = np.eye(3)[D]
    out, cps, step = {}, set(checkpoints), 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, DS.BATCH):
            b = perm[i:i + DS.BATCH]
            x = Z[b]
            a1 = x @ P['w1'].T + P['b1']; h1 = np.maximum(a1, 0)
            a2 = h1 @ P['w2'].T + P['b2']; h = np.maximum(a2, 0)
            o = h @ P['wo'].T + P['bo']
            S = max(cw[b].sum(), 1.0)
            z = o[:, 0]
            do = np.zeros_like(o)
            do[:, 0] = cw[b] * (_sig(z) - y[b])
            if lam > 0:
                do[:, 0] += lam * DS.T_KD ** 2 * cw[b] * ish[b] * (_sig(z / DS.T_KD) - Q[b]) / DS.T_KD
            do[:, 1] = DS.W_SPO2 * w_sp[b] * np.clip(o[:, 1] - sp[b], -1, 1)
            corr = ((z > 0).astype(float) == y[b]).astype(float)
            do[:, 2] = DS.W_ABS * wrow[b] * ish[b] * (_sig(o[:, 2]) - corr)
            do[:, 3] = W_HR * w_dh[b] * np.clip(o[:, 3] - dh[b], -1, 1)
            do /= S
            G = {'wo': do.T @ h, 'bo': do.sum(0)}
            dhh = do @ P['wo']
            if use_dom:
                d1 = h @ P['wd1'].T + P['bd1']; hd = np.maximum(d1, 0)
                dl = hd @ P['wd2'].T + P['bd2']
                e = np.exp(dl - dl.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
                ddl = w_dom[b][:, None] * (pr - onehot[b]) / S
                G['wd2'] = ddl.T @ hd; G['bd2'] = ddl.sum(0)
                dd1 = (ddl @ P['wd2']) * (d1 > 0)
                G['wd1'] = dd1.T @ h; G['bd1'] = dd1.sum(0)
                dhh = dhh - grl * (dd1 @ P['wd1'])                 # gradient reversal into the shared trunk
            da2 = dhh * (a2 > 0)
            G['w2'] = da2.T @ h1; G['b2'] = da2.sum(0)
            da1 = (da2 @ P['w2']) * (a1 > 0)
            G['w1'] = da1.T @ x; G['b1'] = da1.sum(0)
            step += 1
            for k in keys:
                g = G[k] + wd * P[k]
                m1[k] = b1_ * m1[k] + (1 - b1_) * g; m2[k] = b2_ * m2[k] + (1 - b2_) * g * g
                P[k] -= lr * (m1[k] / (1 - b1_ ** step)) / (np.sqrt(m2[k] / (1 - b2_ ** step)) + eps)
        if ep in cps:
            out[ep] = Net(prep, {k: P[k].copy() for k in ('w1', 'b1', 'w2', 'b2', 'wo', 'bo')})
    return out


# ------------------------------------------------------------------------------------------------ nested selection per outer fold
def hof_design(Xd, conds, rows, y, s2):
    Xs, ys, w, q = DS.soft_targets(Xd, conds, rows, y)
    return dict(X=Xs, y=ys, w=w, q=q, spo2=np.tile(s2, len(conds)))


def fit_fold(Xd, conds, tr, y_tr, s_tr, t_tr, g_tr, spec, A):
    seed = spec['seed']
    auxs = GRID_AUX if spec['aux'] else (0.0,)
    grl = GRL if spec['aux'] else 0.0
    keys = [(l, a, e) for l in GRID_LAM for a in auxs for e in GRID_EP]
    oof = {k: {c: (np.zeros(len(tr)), np.zeros(len(tr))) for c in conds} for k in keys}
    for ip in np.unique(g_tr):
        itr = np.nonzero(g_tr != ip)[0]; iva = np.nonzero(g_tr == ip)[0]
        H = hof_design(Xd, conds, tr[itr], y_tr[itr], s_tr[itr])
        for lam in GRID_LAM:
            for wa in auxs:
                for e, m in train_net(H, A, lam, wa, grl, max(GRID_EP), seed, GRID_EP).items():
                    for c in conds:
                        p, _, a = m.predict(Xd[c][tr[iva]])
                        oof[(lam, wa, e)][c][0][iva] = p; oof[(lam, wa, e)][c][1][iva] = a
    crit = {}
    for k, d in oof.items():
        clean = DS.tm(t_tr, y_tr, d['clean'][0] > 0.5)
        aug = np.mean([DS.tm(t_tr, y_tr, d[c][0] > 0.5) for c in conds if c != 'clean'])
        crit[k] = (0.5 * clean + 0.5 * aug, clean, aug)
    best = max(crit, key=lambda k: (round(crit[k][0], 6), -k[2], -k[1], -k[0]))
    pc, ac = oof[best]['clean']
    tau, cert, cov, acc = F.select_tau(ac, (pc > 0.5).astype(int), y_tr)
    H = hof_design(Xd, conds, tr, y_tr, s_tr)
    model = train_net(H, A, best[0], best[1], grl, best[2], seed, [best[2]])[best[2]]
    info = dict(lam=best[0], w_aux=best[1], epochs=best[2], inner_crit=round(crit[best][0], 4), inner_tm_clean=round(crit[best][1], 4),
                inner_tm_aug=round(crit[best][2], 4), grid={f'lam={k[0]},aux={k[1]},ep={k[2]}': [round(v, 4) for v in crit[k]] for k in crit},
                tau=round(float(tau), 4), tau_certified=bool(cert), inner_coverage=round(float(cov), 3), inner_answered_acc=round(float(acc), 3))
    return model, tau, info


def null_perm(n, test_group, draw, salt=11):
    return np.random.default_rng([salt, zlib.crc32(str(test_group).encode()), int(draw)]).permutation(n)


def aux_for(spec, test_group):
    if not spec['aux']:
        return None
    A = {k: v.copy() for k, v in aux().items()}
    A['X'] = A['X'].astype(np.float64)
    if spec['null'] is not None:
        for d in (1, 2):
            idx = np.nonzero(A['dom'] == d)[0]
            p = idx[null_perm(len(idx), f'{test_group}_aux{d}', spec['null'], salt=13)]
            A['spo2'][idx] = A['spo2'][p]; A['dhr'][idx] = A['dhr'][p]
    return A


def fit_model(name, Xd, log=print):
    path = OUT_DIR / 'models' / f'{name}.pkl'
    if path.exists():
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    spec = MODELS[name]
    mt = F.meta()
    y, t, sid, s2 = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str), mt.y_spo2.values.astype(float)
    conds = DS.train_conds(spec['seed'])
    folds = []
    for f in SP.loso(sid):
        tr, te = f.train_idx, f.test_idx
        SP.assert_no_leak(sid, tr, te)
        y_tr, s_tr = y[tr].copy(), s2[tr].copy()
        if spec['null'] is not None:
            p = DS.null_perm(len(tr), f.test_group, spec['null'])
            y_tr, s_tr = y_tr[p], s_tr[p]
        t0 = time.time()
        model, tau, info = fit_fold(Xd, conds, tr, y_tr, s_tr, t[tr], sid[tr], spec, aux_for(spec, f.test_group))
        info['seconds'] = round(time.time() - t0, 1)
        log(f'{name} fold {f.test_group}: lam {info["lam"]} aux {info["w_aux"]} ep {info["epochs"]} inner crit {info["inner_crit"]:.3f} '
            f'(clean {info["inner_tm_clean"]:.3f}, aug {info["inner_tm_aug"]:.3f}); tau {tau:.3f} cert {info["tau_certified"]} cov {info["inner_coverage"]:.2f}; '
            f'{info["seconds"]} s', flush=True)
        folds.append(dict(sid=f.test_group, te=te, model=model, tau=tau, info=info))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(folds, fh)
    return folds


def run_models(names, levels=LEVELS, log=print):
    os.environ.setdefault(DS.PACK_ENV, os.environ.get('MULTIDA_PACK', ''))
    DS._pack()
    mt = F.meta(); n = len(mt)
    seeds = sorted({MODELS[m]['seed'] for m in names})
    conds = list(dict.fromkeys(sum([DS.train_conds(s) for s in seeds], []) + [DS.eval_cond(l) for l in levels]))
    Xd, _ = DS.features(conds, log=log)
    for name in names:
        folds = fit_model(name, Xd, log=log)
        lvs = [0] if MODELS[name]['null'] is not None else levels
        for lv in lvs:
            sc, sp, cf, tau = DS.score_raw(folds, Xd[DS.eval_cond(lv)], n)
            df = DS.frame(mt, sc, sp, cf, cf >= tau)
            DS.write(name, lv, df)
            log(f'  {name} {lv}: TM {DS.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}', flush=True)


def ensemble(levels=LEVELS, log=print):
    os.environ.setdefault(DS.PACK_ENV, os.environ.get('MULTIDA_PACK', ''))
    DS._pack()
    mt = F.meta()
    taus = []
    for s in range(3):
        with open(OUT_DIR / 'models' / f'multida_s{s}.pkl', 'rb') as fh:
            folds = pickle.load(fh)
        tau = np.full(len(mt), np.nan)
        for fo in folds:
            tau[fo['te']] = fo['tau']
        taus.append(tau)
    mtau = np.mean(taus, 0)
    for lv in levels:
        tag = f'L{lv}' if not isinstance(lv, str) else lv
        ds = [pd.read_csv(SCORES / f'multida_s{s}__{tag}.csv') for s in range(3)]
        sc = np.mean([d.score.values for d in ds], 0); sp = np.mean([d.spo2_hat.values for d in ds], 0); cf = np.mean([d.conf.values for d in ds], 0)
        df = DS.frame(mt, sc, sp, cf, cf >= mtau)
        DS.write('multida', lv, df)
        log(f'  multida (ensemble) {lv}: TM {DS.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')


def run(level=0, seed=0):
    """CONTRACT entry point (needs MULTIDA_PACK and MULTIDA_AUX): writes and returns the seed-`seed` model's out-of-fold scores at `level`."""
    run_models([f'multida_s{seed}'], [level], log=lambda *a, **k: None)
    tag = f'L{level}' if not isinstance(level, str) else level
    return pd.read_csv(SCORES / f'multida_s{seed}__{tag}.csv')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--levels', nargs='*', default=None)
    ap.add_argument('--ensemble', action='store_true')
    ap.add_argument('--build-aux', default=None)
    ap.add_argument('--mths', default=None)
    ap.add_argument('--butppg', default=None)
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    lv = LEVELS if a.levels is None else [int(x) if x.isdigit() else x for x in a.levels]
    if a.build_aux:
        print(build_aux(a.build_aux, a.mths, a.butppg)); return
    if a.ensemble:
        ensemble(lv); return
    run_models(a.models or list(MODELS), lv, log=print)


if __name__ == '__main__':
    import sys
    from oxydrop_ml.photon import multida as _m
    sys.exit(_m.main())
