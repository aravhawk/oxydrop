"""Round-2 candidate R2B, PhotonDistill: one small student network per outer fold, distilled from teachers fit on that fold's training people only.

    cd python && python -m oxydrop_ml.photon.distill_student                 # seeds 0-2, ensemble, ablation, nulls; all levels + X1
    cd python && python -m oxydrop_ml.photon.distill_student --models distill_s0 --levels 0

DEPLOYED MODEL (the only thing that runs at inference): `Student`, an MLP on the 39 causal per-window features of FlashGap (`flashgap.COLS`: the 19
baseline-relative chain v1 deltas, 15 absolute signal-quality features, 5 baseline-relative respiratory features; every feature is computed from the
per-frame RGB of the window and the person's own label-free first minute), standardised with training-row statistics (`flashgap.Prep`).
    39 -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> three linear heads
        hyp:     logit of P(SpO2 < 90)                        (score = sigmoid; pred = score > 0.5)
        spo2:    (SpO2 - 90) / 5                              (spo2_hat)
        abstain: logit of P(hyp head correct)                (conf; answer = sigmoid >= tau)
No gate, no teacher at inference, no averaging with chain_v1. The student decides from the signal (SNR, perfusion, DC, HR features together) how much
the heart-rate cues and the DC-colour cues count.

TEACHERS (refit inside every outer fold AND every inner fold, on the training people of that fold only):
    T_chain: logistic chain v1 exactly as shipped (flashgap.ChainExpert: mean of the class-balanced LR on dHR, C = 0.01, and on the 19 deltas, C = 0.1),
             fitted on CLEAN training rows; its probability is the soft target of the clean rows.
    T_dc:    class-balanced LR (C = 0.1, StandardScaler) on dc_R, dc_G, dc_B (the ctl_dc_lr cue of controls.py), fitted on clean + augmented training rows;
             its probability is the soft target of the augmented (degraded) rows.
    The PhotonTwin physics path (twin_inverse.py) was NOT used as a teacher (time budget of round 2; disclosed). Round 1 showed ctl_dc_lr about as robust as
    the physics-only path, so T_dc stands in for the robust teacher.
LOSS (per row, rows weighted 1 clean / 0.25 per augmented copy, class-balanced positive weight from the training rows):
    BCE(hyp, y) + lam_kd * T^2 * BCE(hyp / T, tempered teacher probability) + 0.3 * Huber(spo2) + 0.5 * BCE(abstain, 1[hyp correct], hyp detached)
    with T = 2 (fixed). Adam (lr 3e-3, weight decay nested), batch 512.
NESTED SELECTION (per outer fold, leave-one-inner-person-out over the 5 training people; teachers refit per inner fold):
    v2 (primary, launched 18:53 CT): lam_kd in {0.5, 2, 8} x weight decay in {1e-4, 1e-2} x epochs in {15, 30};
    v1 (18:47 CT, kept as distill_v1_* score files): lam_kd in {0.5, 2} x epochs in {15, 30}, weight decay 1e-4. v1 lost to chain_v1 at L0 (-0.10) and
    picked the largest lam_kd in most folds; v2 widened the grid after seeing v1's held-out numbers (post hoc, disclosed in results/photon/distill_log.md).
    Ties: fewer epochs, more decay, less KD.
    Common to both: criterion 0.5 TM(clean inner OOF) + 0.5 mean TM(augmented inner OOF copies) (same criterion family as
    FlashGap). tau = flashgap.select_tau on the clean inner OOF abstain-head outputs of the chosen configuration (answered accuracy and balanced accuracy
    >= 0.90 with the +1 correction, largest passing coverage). The held-out person is never seen by teachers, student, selection or tau.
AUGMENTATION PRIOR: FlashGap's declared distant-flash prior (flashgap.AUG_PRIOR, flash_augment, K = 4 copies per training recording) with augmentation seed
    = student seed (0, 1, 2), cached features of flashgap; disjoint from stress.py (level, seed) pairs and from X1 by construction (flashgap.assert_disjoint
    is called here). The L1-L5 levels sit inside that prior (in-simulator, optimistic); X1 is out of family.
EVALUATION: held-out person's recordings degraded by stress.degrade(rgb, level, 0) or the X1 function of stress.py (seed 0; whole recording; X1 features come precomputed from the cache); the student is not refit.
REMOTE RUNS: `--build-pack PATH` writes every feature matrix (float32) and the window table to one .npz (disjointness asserted there); with DISTILL_PACK=PATH
the training reads only that pack (no raw recordings, no legacy `oxydrop` package), which is how the score files were produced on a Daytona sandbox.
Models: distill_s0, distill_s1, distill_s2 (seeds), distill (3-seed ensemble: mean score, mean spo2, mean abstain prob vs mean tau), distill_nokd
(ablation: lam_kd = 0, epochs nested), distill_null / distill_d1_null / distill_d2_null (seed 0 pipeline with the training labels and SpO2 permuted
globally across training windows, permutation draws 0-2; the teachers see the permuted labels too).
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

from .. import CACHE
from .. import metrics as M
from .. import splits as SP
from . import flashgap as F

RES = F.RES
SCORES = F.SCORES
CACHE_DIR = Path(CACHE) / 'photon_distill'
LEVELS = [0, 1, 2, 3, 4, 5, 'X1']
HID = 32
T_KD = 2.0
W_SPO2 = 0.3
W_ABS = 0.5
LR, WD, BATCH = 3e-3, 1e-4, 512
GRID_LAM = (0.5, 2.0, 8.0)                 # v1 (18:47 CT run, files distill_v1_*): (0.5, 2.0)
GRID_WD = (1e-4, 1e-2)                     # v1: WD only
GRID_EP = (15, 30)
DC_COLS = [F.COLS.index(c) for c in ('dc_R', 'dc_G', 'dc_B')]
MODELS = {
    'distill_s0': dict(seed=0, kd=True, null=None),
    'distill_s1': dict(seed=1, kd=True, null=None),
    'distill_s2': dict(seed=2, kd=True, null=None),
    'distill_nokd': dict(seed=0, kd=False, null=None),
    'distill_null': dict(seed=0, kd=True, null=0),
    'distill_d1_null': dict(seed=0, kd=True, null=1),
    'distill_d2_null': dict(seed=0, kd=True, null=2),
}


def _torch():
    import torch
    torch.set_num_threads(2)
    return torch


# ------------------------------------------------------------------------------------------------ features (FlashGap's cache, read-only; stress levels in our cache)
def _patch_cache():
    """Clean and augmented features are read from FlashGap's cache (identical code, read only). Stress-level features depend on the current stress.py, so
    they live in this candidate's own cache keyed by the stress.py hash (FlashGap's stress cache is for an older stress.py)."""
    orig = F._cache_path

    def path(cond, ri):
        if cond != 'clean' and cond[0] in ('L', 'X'):
            return CACHE_DIR / f'{F.FEAT_VERSION}_st{F.stress_hash()}' / f'{F.cond_name(cond)}_r{ri:02d}.npz'
        return orig(cond, ri)
    if getattr(F._cache_path, '_distill', False):
        return
    path._distill = True
    F._cache_path = path


PACK_ENV = 'DISTILL_PACK'
_PACK = {}


def pack_conds(seeds=(0, 1, 2), levels=LEVELS):
    return list(dict.fromkeys(sum([train_conds(s) for s in seeds], []) + [eval_cond(l) for l in levels]))


def build_pack(path, seeds=(0, 1, 2), levels=LEVELS, log=print):
    """Compact float32 pack of every feature matrix this candidate needs (for remote training on Daytona without the raw recordings or the legacy
    `oxydrop` package). The augmentation / stress disjointness is asserted here, on the box that has the stress module and the draws."""
    conds = pack_conds(seeds, levels)
    Xd, params = features(conds, jobs=1, log=log)
    keys = [F.aug_key(s, ri, k) for s in seeds for ri in range(len(F.recordings())) for k in range(F.K_COPIES)]
    F.assert_disjoint(keys, [(l, 0) for l in levels], [params[(s, ri, k)] for s in seeds for ri in range(len(F.recordings())) for k in range(F.K_COPIES)])
    mt = F.meta()
    arr = {f'X_{F.cond_name(c)}': Xd[c].astype(np.float32) for c in conds}
    np.savez_compressed(path, rec=np.array(mt.rec.astype(str).tolist(), dtype='U'), t=mt.t.values, sid=mt.sid.values, y_spo2=mt.y_spo2.values, y=mt.y.values, ri=mt.ri.values,
                        stress_hash=F.stress_hash(), disjoint_checked=True, n_aug_params=len(params), **arr)
    return path


def _pack():
    p = os.environ.get(PACK_ENV)
    if not p:
        return None
    if not _PACK:
        z = np.load(p, allow_pickle=False)
        _PACK.update({k: z[k] for k in z.files})
        mt = pd.DataFrame(dict(rec=_PACK['rec'], t=_PACK['t'], sid=_PACK['sid'], y_spo2=_PACK['y_spo2'], y=_PACK['y'], ri=_PACK['ri']))
        F.meta = lambda: mt                                  # pack mode: windows from the pack, raw recordings never loaded
    return _PACK


def features(conds, jobs=2, log=print):
    pk = _pack()
    if pk is not None:
        assert bool(pk['disjoint_checked'])
        return {c: pk[f'X_{F.cond_name(c)}'].astype(np.float64) for c in conds}, None
    _patch_cache()
    return F.compute_features(conds, jobs=jobs, log=log)


def train_conds(seed):
    return ['clean'] + [('aug', seed, k) for k in range(F.K_COPIES)]


def eval_cond(level, seed=0):
    return F.eval_cond(level, seed)


# ------------------------------------------------------------------------------------------------ teachers
class DCTeacher:
    def fit(self, X, y, w):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X[:, DC_COLS])
        self.lr = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000).fit(self.sc.transform(X[:, DC_COLS]), y, sample_weight=w)
        return self

    def predict(self, X):
        return self.lr.predict_proba(self.sc.transform(X[:, DC_COLS]))[:, 1]


def soft_targets(Xd, conds, rows, y):
    """Teachers fitted on `rows` (training people of the current fold) with labels y; returns the stacked design and soft targets of every condition."""
    chain = F.ChainExpert().fit(Xd['clean'][rows], y)
    Xs = np.vstack([Xd[c][rows] for c in conds])
    ys = np.tile(y, len(conds))
    w = np.concatenate([np.full(len(rows), 1.0 if c == 'clean' else F.AUG_WEIGHT) for c in conds])
    dc = DCTeacher().fit(Xs, ys, w)
    q = np.concatenate([chain.predict(Xd[c][rows]) if c == 'clean' else dc.predict(Xd[c][rows]) for c in conds])
    return Xs, ys, w, q


# ------------------------------------------------------------------------------------------------ the student
class Student:
    """Fitted student of one fold: Prep + MLP weights (numpy) so it can be pickled and exported without torch."""

    def __init__(self, prep, W):
        self.prep, self.W = prep, W

    def forward(self, X):
        Z = self.prep.transform(X)
        h = np.maximum(Z @ self.W['w1'].T + self.W['b1'], 0)
        h = np.maximum(h @ self.W['w2'].T + self.W['b2'], 0)
        o = h @ self.W['wo'].T + self.W['bo']
        return o

    def predict(self, X):
        o = self.forward(X)
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return sig(o[:, 0]), 90 + 5 * o[:, 1], sig(o[:, 2])


def train_student(Xs, ys, w, q, spo2, lam, epochs, seed, checkpoints=None, wd=WD):
    """Train one student. Returns {epoch: Student} for each checkpoint epoch."""
    torch = _torch()
    nn = torch.nn
    prep = F.Prep().fit(Xs)
    Z = torch.tensor(prep.transform(Xs), dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32)
    Wt = torch.tensor(w, dtype=torch.float32)
    tq = torch.tensor(np.clip(q, 1e-4, 1 - 1e-4), dtype=torch.float32)
    lq = torch.log(tq / (1 - tq)) / T_KD
    Q = torch.sigmoid(lq)                                   # tempered teacher probability
    S2 = torch.tensor((np.nan_to_num(spo2, nan=90.0) - 90) / 5, dtype=torch.float32)
    pos = float((w * (ys == 1)).sum()); neg = float((w * (ys == 0)).sum())
    pw = neg / max(pos, 1e-9)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 3))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    cw = torch.where(Y == 1, torch.tensor(pw), torch.tensor(1.0)) * Wt
    bce = nn.functional.binary_cross_entropy_with_logits
    out = {}
    cps = set(checkpoints or [epochs])
    n = len(Z)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            o = net(Z[b]); z = o[:, 0]
            l = (bce(z, Y[b], reduction='none') * cw[b]).sum()
            if lam > 0:
                l = l + lam * (T_KD ** 2) * (bce(z / T_KD, Q[b], reduction='none') * cw[b]).sum()
            l = l + W_SPO2 * (nn.functional.huber_loss(o[:, 1], S2[b], reduction='none') * Wt[b]).sum()
            corr = ((z.detach() > 0).float() == Y[b]).float()
            l = l + W_ABS * (bce(o[:, 2], corr, reduction='none') * Wt[b]).sum()
            l = l / cw[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        if ep in cps:
            sd = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
            out[ep] = Student(prep, dict(w1=sd['0.weight'], b1=sd['0.bias'], w2=sd['2.weight'], b2=sd['2.bias'], wo=sd['4.weight'], bo=sd['4.bias']))
    return out


def tm(t, y, pred):
    return F.tm(t, y, pred)


def fit_fold(Xd, conds, tr, y_tr, s_tr, t_tr, g_tr, spec):
    """Nested selection (leave-one-inner-person-out) and final fit for one outer fold. Only rows `tr` (and their labels y_tr, possibly permuted) are used."""
    seed = spec['seed']
    lams = GRID_LAM if spec['kd'] else (0.0,)
    people = np.unique(g_tr)
    oof = {(l, d, e): {c: (np.zeros(len(tr)), np.zeros(len(tr))) for c in conds} for l in lams for d in GRID_WD for e in GRID_EP}
    for ip in people:
        itr = np.nonzero(g_tr != ip)[0]; iva = np.nonzero(g_tr == ip)[0]
        Xs, ys, w, q = soft_targets(Xd, conds, tr[itr], y_tr[itr])
        sp = np.tile(s_tr[itr], len(conds))
        for lam in lams:
            for wd in GRID_WD:
                st = train_student(Xs, ys, w, q, sp, lam, max(GRID_EP), seed, checkpoints=GRID_EP, wd=wd)
                for e, m in st.items():
                    for c in conds:
                        p, a = m.predict(Xd[c][tr[iva]])[0::2]
                        oof[(lam, wd, e)][c][0][iva] = p; oof[(lam, wd, e)][c][1][iva] = a
    crit = {}
    for k, d in oof.items():
        clean = tm(t_tr, y_tr, d['clean'][0] > 0.5)
        aug = np.mean([tm(t_tr, y_tr, d[c][0] > 0.5) for c in conds if c != 'clean'])
        crit[k] = (0.5 * clean + 0.5 * aug, clean, aug)
    best = max(crit, key=lambda k: (round(crit[k][0], 6), -k[2], k[1], -k[0]))    # ties: fewer epochs, more decay, less KD
    pc, ac = oof[best]['clean']
    tau, cert, cov, acc = F.select_tau(ac, (pc > 0.5).astype(int), y_tr)
    Xs, ys, w, q = soft_targets(Xd, conds, tr, y_tr)
    model = train_student(Xs, ys, w, q, np.tile(s_tr, len(conds)), best[0], best[2], seed, wd=best[1])[best[2]]
    info = dict(lam=best[0], wd=best[1], epochs=best[2], inner_crit=round(crit[best][0], 4), inner_tm_clean=round(crit[best][1], 4), inner_tm_aug=round(crit[best][2], 4),
                grid={f'lam={k[0]},wd={k[1]:g},ep={k[2]}': [round(v, 4) for v in crit[k]] for k in crit}, tau=round(tau, 4), tau_certified=bool(cert),
                inner_coverage=round(cov, 3), inner_answered_acc=round(acc, 3))
    return model, tau, info


def null_perm(n, test_group, draw):
    rng = np.random.default_rng([11, zlib.crc32(str(test_group).encode()), int(draw)])
    return rng.permutation(n)


def fit_model(name, Xd, log=print):
    path = CACHE_DIR / 'models' / f'{name}.pkl'
    if path.exists():
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    spec = MODELS[name]
    mt = F.meta()
    y, t, sid, s2 = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str), mt.y_spo2.values.astype(float)
    conds = train_conds(spec['seed'])
    folds = []
    for f in SP.loso(sid):
        tr, te = f.train_idx, f.test_idx
        SP.assert_no_leak(sid, tr, te)
        y_tr, s_tr = y[tr].copy(), s2[tr].copy()
        if spec['null'] is not None:
            p = null_perm(len(tr), f.test_group, spec['null'])
            y_tr, s_tr = y_tr[p], s_tr[p]
        t0 = time.time()
        model, tau, info = fit_fold(Xd, conds, tr, y_tr, s_tr, t[tr], sid[tr], spec)
        info['seconds'] = round(time.time() - t0, 1)
        log(f'{name} fold {f.test_group}: lam {info["lam"]} wd {info["wd"]:g} epochs {info["epochs"]} inner crit {info["inner_crit"]:.3f} (clean {info["inner_tm_clean"]:.3f}, '
            f'aug {info["inner_tm_aug"]:.3f}); tau {tau:.3f} ({"certified" if info["tau_certified"] else "NOT certified"}, inner cov {info["inner_coverage"]:.2f}, '
            f'acc {info["inner_answered_acc"]:.3f}); {info["seconds"]} s')
        folds.append(dict(sid=f.test_group, te=te, model=model, tau=tau, info=info))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(folds, fh)
    return folds


def frame(mt, score, spo2, conf, ans):
    return pd.DataFrame(dict(rec=mt.rec.values, t=mt.t.values, sid=mt.sid.values, y_spo2=mt.y_spo2.values, y=mt.y.values, score=score,
                             pred=(score > 0.5).astype(int), answer=ans.astype(bool), spo2_hat=spo2, conf=conf))


def score_raw(folds, X, n):
    sc, sp, cf, tau = (np.full(n, np.nan) for _ in range(4))
    for fo in folds:
        te = fo['te']
        p, s, a = fo['model'].predict(X[te])
        sc[te], sp[te], cf[te], tau[te] = p, s, a, fo['tau']
    assert np.isfinite(sc).all()
    return sc, sp, cf, tau


def write(name, lv, df):
    SCORES.mkdir(parents=True, exist_ok=True)
    tag = f'L{lv}' if not isinstance(lv, str) else lv
    p = SCORES / f'{name}__{tag}.csv'
    df.to_csv(p, index=False, float_format='%.6g')
    return p


def run_models(names, levels=LEVELS, log=print):
    _pack()
    mt = F.meta(); n = len(mt)
    seeds = sorted({MODELS[m]['seed'] for m in names})
    conds = list(dict.fromkeys(sum([train_conds(s) for s in seeds], []) + [eval_cond(l) for l in levels]))
    Xd, params = features(conds, log=log)
    if params is not None:                                   # pack mode: asserted when the pack was built (build_pack)
        keys = [F.aug_key(s, ri, k) for s in seeds for ri in range(len(F.recordings())) for k in range(F.K_COPIES)]
        F.assert_disjoint(keys, [(l, 0) for l in levels], [params[(s, ri, k)] for s in seeds for ri in range(len(F.recordings())) for k in range(F.K_COPIES)])
    raw = {}
    for name in names:
        folds = fit_model(name, Xd, log=log)
        lvs = [0] if MODELS[name]['null'] is not None else levels
        for lv in lvs:
            sc, sp, cf, tau = score_raw(folds, Xd[eval_cond(lv)], n)
            raw[(name, lv)] = (sc, sp, cf, tau)
            df = frame(mt, sc, sp, cf, cf >= tau)
            write(name, lv, df)
            log(f'  {name} {lv}: TM {tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')
    return raw


def ensemble(levels=LEVELS, log=print):
    """distill = 3-seed ensemble from the per-seed score files (mean score, mean spo2_hat, mean abstain prob; answer = mean conf >= mean tau)."""
    _pack()
    mt = F.meta()
    taus = {}
    for s in range(3):
        with open(CACHE_DIR / 'models' / f'distill_s{s}.pkl', 'rb') as fh:
            folds = pickle.load(fh)
        tau = np.full(len(mt), np.nan)
        for fo in folds:
            tau[fo['te']] = fo['tau']
        taus[s] = tau
    mtau = np.mean([taus[s] for s in range(3)], 0)
    for lv in levels:
        tag = f'L{lv}' if not isinstance(lv, str) else lv
        ds = [pd.read_csv(SCORES / f'distill_s{s}__{tag}.csv') for s in range(3)]
        sc = np.mean([d.score.values for d in ds], 0); sp = np.mean([d.spo2_hat.values for d in ds], 0); cf = np.mean([d.conf.values for d in ds], 0)
        df = frame(mt, sc, sp, cf, cf >= mtau)
        write('distill', lv, df)
        log(f'  distill (ensemble) {lv}: TM {tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')


def run(level=0, seed=0):
    """CONTRACT entry point: out-of-fold scores of the seed-`seed` student at `level` (stress seed 0)."""
    raw = run_models([f'distill_s{seed}'], [level], log=lambda *a: None)
    sc, sp, cf, tau = raw[(f'distill_s{seed}', level)]
    return frame(F.meta(), sc, sp, cf, cf >= tau)


def fold_info(names):
    out = {}
    for n in names:
        p = CACHE_DIR / 'models' / f'{n}.pkl'
        if p.exists():
            with open(p, 'rb') as fh:
                out[n] = [dict(sid=fo['sid'], **fo['info']) for fo in pickle.load(fh)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--levels', nargs='*', default=None)
    ap.add_argument('--ensemble', action='store_true')
    ap.add_argument('--info', action='store_true')
    ap.add_argument('--build-pack', default=None, help='write the compact feature pack to this path and exit')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    lv = LEVELS if a.levels is None else [int(x) if x.isdigit() else x for x in a.levels]
    if a.build_pack:
        print(build_pack(a.build_pack)); return
    if a.info:
        print(json.dumps(fold_info(list(MODELS)), indent=1)); return
    if a.ensemble:
        ensemble(lv); return
    run_models(a.models or list(MODELS), lv)
    if a.models is None:
        ensemble(lv)


if __name__ == '__main__':
    import sys
    from oxydrop_ml.photon import distill_student as _m   # pickles refer to the package module, not __main__
    sys.exit(_m.main())
