"""Round-3 candidate R3-DISTILL+, PhotonDistill2: the PhotonDistill student (same 39-feature front end, same 39 -> H -> H -> 3 MLP that
app/js/nn_runtime.js runs through distill_app_export.py's graph) with stronger teachers and a wider nested search.

    cd python && DISTILL_PACK=.../distill_pack.npz python -m oxydrop_ml.photon.distill2 --model distill2_s0 --fold 0   # one outer fold (parallel runs)
    cd python && DISTILL_PACK=... python -m oxydrop_ml.photon.distill2 --collect                                      # score files + ensemble
    cd python && DISTILL_PACK=... DISTILL_FIXTURE=... python -m oxydrop_ml.photon.distill2 --export                    # results/photon/distill2_weights.json

DEPLOYED MODEL: identical in form to distill_student.Student: flashgap.Prep (median impute, clip, standardise with training statistics) on the 39 causal
features (flashgap.COLS), then Linear(H) -> ReLU -> Linear(H) -> ReLU -> Linear(3) (hyp logit, (SpO2 - 90) / 5, abstain logit); answer = sigmoid(abstain) >= tau.
Depth is fixed at 2 hidden layers because the app export format (distill_app_export.graph) is a 2-hidden-layer graph; width H is nested.
No teacher, gate, or chain_v1 at inference.

TEACHERS (refit inside every outer fold and every inner fold, on that fold's training people only):
    base  (= distill): T_chain (flashgap.ChainExpert, clean rows) for clean rows; T_dc (LR on dc_R, dc_G, dc_B, clean + augmented rows) for augmented rows.
    gbdt: the base teachers averaged with a gradient-boosted tree ensemble on all 39 features (sklearn HistGradientBoostingClassifier, the LightGBM
          histogram algorithm; lightgbm itself is not installed on the sandbox): GBDT_clean fit on clean rows teaches clean rows (mean with T_chain),
          GBDT_all fit on clean + augmented rows teaches augmented rows (mean with T_dc).
    NOT used (disclosed in results/photon/distill2_log.md): the twin_inverse physics path (needs the raw per-frame RGB and the optics renderer, not in the
    feature pack, and the 20:15 CT deadline left no time to recompute it remotely); oxyseq / oxytrack outputs (their saved scores are LOSO outputs whose
    models saw other folds' held-out people, so they would leak; refitting them per inner fold was out of time).
LOSS: (1 - pure) * BCE(hyp, y) + lam * T^2 * BCE(hyp / T, sigmoid(logit(q) / T)) + 0.3 Huber(spo2) + 0.5 BCE(abstain, hyp correct); pure KD drops the label
    term and uses lam = 1. Rows weighted 1 clean / 0.25 augmented, class balanced. Adam lr 3e-3, batch 512 (as distill).
NESTED SELECTION (leave-one-inner-person-out over the 5 training people of the outer fold; criterion 0.5 TM(clean inner OOF) + 0.5 mean TM(augmented inner
    OOF copies), as distill), two stages, both on inner folds only:
    stage 1: teacher {base, gbdt} x lam {8, 16, 32, pure} x T {2, 4}, at H = 32, weight decay 1e-4, epochs {15, 30} (checkpoints of one run);
    stage 2: at the stage-1 teacher / lam / T: H {16, 32, 64} x weight decay {1e-4, 1e-2}, epochs {15, 30}.
    Ties: fewer epochs, smaller H, more decay. tau = flashgap.select_tau on the clean inner OOF abstain output of the chosen configuration.
AUGMENTATION, STRESS, NULLS: as distill_student (FlashGap's declared prior, augmentation seed = student seed; held-out people degraded by stress.py seed 0;
    nulls permute labels and SpO2 globally across the outer fold's training windows, draws 0-2, teachers see the permuted labels).
Models: distill2_s0..s2 (seeds), distill2 (3-seed ensemble), distill2_null / distill2_d1_null / distill2_d2_null (seed 0 pipeline, null draws 0-2).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CACHE
from .. import splits as SP
from . import distill_student as D
from . import flashgap as F

RES = F.RES
SCORES = F.SCORES
CACHE2 = Path(CACHE) / 'photon_distill2'
LEVELS = D.LEVELS
LR, BATCH = 3e-3, 512
S1_TEACH, S1_LAM, S1_T = ('base', 'gbdt'), (8.0, 16.0, 32.0, 'pure'), (2.0, 4.0)
S2_H, S2_WD = (16, 32, 64), (1e-4, 1e-2)
EPOCHS = (15, 30)
H0, WD0 = 32, 1e-4
MODELS = {
    'distill2_s0': dict(seed=0, null=None), 'distill2_s1': dict(seed=1, null=None), 'distill2_s2': dict(seed=2, null=None),
    'distill2_null': dict(seed=0, null=0), 'distill2_d1_null': dict(seed=0, null=1), 'distill2_d2_null': dict(seed=0, null=2),
}


def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    return torch


class GBDT:
    def fit(self, X, y, w):
        from sklearn.ensemble import HistGradientBoostingClassifier
        pos = float((w * (y == 1)).sum()); neg = float((w * (y == 0)).sum())
        sw = w * np.where(y == 1, neg / max(pos, 1e-9), 1.0)
        self.m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                                random_state=0).fit(X, y, sample_weight=sw)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def teachers(Xd, conds, rows, y):
    """Both teacher sets fitted on `rows` only. Returns the stacked design, labels, weights and {teacher set: soft targets}."""
    chain = F.ChainExpert().fit(Xd['clean'][rows], y)
    Xs = np.vstack([Xd[c][rows] for c in conds])
    ys = np.tile(y, len(conds))
    w = np.concatenate([np.full(len(rows), 1.0 if c == 'clean' else F.AUG_WEIGHT) for c in conds])
    dc = D.DCTeacher().fit(Xs, ys, w)
    gc = GBDT().fit(Xd['clean'][rows], y, np.ones(len(rows)))
    ga = GBDT().fit(Xs, ys, w)
    base, gb = [], []
    for c in conds:
        X = Xd[c][rows]
        if c == 'clean':
            b = chain.predict(X); g = 0.5 * (b + gc.predict(X))
        else:
            b = dc.predict(X); g = 0.5 * (b + ga.predict(X))
        base.append(b); gb.append(g)
    return Xs, ys, w, dict(base=np.concatenate(base), gbdt=np.concatenate(gb))


def train(Xs, ys, w, q, spo2, lam, T, H, wd, seed, checkpoints):
    torch = _torch()
    nn = torch.nn
    prep = F.Prep().fit(Xs)
    Z = torch.tensor(prep.transform(Xs), dtype=torch.float32)
    Y = torch.tensor(ys, dtype=torch.float32)
    Wt = torch.tensor(w, dtype=torch.float32)
    tq = np.clip(q, 1e-4, 1 - 1e-4)
    Q = torch.sigmoid(torch.tensor(np.log(tq / (1 - tq)) / T, dtype=torch.float32))
    S2 = torch.tensor((np.nan_to_num(spo2, nan=90.0) - 90) / 5, dtype=torch.float32)
    pure = lam == 'pure'
    lamv = 1.0 if pure else float(lam)
    pos = float((w * (ys == 1)).sum()); neg = float((w * (ys == 0)).sum())
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 3))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    cw = torch.where(Y == 1, torch.tensor(neg / max(pos, 1e-9)), torch.tensor(1.0)) * Wt
    bce = nn.functional.binary_cross_entropy_with_logits
    out, n = {}, len(Z)
    for ep in range(1, max(checkpoints) + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            o = net(Z[b]); z = o[:, 0]
            l = lamv * (T ** 2) * (bce(z / T, Q[b], reduction='none') * cw[b]).sum()
            if not pure:
                l = l + (bce(z, Y[b], reduction='none') * cw[b]).sum()
            l = l + D.W_SPO2 * (nn.functional.huber_loss(o[:, 1], S2[b], reduction='none') * Wt[b]).sum()
            corr = ((z.detach() > 0).float() == Y[b]).float()
            l = l + D.W_ABS * (bce(o[:, 2], corr, reduction='none') * Wt[b]).sum()
            l = l / cw[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        if ep in checkpoints:
            sd = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
            out[ep] = D.Student(prep, dict(w1=sd['0.weight'], b1=sd['0.bias'], w2=sd['2.weight'], b2=sd['2.bias'], wo=sd['4.weight'], bo=sd['4.bias']))
    return out


def _key(k):
    return f'teach={k[0]},lam={k[1]},T={k[2]:g},H={k[3]},wd={k[4]:g},ep={k[5]}'


def fit_fold(Xd, conds, tr, y_tr, s_tr, t_tr, g_tr, seed, log=print):
    people = np.unique(g_tr)
    inner = []
    for ip in people:
        itr = np.nonzero(g_tr != ip)[0]; iva = np.nonzero(g_tr == ip)[0]
        Xs, ys, w, qs = teachers(Xd, conds, tr[itr], y_tr[itr])
        inner.append((iva, Xs, ys, w, qs, np.tile(s_tr[itr], len(conds))))
    oof, crit = {}, {}

    def evaluate(cfgs):
        for (teach, lam, T, H, wd) in cfgs:
            ks = [(teach, lam, T, H, wd, e) for e in EPOCHS]
            for k in ks:
                oof[k] = {c: (np.zeros(len(tr)), np.zeros(len(tr))) for c in conds}
            for iva, Xs, ys, w, qs, sp in inner:
                st = train(Xs, ys, w, qs[teach], sp, lam, T, H, wd, seed, EPOCHS)
                for e, m in st.items():
                    for c in conds:
                        p, _, a = m.predict(Xd[c][tr[iva]])
                        oof[(teach, lam, T, H, wd, e)][c][0][iva] = p; oof[(teach, lam, T, H, wd, e)][c][1][iva] = a
            for k in ks:
                d = oof[k]
                clean = D.tm(t_tr, y_tr, d['clean'][0] > 0.5)
                aug = np.mean([D.tm(t_tr, y_tr, d[c][0] > 0.5) for c in conds if c != 'clean'])
                crit[k] = (0.5 * clean + 0.5 * aug, clean, aug)

    rank = lambda k: (round(crit[k][0], 6), -k[5], -k[3], k[4])          # ties: fewer epochs, smaller H, more decay
    evaluate([(te, l, T, H0, WD0) for te in S1_TEACH for l in S1_LAM for T in S1_T])
    b1 = max(crit, key=rank)
    evaluate([(b1[0], b1[1], b1[2], H, wd) for H in S2_H for wd in S2_WD if (H, wd) != (H0, WD0)])
    best = max(crit, key=rank)
    pc, ac = oof[best]['clean']
    tau, cert, cov, acc = F.select_tau(ac, (pc > 0.5).astype(int), y_tr)
    Xs, ys, w, qs = teachers(Xd, conds, tr, y_tr)
    teach, lam, T, H, wd, ep = best
    model = train(Xs, ys, w, qs[teach], np.tile(s_tr, len(conds)), lam, T, H, wd, seed, (ep,))[ep]
    info = dict(teacher=teach, lam=lam, T=T, H=H, wd=wd, epochs=ep, stage1=_key(b1), inner_crit=round(crit[best][0], 4), inner_tm_clean=round(crit[best][1], 4),
                inner_tm_aug=round(crit[best][2], 4), grid={_key(k): [round(v, 4) for v in crit[k]] for k in crit}, tau=round(tau, 4), tau_certified=bool(cert),
                inner_coverage=round(cov, 3), inner_answered_acc=round(acc, 3))
    return model, tau, info


def fold_path(name, k):
    return CACHE2 / 'models' / f'{name}_f{k}.pkl'


def run_fold(name, k, log=print):
    D._pack()
    spec = MODELS[name]
    mt = F.meta()
    y, t, sid, s2 = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str), mt.y_spo2.values.astype(float)
    conds = D.train_conds(spec['seed'])
    Xd, _ = D.features(conds, log=log)
    f = list(SP.loso(sid))[k]
    tr, te = f.train_idx, f.test_idx
    SP.assert_no_leak(sid, tr, te)
    y_tr, s_tr = y[tr].copy(), s2[tr].copy()
    if spec['null'] is not None:
        p = D.null_perm(len(tr), f.test_group, spec['null'])
        y_tr, s_tr = y_tr[p], s_tr[p]
    t0 = time.time()
    model, tau, info = fit_fold(Xd, conds, tr, y_tr, s_tr, t[tr], sid[tr], spec['seed'], log)
    info['seconds'] = round(time.time() - t0, 1)
    log(f'{name} fold {f.test_group}: {info["teacher"]} lam {info["lam"]} T {info["T"]} H {info["H"]} wd {info["wd"]:g} ep {info["epochs"]}; inner crit '
        f'{info["inner_crit"]:.3f} (clean {info["inner_tm_clean"]:.3f}, aug {info["inner_tm_aug"]:.3f}); tau {tau:.3f} cert {info["tau_certified"]} '
        f'cov {info["inner_coverage"]:.2f}; {info["seconds"]} s', flush=True)
    path = fold_path(name, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(dict(sid=f.test_group, te=te, model=model, tau=tau, info=info), fh)


def load_folds(name):
    out = []
    for k in range(6):
        with open(fold_path(name, k), 'rb') as fh:
            out.append(pickle.load(fh))
    return out


def collect(names=None, log=print):
    D._pack()
    mt = F.meta(); n = len(mt)
    names = [m for m in (names or MODELS) if all(fold_path(m, k).exists() for k in range(6))]
    Xd, _ = D.features([D.eval_cond(l) for l in LEVELS], log=log)
    taus = {}
    for name in names:
        folds = load_folds(name)
        lvs = [0] if MODELS[name]['null'] is not None else LEVELS
        for lv in lvs:
            sc, sp, cf, tau = D.score_raw(folds, Xd[D.eval_cond(lv)], n)
            taus[name] = tau
            df = D.frame(mt, sc, sp, cf, cf >= tau)
            D.write(name, lv, df)
            log(f'  {name} {lv}: TM {D.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')
    seeds = [f'distill2_s{s}' for s in range(3)]
    if all(s in taus for s in seeds):
        mtau = np.mean([taus[s] for s in seeds], 0)
        for lv in LEVELS:
            tag = f'L{lv}' if not isinstance(lv, str) else lv
            ds = [pd.read_csv(SCORES / f'{s}__{tag}.csv') for s in seeds]
            sc = np.mean([d.score.values for d in ds], 0); sp = np.mean([d.spo2_hat.values for d in ds], 0); cf = np.mean([d.conf.values for d in ds], 0)
            df = D.frame(mt, sc, sp, cf, cf >= mtau)
            D.write('distill2', lv, df)
            log(f'  distill2 (ensemble) {lv}: TM {D.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')
    info = {m: [dict(sid=fo['sid'], **fo['info']) for fo in load_folds(m)] for m in names}
    (RES / 'distill2_fold_info.json').write_text(json.dumps(info, indent=1, default=str))


def modal(vals):
    c = Counter(vals); top = max(c.values())
    return sorted([v for v in c if c[v] == top], key=str)[0]


def export(out=None):
    """Refit on all 6 people with the modal nested recipe of each seed (tau = median fold tau); same JSON format as distill_export (read by distill_app_export)."""
    from . import distill_export as DE
    D._pack()
    mt = F.meta()
    rows = np.arange(len(mt)); y = mt.y.values.astype(int); s2 = mt.y_spo2.values.astype(float)
    students, taus, rec = [], [], []
    for s in range(3):
        folds = load_folds(f'distill2_s{s}')
        r = {k: modal([fo['info'][k] for fo in folds]) for k in ('teacher', 'lam', 'T', 'H', 'wd', 'epochs')}
        tau = float(np.median([fo['tau'] for fo in folds]))
        conds = D.train_conds(s)
        Xd, _ = D.features(conds, log=lambda *a: None)
        Xs, ys, w, qs = teachers(Xd, conds, rows, y)
        m = train(Xs, ys, w, qs[r['teacher']], np.tile(s2, len(conds)), r['lam'], r['T'], r['H'], r['wd'], s, (r['epochs'],))[r['epochs']]
        students.append(m); taus.append(tau); rec.append(dict(seed=s, teacher=r['teacher'], lam_kd=r['lam'], T=r['T'], hidden=r['H'], weight_decay=r['wd'],
                                                               epochs=r['epochs'], tau=tau))
    tw, Xw = DE.fixture_windows()
    sc, sp, cf, an = DE.predict_ens(students, taus, Xw)
    js = dict(format=DE.FORMAT, model='distill2', cols=F.COLS, hidden=[int(r['hidden']) for r in rec], recipe=rec, tau=float(np.mean(taus)),
              students=[DE.student_json(m) for m in students],
              reference=dict(fixture=str(DE.FIXTURE.relative_to(F.REPO)), t=DE._l(tw), X=DE._l(Xw), score=DE._l(sc), spo2_hat=DE._l(sp), conf=DE._l(cf),
                             answer=[bool(v) for v in an]))
    out = Path(out or RES / 'distill2_weights.json')
    out.write_text(json.dumps(js))
    print('wrote', out, rec, np.round(sc, 4).tolist())


def compare(models=('distill2', 'distill2_s0', 'distill2_s1', 'distill2_s2'), refs=('chain_v1', 'distill'), B=2000, out=None):
    """Paired person bootstrap (evalkit.Boot, same resampled people) of the TM bal. acc. difference of each model to each reference at every level."""
    from . import evalkit as E
    ref = E.eval_windows()
    boot = E.Boot(ref, B=B)
    rows = []
    for lv in LEVELS:
        tag = f'L{lv}' if not isinstance(lv, str) else lv
        res = {}
        for m in list(models) + list(refs):
            p = SCORES / f'{m}__{tag}.csv'
            if p.exists():
                res[m] = boot.tm(E.load_scores(p, ref).pred.values)
        for m in models:
            if m not in res:
                continue
            pt, bt, per = res[m]
            r = dict(model=m, level=tag, tm=round(pt, 4), tm_lo=round(E.ci(bt)[0], 4), tm_hi=round(E.ci(bt)[1], 4), worst=round(float(np.nanmin(per)), 4))
            for rf in refs:
                if rf in res:
                    lo, hi = E.ci(bt - res[rf][1])
                    pr = res[rf][2]; ok = np.isfinite(per) & np.isfinite(pr) & (per != pr)
                    r.update({f'd_{rf}': round(pt - res[rf][0], 4), f'd_{rf}_lo': round(lo, 4), f'd_{rf}_hi': round(hi, 4),
                              f'better_{rf}': f'{int((per[ok] > pr[ok]).sum())}/{int(ok.sum())}'})
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out or RES / 'distill2_compare.csv', index=False)
    print(T.to_string(index=False))
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--model', default=None)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--export', action='store_true')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    if a.compare:
        compare(); return
    if a.collect:
        collect(); return
    if a.export:
        export(); return
    names = [a.model] if a.model else list(MODELS)
    for nm in names:
        for k in ([a.fold] if a.fold is not None else range(6)):
            if not fold_path(nm, k).exists():
                run_fold(nm, k)


if __name__ == '__main__':
    import sys
    from oxydrop_ml.photon import distill2 as _m
    sys.exit(_m.main())
