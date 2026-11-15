"""Conformal 3-state output (Normal / Possible low oxygen / Unsure) on the final hybrid score (FEAT-007, python/oxydrop_ml/amendments/AMENDMENT_5_conformal_openox.md).

    python -m oxydrop_ml.conformal calib  --runs RUNS/hyb_A --seeds 0 1 2 --res ../results/ml      # scores/conformal_inputs_hoffman.csv (host, needs the FEAT-008 run tables)
    python -m oxydrop_ml.conformal run    --res ../results/ml                                                                           # all CSVs of the nested evaluation (score CSVs only)
    python -m oxydrop_ml.conformal report --res ../results/ml                                                                           # CONFORMAL.md from the CSVs

Method. Input is the decision logit z of a window (> 0 = low oxygen), p = sigmoid(z), label y = 1 for window-mean reference SpO2 < 90.
Nonconformity s(x, y) = 1 - p_y. The prediction set is {y : s(x, y) <= q_y} with the finite-sample quantile level ceil((n + 1)(1 - alpha)) / n.
  mondrian (primary): one quantile per class (class-conditional coverage), marginal (control): one quantile for all calibration windows.
In one dimension both reduce to the thresholds t0 (set contains 0 iff z <= t0) and t1 (set contains 1 iff z >= t1):
  set {0} -> Normal, set {1} -> Possible low oxygen, set {0, 1} or {} -> Unsure (the only abstention).
Nested protocol. For the outer Hoffman person o the calibration windows are the five other people, each scored by the pipeline selected for o with a
leave-one-inner-person-out meta-learner whose CNN logits come from fits that exclude {o, q}: o's windows never touch its own thresholds.
Everything below the `calib` step works on score CSVs only.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import metrics as M

ALPHAS = np.round(np.arange(0.005, 0.5001, 0.005), 3)
HEADLINE = 0.10                       # declared in Amendment 5 before any result
REPORT_ALPHAS = (0.05, 0.10, 0.20)
METHODS = ('mondrian', 'marginal')
FIELDS = ['n0', 'n1', 'N0', 'N1', 'L0', 'L1', 'U0', 'U1', 'cov0', 'cov1']
STATE_NAMES = {0: 'normal', 1: 'low', 2: 'unsure'}
B_DEFAULT = 2000


# ------------------------------------------------------------------------------------------------ the method (pure functions)
def q_high(v, alpha):
    """Finite-sample conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest value, +inf when that rank exceeds n (or the set is empty)."""
    v = np.sort(np.asarray(v, float))
    n = len(v)
    if n == 0:
        return np.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    return np.inf if k > n else float(v[max(k, 1) - 1])


def thresholds(z, y, alpha, method='mondrian'):
    """(t0, t1) from calibration logits z and labels y. The set of a window contains 0 iff z <= t0 and contains 1 iff z >= t1."""
    z = np.asarray(z, float); y = np.asarray(y).astype(int)
    if method == 'mondrian':
        t0 = q_high(z[y == 0], alpha)                      # s(x, 0) = p increases with z: contains 0 iff z <= q
        t1 = -q_high(-z[y == 1], alpha)                    # s(x, 1) = 1 - p, its order is that of -z: contains 1 iff z >= -q
        return float(t0), float(t1)
    if method == 'marginal':
        p = expit(z)
        q = q_high(np.where(y == 1, 1.0 - p, p), alpha)
        if not np.isfinite(q) or q >= 1.0:
            return np.inf, -np.inf
        return float(logit(q)) if q > 0 else -np.inf, float(logit(1.0 - q)) if q < 1 else -np.inf
    raise ValueError(method)


def band_thresholds(delta):
    """The fixed abstain band |p - 0.5| < delta as thresholds: Normal when p <= 0.5 - delta, Possible low when p >= 0.5 + delta."""
    return float(logit(0.5 - delta)) if delta < 0.5 else -np.inf, float(logit(0.5 + delta)) if delta < 0.5 else np.inf


def states(z, t0, t1):
    """0 normal, 1 possible low oxygen, 2 unsure, and the two set-membership masks (contains 0, contains 1)."""
    z = np.asarray(z, float)
    c0, c1 = z <= t0, z >= t1
    s = np.full(len(z), 2, int)
    s[c0 & ~c1] = 0
    s[c1 & ~c0] = 1
    return s, c0, c1


def count_vector(z, y, t0, t1):
    y = np.asarray(y).astype(int)
    s, c0, c1 = states(z, t0, t1)
    return np.array([(y == 0).sum(), (y == 1).sum(), ((s == 0) & (y == 0)).sum(), ((s == 0) & (y == 1)).sum(), ((s == 1) & (y == 0)).sum(), ((s == 1) & (y == 1)).sum(),
                     ((s == 2) & (y == 0)).sum(), ((s == 2) & (y == 1)).sum(), (c0 & (y == 0)).sum(), (c1 & (y == 1)).sum()], float)


def _div(a, b):
    with np.errstate(all='ignore'):
        return np.where(np.asarray(b) > 0, np.asarray(a, float) / np.where(np.asarray(b) > 0, b, 1), np.nan)


def metrics_from_counts(c):
    """c: [..., 10] counts in FIELDS order -> dict of arrays (works on a bootstrap stack)."""
    c = np.asarray(c, float)
    n0, n1, N0, N1, L0, L1, U0, U1, cov0, cov1 = [c[..., i] for i in range(10)]
    n = n0 + n1
    ans = N0 + N1 + L0 + L1
    sens, spec = _div(L1, L1 + N1), _div(N0, N0 + L0)
    return dict(n=n, coverage=_div(ans, n), acc=_div(N0 + L1, ans), sens_answered=sens, spec_answered=spec, bacc_answered=0.5 * (sens + spec),
                alert_share_all=_div(L1, n1), confident_miss=_div(N1, n1), confident_false_alarm=_div(L0, n0), set_cov_hypoxic=_div(cov1, n1), set_cov_normal=_div(cov0, n0),
                share_normal=_div(N0 + N1, n), share_low=_div(L0 + L1, n), share_unsure=_div(U0 + U1, n), unsure_of_hypoxic=_div(U1, n1), unsure_of_normal=_div(U0, n0),
                n_answered=ans)


# ------------------------------------------------------------------------------------------------ nested evaluation
def person_counts(S, Cal, method, param, kind='conformal'):
    """Per outer person the count vector and the thresholds. S: frame(outer, subject, y, z); Cal: frame(outer, y, z). kind 'conformal' (param = alpha) or 'band' (param = delta)."""
    rows, thr = [], []
    for o, g in S.groupby('outer', sort=True):
        if kind == 'conformal':
            c = Cal[Cal.outer == o]
            t0, t1 = thresholds(c.z.values, c.y.values, param, method)
        else:
            t0, t1 = band_thresholds(param)
        rows.append(count_vector(g.z.values, g.y.values, t0, t1))
        thr.append(dict(outer=o, t0=t0, t1=t1, n_cal0=int((Cal.y[Cal.outer == o] == 0).sum()) if kind == 'conformal' else 0,
                        n_cal1=int((Cal.y[Cal.outer == o] == 1).sum()) if kind == 'conformal' else 0))
    return np.vstack(rows), pd.DataFrame(thr)


def boot_metrics(counts, B=B_DEFAULT, seed=0):
    """Pooled metrics of the persons' count matrix [P, 10] with the 95 % person-bootstrap interval of each metric (people resampled)."""
    pooled = metrics_from_counts(counts.sum(0))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(counts), (B, len(counts)))
    bm = metrics_from_counts(counts[pick].sum(1))
    out = {}
    for k, v in pooled.items():
        out[k] = float(v)
        if k in ('n', 'n_answered'):
            continue
        out[k + '_lo'], out[k + '_hi'] = (float(np.nanpercentile(bm[k], 2.5)), float(np.nanpercentile(bm[k], 97.5))) if np.isfinite(bm[k]).any() else (np.nan, np.nan)
    return out, bm


def curves(S, Cal, alphas=ALPHAS, B=B_DEFAULT, seed=0, systems=None):
    """Tidy curve table for the conformal methods (nested) and the fixed band on the same score. Also returns the per-(method, alpha) bootstrap stacks for the matched-coverage step."""
    rows, stacks, thr_rows = [], {}, []
    for method in METHODS:
        for a in alphas:
            cnt, thr = person_counts(S, Cal, method, a)
            m, bm = boot_metrics(cnt, B, seed)
            rows.append(dict(method=method, param=float(a), target_coverage=1 - float(a), **m))
            stacks[(method, float(a))] = bm
            thr_rows.append(thr.assign(method=method, alpha=float(a)))
    for d in np.round(np.arange(0.0, 0.4901, 0.01), 2):
        cnt, _ = person_counts(S, Cal, 'band', d, kind='band')
        m, bm = boot_metrics(cnt, B, seed)
        rows.append(dict(method='band', param=float(d), target_coverage=np.nan, **m))
        stacks[('band', float(d))] = bm
    return pd.DataFrame(rows), stacks, pd.concat(thr_rows, ignore_index=True)


def per_person(S, Cal, alphas=REPORT_ALPHAS, methods=METHODS):
    rows = []
    people = sorted(S.outer.unique())
    for method in methods:
        for a in alphas:
            cnt, thr = person_counts(S, Cal, method, a)
            for o, c, (_, t) in zip(people, cnt, thr.iterrows()):
                m = metrics_from_counts(c)
                rows.append(dict(method=method, alpha=a, person=o, t0=t.t0, t1=t.t1, **{k: float(v) for k, v in m.items()}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ matched coverage
def acc_at_coverage(cov, acc, target):
    """Linear interpolation of accuracy at a coverage (cov decreasing in alpha is not assumed: points are sorted by coverage). NaN outside the range."""
    cov, acc = np.asarray(cov, float), np.asarray(acc, float)
    ok = np.isfinite(cov) & np.isfinite(acc)
    if ok.sum() < 2:
        return np.nan
    o = np.argsort(cov[ok], kind='stable')
    c, a = cov[ok][o], acc[ok][o]
    if target < c[0] or target > c[-1]:
        return np.nan
    return float(np.interp(target, c, a))


def ranked_at_coverage(conf, y, pred, target):
    """Accuracy and alert share of the top `target` share of windows by confidence (a coverage-exact, non-nested control)."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    k = max(1, int(round(target * len(y))))
    o = np.argsort(-np.asarray(conf, float), kind='stable')[:k]
    a = np.zeros(len(y), bool); a[o] = True
    n1 = max(int((y == 1).sum()), 1)
    return float((pred[a] == y[a]).mean()), float(((pred == 1) & (y == 1) & a).sum() / n1)


def ranked_curve(conf, y, pred, subj, grid, B=B_DEFAULT, seed=0):
    """Selective-accuracy curve of a confidence score with person-bootstrap intervals; rows per coverage."""
    rows = []
    for c in grid:
        st = lambda idx, c=c: ranked_at_coverage(conf[idx], y[idx], pred[idx], c)[0]
        sa = lambda idx, c=c: ranked_at_coverage(conf[idx], y[idx], pred[idx], c)[1]
        bt = M.person_bootstrap(subj, st, B=B, seed=seed)
        ba = M.person_bootstrap(subj, sa, B=B, seed=seed)
        rows.append(dict(coverage=float(c), acc=bt['point'], acc_lo=bt['lo'], acc_hi=bt['hi'], alert_share_all=ba['point'], alert_lo=ba['lo'], alert_hi=ba['hi']))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ inputs
def load_scores(res_dir, system='final'):
    """Outer scores and inner calibration scores of one system from `scores/conformal_inputs_hoffman.csv`: S(outer, subject, rec, t_start, y, z), Cal(outer, inner, y, z)."""
    d = pd.read_csv(os.path.join(res_dir, 'scores', 'conformal_inputs_hoffman.csv'))
    d = d[d.system == system]
    o = d[d.kind == 'outer']; i = d[d.kind == 'inner']
    S = pd.DataFrame(dict(outer=o.outer.values, subject=o.outer.values, rec=o.rec.values, t_start=o.t_start.values, y=o.low.values.astype(int), z=o.z.values.astype(float)))
    Cal = pd.DataFrame(dict(outer=i.outer.values, inner=i.inner.values, y=i.low.values.astype(int), z=i.z.values.astype(float)))
    return S, Cal


def baseline_frame(res_dir, S):
    """Baseline P1 scores on exactly the windows of S (merge on subject, rec, t_start)."""
    b = pd.read_csv(os.path.join(res_dir, 'scores', 'baseline_P1_hoffman.csv'))
    s = S.copy(); s['t_start'] = s.t_start.astype(float); b['t_start'] = b.t_start.astype(float)
    d = s.merge(b, on=['subject', 'rec', 't_start'], how='left')
    assert len(d) == len(S) and d['chain_v1__score'].notna().all(), 'baseline scores must cover every window'
    return d


def band_from_decisions(dec, y):
    """Coverage and accuracy of a baseline decision column (-1 = abstain)."""
    dec = np.asarray(dec); y = np.asarray(y)
    a = dec != -1
    return float(a.mean()), float((dec[a] == y[a]).mean()), float(((dec == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))


# ------------------------------------------------------------------------------------------------ nested calibration scores (needs the FEAT-008 run tables)
SYSTEMS = ('final', 'deployed_R', 'chain_v1')
SYSTEM_TEXT = {'final': 'HYBRID_FINAL: pipeline selected per outer person by the inner folds (Amendment 2), the confirmatory headline',
               'deployed_R': 'deployed pipeline R fixed for every person (source R, no per-user calibration, k = 1, what the app runs); exploratory, R was chosen on all six people',
               'chain_v1': 'v1 chain logit alone (control; the same procedure on the incumbent)'}


def inner_calibration(runs_dirs, seeds, choices_csv, final_scores_csv, families=('P', 'R', 'D')):
    """For every system and outer person: (a) kind 'inner': leave-one-inner-person-out logits of the pipeline on the inner out-of-fold table of the five training people (the calibration set;
    no outer window is used); (b) kind 'outer': the outer person's logits of the same pipeline. A consistency check reproduces `scores/hybrid_final_hoffman.csv` exactly."""
    from . import hybrid as H
    runs, counts = H.load_runs(runs_dirs, families=families, seeds=seeds)
    meta, X19 = H.hoffman_meta()
    w = H.add_chain(H.wide(runs), meta, X19, None)
    del runs
    F = H.folds(w)
    ch = pd.read_csv(choices_csv)
    ch = ch[ch.pipeline == 'HYBRID_FINAL'].set_index('outer')
    final = pd.read_csv(final_scores_csv)
    out, maxdiff = [], 0.0
    for o, d in F.items():
        inner, outer = d['inner'], d['outer']
        ev = inner['eval'].values.astype(bool)
        evo = outer['eval'].values.astype(bool)
        y = inner.low.values.astype(int)
        who = inner.inner.values
        for system in SYSTEMS:
            if system == 'final':
                src, cal, k = str(ch.loc[o, 'source']), bool(ch.loc[o, 'cal']), int(ch.loc[o, 'k'])
            elif system == 'deployed_R':
                src, cal, k = 'R', False, 1
            else:
                src, cal, k = 'chain_shipped', False, 1
            z = np.full(len(inner), np.nan)
            if src == 'chain_shipped':
                z = inner.c.values.astype(float)
                zo = outer.c.values.astype(float)
            else:
                Fe = H.lever_features(inner, H.cand_cols(src), cal, k)
                for q in np.unique(who[ev]):
                    te, tr = ev & (who == q), ev & (who != q)
                    z[te] = H.meta_fit(Fe[tr], y[tr])(Fe[te])
                zo, _ = H.apply_pipeline(inner, outer, src, cal, k)
            t = inner[ev][['outer', 'inner', 'rec', 'local', 't_start', 'low']].copy()
            t['z'] = z[ev]; t['kind'] = 'inner'
            u = outer[evo][['outer', 'rec', 'local', 't_start', 'low']].copy()
            u['inner'] = ''; u['z'] = np.asarray(zo)[evo]; u['kind'] = 'outer'
            tt = pd.concat([t, u], ignore_index=True)
            tt['system'] = system; tt['source'] = src; tt['cal'] = cal; tt['k'] = k
            out.append(tt)
            if system == 'final':
                ref = final[final.outer == o].set_index('local').logit
                got = pd.Series(np.asarray(zo)[evo], index=outer.local.values[evo])
                maxdiff = max(maxdiff, float(np.abs(got - ref.reindex(got.index)).max()))
    print('max |recomputed outer logit - committed logit| (system final)', maxdiff)
    cols = ['system', 'kind', 'outer', 'inner', 'rec', 'local', 't_start', 'low', 'z', 'source', 'cal', 'k']
    return pd.concat(out, ignore_index=True)[cols], maxdiff


# ------------------------------------------------------------------------------------------------ the full nested evaluation
def add_regime(curve):
    """Coverage first rises as alpha grows (the middle band where both class sets hold shrinks) and falls again once the two class thresholds cross (t0 < t1: the middle
    band becomes windows with an EMPTY set, which are also unsure). The alpha of the largest pooled coverage ends the 'main' regime; larger alpha are labelled 'past_crossing'
    and are not used for the coverage-matched comparison."""
    curve = curve.copy()
    curve['regime'] = ''
    for m in METHODS:
        cu = curve[curve.method == m].sort_values('param')
        amax = float(cu.param.values[int(np.nanargmax(cu.coverage.values))])
        curve.loc[cu.index, 'regime'] = np.where(cu.param.values <= amax, 'main', 'past_crossing')
    curve.loc[curve.method == 'band', 'regime'] = 'band'
    return curve


def matched_from_curve(curve, stacks, method, target, B=B_DEFAULT):
    """Accuracy (and alert share, sensitivity, specificity on answered windows) of a nested conformal method at a target coverage, with the person-bootstrap interval of the interpolation.
    Only the 'main' regime enters. Returns dict name -> (point, lo, hi) and the per-resample accuracies."""
    cu = curve[(curve.method == method) & (curve.regime == 'main')].sort_values('param')
    keys = [(method, float(a)) for a in cu.param]
    out = {}
    accs = None
    cov_b = np.stack([stacks[k]['coverage'] for k in keys], 1)
    for name in ('acc', 'alert_share_all', 'bacc_answered', 'sens_answered', 'spec_answered'):
        pt = acc_at_coverage(cu.coverage.values, cu[name].values, target)
        val_b = np.stack([stacks[k][name] for k in keys], 1)
        vals = np.array([acc_at_coverage(cov_b[b], val_b[b], target) for b in range(B)])
        out[name] = (pt, float(np.nanpercentile(vals, 2.5)) if np.isfinite(vals).any() else np.nan, float(np.nanpercentile(vals, 97.5)) if np.isfinite(vals).any() else np.nan)
        if name == 'acc':
            accs = vals
    return out, accs


def run(res_dir, B=B_DEFAULT, seed=0):
    out = {k: [] for k in ('curve', 'thr', 'pp', 'matched', 'ranked', 'calib')}
    S0, _ = load_scores(res_dir, 'final')
    base = baseline_frame(res_dir, S0)
    y, subj, outer_ids = S0.y.values, S0.subject.values, S0.outer.values
    people = sorted(S0.outer.unique())
    refs = {'v1': base.chain_v1__band_shipped.values, 'v2': base.chain_v2__nolock_band_shipped.values}
    ref_rows = []
    for nm, dec in refs.items():
        cov, acc, alert = band_from_decisions(dec, y)
        bt = M.person_bootstrap(subj, lambda idx, dec=dec: band_from_decisions(dec[idx], y[idx])[1], B=B, seed=seed)
        bc = M.person_bootstrap(subj, lambda idx, dec=dec: band_from_decisions(dec[idx], y[idx])[0], B=B, seed=seed)
        ref_rows.append(dict(system=f'{nm}_shipped_band', coverage=cov, coverage_lo=bc['lo'], coverage_hi=bc['hi'], acc=acc, acc_lo=bt['lo'], acc_hi=bt['hi'], alert_share_all=alert))
    refdf = pd.DataFrame(ref_rows).set_index('system')
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(people), (B, len(people)))                         # the same people resamples as boot_metrics
    for system in SYSTEMS:
        S, Cal = load_scores(res_dir, system)
        assert np.array_equal(S.rec.values, S0.rec.values) and np.array_equal(S.t_start.values, S0.t_start.values), 'systems must score the same windows in the same order'
        z = S.z.values
        curve, stacks, thr = curves(S, Cal, B=B, seed=seed)
        curve = add_regime(curve)
        out['curve'].append(curve.assign(system=system)); out['thr'].append(thr.assign(system=system))
        out['pp'].append(per_person(S, Cal).assign(system=system))
        conf, pred = np.abs(z), (z > 0).astype(int)
        out['ranked'].append(ranked_curve(conf, y, pred, subj, grid, B=B, seed=seed).assign(system=system))
        for rname in ('v1', 'v2'):
            r = refdf.loc[f'{rname}_shipped_band']
            tcov = float(r.coverage)
            dec = refs[rname]
            ref_cnt = np.array([[(dec[outer_ids == o] != -1).sum(), ((dec[outer_ids == o] == y[outer_ids == o]) & (dec[outer_ids == o] != -1)).sum()] for o in people], float)
            for method in METHODS:
                mm, accs = matched_from_curve(curve, stacks, method, tcov, B)
                sums = ref_cnt[pick].sum(1)
                d = accs - sums[:, 1] / np.where(sums[:, 0] > 0, sums[:, 0], np.nan)
                out['matched'].append(dict(system=system, reference=rname, target_coverage=tcov, variant=f'conformal_{method}', coverage=tcov, acc=mm['acc'][0], acc_lo=mm['acc'][1], acc_hi=mm['acc'][2],
                                           alert_share_all=mm['alert_share_all'][0], reference_acc=float(r.acc), diff_acc=mm['acc'][0] - float(r.acc), diff_lo=float(np.nanpercentile(d, 2.5)), diff_hi=float(np.nanpercentile(d, 97.5)),
                                           note='nested conformal, interpolated in coverage'))
            a_, al = ranked_at_coverage(conf, y, pred, tcov)
            st = lambda idx, c=conf, p=pred, t=tcov: ranked_at_coverage(c[idx], y[idx], p[idx], t)[0]
            ds = lambda idx, c=conf, p=pred, t=tcov, dec=dec: ranked_at_coverage(c[idx], y[idx], p[idx], t)[0] - band_from_decisions(dec[idx], y[idx])[1]
            bt, pb = M.person_bootstrap(subj, st, B=B, seed=seed), M.person_bootstrap(subj, ds, B=B, seed=seed)
            out['matched'].append(dict(system=system, reference=rname, target_coverage=tcov, variant='confidence_ranked', coverage=tcov, acc=a_, acc_lo=bt['lo'], acc_hi=bt['hi'], alert_share_all=al,
                                       reference_acc=float(r.acc), diff_acc=pb['point'], diff_lo=pb['lo'], diff_hi=pb['hi'], note='ranked by |logit|, coverage exact, not nested (control)'))
        p = expit(z)
        cnt, thr0 = person_counts(S, Cal, 'mondrian', HEADLINE)
        t0o = thr0.set_index('outer')
        ans = np.zeros(len(S), bool)
        for o in t0o.index:
            mk = S.outer.values == o
            ans[np.where(mk)[0]] = states(z[mk], t0o.loc[o, 't0'], t0o.loc[o, 't1'])[0] != 2
        out['calib'].append(dict(system=system, ece_all_windows=M.ece(y, p), ece_answered_mondrian_alpha010=M.ece(y[ans], p[ans]), auroc_all=M.auroc(y, z), n_windows=len(S), n_people=len(people),
                                 prevalence=float(y.mean())))
        print(system, 'done', flush=True)
    # the baselines' own confidence-ranked curves (same windows), for the figure
    rk_b = []
    for nm, col in (('chain_v2', 'chain_v2__score'),):
        pv = base[col].values
        rk_b.append(ranked_curve(np.abs(pv - 0.5), y, (pv > 0.5).astype(int), subj, grid, B=B, seed=seed).assign(system=nm))
    res = dict(curve=pd.concat(out['curve'], ignore_index=True), thr=pd.concat(out['thr'], ignore_index=True), per_person=pd.concat(out['pp'], ignore_index=True),
               matched=pd.DataFrame(out['matched']), ranked=pd.concat(out['ranked'] + rk_b, ignore_index=True), ref=refdf.reset_index(), calib=pd.DataFrame(out['calib']))
    names = dict(curve='conformal_curve.csv', thr='conformal_thresholds.csv', per_person='conformal_per_person.csv', matched='conformal_matched.csv', ranked='conformal_ranked.csv',
                 ref='conformal_reference_bands.csv', calib='conformal_calibration.csv')
    for k, f in names.items():
        res[k].to_csv(os.path.join(res_dir, f), index=False)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('calib'); c.add_argument('--runs', required=True); c.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); c.add_argument('--res', default='../results/ml')
    r = sp.add_parser('run'); r.add_argument('--res', default='../results/ml'); r.add_argument('--B', type=int, default=B_DEFAULT)
    p = sp.add_parser('report'); p.add_argument('--res', default='../results/ml')
    a = ap.parse_args(argv)
    if a.cmd == 'calib':
        t, md = inner_calibration(a.runs.split(','), a.seeds, os.path.join(a.res, 'hybrid_p1_final_choices.csv'), os.path.join(a.res, 'scores', 'hybrid_final_hoffman.csv'))
        t.to_csv(os.path.join(a.res, 'scores', 'conformal_inputs_hoffman.csv'), index=False)
        print('wrote', len(t), 'rows; consistency max diff', md)
    elif a.cmd == 'run':
        run(a.res, a.B)
    else:
        from .conformal_md import report
        report(a.res)


if __name__ == '__main__':
    main()
