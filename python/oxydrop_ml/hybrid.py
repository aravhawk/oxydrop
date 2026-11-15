"""Hybrid (physics chain + CNN) P1 pipeline of FEAT-008: candidates, inner-fold selection, outer scoring, tables.

Protocol, declared in amendments/AMENDMENT_2_hybrid.md BEFORE any hybrid outer score existed:
  per outer person o (leave-one-person-out over the 6 Hoffman people)
    inner OOF table : the 5 other people q, each scored by the fits that exclude {o, q}        (hybrid_run.py, kind 'inner')
    outer table     : o scored by the fit that excludes o                                       (kind 'outer')
  features per window: c = v1 chain logit (chain fit on the people the CNN was fit on), p = plain CNN logit, d = distilled CNN logit,
  corr = residual network output, r = c + corr (residual hybrid logit); CNN logits are means over seeds.
  pipeline = (source candidate) -> (per-user baseline calibration off/on) -> (causal mean over k windows) -> standardised class-balanced logistic meta-learner
  fit on the inner OOF windows, decision at 0. Every choice is made on the inner OOF table only (leave-one-inner-person-out CV time-matched balanced accuracy).
The outer person's windows only enter after the choice is frozen. `evaluate` returns every table the report needs.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import DATA_DIR
from . import metrics as M
from . import tta
from .distill import Chain

KS = (1, 6, 11, 21)
MARGIN = 0.01
COMPONENTS = {'c': 'chain', 'p': 'P', 'd': 'D', 'corr': 'R'}
CANDS = {
    'chain_shipped': None,                 # raw chain logit, no meta-learner: the incumbent (control)
    'chain_platt': ['c'],                  # control
    'P': ['p'], 'D': ['d'], 'R': ['r'],
    'c+p': ['c', 'p'], 'c+d': ['c', 'd'], 'c+corr': ['c', 'corr'], 'all': ['c', 'p', 'd', 'corr'],
}
CNN_ORDER = ['P', 'D', 'R', 'c+p', 'c+d', 'c+corr', 'all']          # simplest first (tie rule)
CHAIN_ORDER = ['chain_platt']


# ------------------------------------------------------------------------------------------------ loading
def load_runs(dirs, families=('P', 'R', 'D'), seeds=(0,), require_complete=True):
    """Concatenate the job CSVs of hybrid_run.py found under the run directories. Returns (DataFrame, dict (family, seed) -> number of job files)."""
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    frames, counts = [], {}
    for fam in families:
        for s in seeds:
            files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, f'{fam}_s{s}', '*.csv')))
            counts[(fam, s)] = len(files)
            if files:
                frames.append(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    if require_complete:
        bad = {k: v for k, v in counts.items() if v != 36}
        if bad:
            raise RuntimeError(f'incomplete runs (expected 36 job files per family and seed): {bad}')
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), counts


def hoffman_meta(data_dir=None):
    d = os.path.join(data_dir or DATA_DIR, 'hoffman')
    meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    return meta, np.load(os.path.join(d, 'chain19.npy'))


KEYS = ['outer', 'inner', 'target', 'local']
INFO = ['subject', 'rec', 't_start', 'low', 'eval', 'in_base', 'hr_ref']


def wide(df):
    """One row per (outer, inner, target, local): seed means of the logit per family as columns p, d, corr (and hr / p_sqi of P)."""
    if df.empty:
        return df
    df = df.assign(inner=df['inner'].fillna(''))
    g = df.groupby(KEYS + ['family'], sort=False)
    m = g.agg(logit=('logit', 'mean'), hr=('hr', 'mean'), p_sqi=('p_sqi', 'mean'), n_seeds=('seed', 'nunique')).reset_index()
    base = df.drop_duplicates(KEYS)[KEYS + INFO]
    out = base.copy()
    for fam, col in (('P', 'p'), ('D', 'd'), ('R', 'corr')):
        s = m[m.family == fam].set_index(KEYS)
        if len(s):
            out = out.join(s[['logit']].rename(columns={'logit': col}), on=KEYS)
    s = m[m.family == 'P'].set_index(KEYS)
    if len(s):
        out = out.join(s[['hr', 'p_sqi']], on=KEYS)
    out['inner'] = out['inner'].fillna('')
    return out.reset_index(drop=True)


def add_chain(w, meta, X19, hof):
    """Chain logit c for every row of the wide table with the matching training people: inner rows: hof minus {o, q}; outer, member and train rows: hof minus {o}."""
    y = meta.low.values.astype(int)
    ev = (meta['eval'].values & meta.has_low.values)
    subj = meta.subject.values
    cache = {}

    def chain(excl):
        k = tuple(sorted(excl))
        if k not in cache:
            r = np.nonzero(ev & ~np.isin(subj, list(excl)))[0]
            cache[k] = Chain(X19[r], y[r])
        return cache[k]
    c = np.zeros(len(w))
    for (o, q), idx in w.groupby(['outer', 'inner']).indices.items():
        ch = chain([o, q] if q else [o])
        sel = idx
        mem = w.target.values[sel] == 'member'
        if q and mem.any():                         # fold-ensemble members score o: use the outer chain
            c[sel[mem]] = chain([o]).logit(X19[w.local.values[sel[mem]]])
            sel = sel[~mem]
        c[sel] = ch.logit(X19[w.local.values[sel]])
    w = w.copy()
    w['c'] = c
    if 'corr' in w:
        w['r'] = w['c'] + w['corr']
    return w


# ------------------------------------------------------------------------------------------------ pipeline pieces
def tm_bacc(t, y, score, thr=0.0):
    return M.time_matched(t, y, (np.asarray(score) > thr).astype(int), score)['tm_bacc']


def feats(tbl, cols):
    return np.column_stack([tbl[c].values.astype(float) for c in cols])


def lever_features(tbl, cols, cal, k):
    """Features of the table after the declared per-user calibration and causal mean (rows keep their order)."""
    return tta.calibrate_and_smooth(feats(tbl, cols), tbl.rec.values, tbl.t_start.values, tbl.in_base.values, cal, k)


def meta_fit(F, y, C=1.0):
    sc = StandardScaler().fit(F)
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(sc.transform(F), y)
    f = lambda G: lr.decision_function(sc.transform(G))
    f.params = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    return f


def cv_score(tbl, F):
    """Leave-one-inner-person-out CV time-matched balanced accuracy of the meta-learner on the labelled evaluation windows of the inner table."""
    ev = tbl['eval'].values.astype(bool)
    who = tbl.inner.values
    y = tbl.low.values.astype(int)
    cv = np.zeros(len(tbl))
    for q in np.unique(who[ev]):
        te = ev & (who == q)
        tr = ev & (who != q)
        cv[te] = meta_fit(F[tr], y[tr])(F[te])
    return tm_bacc(tbl.t_start.values[ev], y[ev], cv[ev])


def cand_cols(name):
    cols = CANDS[name]
    return cols


def cand_score(tbl, name, cal, k):
    if name == 'chain_shipped':
        return cv_tm_raw(tbl)
    return cv_score(tbl, lever_features(tbl, cand_cols(name), cal, k))


def cv_tm_raw(tbl):
    ev = tbl['eval'].values.astype(bool)
    return tm_bacc(tbl.t_start.values[ev], tbl.low.values[ev].astype(int), tbl.c.values[ev])


def select(inner, order, margin=MARGIN, ks=KS):
    """The declared three-step inner-fold selection. Returns dict(source, cal, k, trace)."""
    s1 = {n: cand_score(inner, n, False, 1) for n in order}
    best = max(s1.values())
    src = next(n for n in order if s1[n] >= best - margin)
    on = cand_score(inner, src, True, 1)
    cal = bool(on >= s1[src] + margin)
    s3 = {k: cand_score(inner, src, cal, k) for k in ks}
    bk = max(s3.values())
    k = next(k for k in ks if s3[k] >= bk - margin)
    return dict(source=src, cal=cal, k=int(k), inner_cv=float(s3[k]), stage1=s1, cal_on=float(on), stage3=s3)


def apply_pipeline(inner, outer, source, cal, k):
    """Fit the meta-learner of the pipeline on the inner OOF table and return the decision scores of every row of `outer` (and the fitted function)."""
    if source == 'chain_shipped':
        return outer.c.values.astype(float), None
    ev = inner['eval'].values.astype(bool)
    cols = cand_cols(source)
    f = meta_fit(lever_features(inner, cols, cal, k)[ev], inner.low.values[ev].astype(int))
    return f(lever_features(outer, cols, cal, k)), f


# ------------------------------------------------------------------------------------------------ evaluation of all outer folds
def folds(w):
    out = {}
    for o in sorted(w.outer.unique()):
        s = w[w.outer == o]
        out[o] = dict(inner=s[s.target == 'inner'].reset_index(drop=True), outer=s[s.target == 'outer'].reset_index(drop=True),
                      member=s[s.target == 'member'].reset_index(drop=True), train=s[s.target == 'train'].reset_index(drop=True))
    return out


def fold_stats(scores, tbl):
    """Pooled and per-person time-matched metrics of decision scores on the labelled evaluation windows of tbl (rows of one or several people)."""
    ev = tbl['eval'].values.astype(bool)
    y, t, s, sub = tbl.low.values[ev].astype(int), tbl.t_start.values[ev], np.asarray(scores)[ev], tbl.subject.values[ev]
    return y, t, s, sub


def run_pipeline_over_folds(F, order, fixed=None, with_train=False):
    """For every outer fold: select (or take `fixed` = (source, cal, k)), score the outer person. Returns (outer score frame, choices DataFrame, per-fold train info)."""
    frames, choices, trains = [], [], []
    for o, d in F.items():
        ch = select(d['inner'], order) if fixed is None else dict(source=fixed[0], cal=fixed[1], k=fixed[2], inner_cv=cand_score(d['inner'], *fixed), stage1={}, stage3={}, cal_on=np.nan)
        sc, f = apply_pipeline(d['inner'], d['outer'], ch['source'], ch['cal'], ch['k'])
        fr = d['outer'][['outer', 'subject', 'rec', 'local', 't_start', 'low', 'eval', 'in_base']].copy()
        fr['score'] = sc
        frames.append(fr)
        choices.append(dict(outer=o, source=ch['source'], cal=ch['cal'], k=ch['k'], inner_cv=ch['inner_cv'], **{f'stage1_{k}': v for k, v in ch['stage1'].items()},
                            cal_on_cv=ch['cal_on'], **{f'stage3_k{k}': v for k, v in ch['stage3'].items()}))
        if with_train and len(d['train']):
            tr = d['train']
            sc_tr, _ = apply_pipeline(d['inner'], tr, ch['source'], ch['cal'], ch['k'])
            y, t, s, sub = fold_stats(sc_tr, tr)
            trains.append(dict(outer=o, train_tm_bacc=tm_bacc(t, y, s)))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(choices), pd.DataFrame(trains)


def summarize(frame, B=2000, seed=0, name=''):
    """Pooled TM balanced accuracy with the person-bootstrap CI, per-person values and spread, classification extras."""
    ev = frame[frame['eval'].astype(bool)]
    y, t, s, sub = ev.low.values.astype(int), ev.t_start.values, ev.score.values, ev.subject.values
    pooled = M.time_matched(t, y, (s > 0).astype(int), s)
    bt = M.person_bootstrap(sub, M.tm_stat(t, y, s, 0.0, 'tm_bacc'), B=B, seed=seed) if B else dict(lo=np.nan, hi=np.nan)
    per = {p: M.time_matched(t[sub == p], y[sub == p], (s[sub == p] > 0).astype(int), s[sub == p])['tm_bacc'] for p in np.unique(sub)}
    sm = M.summarize_folds(list(per.values()), seed=seed)
    cl = M.classification_metrics(y, s, thr=0.0)
    r = dict(model=name, tm_bacc=pooled['tm_bacc'], ci_lo=bt['lo'], ci_hi=bt['hi'], tm_sens=pooled['tm_sens'], tm_spec=pooled['tm_spec'], tm_auroc=pooled['tm_auroc'],
             fold_mean=sm['mean'], fold_sd=sm['sd'], fold_min=sm['min'], fold_max=sm['max'], acc=cl['acc'], bacc=cl['bacc'], auroc=cl['auroc'], n_windows=len(ev), n_people=len(per))
    for p, v in per.items():
        r[f'tm_bacc_{p}'] = v
    return r


def paired_vs_baselines(frame, base, B=2000, name='', best=None):
    """Paired person-level bootstrap of (pipeline - baseline) TM balanced accuracy against every baseline column `*__full` of baselines/scores/baseline_P1_hoffman.csv."""
    f = frame[frame['eval'].astype(bool)].copy()
    f['t_start'] = f.t_start.astype(float)
    b = base.copy(); b['t_start'] = b.t_start.astype(float)
    d = f.merge(b, on=['subject', 'rec', 't_start'], how='inner')
    assert len(d) == len(f), 'baseline scores must cover every evaluated window'
    y, t, s, sub = d.low.values.astype(int), d.t_start.values, d.score.values, d.subject.values
    me = M.tm_stat(t, y, s, 0.0, 'tm_bacc')
    rows = []
    for col in [c for c in b.columns if c.endswith('__full') and not c.startswith('majority')]:
        nm = col[:-6]
        dec = d[col].values.astype(float)
        sc = d[nm + '__score'].values.astype(float) if (nm + '__score') in d else dec
        base_stat = lambda idx, dec=dec, sc=sc: M.time_matched(t[idx], y[idx], dec[idx].astype(int), sc[idx])['tm_bacc']
        pb = M.paired_person_bootstrap(sub, me, base_stat, B=B, seed=0)
        rows.append(dict(model=name, baseline=nm, model_tm_bacc=pb['a'], baseline_tm_bacc=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'],
                         verdict=('win' if pb['lo'] > 0 else 'loss' if pb['hi'] < 0 else 'tie'), is_best_baseline=(nm == best), n_people=len(np.unique(sub))))
    return pd.DataFrame(rows)


def best_baseline(base):
    """The honest baseline with the highest pooled TM bacc, from the baselines' own decisions (full coverage)."""
    t, y = base.t_start.values, base.y_low.values.astype(int)
    out = {}
    for col in [c for c in base.columns if c.endswith('__full') and not c.startswith('majority')]:
        nm = col[:-6]
        if nm.startswith(('clock', 'ror_')):
            continue                                    # trivial rules, not honest learned or physics baselines for the best-baseline rule
        sc = base[nm + '__score'].values if (nm + '__score') in base else base[col].values
        out[nm] = M.time_matched(t, y, base[col].values.astype(int), sc)['tm_bacc']
    return max(out, key=out.get), out


def evaluate(runs, seeds, hof_meta=None, B=2000, base=None, fold_ens=True):
    """Everything for one set of seeds. runs: DataFrame from load_runs. Returns dict of tables and score frames."""
    meta, X19 = hof_meta or hoffman_meta()
    w = add_chain(wide(runs), meta, X19, sorted(meta.subject.unique()))
    F = folds(w)
    res = {}
    res['w'] = w
    sel, ch, tr = run_pipeline_over_folds(F, CNN_ORDER, with_train=True)
    res['final_scores'], res['final_choices'], res['final_train'] = sel, ch, tr
    cs, cch, ctr = run_pipeline_over_folds(F, CHAIN_ORDER, with_train=True)
    res['chain_levers_scores'], res['chain_levers_choices'], res['chain_levers_train'] = cs, cch, ctr
    cr, _, _ = run_pipeline_over_folds(F, ['chain_shipped'], fixed=('chain_shipped', False, 1))
    res['chain_shipped_scores'] = cr
    # grid of every cell at declared levers (exploratory)
    grid, scores_default = [], {}
    for name in [n for n in CANDS if n != 'chain_shipped']:
        for cal in (False, True):
            for k in KS:
                fr, _, _ = run_pipeline_over_folds(F, [name], fixed=(name, cal, k))
                s = summarize(fr, B=0, name=f'{name}|cal={int(cal)}|k={k}')
                s.update(cand=name, cal=cal, k=k, inner_cv_mean=float(np.mean([cand_score(F[o]['inner'], name, cal, k) for o in F])))
                grid.append(s)
                if not cal and k == 1:
                    scores_default[name] = fr
    res['grid'] = pd.DataFrame(grid)
    res['default_scores'] = scores_default
    return res


def fold_ensemble_scores(F, source):
    """Exploratory: the member fits (inner fits that exclude o) are averaged with the outer fit for P / D; the meta-learner is the one fit on single-model OOF logits."""
    frames = []
    for o, d in F.items():
        out = d['outer'].copy()
        mem = d['member']
        for col in ('p', 'd', 'corr'):
            if col in out and col in mem:
                m = mem.groupby('local')[col].mean()
                out[col] = (out[col].values + 5 * m.reindex(out.local).values) / 6.0
        if 'corr' in out:
            out['r'] = out['c'] + out['corr']
        sc, _ = apply_pipeline(d['inner'], out, source, False, 1)
        fr = out[['outer', 'subject', 'rec', 'local', 't_start', 'low', 'eval', 'in_base']].copy()
        fr['score'] = sc
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)


def paired_models(fa, fb, B=2000, name_a='', name_b=''):
    """Paired person bootstrap of two pipelines evaluated on the same windows (frames with subject, rec, t_start, low, eval, score)."""
    a = fa[fa['eval'].astype(bool)].copy(); b = fb[fb['eval'].astype(bool)].copy()
    a['t_start'] = a.t_start.astype(float); b['t_start'] = b.t_start.astype(float)
    d = a.merge(b[['subject', 'rec', 't_start', 'score']], on=['subject', 'rec', 't_start'], suffixes=('_a', '_b'))
    y, t, sub = d.low.values.astype(int), d.t_start.values, d.subject.values
    pb = M.paired_person_bootstrap(sub, M.tm_stat(t, y, d.score_a.values, 0.0, 'tm_bacc'), M.tm_stat(t, y, d.score_b.values, 0.0, 'tm_bacc'), B=B, seed=0)
    return dict(model=name_a, baseline=name_b, model_tm_bacc=pb['a'], baseline_tm_bacc=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'],
                verdict=('better' if pb['lo'] > 0 else 'worse' if pb['hi'] < 0 else 'tie'), is_best_baseline=False, n_people=len(np.unique(sub)))


def gap_row(name, frame, train, choices):
    """Train-vs-test gap of a whole pipeline: (a) resubstitution: the fold's pipeline scored on its own training people with the outer fit's in-sample logits,
    minus the held-out person's score; (b) inner-CV optimism: the inner CV score used to select minus the held-out score. Means over the 6 folds."""
    ev = frame[frame['eval'].astype(bool)]
    held = {o: M.time_matched(g.t_start.values, g.low.values.astype(int), (g.score.values > 0).astype(int), g.score.values)['tm_bacc'] for o, g in ev.groupby('outer')}
    o_sorted = sorted(held)
    te = np.array([held[o] for o in o_sorted])
    pooled = M.time_matched(ev.t_start.values, ev.low.values.astype(int), (ev.score.values > 0).astype(int), ev.score.values)['tm_bacc']
    out = dict(model=name, heldout_pooled=pooled, heldout_fold_mean=float(te.mean()))
    if len(train):
        tr = train.set_index('outer').train_tm_bacc.reindex(o_sorted).values
        g = M.train_vs_test_gap(tr, te)
        out.update(train_mean=g['train_mean'], gap_resub=g['gap'], gap_resub_sd=g['gap_sd'], gap_resub_lo=g['gap_lo'], gap_resub_hi=g['gap_hi'], gap_resub_pooled=float(tr.mean() - pooled))
    ic = choices.set_index('outer').inner_cv.reindex(o_sorted).values
    g2 = M.train_vs_test_gap(ic, te)
    out.update(inner_cv_mean=g2['train_mean'], gap_inner=g2['gap'], gap_inner_sd=g2['gap_sd'], gap_inner_lo=g2['gap_lo'], gap_inner_hi=g2['gap_hi'])
    return out


def final_score_frame(res):
    """Per-window OOF scores of the final hybrid (for the conformal stream): one row per evaluated Hoffman window."""
    s = res['final_scores']; ch = res['final_choices'].set_index('outer')
    w = res['w']
    o = w[w.target == 'outer'][['outer', 'local', 'p', 'd', 'corr', 'c', 'hr', 'p_sqi'] if 'd' in w else ['outer', 'local', 'c']]
    d = s[s['eval'].astype(bool)].merge(o, on=['outer', 'local'], how='left')
    d['prob'] = 1.0 / (1.0 + np.exp(-d.score))
    d['source'] = d.outer.map(ch.source); d['cal'] = d.outer.map(ch.cal); d['k'] = d.outer.map(ch.k)
    return d.rename(columns={'score': 'logit', 'c': 'chain_logit', 'p': 'cnn_logit', 'd': 'distilled_logit', 'corr': 'residual_corr', 'hr': 'cnn_hr', 'p_sqi': 'cnn_sqi'})


def report_cli(a):
    runs_dirs = a.runs.split(',')
    seeds = a.seeds or [0]
    out = a.out
    os.makedirs(os.path.join(out, 'scores'), exist_ok=True)
    runs, counts = load_runs(runs_dirs, seeds=seeds)
    print('job files per (family, seed):', counts, flush=True)
    base = pd.read_csv(os.path.join(out, 'scores', 'baseline_P1_hoffman.csv'))
    bname, bvals = best_baseline(base)
    print('best honest baseline', bname, round(bvals[bname], 4), flush=True)
    res = evaluate(runs, seeds, B=a.B)
    tag = a.tag
    rows = []
    named = {'HYBRID_FINAL': res['final_scores'], 'CHAIN_SAME_LEVERS': res['chain_levers_scores'], 'CHAIN_SHIPPED': res['chain_shipped_scores']}
    named.update({f'default_{k}': v for k, v in res['default_scores'].items()})
    for nm, fr in named.items():
        r = summarize(fr, B=a.B, name=nm)
        rows.append(r)
    final = pd.DataFrame(rows)
    final['seeds'] = ','.join(map(str, seeds))
    final.to_csv(os.path.join(out, f'hybrid_p1{tag}.csv'), index=False)
    pairs = [paired_vs_baselines(res['final_scores'], base, a.B, 'HYBRID_FINAL', bname), paired_vs_baselines(res['chain_levers_scores'], base, a.B, 'CHAIN_SAME_LEVERS', bname)]
    pairs.append(pd.DataFrame([paired_models(res['final_scores'], res['chain_levers_scores'], a.B, 'HYBRID_FINAL', 'CHAIN_SAME_LEVERS'),
                               paired_models(res['final_scores'], res['default_scores']['P'], a.B, 'HYBRID_FINAL', 'P_alone_default'),
                               paired_models(res['default_scores']['P'], res['chain_shipped_scores'], a.B, 'P_alone_default', 'CHAIN_SHIPPED')]))
    pd.concat(pairs, ignore_index=True).to_csv(os.path.join(out, f'hybrid_p1_paired{tag}.csv'), index=False)
    ch = res['final_choices'].assign(pipeline='HYBRID_FINAL')
    ch = pd.concat([ch, res['chain_levers_choices'].assign(pipeline='CHAIN_SAME_LEVERS')], ignore_index=True)
    ch.to_csv(os.path.join(out, f'hybrid_p1_choices{tag}.csv'), index=False)
    res['grid'].to_csv(os.path.join(out, f'hybrid_p1_grid{tag}.csv'), index=False)
    gaps = pd.DataFrame([gap_row('HYBRID_FINAL', res['final_scores'], res['final_train'], res['final_choices']),
                         gap_row('CHAIN_SAME_LEVERS', res['chain_levers_scores'], res['chain_levers_train'], res['chain_levers_choices'])])
    gaps.to_csv(os.path.join(out, f'hybrid_p1_gap{tag}.csv'), index=False)
    final_score_frame(res).to_csv(os.path.join(out, 'scores', f'hybrid_final{tag}_hoffman.csv'), index=False)
    # single seeds (exploratory): default levers
    srows = []
    meta, X19 = hoffman_meta()
    for s in seeds:
        r1, _ = load_runs(runs_dirs, seeds=[s])
        F1 = folds(add_chain(wide(r1), meta, X19, None))
        for nm in ('P', 'D', 'R', 'c+p', 'c+d', 'c+corr', 'all'):
            fr, _, _ = run_pipeline_over_folds(F1, [nm], fixed=(nm, False, 1))
            srows.append(dict(seed=s, **summarize(fr, B=0, name=nm)))
    Fall = folds(res['w'])
    for nm in ('P', 'D', 'R', 'c+p'):
        fr, _, _ = run_pipeline_over_folds(Fall, [nm], fixed=(nm, False, 1))
        srows.append(dict(seed=-1, **summarize(fr, B=0, name=nm)))
    fe = fold_ensemble_scores(Fall, 'P')
    srows.append(dict(seed=-1, **summarize(fe, B=0, name='P+fold_ensemble')))
    pd.DataFrame(srows).to_csv(os.path.join(out, f'hybrid_p1_seeds{tag}.csv'), index=False)
    print(final[['model', 'tm_bacc', 'ci_lo', 'ci_hi', 'fold_mean', 'fold_sd', 'tm_auroc']].round(3).to_string())
    print(res['final_choices'][['outer', 'source', 'cal', 'k', 'inner_cv']].to_string())
