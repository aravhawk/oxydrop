"""P2 (heart rate) of the hybrid: the final HR rule (amendments/AMENDMENT_2_hybrid.md), selected on Hoffman inner out-of-fold windows only and applied unchanged
to the true-LOSO BUT PPG + MTHS predictions of the CNN (FEAT-006 runs, 3 seeds, 100 people) next to the v2 spectral estimator.

Rules (cnn = median over seeds of the CNN HR, spec = v2 spectral + autocorrelation estimator with forced answer):
  cnn    the CNN
  spec   the spectral estimator
  guard  CNN unless it lies within 8 bpm of 2x or 0.5x the spectral estimate (a harmonic disagreement), then the spectral estimate
  agree  mean of the two when they are within 5 bpm of each other, else the CNN
  gate   (Amendment 3) the spectral estimate when the shipped v2 quality gate passes, else the CNN
The headline rule is the best CNN-containing rule on Hoffman inner windows (ties within 0.005: earliest of CNN_RULES); `spec` is the control.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import metrics as M

CNN_RULES = ('cnn', 'gate', 'guard', 'agree')
RULES = CNN_RULES + ('spec',)
TIE = 0.005
BASE_COLS = ['fft_plain__forced', 'v1_peak__forced', 'v2_spectral_acf__forced', 'v2_spectral_acf__gate_shipped', 'v2_spectral_acf__gate_nested']


def apply_rule(rule, cnn, spec, valid=None):
    cnn = np.asarray(cnn, float); spec = np.asarray(spec, float)
    if rule == 'cnn':
        return cnn.copy()
    if rule == 'spec':
        return spec.copy()
    if rule == 'guard':
        harm = (np.abs(cnn - 2 * spec) < 8) | (np.abs(cnn - 0.5 * spec) < 8)
        return np.where(harm & np.isfinite(spec), spec, cnn)
    if rule == 'gate':
        return np.where(np.asarray(valid, bool) & np.isfinite(spec), spec, cnn)
    if rule == 'agree':
        close = np.abs(cnn - spec) <= 5
        return np.where(close, 0.5 * (cnn + spec), cnn)
    raise ValueError(rule)


def hoffman_rule_table(runs, spec_hoffman, valid_hoffman):
    """Rule selection table on the Hoffman inner OOF rows of the P family. runs: DataFrame from hybrid.load_runs (family P). spec_hoffman, valid_hoffman: arrays indexed by shard-local row id (forced v2 estimate, shipped gate passes)."""
    p = runs[(runs.family == 'P') & (runs.target == 'inner')]
    cnn = p.groupby(['outer', 'inner', 'local']).hr.median().reset_index()
    ref = p.drop_duplicates(['outer', 'inner', 'local'])[['outer', 'inner', 'local', 'hr_ref', 'subject']]
    d = cnn.merge(ref, on=['outer', 'inner', 'local'])
    d = d[np.isfinite(d.hr_ref)]
    spec = spec_hoffman[d.local.values]
    rows = []
    for r in RULES:
        est = apply_rule(r, d.hr.values, spec, valid_hoffman[d.local.values])
        m = M.hr_metrics(est, d.hr_ref.values)
        rows.append(dict(rule=r, within5_full=m['within5_full'], mae=m['mae'], doubling=m['doubling'], n=len(d), n_people=d.subject.nunique()))
    t = pd.DataFrame(rows)
    c = t[t.rule.isin(CNN_RULES)]
    pick = next(r for r in CNN_RULES if float(c[c.rule == r].within5_full.iloc[0]) >= c.within5_full.max() - TIE)
    t['selected'] = t.rule == pick
    return t


def p2_frame(cnn_scores, base_scores):
    """Merge CNN LOSO scores (long, seed column) with the baseline estimates on (subject, rec, t_start): one row per window with cnn (seed median), spec, and the baselines."""
    c = cnn_scores[cnn_scores.hr_ref.notna()]
    g = c.groupby(['subject', 'rec', 't_start']).agg(cnn=('hr_pred', 'median'), p_sqi=('p_sqi', 'mean'), hr_ref=('hr_ref', 'first'), n_seeds=('seed', 'nunique')).reset_index()
    b = base_scores.drop(columns=['hr_ref'], errors='ignore')
    d = g.merge(b, on=['subject', 'rec', 't_start'], how='inner')
    assert len(d) == len(g), 'baseline estimates must cover every CNN window'
    d['spec'] = d['v2_spectral_acf__forced']
    d['valid'] = np.isfinite(d['v2_spectral_acf__gate_shipped'].values)
    return d


def p2_tables(frame, rule, B=2000, seed=0):
    """Metrics with person-bootstrap CIs for the chosen rule and every baseline, per scope (all, butppg, mths), plus paired differences."""
    rows, pairs = [], []
    ests = {f'HYBRID_{rule}': apply_rule(rule, frame.cnn.values, frame.spec.values, frame.valid.values), 'cnn_alone': frame.cnn.values}
    for c in BASE_COLS:
        ests[c] = frame[c].values.astype(float)
    ref, subj, ds = frame.hr_ref.values, frame.subject.values, frame.ds.values
    best = max(BASE_COLS, key=lambda c: M.hr_metrics(ests[c], ref)['within5_full'])
    for scope in ('all', 'butppg', 'mths'):
        m = np.ones(len(frame), bool) if scope == 'all' else ds == scope
        sub, rf = subj[m], ref[m]
        for name, e in ests.items():
            e = e[m]
            pooled = M.hr_metrics(e, rf)
            bt = M.person_bootstrap(sub, lambda idx, e=e: M.hr_metrics(e[idx], rf[idx])['within5_full'], B=B, seed=seed)
            per = [M.hr_metrics(e[sub == p], rf[sub == p])['within5_full'] for p in np.unique(sub)]
            sm = M.summarize_folds(per, seed=seed)
            rows.append(dict(scope=scope, model=name, within5_full=pooled['within5_full'], lo=bt['lo'], hi=bt['hi'], mae=pooled['mae'], doubling=pooled['doubling'],
                             halving=pooled['halving'], coverage=pooled['coverage'], fold_mean=sm['mean'], fold_sd=sm['sd'], n_people=len(per), n_windows=int(m.sum()),
                             is_best_baseline=(name == best)))
        for name in [f'HYBRID_{rule}'] + ['cnn_alone']:
            for bn in BASE_COLS + (['cnn_alone'] if name != 'cnn_alone' else []):
                a, b = ests[name][m], ests[bn][m]
                pb = M.paired_person_bootstrap(sub, lambda idx, a=a: M.hr_metrics(a[idx], rf[idx])['within5_full'], lambda idx, b=b: M.hr_metrics(b[idx], rf[idx])['within5_full'], B=B, seed=seed)
                pairs.append(dict(scope=scope, model=name, baseline=bn, model_within5=pb['a'], baseline_within5=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'],
                                  verdict=('win' if pb['lo'] > 0 else 'loss' if pb['hi'] < 0 else 'tie'), is_best_baseline=(bn == best)))
    return pd.DataFrame(rows), pd.DataFrame(pairs), best


def load_p2_inputs(res_dir, cache_scores, cnn_variant='full'):
    """CNN LOSO scores and baseline estimates of BUT PPG (committed) and MTHS (cache, licence) as one frame with a `ds` column."""
    parts = []
    for ds in ('butppg', 'mths'):
        cp = os.path.join(res_dir, 'scores', f'cnn_{cnn_variant}_{ds}.csv') if ds == 'butppg' else os.path.join(cache_scores, f'cnn_{cnn_variant}_{ds}.csv')
        bp = os.path.join(res_dir, 'scores', 'baseline_P2_butppg.csv') if ds == 'butppg' else os.path.join(os.path.dirname(cache_scores), 'baselines', 'scores', 'baseline_P2_mths.csv')
        if not (os.path.exists(cp) and os.path.exists(bp)):
            continue
        d = p2_frame(pd.read_csv(cp), pd.read_csv(bp))
        d['ds'] = ds
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
