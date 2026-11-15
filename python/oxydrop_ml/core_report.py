"""Build results/ml/ablation_core.csv, results/ml/CORE_RESULTS.md and the committed per-window score files from the LOSO run directories.

    python -m oxydrop_ml.core_report --runs RUNS_ROOT --out results/ml [--baselines results/ml] [--cache-scores CACHE/scores]

RUNS_ROOT layout (written by loso_run.py):  <variant>/s<seed>/{p1,p23,p23k}/fold_*.{json,csv}  and  nested/nested_*.json.
Every number in the tables is computed here from those files; the narrative verdicts (win / tie / loss) come from the paired person-level bootstrap
against the best honest baseline of the same task, using the baseline score files of FEAT-005 on the SAME windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import metrics as M
from . import loso_run as LR

ROWS = [   # (variant dir, ablation number, label)
    ('full', '1', 'full CNN (SSL + OpenOx stage B + physics channels + baseline conditioning + augmentation)'),
    ('no_ssl', '2', 'no SSL pretraining (stage B from scratch)'),
    ('no_physics', '3', 'no physics channels (ratio-of-ratios channels, beat_ok and their side features removed)'),
    ('no_openox', '4', 'no OpenOx data (no SSL on OpenOx, no stage B, no OpenOx in supervised training)'),
    ('no_aug', '5', 'no device augmentation'),
    ('no_basecond', '6', 'no baseline-relative conditioning (baseline-relative inputs and emb_base removed)'),
    ('hr_only', '7', 'HR-only CNN (SpO2, low-oxygen and auxiliary heads untrained; P1 and P3 not defined)'),
    ('win20', '8', '20 s window (instead of 10 s)'),
    ('adv', '9 (exploratory)', 'gradient-reversal domain-adversarial head on dataset id'),
]


def seed_dirs(root, variant):
    return sorted(glob.glob(os.path.join(root, variant, 's*')))


def _has(d):
    return os.path.isdir(d) and bool(glob.glob(d + '/fold_*.json'))


def variant_stats(root, variant, B=1000, kind='p23k'):
    """Per-seed summaries. P1 comes from <seed>/p1 (always LOSO), P2/P3 from <seed>/<kind> (p23 = LOSO over 100 people, p23k = 10 grouped folds, NOT LOSO)."""
    per = []
    for sd in seed_dirs(root, variant):
        dirs = [d for d in (os.path.join(sd, 'p1'), os.path.join(sd, kind)) if _has(d)]
        if not dirs:
            continue
        st = LR.summarize(dirs, B=B)
        st['seed_dir'] = sd
        st['p23_protocol'] = ('LOSO' if kind == 'p23' else 'grouped10 (not LOSO)') if _has(os.path.join(sd, kind)) else 'none'
        st['p1_protocol'] = 'LOSO' if _has(os.path.join(sd, 'p1')) else 'none'
        st['seed'] = int(os.path.basename(sd)[1:])
        per.append(st)
    return per


def agg(per, key):
    v = np.array([s[key] for s in per if key in s and s[key] is not None and np.isfinite(s[key])], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, np.nan, 0)


def ablation_row(root, variant, num, label, B=1000, kind='p23k'):
    per = variant_stats(root, variant, B, kind)
    if not per:
        return None
    r = dict(ablation=num, variant=variant, p23_kind=kind, description=label, n_seeds=len(per), seeds=','.join(str(s['seed']) for s in per),
             p1_protocol=per[0].get('p1_protocol'), p23_protocol=per[0].get('p23_protocol'), n_folds=int(np.mean([s['n_folds'] for s in per])))
    for key, tag in (('p1', 'p1_tm_bacc'), ('p2', 'p2_within5'), ('p3', 'p3_mae')):
        m, sd, n = agg(per, key)
        r[tag] = m; r[tag + '_seed_sd'] = sd
        for suf in ('lo', 'hi', 'fold_mean', 'fold_sd', 'train', 'gap', 'gap_lo', 'gap_hi'):
            r[f'{tag}_{suf}'] = agg(per, f'{key}_{suf}')[0]
    for k in ('p1_auroc', 'p2_doubling', 'p2_mae', 'p2_mths', 'p2_but'):
        r[k] = agg(per, k)[0]
    return r


def baseline_scores(bdir, cache_scores):
    out = {}
    p = os.path.join(bdir, 'scores', 'baseline_P1_hoffman.csv')
    if os.path.exists(p):
        out['p1'] = pd.read_csv(p)
    p = os.path.join(bdir, 'scores', 'baseline_P2_butppg.csv')
    q = os.path.join(cache_scores, 'baseline_P2_mths.csv')
    if os.path.exists(p):
        d = pd.read_csv(p)
        if os.path.exists(q):
            d = pd.concat([d, pd.read_csv(q)], ignore_index=True)
        out['p2'] = d
    q = os.path.join(cache_scores, 'baseline_P3_mths.csv')
    if os.path.exists(q):
        out['p3'] = pd.read_csv(q)
    return out


def paired_vs_baselines(root, variant, bdir, cache_scores, B=1000):
    """Paired person-level bootstrap of the CNN minus each honest baseline on P1, P2 and P3 (same windows). Returns a DataFrame."""
    bs = baseline_scores(bdir, cache_scores)
    sdirs = seed_dirs(root, variant)
    rows = []
    for sd in sdirs:
        seed = int(os.path.basename(sd)[1:])
        p23 = os.path.join(sd, 'p23') if glob.glob(os.path.join(sd, 'p23', 'fold_*.json')) else os.path.join(sd, 'p23k')
        df, _ = LR.load_run([d for d in (os.path.join(sd, 'p1'), p23) if os.path.isdir(d)])
        if df.empty:
            continue
        # ---- P1
        if 'p1' in bs:
            b = bs['p1']
            d = df[(df.ds == 'hoffman') & df['eval'] & df.low.notna()].merge(b, on=['subject', 'rec', 't_start'], how='inner')
            if len(d):
                y, t, s, subj = d.low.values.astype(int), d.t_start.values, d.p_low.values, d.subject.values
                cnn = M.tm_stat(t, y, s, 0.5, 'tm_bacc')
                for col in [c for c in b.columns if c.endswith('__full') and not c.startswith('majority')]:
                    name = col[:-6]
                    dec = d[col].values.astype(float)
                    sc = d[name + '__score'].values.astype(float) if (name + '__score') in d else dec
                    base = lambda idx, dec=dec, sc=sc: M.time_matched(t[idx], y[idx], dec[idx].astype(int), sc[idx])['tm_bacc']
                    pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=seed)
                    rows.append(dict(task='P1', seed=seed, baseline=name, metric='tm_bacc (higher better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d)))
        # ---- P2
        if 'p2' in bs:
            b = bs['p2']
            d = df[df.ds.isin(['mths', 'butppg']) & df.hr_ref.notna()].merge(b.drop(columns=['hr_ref']), on=['subject', 'rec', 't_start'], how='inner')
            if len(d):
                ref, pred, subj = d.hr_ref.values, d.hr_pred.values, d.subject.values
                cnn = lambda idx: M.hr_metrics(pred[idx], ref[idx])['within5_full']
                for col in [c for c in b.columns if c.endswith('__forced')]:
                    bp = d[col].values.astype(float)
                    base = lambda idx, bp=bp: M.hr_metrics(bp[idx], ref[idx])['within5_full']
                    pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=seed)
                    rows.append(dict(task='P2', seed=seed, baseline=col[:-8], metric='within5_full (higher better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d)))
        # ---- P3
        if 'p3' in bs:
            b = bs['p3']
            d = df[(df.ds == 'mths') & df['eval'] & df.spo2_ref.notna()].merge(b.drop(columns=[c for c in ('spo2_ref',) if c in b.columns]), on=['subject', 't_start'], how='inner', suffixes=('', '_b'))
            if len(d):
                ref, pred, subj = d.spo2_ref.values, d.spo2_pred.values, d.subject.values
                cnn = lambda idx: float(np.abs(pred[idx] - ref[idx]).mean())
                for col in [c for c in b.columns if c.endswith('__full') or c.endswith('__pred')]:
                    bp = d[col].values.astype(float)
                    base = lambda idx, bp=bp: float(np.nanmean(np.abs(bp[idx] - ref[idx])))
                    pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=seed, higher_is_better=False)
                    rows.append(dict(task='P3', seed=seed, baseline=col.rsplit('__', 1)[0], metric='mae (lower better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d)))
    return pd.DataFrame(rows)


def export_scores(root, out, cache_scores):
    """Per-window out-of-fold predictions. Hoffman and BUT rows are committed (results/ml/scores/cnn_<variant>_{hoffman,butppg}.csv, long format with a seed
    column); MTHS rows stay in the cache (CC BY-NC-ND), OpenOx rows are never written."""
    os.makedirs(os.path.join(out, 'scores'), exist_ok=True)
    os.makedirs(cache_scores, exist_ok=True)
    cols = ['seed', 'subject', 'rec', 't_start', 'eval', 'hr_ref', 'hr_pred', 'spo2_ref', 'spo2_pred', 'spo2_logvar', 'low', 'p_low', 'p_sqi']
    for v, _, _ in ROWS:
        parts = {'hoffman': [], 'butppg': [], 'mths': []}
        for sd in seed_dirs(root, v):
            p23 = os.path.join(sd, 'p23') if _has(os.path.join(sd, 'p23')) else os.path.join(sd, 'p23k')
            df, _f = LR.load_run([d for d in (os.path.join(sd, 'p1'), p23) if _has(d)])
            if df.empty:
                continue
            for ds in parts:
                q = df[df.ds == ds]
                if len(q):
                    parts[ds].append(q[cols[1:]].assign(seed=int(os.path.basename(sd)[1:]))[cols])
        for ds, L in parts.items():
            if not L:
                continue
            d = pd.concat(L, ignore_index=True)
            for c in ('hr_pred', 'spo2_pred', 'spo2_logvar', 'p_low', 'p_sqi'):
                d[c] = d[c].round(4)
            d.to_csv(os.path.join(cache_scores if ds == 'mths' else os.path.join(out, 'scores'), f'cnn_{v}_{ds}.csv'), index=False)


def verdict(lo, hi):
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie (interval includes zero)')


def fmt(x, d=3):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def write_markdown(out, df, pv, runs, bdir):
    L = ['# Core CNN results (FEAT-006)', '', 'Generated by `python -m oxydrop_ml.core_report` (`bash scripts/reproduce_ml.sh core-report`). Every number below is read from `results/ml/ablation_core.csv`,',
         '`results/ml/cnn_vs_baselines_paired.csv`, `results/ml/nested_selection*.csv` and `results/ml/ssl_manifest.json`. Protocol: `python/oxydrop_ml/PREREGISTRATION.md`.', '']
    if len(pv):
        L += ['## Headline: the CNN against the honest baselines (paired person-level bootstrap, B = 1000, same windows)', '',
              '| task | metric | baseline | baseline value | CNN (full, mean over seeds) | CNN minus baseline [95% CI] | verdict |', '|---|---|---|---|---|---|---|']
        g = pv.groupby(['task', 'baseline']).agg(metric=('metric', 'first'), cnn=('cnn', 'mean'), base=('base', 'first'), diff=('diff', 'mean'), lo=('lo', 'mean'), hi=('hi', 'mean'), seeds=('seed', 'nunique')).reset_index()
        for task in ('P1', 'P2', 'P3'):
            t = g[g.task == task]
            if not len(t):
                continue
            hib = task != 'P3'
            best = t.sort_values('base', ascending=not hib).head(4)
            for _, r in best.iterrows():
                lo, hi = (r.lo, r.hi)
                L.append(f"| {task} | {r.metric} | {r.baseline} | {fmt(r.base)} | {fmt(r.cnn)} | {fmt(r['diff'])} [{fmt(lo)}, {fmt(hi)}] | {verdict(lo, hi)} |")
        L.append('')
    if len(df):
        L += ['## Ablation table (descriptive; selection optimism applies: several variants on 6 Hoffman people)', '',
              '| # | variant | P2/P3 protocol | seeds | P1 TM bal. acc [95% CI] | P1 per-person mean ± SD | P1 train-test gap | P2 within-5 [95% CI] | P2 per-person mean ± SD | P2 gap | P3 MAE [95% CI] | P3 per-person mean ± SD | P3 gap |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in df.iterrows():
            L.append(f"| {r.ablation} | {r.variant} | {r.p23_protocol} | {r.n_seeds} | {fmt(r.p1_tm_bacc)} [{fmt(r.p1_tm_bacc_lo)}, {fmt(r.p1_tm_bacc_hi)}] ± seed SD {fmt(r.p1_tm_bacc_seed_sd)} | "
                     f"{fmt(r.p1_tm_bacc_fold_mean)} ± {fmt(r.p1_tm_bacc_fold_sd)} | {fmt(r.p1_tm_bacc_gap)} | {fmt(r.p2_within5)} [{fmt(r.p2_within5_lo)}, {fmt(r.p2_within5_hi)}] | "
                     f"{fmt(r.p2_within5_fold_mean)} ± {fmt(r.p2_within5_fold_sd)} | {fmt(r.p2_within5_gap)} | {fmt(r.p3_mae)} [{fmt(r.p3_mae_lo)}, {fmt(r.p3_mae_hi)}] | {fmt(r.p3_mae_fold_mean)} ± {fmt(r.p3_mae_fold_sd)} | {fmt(r.p3_mae_gap)} |")
        L.append('')
    open(os.path.join(out, 'CORE_RESULTS_tables.md'), 'w').write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--baselines', default=None, help='results/ml directory of FEAT-005 (default: --out)')
    ap.add_argument('--cache-scores', default=None)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--headline', default='full')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for v, num, label in ROWS:
      for kind in ('p23', 'p23k'):
        r = ablation_row(a.runs, v, num, label, a.B, kind)
        if r and (kind == 'p23' or not rows or True):
            rows.append(r)
            print('row', v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k in ('p1_tm_bacc', 'p2_within5', 'p3_mae', 'n_seeds')}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'ablation_core.csv'), index=False)
    print('wrote', os.path.join(a.out, 'ablation_core.csv'))
    bdir = a.baselines or a.out
    cs = a.cache_scores or os.path.join(os.environ.get('OXY_CACHE', '/home/aravhawk/.t3/scratch/veritas/ml_cache'), 'baselines', 'scores')
    pv = paired_vs_baselines(a.runs, a.headline, bdir, cs, a.B)
    pv.to_csv(os.path.join(a.out, 'cnn_vs_baselines_paired.csv'), index=False)
    print('wrote paired', len(pv))
    export_scores(a.runs, a.out, cs.replace('baselines/scores', 'scores'))
    write_markdown(a.out, df, pv, a.runs, bdir)


if __name__ == '__main__':
    main()
