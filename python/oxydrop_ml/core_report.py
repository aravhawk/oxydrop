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
    ('no_pretrain', '10 (exploratory)', 'no pretraining at all (no SSL, no OpenOx stage B; random init, 400 fine-tuning steps; the first run of row 2 accidentally measured this because of a variant-definition bug, it was caught and row 2 was rerun with stage B from scratch)'),
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
            d = df[(df.ds == 'mths') & df['eval'] & df.spo2_ref.notna()].merge(b.drop(columns=[c for c in ('spo2_ref', 'rec') if c in b.columns]), on=['subject', 't_start'], how='inner')
            if len(d):
                ref, pred, subj = d.spo2_ref.values, d.spo2_pred.values, d.subject.values
                cnn = lambda idx: float(np.abs(pred[idx] - ref[idx]).mean())
                for col in [c for c in b.columns if c not in ('subject', 'rec', 't_start', 'spo2_ref')]:
                    bp = d[col].values.astype(float)
                    base = lambda idx, bp=bp: float(np.nanmean(np.abs(bp[idx] - ref[idx])))
                    pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=seed, higher_is_better=False)
                    rows.append(dict(task='P3', seed=seed, baseline=col, metric='mae (lower better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d)))
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


def task_verdicts(pv):
    """Per task: best baseline, the CNN's paired difference to it, and which baselines it beats / ties / loses to (paired person bootstrap, mean over seeds)."""
    out = {}
    if not len(pv):
        return out
    g = pv.groupby(['task', 'baseline']).agg(metric=('metric', 'first'), cnn=('cnn', 'mean'), base=('base', 'first'), diff=('diff', 'mean'), lo=('lo', 'mean'), hi=('hi', 'mean')).reset_index()
    for task, t in g.groupby('task'):
        hib = task != 'P3'
        t = t.sort_values('base', ascending=not hib)
        best = t.iloc[0]
        out[task] = dict(best=best, wins=[r.baseline for _, r in t.iterrows() if r.lo > 0], ties=[r.baseline for _, r in t.iterrows() if r.lo <= 0 <= r.hi],
                         losses=[r.baseline for _, r in t.iterrows() if r.hi < 0], table=t)
    return out


NOTES = """
## What was run

* **Pipeline** (`python/oxydrop_ml/`): `ssl_pretrain.py` (contrastive SSL) -> `train.py --stageb` (OpenOx supervised warm-up) -> `loso_run.py` (per-fold fine-tuning on pooled phone + OpenOx windows, LOSO or grouped folds,
  nested selection) -> `core_report.py` (tables, paired bootstrap, scores). Student = `OxyNet` with 167,588 shipped parameters (`model.py`).
* **Per fold**: the held-out person (both Hoffman hands together) and an inner-validation set drawn from the other people (1 Hoffman, 6 MTHS, 4 BUT people) never enter training, the input
  scale, the baseline-embedding table or the OpenOx warm-up. Early stopping uses the inner-validation set only. 400 fine-tuning steps (batch 128, AdamW lr 1e-3, cosine) from the stage-B checkpoint.
* **Pretraining never touches an outer test person**: the corpus holds no Hoffman, MTHS or BUT person; 34 OpenOx people (about 25 %, stratified by Monk tone, `train.openox_holdout`) are excluded from SSL,
  stage B and every supervised run so FEAT-007/008 can evaluate OpenOx-specific tasks on people the student never saw. `tests/test_train.py` shows that changing a held-out person's signal and labels leaves
  every trained weight byte-identical.
* **Recipe**: chosen by nested CV before any outer comparison (`nested_selection.csv`): for each of the 6 outer Hoffman folds an inner leave-one-Hoffman-person-out CV (2 inner people) plus 8 MTHS and 6 BUT validation people
  scores 4 candidate recipes; the candidate with the best mean inner score (mean of time-matched balanced accuracy, HR within-5 and 1 - SpO2 MAE/5) is `C_lowhi` (low-oxygen loss weight 3). It is the per-fold winner for 5 of 6 outer folds
  (`D_steps800` for hof_100002); one recipe was applied to every fold and every ablation for simplicity and compute (a strictly per-fold choice would change one fold's recipe).
* **Protocols**: P1 is true LOSO over the 6 Hoffman people for every row. P2/P3: the `full` row is true LOSO over 100 people (62 MTHS + 38 BUT); ablation rows use 10 subject-grouped folds over the same 100 people
  and are labelled "not LOSO" in the `p23_protocol` column (compute); `full` is also reported under the same 10-fold protocol so each ablation is compared like for like.
* **Seeds**: the `full` row and the five P1 ablations that have them use seeds 0, 1, 2 (mean over seeds in the table, seed SD in `*_seed_sd`). Other rows use seed 0 only.
* **Windows**: P1 = 5,644 Hoffman windows (t >= 60 s), decision = low-oxygen probability > 0.5 (fixed, not tuned), time-matched balanced accuracy pooled over people; P2 = every window with a reference HR of the 100 people
  at full coverage (no abstention); P3 = MTHS windows with t >= 30 s.
* **Train-vs-test gap**: each fold's model is scored on its own training rows with the same metric (P1: time-matched balanced accuracy on the training Hoffman people; P2: within-5 on training MTHS+BUT rows; P3: MAE on training
  MTHS rows) and compared with the held-out person's score of the fold (per-person values); positive gap = over-fitting.
"""


def write_markdown(out, df, pv, runs, bdir):
    L = ['# Core CNN results (FEAT-006)', '',
         'Generated by `python -m oxydrop_ml.core_report` (`bash scripts/reproduce_ml.sh core-report`). Every number below is read from `results/ml/ablation_core.csv`,',
         '`results/ml/cnn_vs_baselines_paired.csv`, `results/ml/nested_selection*.csv` and `results/ml/ssl_manifest.json`; baselines are `results/ml/baselines_P{1,2,3}.csv` (FEAT-005).',
         'Protocol: `python/oxydrop_ml/PREREGISTRATION.md`. A win on a pre-registered metric needs the paired person-level bootstrap 95 % interval of the difference to the best honest baseline to exclude zero.', '']
    tv = task_verdicts(pv)
    L += ['## Verdict', '']
    names = {'P1': 'P1 low-oxygen detection (Hoffman, 6 people, time-matched balanced accuracy)', 'P2': 'P2 heart rate (MTHS + BUT, 100 people, within 5 bpm at full coverage)', 'P3': 'P3 absolute SpO2 (MTHS, 62 people, MAE, lower is better)'}
    full = df[(df.variant == 'full')].sort_values('p23_kind') if len(df) else df
    for task, key, cnn_key in (('P1', 'p1', 'p1_tm_bacc'), ('P2', 'p2', 'p2_within5'), ('P3', 'p3', 'p3_mae')):
        if task not in tv:
            continue
        v = tv[task]; b = v['best']
        row = full[full[cnn_key].notna()].iloc[0] if len(full) and full[cnn_key].notna().any() else None
        cnn = f"{fmt(row[cnn_key])} [{fmt(row[cnn_key + '_lo'])}, {fmt(row[cnn_key + '_hi'])}]" if row is not None else fmt(b.cnn)
        vd = verdict(b.lo if task != 'P3' else b.lo, b.hi)
        L.append(f"* **{names[task]}**: CNN (full) {cnn}; best honest baseline `{b.baseline}` {fmt(b.base)}; paired difference CNN minus baseline {fmt(b['diff'])} [{fmt(b.lo)}, {fmt(b.hi)}] (positive = CNN better) -> **{vd}**. "
                 f"Baselines the CNN beats with the interval excluding zero: {', '.join(v['wins']) or 'none'}. Ties: {', '.join(v['ties']) or 'none'}. Losses: {', '.join(v['losses']) or 'none'}.")
    L.append('')
    if len(pv):
        L += ['## Headline: the CNN against the honest baselines (paired person-level bootstrap, B = 1000, same windows, mean over seeds)', '',
              '| task | metric | baseline | baseline value | CNN | CNN minus baseline [95% CI] (positive = CNN better) | verdict |', '|---|---|---|---|---|---|---|']
        for task in ('P1', 'P2', 'P3'):
            if task not in tv:
                continue
            for _, r in tv[task]['table'].iterrows():
                lo, hi = r.lo, r.hi
                L.append(f"| {task} | {r.metric} | {r.baseline} | {fmt(r.base)} | {fmt(r.cnn)} | {fmt(r['diff'])} [{fmt(lo)}, {fmt(hi)}] | {verdict(lo, hi)} |")
        L.append('')
    if len(df):
        L += ['## Ablation table (descriptive; with several variants on 6 Hoffman people this carries selection optimism)', '',
              'Pooled values with 95 % person-bootstrap CI (mean of the seeds\' pooled values and CI bounds); "per-person" is the mean ± SD over folds (people). Gap = train-fold score minus held-out score (positive = over-fitting; for MAE test minus train).', '',
              '| # | variant | P2/P3 protocol | seeds | P1 TM bal. acc [95% CI] (seed SD) | P1 per-person mean ± SD | P1 gap | P2 within-5 [95% CI] (seed SD) | P2 per-person mean ± SD | P2 gap | P3 MAE [95% CI] | P3 per-person mean ± SD | P3 gap |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in df.iterrows():
            L.append(f"| {r.ablation} | {r.variant} | {r.p23_protocol} | {r.n_seeds} | {fmt(r.p1_tm_bacc)} [{fmt(r.p1_tm_bacc_lo)}, {fmt(r.p1_tm_bacc_hi)}] ({fmt(r.p1_tm_bacc_seed_sd)}) | "
                     f"{fmt(r.p1_tm_bacc_fold_mean)} ± {fmt(r.p1_tm_bacc_fold_sd)} | {fmt(r.p1_tm_bacc_gap)} | {fmt(r.p2_within5)} [{fmt(r.p2_within5_lo)}, {fmt(r.p2_within5_hi)}] ({fmt(r.p2_within5_seed_sd)}) | "
                     f"{fmt(r.p2_within5_fold_mean)} ± {fmt(r.p2_within5_fold_sd)} | {fmt(r.p2_within5_gap)} | {fmt(r.p3_mae)} [{fmt(r.p3_mae_lo)}, {fmt(r.p3_mae_hi)}] | {fmt(r.p3_mae_fold_mean)} ± {fmt(r.p3_mae_fold_sd)} | {fmt(r.p3_mae_gap)} |")
        L += ['', 'Variant definitions: `no_ssl` = stage B from scratch; `no_physics` = ratio-of-ratios channels, beat_ok and their side features zeroed; `no_openox` = SSL without OpenOx, no stage B, no OpenOx in training; '
              '`no_aug` = no device augmentation; `no_basecond` = baseline-relative inputs and emb_base removed; `hr_only` = only the HR and quality heads are trained (P1 and P3 are then not defined; the untrained SpO2 and '
              'low-oxygen heads are not scored); `win20` = two consecutive 10 s feature windows (20 s of input), BUT clips are 10 s so P2 for this row is MTHS only; `adv` = exploratory gradient-reversal head on dataset id.', '']
    L.append(NOTES)
    ext = os.path.join(out, 'CORE_RESULTS_extra.md')
    if os.path.exists(ext):
        L.append(open(ext).read())
    open(os.path.join(out, 'CORE_RESULTS.md'), 'w').write('\n'.join(L) + '\n')


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
        kinds = [k for k in ('p23', 'p23k') if any(_has(os.path.join(sd, k)) for sd in seed_dirs(a.runs, v))] or ['none']
        for kind in kinds:
            r = ablation_row(a.runs, v, num, label, a.B, kind)
            if r:
                rows.append(r)
                print('row', v, kind, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k in ('p1_tm_bacc', 'p2_within5', 'p3_mae', 'n_seeds')}, flush=True)
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
