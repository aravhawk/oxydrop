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


def variant_stats(root, variant, B=1000, kind='p23k', seeds=None):
    """Per-seed summaries. P1 comes from <seed>/p1 (always LOSO), P2/P3 from <seed>/<kind> (p23 = LOSO over 100 people, p23k = 10 grouped folds, NOT LOSO)."""
    per = []
    for sd in seed_dirs(root, variant):
        if seeds is not None and int(os.path.basename(sd)[1:]) not in seeds:
            continue
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


def ablation_row(root, variant, num, label, B=1000, kind='p23k', seeds=None, name=None):
    per = variant_stats(root, variant, B, kind, seeds)
    if not per:
        return None
    r = dict(ablation=num, variant=name or variant, p23_kind=kind, description=label, n_seeds=len(per), seeds=','.join(str(s['seed']) for s in per),
             p1_protocol=per[0].get('p1_protocol'), p23_protocol=per[0].get('p23_protocol'), n_folds=int(np.mean([s['n_folds'] for s in per])))
    for key, tag in (('p1', 'p1_tm_bacc'), ('p2', 'p2_within5'), ('p3', 'p3_mae')):
        m, sd, n = agg(per, key)
        r[tag] = m; r[tag + '_seed_sd'] = sd if n >= 2 else np.nan; r[tag + '_n_seeds'] = n
        for suf in ('lo', 'hi', 'fold_mean', 'fold_sd', 'train', 'gap', 'gap_lo', 'gap_hi'):
            r[f'{tag}_{suf}'] = agg(per, f'{key}_{suf}')[0]
    for k in ('p1_auroc', 'p2_doubling', 'p2_mae', 'p2_mths', 'p2_but'):
        r[k] = agg(per, k)[0]
    if variant == 'hr_only':          # the SpO2 and low-oxygen heads are never trained in this variant: P1 and P3 are not defined
        for k in list(r):
            if k.startswith(('p1_', 'p3_')) and not k.endswith('protocol'):
                r[k] = np.nan
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


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def paired_one(df, bs, seed, B=1000, subsets=True):
    """Paired person-level bootstrap of the CNN minus each honest baseline on P1, P2 and P3 for one prediction table `df` (rows of fold CSVs, or the seed ensemble)."""
    rows = []
    if 'p1' in bs and len(df):
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
                pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=0)
                rows.append(dict(task='P1', seed=seed, baseline=name, metric='tm_bacc (higher better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d), n_people=len(np.unique(subj))))
    if 'p2' in bs and len(df):
        b = bs['p2']
        d = df[df.ds.isin(['mths', 'butppg']) & df.hr_ref.notna()].merge(b.drop(columns=['hr_ref']), on=['subject', 'rec', 't_start'], how='inner')
        if len(d):
            ref, pred, subj, dsn = d.hr_ref.values, d.hr_pred.values, d.subject.values, d.ds.values
            for scope, mask in (('P2', np.ones(len(d), bool)), ('P2-mths', dsn == 'mths'), ('P2-but', dsn == 'butppg')):
                if not subsets and scope != 'P2':
                    continue
                dd = d[mask].reset_index(drop=True)
                r_, p_, sj = dd.hr_ref.values, dd.hr_pred.values, dd.subject.values
                cnn = lambda idx, r_=r_, p_=p_: M.hr_metrics(p_[idx], r_[idx])['within5_full']
                for col in [c for c in b.columns if c.endswith('__forced')]:
                    bp = dd[col].values.astype(float)
                    base = lambda idx, bp=bp, r_=r_: M.hr_metrics(bp[idx], r_[idx])['within5_full']
                    pb = M.paired_person_bootstrap(sj, cnn, base, B=B, seed=0)
                    rows.append(dict(task=scope, seed=seed, baseline=col[:-8], metric='within5_full (higher better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(dd), n_people=len(np.unique(sj))))
    if 'p3' in bs and len(df):
        b = bs['p3']
        d = df[(df.ds == 'mths') & df['eval'] & df.spo2_ref.notna()].merge(b.drop(columns=[c for c in ('spo2_ref', 'rec') if c in b.columns]), on=['subject', 't_start'], how='inner')
        if len(d):
            ref, pred, subj = d.spo2_ref.values, d.spo2_pred.values, d.subject.values
            cnn = lambda idx: float(np.abs(pred[idx] - ref[idx]).mean())
            for col in [c for c in b.columns if c not in ('subject', 'rec', 't_start', 'spo2_ref')]:
                bp = d[col].values.astype(float)
                base = lambda idx, bp=bp: float(np.nanmean(np.abs(bp[idx] - ref[idx])))
                pb = M.paired_person_bootstrap(subj, cnn, base, B=B, seed=0, higher_is_better=False)
                rows.append(dict(task='P3', seed=seed, baseline=col, metric='mae (lower better)', cnn=pb['a'], base=pb['b'], diff=pb['diff'], lo=pb['lo'], hi=pb['hi'], win=pb['win'], n=len(d), n_people=len(np.unique(subj))))
    return rows


def paired_vs_baselines(root, variant, bdir, cache_scores, B=1000):
    """Paired comparisons of the CNN with every honest baseline: per seed (seed = 0, 1, 2) and for the seed ensemble (seed = -1: mean logit / mean HR / mean SpO2
    of the seeds on identical windows). P1 uses every seed with P1 folds, P2/P3 only true-LOSO (p23) seeds; grouped folds are never mixed in."""
    bs = baseline_scores(bdir, cache_scores)
    rows, tabs = [], {}
    for sd in seed_dirs(root, variant):
        seed = int(os.path.basename(sd)[1:])
        dirs = [d for d in (os.path.join(sd, 'p1'), os.path.join(sd, 'p23')) if _has(d)]
        df, _ = LR.load_run(dirs)
        if df.empty:
            continue
        tabs[seed] = df
        rows += paired_one(df, bs, seed, B)
    if len(tabs) >= 2:
        allp = pd.concat([d.assign(seed=k) for k, d in tabs.items()], ignore_index=True)
        allp['lg'] = _logit(allp.p_low)
        keys = ['ds', 'subject', 'rec', 't_start']
        g = allp
        agg_ = g.groupby(keys).agg(eval=('eval', 'first'), low=('low', 'first'), hr_ref=('hr_ref', 'first'), spo2_ref=('spo2_ref', 'first'), lg=('lg', 'mean'),
                                   hr_pred=('hr_pred', 'mean'), spo2_pred=('spo2_pred', 'mean'), n=('seed', 'nunique')).reset_index()
        ens = agg_[agg_.n >= 2].copy()
        ens['p_low'] = 1 / (1 + np.exp(-ens.lg))
        rows += paired_one(ens, bs, -1, B)
    return pd.DataFrame(rows)


def seed_dirs_with(root, variant, kind):
    return [sd for sd in seed_dirs(root, variant) if _has(os.path.join(sd, kind))]


def per_person_p1(root, variant):
    """Per seed and person: plain AUROC, balanced accuracy, sensitivity, specificity (p_low > 0.5) and the person's own time-matched balanced accuracy."""
    out = []
    for sd in seed_dirs(root, variant):
        if not _has(os.path.join(sd, 'p1')):
            continue
        df, _ = LR.load_run(os.path.join(sd, 'p1'))
        d = df[(df.ds == 'hoffman') & df['eval'] & df.low.notna()]
        for p, g in d.groupby('subject'):
            c = M.classification_metrics(g.low.values, g.p_low.values)
            tm = M.time_matched(g.t_start.values, g.low.values.astype(int), (g.p_low.values > 0.5).astype(int), g.p_low.values)
            out.append(dict(seed=int(os.path.basename(sd)[1:]), subject=p, n=len(g), hypoxic_share=float(g.low.mean()), auroc=c['auroc'], bacc=c['bacc'], sens=c['sens'], spec=c['spec'], tm_bacc_person=tm['tm_bacc']))
    return pd.DataFrame(out)


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


def label(task, base_is_best, lo, hi):
    """Pre-registered wording: a 'win' only against the best honest baseline; against a weaker baseline the same interval is reported as 'better than a weaker baseline'."""
    if lo > 0:
        return 'win' if base_is_best else 'CNN better than this weaker baseline (not a pre-registered win)'
    if hi < 0:
        return 'loss' if base_is_best else 'baseline better'
    return 'tie (interval includes zero)'


def task_verdicts(pv):
    """Per task: best baseline (highest baseline value on the primary rows), primary paired difference (seed ensemble if present, else the lowest seed) and the per-seed outcomes."""
    out = {}
    if not len(pv):
        return out
    for task, t in pv.groupby('task'):
        hib = task != 'P3'
        seeds = sorted(t.seed.unique())
        prim = -1 if -1 in seeds else seeds[0]
        pr = t[t.seed == prim].sort_values('base', ascending=not hib).reset_index(drop=True)
        best = pr.iloc[0]
        per_seed = {int(sd): t[(t.seed == sd) & (t.baseline == best.baseline)].iloc[0] for sd in seeds if sd != -1 and len(t[(t.seed == sd) & (t.baseline == best.baseline)])}
        out[task] = dict(best=best, primary_seed=prim, table=pr, per_seed=per_seed,
                         wins=[r.baseline for _, r in pr.iterrows() if r.lo > 0], ties=[r.baseline for _, r in pr.iterrows() if r.lo <= 0 <= r.hi], losses=[r.baseline for _, r in pr.iterrows() if r.hi < 0])
    return out


NOTES = """
## What was run

* **Pipeline** (`python/oxydrop_ml/`): `ssl_pretrain.py` (contrastive SSL) -> `train.py --stageb` (OpenOx supervised warm-up) -> `loso_run.py` (per-fold fine-tuning on pooled phone + OpenOx windows, LOSO or grouped folds,
  nested selection) -> `core_report.py` (tables, paired bootstrap, scores); `stack_run.py` (hybrid chain + CNN, BatchNorm adaptation). Student = `OxyNet` with 167,588 shipped parameters (`model.py`).
* **Per fold**: the held-out person (both Hoffman hands together) and an inner-validation set drawn from the other people (1 Hoffman, 6 MTHS, 4 BUT people; the same people in every fold, so the same Hoffman person
  is the early-stopping person in 5 of 6 P1 folds and each P1 model trains on 4 Hoffman people) never enter training, the input scale, the baseline-embedding table or the OpenOx warm-up. 400 fine-tuning steps
  (batch 128, AdamW lr 1e-3, cosine) from the stage-B checkpoint; the weights of the best inner-validation step are kept.
* **Pretraining never touches an outer test person**: the corpus holds no Hoffman, MTHS or BUT person (subject-prefix filter and dataset provenance, not a person-level cross-check); 34 OpenOx people (about 25 %, stratified by Monk tone,
  `train.openox_holdout`) are excluded from SSL, stage B and every supervised run so later OpenOx-only tasks can use people the student never saw. `tests/test_train.py` shows that changing a held-out person's signal and labels
  leaves every trained weight byte-identical, and that stage B and nested selection read only the rows they should.
* **Recipe selection (deviates from the pre-registered wording, see PREREGISTRATION.md Amendment 1)**: 4 candidate recipes were scored by inner folds (for each of the 6 outer Hoffman people: 2 inner Hoffman people, plus 8 MTHS and 6 BUT validation
  people; score = mean of time-matched balanced accuracy, HR within-5 and 1 - SpO2 MAE/5). The choice for one outer person uses only its own inner jobs (`nested_selection.csv`: 5 of 6 outer people pick `C_lowhi`, hof_100002 picks `D_steps800`). The single
  recipe applied to all folds and ablations (`C_lowhi`: low-oxygen loss weight 3) is the mean over ALL outer folds' inner scores, which scores every Hoffman person as an inner person, and the 14 MTHS/BUT validation people are P2/P3 evaluation people.
  `nested_selection_sensitivity.csv` shows that dropping every selection job that touches any one person never changes the winner. Any bias favours the CNN slightly and cannot create a win against a baseline by itself.
* **Protocols**: P1 is true LOSO over the 6 Hoffman people for every row. P2/P3: the `full` row is true LOSO over 100 people (62 MTHS + 38 BUT; P3 uses the 61 MTHS people that have a window at t >= 30 s); ablation rows use 10 subject-grouped folds over the same people
  and are labelled "not LOSO" in the protocol column (compute); `full` is also reported under that protocol. LOSO and grouped P2 differ by more than seed noise (see table) and are not interchangeable.
* **Seeds**: `full` has seeds 0, 1, 2 for P1 and for the grouped P2/P3 protocol; the number of seeds of the LOSO P2/P3 rows is in the seeds column (pre-registration rule 7 asks for 3: see the table and Amendment 1). Ablation rows have seed 0 for P2/P3
  and 3 seeds for P1 except `hr_only`, `win20`, `adv`. Single-seed rows should be compared with the `full (seed 0)` row.
* **Windows**: P1 = 5,644 Hoffman windows (t >= 60 s), decision = low-oxygen probability > 0.5 (fixed, not tuned), time-matched balanced accuracy pooled over people; P2 = every window with a reference HR of the 100 people
  at full coverage (no abstention); P3 = MTHS windows with t >= 30 s.
* **Train-vs-test gap**: each fold model is scored on its own training rows (P1: time-matched balanced accuracy on the training Hoffman people, pooled; P2: within-5 on up to 1,500 training MTHS+BUT windows) and the mean over folds is compared with
  the held-out score pooled over all held-out people. The training MAE of P3 in these runs lacked the t >= 30 s filter, so no P3 gap is reported (fixed in `train_scores` for future runs).
"""


def write_markdown(out, df, pv, runs, bdir):
    L = ['# Core CNN results (FEAT-006)', '',
         'Generated by `python -m oxydrop_ml.core_report` (`bash scripts/reproduce_ml.sh core-report`). Every table below is computed from the run files and written to a CSV in `results/ml/`',
         '(`ablation_core.csv`, `cnn_vs_baselines_paired.csv`, `p1_per_person_seeds.csv`, `hybrid_p1*.csv`, `nested_selection*.csv`, `ssl_manifest.json`); baselines are `results/ml/baselines_P{1,2,3}.csv` (FEAT-005).',
         'The section "Reading the results" is hand-written and only quotes numbers that appear in those tables. Protocol: `python/oxydrop_ml/PREREGISTRATION.md` (see its Amendment 1 for the deviations).',
         'A **win** on a pre-registered metric needs the paired person-level bootstrap 95 % interval of the difference to the **best honest baseline** to exclude zero; against weaker baselines the same interval is only reported as "better than a weaker baseline".', '']
    tv = task_verdicts(pv)
    names = {'P1': 'P1 low-oxygen detection (Hoffman, 6 people, time-matched balanced accuracy)', 'P2': 'P2 heart rate (MTHS + BUT, 100 people, within 5 bpm at full coverage)',
             'P3': 'P3 absolute SpO2 (MTHS, people with a window at t >= 30 s, MAE, lower is better)'}
    L += ['## Verdict (primary = 3-seed ensemble where several seeds exist, else seed 0; per-seed outcomes in brackets)', '']
    for task in ('P1', 'P2', 'P3'):
        if task not in tv:
            continue
        v = tv[task]; b = v['best']
        ps = ', '.join(f"seed {k}: {label(task, True, r.lo, r.hi)}" for k, r in sorted(v['per_seed'].items()))
        npe = int(b.n_people) if 'n_people' in b else 0
        L.append(f"* **{names[task]}** ({npe} people): best honest baseline `{b.baseline}` {fmt(b.base)}; CNN {fmt(b.cnn)} ({'seed ensemble' if v['primary_seed'] == -1 else 'seed ' + str(v['primary_seed'])}); paired difference CNN minus baseline "
                 f"{fmt(b['diff'])} [{fmt(b.lo)}, {fmt(b.hi)}] (positive = CNN better) -> **{label(task, True, b.lo, b.hi)}**. Per seed against the same baseline: {ps}. "
                 f"Other baselines the CNN beats with the interval excluding zero (not pre-registered wins): {', '.join(x for x in v['wins'] if x != b.baseline) or 'none'}; baselines that beat the CNN: {', '.join(v['losses']) or 'none'}.")
    L.append('')
    if len(pv):
        L += ['## Headline table: the CNN against every honest baseline (paired person-level bootstrap, B = 1000, same windows)', '',
              'CNN value = mean over the seeds that exist for the task; difference and interval = primary row (seed ensemble when available). The last column lists the per-seed outcome.', '',
              '| task | baseline | baseline value | CNN | CNN minus baseline [95% CI] (positive = CNN better) | outcome | per seed |', '|---|---|---|---|---|---|---|']
        for task in ('P1', 'P2', 'P2-mths', 'P2-but', 'P3'):
            t = pv[pv.task == task]
            if not len(t):
                continue
            seeds = sorted(t.seed.unique()); prim = -1 if -1 in seeds else seeds[0]
            hib = task != 'P3'
            pr = t[t.seed == prim].sort_values('base', ascending=not hib).reset_index(drop=True)
            for k, r in pr.iterrows():
                cnn_mean = t[(t.baseline == r.baseline) & (t.seed != -1)].cnn.mean()
                per = ', '.join(f"{int(x.seed)}:{'+' if x.lo > 0 else ('-' if x.hi < 0 else '0')}" for _, x in t[(t.baseline == r.baseline) & (t.seed != -1)].sort_values('seed').iterrows())
                lab = label(task, k == 0, r.lo, r.hi)
                L.append(f"| {task} | {r.baseline} | {fmt(r.base)} | {fmt(cnn_mean)} | {fmt(r['diff'])} [{fmt(r.lo)}, {fmt(r.hi)}] | {lab} | {per} |")
        L += ['', 'Per seed: `+` the CNN is better with the interval excluding zero, `0` interval includes zero, `-` the baseline is better. `P2-mths` / `P2-but` are descriptive per-dataset subsets of P2 (not pre-registered).', '']
    if len(df):
        L += ['## Ablation table (descriptive; with several variants on 6 Hoffman people this carries selection optimism)', '',
              'Pooled values with 95 % person-bootstrap CI (mean of the seeds\' pooled values and CI bounds, seed SD in brackets, n/a with one seed); "per-person" = mean ± SD over held-out people. '
              'Seeds column = seeds for P1 / for P2+P3 (rule 7 asks for >= 3 on the final row). Gap = mean over folds of the fold model\'s score on its own training rows minus the pooled held-out score (positive = over-fitting; MAE: held-out minus train; P3 gap n/a: the training MAE of these runs did not use the t >= 30 s filter). '
              'Compare single-seed rows with the `full (seed 0)` row, not with the 3-seed `full` row.', '',
              '| # | variant | P2/P3 protocol | seeds | P1 TM bal. acc [95% CI] (seed SD) | P1 per-person mean ± SD | P1 gap | P2 within-5 [95% CI] (seed SD) | P2 per-person mean ± SD | P2 gap | P3 MAE [95% CI] | P3 per-person mean ± SD |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in df.iterrows():
            L.append(f"| {r.ablation} | {r.variant} | {r.p23_protocol} | {int(r.p1_tm_bacc_n_seeds) if r.p1_tm_bacc_n_seeds == r.p1_tm_bacc_n_seeds else 0} / {int(r.p2_within5_n_seeds) if r.p2_within5_n_seeds == r.p2_within5_n_seeds else 0} | "
                     f"{fmt(r.p1_tm_bacc)} [{fmt(r.p1_tm_bacc_lo)}, {fmt(r.p1_tm_bacc_hi)}] ({fmt(r.p1_tm_bacc_seed_sd)}) | {fmt(r.p1_tm_bacc_fold_mean)} ± {fmt(r.p1_tm_bacc_fold_sd)} | {fmt(r.p1_tm_bacc_gap)} | "
                     f"{fmt(r.p2_within5)} [{fmt(r.p2_within5_lo)}, {fmt(r.p2_within5_hi)}] ({fmt(r.p2_within5_seed_sd)}) | {fmt(r.p2_within5_fold_mean)} ± {fmt(r.p2_within5_fold_sd)} | {fmt(r.p2_within5_gap)} | "
                     f"{fmt(r.p3_mae)} [{fmt(r.p3_mae_lo)}, {fmt(r.p3_mae_hi)}] | {fmt(r.p3_mae_fold_mean)} ± {fmt(r.p3_mae_fold_sd)} |")
        L += ['', 'Variant definitions: `no_ssl` = stage B (OpenOx supervised warm-up) from scratch, no SSL; `no_pretrain` = neither SSL nor stage B (exploratory; the first run of row 2 accidentally measured this, see below); `no_physics` = ratio-of-ratios channels, '
              'beat_ok and their side features zeroed (the baseline-relative DC channels stay, so this is not "raw RGB only"); `no_openox` = SSL without OpenOx, no stage B, no OpenOx in training; `no_aug` = no device augmentation; '
              '`no_basecond` = baseline-relative inputs and emb_base removed; `hr_only` = only the HR and quality heads are trained (P1 and P3 not defined); `win20` = two consecutive 10 s feature windows, fine-tuned from a stage-B '
              'checkpoint trained on 10 s windows; BUT clips are 10 s so P2 for this row is MTHS only; `adv` = exploratory gradient-reversal head on dataset id (single seed).', '']
    pp = os.path.join(out, 'p1_per_person_seeds.csv')
    if os.path.exists(pp):
        q = pd.read_csv(pp)
        L += ['## P1 per person and seed (`full`, threshold 0.5)', '', '| person | ' + ' | '.join(f'seed {int(s_)} AUROC / sens / spec' for s_ in sorted(q.seed.unique())) + ' |', '|---|' + '---|' * q.seed.nunique()]
        for per_, g in q.groupby('subject'):
            L.append(f'| {per_} | ' + ' | '.join(f"{fmt(r.auroc, 2)} / {fmt(r.sens, 2)} / {fmt(r.spec, 2)}" for _, r in g.sort_values('seed').iterrows()) + ' |')
        L.append('')
    hp_ = os.path.join(out, 'hybrid_p1.csv')
    if os.path.exists(hp_):
        h = pd.read_csv(hp_)
        L += ['## Hybrid physics + CNN and test-time adaptation on P1 (exploratory, seed 0, `stack_run.py`)', '',
              'For each outer Hoffman person the 5 other people give inner out-of-fold CNN and chain predictions (CNN refit without the outer and the inner person); a logistic meta-learner on [CNN logit, v1 chain logit] is fit on them, '
              'and the candidate used for the outer person is the one with the best leave-one-inner-person-out time-matched balanced accuracy. `cnn_bnA`: BatchNorm statistics re-estimated on the person\'s own first-minute windows, blended by A %. '
              '`chain` is the FEAT-005 v1 chain refit with the same folds (reproduces 0.796). The last row is the inner-CV choice; the others are descriptive.', '',
              '| candidate | TM bal. acc [95% CI] | TM AUROC | per-person mean ± SD | plain bal. acc | plain AUROC |', '|---|---|---|---|---|---|']
        for _, r in h.iterrows():
            L.append(f"| {r.candidate} | {fmt(r.tm_bacc)} [{fmt(r.lo)}, {fmt(r.hi)}] | {fmt(r.tm_auroc)} | {fmt(r.per_person_mean)} ± {fmt(r.per_person_sd)} | {fmt(r.bacc)} | {fmt(r.auroc)} |")
        sp = os.path.join(out, 'hybrid_p1_selection.csv')
        if os.path.exists(sp):
            c = pd.read_csv(sp); ch = c[c.chosen]
            L += ['', 'Inner-CV choice per outer person: ' + ', '.join(f"{r.outer[-6:]} -> {r.candidate}" for _, r in ch.iterrows()) + '.']
        L.append('')
    sn = os.path.join(out, 'nested_selection_sensitivity.csv')
    if os.path.exists(sn):
        q = pd.read_csv(sn)
        L += ['## Recipe selection: sensitivity to each person (see Amendment 1)', '', '| selection jobs touching this person dropped | winner | margin to second |', '|---|---|---|']
        for _, r in q.iterrows():
            L.append(f"| {r.dropped_person} | {r.winner} | {fmt(r.margin_to_second, 4)} |")
        L.append('')
    mp = os.path.join(out, 'ssl_manifest.json')
    if os.path.exists(mp):
        m = json.load(open(mp))
        sf, pr = m.get('ssl_full', {}), m.get('linear_probe_hr', {})
        L += ['## Self-supervised pretraining', '',
              f"* SimCLR-style contrastive pretraining, {sf.get('steps')} steps, batch {sf.get('bs')}, {sf.get('rows'):,} windows from {sf.get('subjects')} people (sources {sf.get('sources')}); NT-Xent loss {sf.get('first_loss', float('nan')):.2f} -> {sf.get('last_loss', float('nan')):.2f}, top-1 positive retrieval {sf.get('last_top1', float('nan')):.2f}. "
              f"Checkpoint SHA-256 `{sf.get('sha256')}`. No-OpenOx variant: loss {m.get('ssl_noox', {}).get('last_loss', float('nan')):.2f}. Loss curves: `ssl_loss_curve_full.csv`, `ssl_loss_curve_noox.csv`."]
        if pr:
            L.append(f"* Linear-probe sanity check (frozen 64-d embedding, ridge, fit on MTHS+BUT windows, tested on {pr.get('n_test')} Hoffman windows; nobody shared): HR MAE {pr['ssl']['mae']:.1f} bpm (within-5 {pr['ssl']['within5']:.2f}) "
                     f"versus random-init encoder {pr['random_init']['mae']:.1f} ({pr['random_init']['within5']:.2f}) and train-mean {pr['train_mean']['mae']:.1f} ({pr['train_mean']['within5']:.2f}).")
        L.append('')
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
    for kind in ('p23', 'p23k'):
        r = ablation_row(a.runs, 'full', '1 (seed 0 only)', 'full CNN, seed 0 only: the reference for the single-seed rows', a.B, kind, seeds=[0], name='full (seed 0)')
        if r and any(_has(os.path.join(sd, kind)) for sd in seed_dirs(a.runs, 'full')):
            rows.insert(1 if kind == 'p23' else 3, r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'ablation_core.csv'), index=False)
    pp = per_person_p1(a.runs, a.headline)
    pp.to_csv(os.path.join(a.out, 'p1_per_person_seeds.csv'), index=False)
    nd = os.path.join(a.runs, 'nested')
    if os.path.isdir(nd):
        LR.selection_sensitivity(nd).to_csv(os.path.join(a.out, 'nested_selection_sensitivity.csv'), index=False)
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
