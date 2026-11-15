"""Write results/ml/HYBRID.md from the hybrid_*.csv files (every number in the tables and in the generated verdict sentences is read from a CSV).

    python -m oxydrop_ml.hybrid_md --res results/ml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def f3(x, d=3):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def ci(lo, hi):
    return f'[{f3(lo)}, {f3(hi)}]'


def md_table(df, cols, heads=None):
    heads = heads or cols
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return out


def verdict_text(v, best):
    if v == 'win' and not best:
        return 'better than a weaker baseline (not a pre-registered win)'
    return v


STATIC = """## What is and is not claimed (prior work)

None of the components is new. The contribution is the evaluation: every selection is nested, every number is leave-one-person-out, and the CNN is reported where it ties or loses.
* Learning a correction on top of a physics or domain model (the residual family `R`): theory-guided data science, Karpatne et al., IEEE TKDE 29(10), 2017.
* Distilling a teacher into a student (`D`, with the chain as teacher): Hinton, Vinyals and Dean, arXiv:1503.02531, 2015.
* Stacked generalisation with a meta-learner fit on out-of-fold predictions: Wolpert, Neural Networks 5(2), 1992. Nested cross-validation for selection without optimism: Varma and Simon, BMC Bioinformatics 7:91, 2006.
* Test-time BatchNorm adaptation (tried in FEAT-006, it hurts): Schneider et al., NeurIPS 2020; Wang et al. (Tent), ICLR 2021.
* Data and protocol: Hoffman et al., Smartphone camera oximetry in an induced hypoxemia study, npj Digital Medicine 5:146, 2022 (UW Hoffman data, MIT licence).
* Not claimed: a device-agnostic-augmentation novelty (FEAT-006 found it within noise), phone SpO2 accuracy, or any result on a phone outside the training and evaluation data. Android phones and iPhone models that are not in the data are an untested limitation. The three team iPhones are a held-out sanity check after the fact, never a tuning set.

## Files

| file | content |
|---|---|
| `hybrid_p1_final.csv` | P1 summary per model: pooled TM balanced accuracy, person-bootstrap CI, per-person mean/SD/min/max, one column per held-out person |
| `hybrid_p1_final_paired.csv` | paired person-bootstrap differences to every baseline, to the same-lever chain control and to the plain CNN |
| `hybrid_p1_final_choices.csv` | per outer person: the pipeline chosen from inner folds and every inner-CV score of the selection steps |
| `hybrid_p1_final_grid.csv`, `hybrid_p1_final_seeds.csv`, `hybrid_p1_final_gap.csv` | exploratory grid of candidates x calibration x k, single seeds / ensembles, train-vs-test gaps |
| `hybrid_p2.csv`, `hybrid_p2_paired.csv`, `hybrid_p2_rules.csv` | P2 HR tables, paired differences, Hoffman rule-selection table |
| `scores/hybrid_final_hoffman.csv` | per-window out-of-fold scores of the final hybrid on the 5,644 Hoffman windows (for conformal calibration): `logit` (> 0 = possible low oxygen), `prob` = sigmoid(logit) (class balanced, not prevalence calibrated), the components `chain_logit`, `cnn_logit`, `distilled_logit`, `residual_corr`, the chosen `source`, `cal`, `k` |
| `scores/hybrid_final_butppg.csv` | per-window final HR of the 3,840 BUT PPG windows: `hr_ref`, `hr_cnn`, `hr_spec`, `spec_gate_ok`, `hr_final`, `cnn_sqi`, `rule`. MTHS rows are written to the cache only (licence) |

## Export for the app: `app/data/hybrid_weights.json`

Written by `python -m oxydrop_ml.export_hybrid` (format `oxydrop-hybrid-v1`):

| key | content |
|---|---|
| `pipeline.source`, `pipeline.features` | the selected candidate and its feature names, in order (`c` chain logit, `p` plain CNN logit, `d` distilled CNN logit, `corr` residual network output, `r` = `c` + `corr`) |
| `pipeline.calibration` | true: subtract from every feature its median over the windows of the same recording that end inside the first 60 s |
| `pipeline.aggregate_windows` | k: causal mean of the last k windows (2 s hop, baseline windows included, fewer at the start) |
| `pipeline.meta` | `mean`, `scale`, `coef`, `intercept`: logit = intercept + sum(coef * (f - mean) / scale); decision: logit > 0 means possible low oxygen |
| `chain.models` | the two v1 chain models (`cols`, `mean`, `scale`, `coef`, `intercept`); `c` = logit of the mean of the two probabilities, clipped to [1e-6, 1 - 1e-6]; `max_abs_coef_diff_vs_app_weights_json` compares with `app/data/weights.json` |
| `members[]` | `family` (P, D or R), `seed`, `role` (which feature the member feeds), `weights` (file with the `oxynet-v1` graph, phone input scale folded in, consumes raw features), `ckpt_sha256`; members of a family are averaged in logit space |
| `selection` | the inner-CV scores that chose the pipeline when the rule is run on all six out-of-fold tables |

Reference combiner (per recording, windows in time order, the chain logit `c` and the member logits already computed):

```
f[i]      = [feature_j(i) for j in pipeline.features]              # c, p, d, corr, r = c + corr
if calibration: f[i] -= median(f[w] for windows w that end inside the first 60 s)
if k > 1:       f[i]  = mean(f[max(0, i - k + 1) .. i])            # causal, baseline windows included
logit[i]  = intercept + sum_j coef[j] * (f[i][j] - mean[j]) / scale[j]       # > 0: possible low oxygen
```

`tests/fixtures/hybrid_ref.json` holds one recording of feature rows with the expected logit for the combiner (the chain and the CNN have their own parity tests). Since the chain is only defined after the first minute, the app evaluates the chain on the stored calibration windows once the baseline exists (retroactively) before it applies calibration and aggregation.

## Reproduce

```
source scripts/ml_env.sh
bash scripts/reproduce_ml.sh hybrid-test      # unit tests of the hybrid
bash scripts/reproduce_ml.sh hybrid-fit       # 3 families x 3 seeds x 36 fits (Daytona: scripts/ml_daytona.py up|bg|tail|pull|down; about 1 minute per fit on one vCPU)
bash scripts/reproduce_ml.sh hybrid-report    # hybrid_*.csv, scores/hybrid_final_*.csv, HYBRID.md
bash scripts/reproduce_ml.sh hybrid-export    # app/data/hybrid_weights.json, member weights, tests/fixtures/hybrid_ref.json
```
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/ml'); ap.add_argument('--tag', default='')
    a = ap.parse_args(argv)
    R, t = a.res, a.tag
    rd = lambda n: pd.read_csv(os.path.join(R, n.replace('.csv', f'{t}.csv')))
    p1, pr, ch, gr, gp, sd = rd('hybrid_p1_final.csv'), rd('hybrid_p1_final_paired.csv'), rd('hybrid_p1_final_choices.csv'), rd('hybrid_p1_final_grid.csv'), rd('hybrid_p1_final_gap.csv'), rd('hybrid_p1_final_seeds.csv')
    p2, p2p, p2r = rd('hybrid_p2.csv'), rd('hybrid_p2_paired.csv'), rd('hybrid_p2_rules.csv')
    core = pd.read_csv(os.path.join(R, 'cnn_vs_baselines_paired.csv')) if os.path.exists(os.path.join(R, 'cnn_vs_baselines_paired.csv')) else None
    seeds = p1.seeds.iloc[0]
    row = lambda m: p1[p1.model == m].iloc[0]
    fin, cl, cs = row('HYBRID_FINAL'), row('CHAIN_SAME_LEVERS'), row('CHAIN_SHIPPED')
    bp = pr[(pr.model == 'HYBRID_FINAL') & pr.is_best_baseline].iloc[0]
    pc = pr[(pr.model == 'HYBRID_FINAL') & (pr.baseline == 'CHAIN_SAME_LEVERS')].iloc[0]
    pp = pr[(pr.model == 'HYBRID_FINAL') & (pr.baseline == 'P_alone_default')].iloc[0]
    g_h, g_c = gp[gp.model == 'HYBRID_FINAL'].iloc[0], gp[gp.model == 'CHAIN_SAME_LEVERS'].iloc[0]
    L = ['# OxyDrop hybrid: physics chain + CNN (FEAT-008)', '',
         '*Generated by `python/oxydrop_ml/hybrid_md.py` from the `results/ml/hybrid_p1_final*.csv` and `hybrid_p2*.csv` files; do not edit by hand. Research prototype, not a medical device.*', '',
         'Protocol: `python/oxydrop_ml/PREREGISTRATION.md` with Amendments 1 to 3 (`python/oxydrop_ml/amendments/`). Amendment 2 was committed before any hybrid outer-fold score existed; Amendment 3 (the P2 final HR rule) was written after the Hoffman rule table and the pooled CNN-vs-spectral P2 table had been seen, and says so. All numbers are leave-one-person-out, selections use inner folds only, intervals are 95 % person-level bootstraps (B = 2000, resampling people), the primary metric is unchanged: **P1 time-matched balanced accuracy at full coverage** on the 5,644 Hoffman windows (6 people), best honest baseline = the shipped v1 logistic chain.', '',
         f'CNN seeds used: {seeds}. Recipe, selection rule and candidate set are in Amendment 2; the CNN is the 167,588 parameter student, unchanged in the exported graph.', '',
         '## Headline (P1, primary)', '',
         f'* **Selected hybrid (CNN in the inference path): {f3(fin.tm_bacc)} {ci(fin.ci_lo, fin.ci_hi)}**, per-person mean {f3(fin.fold_mean)} +- {f3(fin.fold_sd)} (min {f3(fin.fold_min)}, max {f3(fin.fold_max)}).',
         f'* Best pre-registered baseline `{bp.baseline}`: {f3(bp.baseline_tm_bacc)}. Paired difference hybrid minus baseline {f3(bp["diff"])} {ci(bp.lo, bp.hi)}: **{bp.verdict}**.',
         f'* Same inner-fold selection restricted to the chain (a recalibrated chain, matched levers): {f3(cl.tm_bacc)} {ci(cl.ci_lo, cl.ci_hi)}; the hybrid minus this control is {f3(pc["diff"])} {ci(pc.lo, pc.hi)} ({pc.verdict}).',
         f'* Hybrid minus the plain CNN (`P`, default levers, same seeds): {f3(pp["diff"])} {ci(pp.lo, pp.hi)} ({pp.verdict}).', '']
    L += ['## P1 table: hybrid, controls and candidates (outer LOSO, selection on inner folds only)', '']
    t1 = p1.copy()
    t1['TM bacc [95% CI]'] = [f'{f3(r.tm_bacc)} {ci(r.ci_lo, r.ci_hi)}' for r in t1.itertuples()]
    t1['per-person mean +- SD'] = [f'{f3(r.fold_mean)} +- {f3(r.fold_sd)}' for r in t1.itertuples()]
    t1['min / max'] = [f'{f3(r.fold_min)} / {f3(r.fold_max)}' for r in t1.itertuples()]
    t1['TM AUROC'] = t1.tm_auroc.map(f3)
    t1['sens / spec (TM)'] = [f'{f3(r.tm_sens)} / {f3(r.tm_spec)}' for r in t1.itertuples()]
    L += md_table(t1, ['model', 'TM bacc [95% CI]', 'per-person mean +- SD', 'min / max', 'TM AUROC', 'sens / spec (TM)'])
    L += ['', '`HYBRID_FINAL` is the pipeline chosen per outer person by the declared rule; `CHAIN_SAME_LEVERS` applies the same rule to the chain only; `CHAIN_SHIPPED` is the baseline as shipped (raw chain logit, threshold 0); `default_*` rows are the candidates at default levers (no calibration, k = 1), exploratory, never used for selection.', '']
    person_cols = [c for c in p1.columns if c.startswith('tm_bacc_hof_')]
    pp_t = p1[p1.model.isin(['HYBRID_FINAL', 'CHAIN_SAME_LEVERS', 'CHAIN_SHIPPED', 'default_P'])][['model'] + person_cols].copy()
    for c in person_cols:
        pp_t[c] = pp_t[c].map(f3)
    L += ['### Per-person values (time-matched balanced accuracy of the held-out person)', ''] + md_table(pp_t, ['model'] + person_cols, ['model'] + [c.replace('tm_bacc_', '') for c in person_cols]) + ['']
    L += ['## Paired comparison with every baseline (P1)', '',
          'Difference = model minus baseline, paired person-level bootstrap on the same windows. A **win** is only claimed against the best baseline; a "win" row against a weaker baseline means better than a weaker baseline, not a pre-registered win.', '']
    pt = pr[pr.model.isin(['HYBRID_FINAL', 'CHAIN_SAME_LEVERS'])].copy()
    pt['model'] = pt.model.where(~pt.is_best_baseline, pt.model)
    pt['baseline'] = [f'{b}{" (best)" if i else ""}' for b, i in zip(pt.baseline, pt.is_best_baseline)]
    pt['diff [95% CI]'] = [f'{f3(r.diff)} {ci(r.lo, r.hi)}' for r in pt.itertuples()]
    pt['verdict'] = [verdict_text(v, b) for v, b in zip(pt.verdict, pt.is_best_baseline)]
    pt['model TM'] = pt.model_tm_bacc.map(f3); pt['baseline TM'] = pt.baseline_tm_bacc.map(f3)
    L += md_table(pt, ['model', 'baseline', 'model TM', 'baseline TM', 'diff [95% CI]', 'verdict']) + ['']
    L += ['### Inner-fold selection, per outer person', '']
    c2 = ch[ch.pipeline == 'HYBRID_FINAL'].copy()
    c2['inner CV TM bacc'] = c2.inner_cv.map(f3)
    L += md_table(c2, ['outer', 'source', 'cal', 'k', 'inner CV TM bacc'], ['held-out person', 'source', 'per-user calibration', 'k windows', 'inner CV TM bacc']) + ['']
    L += [f'Sources chosen: {", ".join(f"{k} x{v}" for k, v in c2.source.value_counts().items())}. Calibration on: {int(c2.cal.sum())} of {len(c2)} folds. k windows: {", ".join(f"{k} x{v}" for k, v in c2.k.value_counts().sort_index().items())}.', '']
    L += ['### Train-vs-test gap (P1)', '',
          'Like with like (Amendment 1, item 5). *Per person*: mean over the 5 training people of their own time-matched balanced accuracy (fold pipeline, outer CNN fit, in-sample logits) minus the held-out person\'s, mean +- SD over the 6 folds (bootstrap CI over folds). *Pooled*: training people pooled vs the pooled held-out windows. *Selection optimism*: inner-CV score used by the selection minus the pooled held-out score. Positive = optimistic.', '']
    gp2 = gp.copy()
    gp2['train, per person'] = gp2.train_person_mean.map(f3)
    gp2['held-out, per person'] = gp2.heldout_person_mean.map(f3)
    gp2['gap, per person'] = [f'{f3(r.gap_person)} +- {f3(r.gap_person_sd)} [{f3(r.gap_person_lo)}, {f3(r.gap_person_hi)}]' for r in gp2.itertuples()]
    gp2['train, pooled'] = gp2.train_pooled.map(f3); gp2['held-out, pooled'] = gp2.heldout_pooled.map(f3); gp2['gap, pooled'] = gp2.gap_pooled.map(f3)
    gp2['inner CV (mean +- SD)'] = [f'{f3(r.inner_cv_mean)} +- {f3(r.inner_cv_sd)}' for r in gp2.itertuples()]
    gp2['selection optimism'] = gp2.gap_inner_pooled.map(f3)
    L += md_table(gp2, ['model', 'train, per person', 'held-out, per person', 'gap, per person', 'train, pooled', 'held-out, pooled', 'gap, pooled', 'inner CV (mean +- SD)', 'selection optimism']) + ['']
    L += ['### Single seeds, seed ensemble and fold ensemble (exploratory, default levers)', '']
    s2 = sd.copy()
    s2['seed'] = s2.seed.map(lambda v: 'ensemble' if v == -1 else int(v))
    s2['TM bacc'] = s2.tm_bacc.map(f3); s2['per-person mean +- SD'] = [f'{f3(r.fold_mean)} +- {f3(r.fold_sd)}' for r in s2.itertuples()]
    L += md_table(s2, ['model', 'seed', 'TM bacc', 'per-person mean +- SD']) + ['']
    L += ['### Lever effects at the outer level (exploratory grid, point estimates)', '',
          'Every candidate x per-user calibration x k. `inner CV` is the mean over folds of the selection score. Descriptive only: comparing many cells on 6 people carries selection optimism.', '']
    g2 = gr.copy()
    g2['TM bacc'] = g2.tm_bacc.map(f3); g2['inner CV'] = g2.inner_cv_mean.map(f3)
    g2['per-person mean +- SD'] = [f'{f3(r.fold_mean)} +- {f3(r.fold_sd)}' for r in g2.itertuples()]
    g2['cal'] = g2.cal.map(lambda v: 'on' if v else 'off')
    L += md_table(g2, ['cand', 'cal', 'k', 'TM bacc', 'per-person mean +- SD', 'inner CV']) + ['']
    L += ['## P2: heart rate (within 5 bpm at full coverage, BUT PPG + MTHS, 100 people, true LOSO)', '',
          'Final HR rule chosen on Hoffman inner out-of-fold windows only (the P2 people never choose). `spec` is the v2 spectral estimator, `cnn` the seed median of the CNN.', '']
    r2 = p2r.copy()
    r2['within-5'] = r2.within5_full.map(f3); r2['MAE (bpm)'] = r2.mae.map(lambda v: f3(v, 2)); r2['doubling'] = r2.doubling.map(f3)
    r2['selected'] = r2.selected.map(lambda v: 'yes' if v else '')
    L += md_table(r2, ['rule', 'within-5', 'MAE (bpm)', 'doubling', 'selected']) + ['']
    rule = p2r.rule[p2r.selected].iloc[0]
    t2 = p2.copy()
    t2['within-5 [95% CI]'] = [f'{f3(r.within5_full)} {ci(r.lo, r.hi)}' for r in t2.itertuples()]
    t2['per-person mean +- SD'] = [f'{f3(r.fold_mean)} +- {f3(r.fold_sd)}' for r in t2.itertuples()]
    t2['MAE'] = t2.mae.map(lambda v: f3(v, 2)); t2['doubling / halving'] = [f'{f3(r.doubling)} / {f3(r.halving)}' for r in t2.itertuples()]
    t2['model'] = [m + (' (best baseline)' if b else '') for m, b in zip(t2.model, t2.is_best_baseline)]
    L += md_table(t2, ['scope', 'model', 'within-5 [95% CI]', 'per-person mean +- SD', 'MAE', 'doubling / halving', 'n_people'], ['scope', 'model', 'within-5 [95% CI]', 'per-person mean +- SD', 'MAE (bpm)', 'doubling / halving', 'people']) + ['']
    pb = p2p[(p2p.model == f'HYBRID_{rule}') & p2p.is_best_baseline]
    L += ['### Paired differences (P2)', ''] + md_table(
        p2p[p2p.model == f'HYBRID_{rule}'].assign(**{'diff [95% CI]': lambda d: [f'{f3(r.diff)} {ci(r.lo, r.hi)}' for r in d.itertuples()]}), ['scope', 'baseline', 'model_within5', 'baseline_within5', 'diff [95% CI]', 'verdict'],
        ['scope', 'baseline', 'hybrid', 'baseline value', 'diff [95% CI]', 'verdict']) + ['']
    L += ['## Reading the results', '']
    ab = pb[pb.scope == 'all'].iloc[0]
    L += [f'* P1: the hybrid is a **{bp.verdict}** with the best baseline (`{bp.baseline}`) by the pre-registered rule: {f3(fin.tm_bacc)} vs {f3(bp.baseline_tm_bacc)}, difference {f3(bp["diff"])} {ci(bp.lo, bp.hi)}.',
          f'* P1 control: re-fitting the chain\'s threshold on inner folds (`CHAIN_SAME_LEVERS`) reaches {f3(cl.tm_bacc)}; the hybrid minus that control is {f3(pc["diff"])} {ci(pc.lo, pc.hi)} ({pc.verdict}). The CNN is kept in the inference path by design (Amendment 2), not because it wins.',
          f'* P2: rule `{rule}` reaches {f3(ab.model_within5)} against {f3(ab.baseline_within5)} for the best baseline `{ab.baseline}`: difference {f3(ab["diff"])} {ci(ab.lo, ab.hi)} ({ab.verdict}). The Hoffman people alone would choose `spec` (see the rule table), the CNN-containing rule is the declared headline (Amendment 3, forking path disclosed).',
          '* No hybrid result here is validated on a phone that is not in the Hoffman, MTHS or BUT PPG data; every interval is wide because P1 has six people.', '']
    L += STATIC
    open(os.path.join(R, f'HYBRID{t}.md'), 'w').write('\n'.join(L) + '\n')
    print('wrote', os.path.join(R, f'HYBRID{t}.md'))


if __name__ == '__main__':
    main()
