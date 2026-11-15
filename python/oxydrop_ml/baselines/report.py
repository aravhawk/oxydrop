"""Build results/ml/BASELINES.md from the CSVs written by run_baselines (every number in the document is read from results/ml/baselines_*.csv)."""
import json

import numpy as np
import pandas as pd

from . import common as C

RES = C.RES
P1_ORDER = ['majority', 'clock', 'ror_textbook_RB', 'ror_textbook_RG', 'ror_calibrated_RB', 'ror_calibrated_RG', 'chain_v1', 'chain_v1_nestedC', 'chain_v2', 'chain_v2_nestedC',
            'xgb_fusion19_v1', 'xgb_fusion19_v2', 'xgb_hand26_cal', 'lgbm_fusion19_v1', 'lgbm_fusion19_v2', 'lgbm_hand26_cal']
LABEL = {
    'majority': 'majority class (always hypoxic)', 'clock': 'time-only clock', 'ror_textbook_RB': 'ratio-of-ratios 110-25R (red/blue)',
    'ror_textbook_RG': 'ratio-of-ratios 110-25R (red/green)', 'ror_calibrated_RB': 'ratio-of-ratios, personally calibrated (red/blue)',
    'ror_calibrated_RG': 'ratio-of-ratios, personally calibrated (red/green)', 'chain_v1': '**v1 logistic chain** (shipped C)', 'chain_v1_nestedC': 'v1 chain, nested-CV C',
    'chain_v2': '**v2 logistic chain** (shipped C, forced answer)', 'chain_v2_nestedC': 'v2 chain, nested-CV C (forced answer)',
    'xgb_fusion19_v1': 'XGBoost, 19 FUSION features (v1)', 'xgb_fusion19_v2': 'XGBoost, 19 FUSION features (v2)', 'xgb_hand26_cal': 'XGBoost, 26-feature recipe',
    'lgbm_fusion19_v1': 'LightGBM, 19 FUSION features (v1)', 'lgbm_fusion19_v2': 'LightGBM, 19 FUSION features (v2)', 'lgbm_hand26_cal': 'LightGBM, 26-feature recipe',
    'fft_plain': 'plain FFT peak', 'v1_peak': 'v1 peak-interval counter', 'v2_spectral_acf': 'v2 spectral + autocorrelation',
    'train_mean': 'train-mean predictor', 'ridge_ratio': 'ridge on six ratio features', 'lgbm_hand26': 'LightGBM, 26+26 handcrafted', 'xgb_hand26': 'XGBoost, 26+26 handcrafted',
}


def f3(v, d=3):
    return '' if v is None or not np.isfinite(v) else f'{v:.{d}f}'


def ci(r, d=3):
    """pooled [lo, hi] of a summary row (person bootstrap)."""
    if r is None or not np.isfinite(r.pooled):
        return ''
    return f'{r.pooled:.{d}f} [{r.pooled_lo:.{d}f}, {r.pooled_hi:.{d}f}]'


def msd(r, d=3):
    if r is None or not np.isfinite(r.fold_mean):
        return ''
    return f'{r.fold_mean:.{d}f} ± {r.fold_sd:.{d}f}'


def gap(r, d=3):
    if r is None or not np.isfinite(r.gap):
        return ''
    return f'{r.gap:+.{d}f} ± {r.gap_sd:.{d}f}'


def get(S, model, variant, metric, dataset=None):
    q = S[(S.model == model) & (S.variant == variant) & (S.metric == metric)]
    if dataset is not None:
        q = q[q.dataset == dataset]
    return q.iloc[0] if len(q) else None


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def p1_section():
    S = pd.read_csv(RES / 'baselines_P1.csv'); F = pd.read_csv(RES / 'baselines_P1_perperson.csv')
    info = json.load(open(RES / 'baselines_P1_info.json'))
    models = [m for m in P1_ORDER if m in set(S.model)]
    L = ['## P1: low-oxygen detection (reference SpO2 < 90), UW Hoffman, leave-one-person-out', '',
         f'{info["n_windows"]:,} windows (t >= 60 s) from {info["n_people"]} people; **{info["majority_share"] * 100:.2f}% of the evaluated windows are hypoxic, so "always alert" is the '
         f'majority-class rule (accuracy {info["majority_share"] * 100:.2f}%, balanced accuracy 0.500).** Primary metric: time-matched (TM) balanced accuracy at full coverage '
         f'(a clock-only model scores 0.5 by construction). Intervals in square brackets are 95% person-bootstrap intervals (6 people resampled, B=1000) of the pooled value; '
         f'"per person" is mean ± SD over the 6 held-out people; "gap" is train minus held-out for the same metric per fold (positive = over-fit), mean ± SD over folds. '
         f'Chain features come from the {info["feature_source"]}.', '',
         '### Full coverage (every window answered)', '']
    rows = []
    for m in models:
        g = lambda k, v='full': get(S, m, v, k)
        rows.append([LABEL.get(m, m), ci(g('tm_bacc')), ci(g('bacc')), ci(g('acc')), ci(g('auroc')), f'{f3(g("sens").pooled)} / {f3(g("spec").pooled)}', msd(g('bacc')),
                     gap(g('bacc')), gap(g('tm_bacc'))])
    L.append(table(['model', 'TM balanced acc [95% CI]', 'balanced acc [CI]', 'accuracy [CI]', 'AUROC [CI]', 'sens / spec', 'per-person bal. acc', 'gap bal. acc', 'gap TM bal. acc (per person)'], rows))
    L += ['', 'TM balanced accuracy pooled over all people balances each 100 s elapsed-time bin across the six people; the per-person TM column in `baselines_P1_perperson.csv` balances within one person and is lower and much noisier (few usable bins).', '']
    # paired differences
    d = S[S.metric.str.startswith('tm_bacc_diff_vs_')]
    if len(d):
        L += ['### Paired difference of TM balanced accuracy to the shipped v1 chain (same windows, same resampled people)', '']
        rows = [[LABEL.get(r.model, r.model), f'{r.pooled:+.3f} [{r.pooled_lo:+.3f}, {r.pooled_hi:+.3f}]', 'yes' if (r.pooled_lo > 0 or r.pooled_hi < 0) else 'no'] for r in d.itertuples()]
        L += [table(['model', 'TM bal. acc minus v1 chain [95% CI]', 'interval excludes zero'], rows), '',
              'A learned model must beat the BEST honest baseline in this sense to claim a win (pre-registration); with 6 people no baseline separates from the v1 chain except the clearly weaker ones.', '']
    # abstention variants
    L += ['### With abstention (answered windows only; unanswered windows are not scored as wrong here, but the last two columns show what that costs)', '',
          '`band_shipped` = shipped half-width 0.10 around p = 0.5 (chains only, not tuned). `band_nested` = half-width chosen on the inner out-of-fold scores of the training people '
          '(smallest c whose answered windows reach accuracy and balanced accuracy >= 0.90, else the best balanced accuracy among c with coverage >= 20%). `nolock_*`: v2 also abstains '
          'when the pulse is not locked. "alert share" = share of ALL hypoxic windows that receive a low-oxygen alert (unanswered hypoxic windows count as missed). '
          '"accuracy, unanswered wrong" = correct answers divided by all windows.', '']
    rows = []
    for m in models:
        for v in ('band_shipped', 'band_nested', 'nolock_abstain', 'nolock_band_shipped', 'nolock_band_nested'):
            if get(S, m, v, 'coverage') is None:
                continue
            g = lambda k: get(S, m, v, k)
            rows.append([LABEL.get(m, m), v, ci(g('coverage')), ci(g('acc')), ci(g('bacc')), ci(g('alert_share_all')), f3(g('acc_all').pooled), f3(g('worst_person_acc').pooled)])
    L.append(table(['model', 'variant', 'coverage [CI]', 'accuracy answered [CI]', 'balanced acc answered [CI]', 'alert share of all hypoxic windows [CI]', 'accuracy, unanswered wrong', 'worst person accuracy'], rows))
    # spec90
    L += ['', '### Sensitivity at fixed specificity (threshold from the training people only: 90% specificity on inner out-of-fold scores)', '']
    rows = [[LABEL.get(m, m), ci(get(S, m, 'spec90', 'sens')), ci(get(S, m, 'spec90', 'spec')), ci(get(S, m, 'spec90', 'tm_bacc'))] for m in models]
    L += [table(['model', 'sensitivity [CI]', 'specificity actually reached [CI]', 'TM bal. acc [CI]'], rows), '']
    # per person
    Fm = F[F.variant == 'full']
    people = sorted(Fm.subject.unique())
    L += ['### Per-person accuracy at full coverage (held-out person)', '']
    show = ['majority', 'clock', 'chain_v1', 'chain_v2', 'xgb_fusion19_v1', 'xgb_hand26_cal', 'lgbm_hand26_cal']
    rows = []
    for p in people:
        r = [p.replace('hof_', '')]
        for m in show:
            q = Fm[(Fm.model == m) & (Fm.subject == p)]
            r.append(f3(q.acc.iloc[0]) if len(q) else '')
        nl = F[(F.model == 'chain_v2') & (F.variant == 'nolock_abstain') & (F.subject == p)]
        r.append(f3(nl.coverage.iloc[0]) if len(nl) else '')
        rows.append(r)
    L += [table(['person (sid)'] + [LABEL.get(m, m) for m in show] + ['v2 pulse-lock coverage'], rows), '',
          'The v2 pulse-lock coverage column is the share of that person\'s windows on which the v2 pulse detector locks. A person whose resting pulse is below the 51 bpm band edge of `hr.py` '
          '(person 4, about 43 bpm) never locks: v2 abstains on all of that person\'s windows under `nolock_*` and answers from forward-filled features under the forced `full` variant.', '']
    sk = RES / 'baselines_P1_skin.csv'
    if sk.exists():
        K = pd.read_csv(sk)
        K = K[(K.variant == 'full') & K.model.isin(['chain_v1', 'chain_v2', 'xgb_fusion19_v1', 'xgb_hand26_cal', 'lgbm_hand26_cal'])]
        rows = [[LABEL.get(r.model, r.model), f3(r.dark_acc), f3(r.light_mean_acc), f'{r.gap_acc:+.3f}', f3(r.dark_bacc), f3(r.light_mean_bacc), f'{r.gap_bacc:+.3f}',
                 f3(r.dark_sens), f3(r.light_mean_sens), f3(r.dark_spec), f3(r.light_mean_spec)] for r in K.itertuples()]
        L += ['### Skin-tone gap (Hoffman documents one dark-skinned participant, hex #32210F, against five lighter ones: n = 1 versus 5, so this is an anecdote, not an estimate)', '',
              table(['model', 'dark acc', 'light mean acc', 'gap (light - dark)', 'dark bal. acc', 'light bal. acc', 'gap', 'dark sens', 'light sens', 'dark spec', 'light spec'], rows), '',
              'Per-skin-tone gaps with enough people (Monk and ITA groups) can only be measured on OpenOximetry (contact sensor); the BUT PPG and MTHS releases publish no skin tone.', '']
    # train vs test gaps
    L += ['### Train-vs-held-out gap, full coverage (per fold, mean ± SD)', '']
    rows = []
    for m in models:
        a = get(S, m, 'full', 'acc'); b = get(S, m, 'full', 'auroc')
        rows.append([LABEL.get(m, m), f3(a.train_mean), f3(a.test_mean), gap(a), f3(b.train_mean), f3(b.test_mean), gap(b)])
    L += [table(['model', 'train acc', 'held-out acc', 'gap acc', 'train AUROC', 'held-out AUROC', 'gap AUROC'], rows), '']
    return L


def p2_section():
    S = pd.read_csv(RES / 'baselines_P2.csv'); info = json.load(open(RES / 'baselines_P2_info.json'))
    L = ['## P2: heart rate on BUT PPG + MTHS, leave-one-subject-out', '',
         f'{info["n_windows"]:,} windows with a reference HR from {info["n_people"]} people (BUT PPG clips: 38 people with SpO2, ECG HR; MTHS windows: 62 people, oximeter HR). '
         'Primary metric: share of windows within 5 bpm at full coverage (no answer counts as wrong) and the doubling rate. The estimators have no fitted parameters; the only selection is the '
         'v2 SQI/ACF gate, chosen on the training people of each fold. "per subject" is mean ± SD over held-out subjects (each subject counts once), "pooled" weights windows '
         '(BUT has more windows per person). v1 sees a 15 s trailing buffer (the shipped v1 uses 30 s); BUT clips are only 10 s, with a 5 s reflected warm-up.', '']
    models = [('fft_plain', 'forced'), ('v1_peak', 'forced'), ('v2_spectral_acf', 'forced'), ('v2_spectral_acf', 'gate_shipped'), ('v2_spectral_acf', 'gate_nested')]
    for scope, title in (('all', 'BUT + MTHS pooled (100 people)'), ('butppg', 'BUT PPG only (38 people)'), ('mths', 'MTHS only (62 people)')):
        rows = []
        for m, v in models:
            g = lambda k: get(S, m, v, k, scope)
            rows.append([LABEL.get(m, m) + ('' if v == 'forced' else f' ({v})'), ci(g('within5_full')), msd(g('within5_full')), ci(g('coverage')), ci(g('within5')), ci(g('mae'), 2), ci(g('doubling')), ci(g('halving')),
                         gap(g('within5_full')), gap(g('mae'), 2)])
        L += [f'### {title}', '', table(['estimator', 'within 5 bpm, full coverage [CI]', 'per subject', 'coverage [CI]', 'within 5 bpm, answered [CI]', 'MAE bpm, answered [CI]', 'doubling rate [CI]', 'halving rate [CI]',
                                         'gap within-5 (full)', 'gap MAE'], rows), '']
    d = S[S.metric.str.startswith('within5_full_diff_vs')]
    rows = [[r.dataset, LABEL.get(r.model, r.model) + f' ({r.variant})', f'{r.pooled:+.3f} [{r.pooled_lo:+.3f}, {r.pooled_hi:+.3f}]', 'yes' if (r.pooled_lo > 0 or r.pooled_hi < 0) else 'no'] for r in d.itertuples()]
    L += ['### Paired difference of within-5 at full coverage to the shipped v2 estimator (forced answer)', '', table(['scope', 'estimator', 'difference [95% CI]', 'excludes zero'], rows), '',
          'Gate selection per fold (SQI, ACF): ' + ', '.join(f'{k}: {v} folds' for k, v in info['gate_counts'].items()) + '.', '']
    return L


def p3_section():
    S = pd.read_csv(RES / 'baselines_P3.csv'); info = json.load(open(RES / 'baselines_P3_info.json'))
    order = ['train_mean', 'ror_textbook_RB', 'ror_textbook_RG', 'ridge_ratio', 'lgbm_hand26', 'xgb_hand26']
    L = ['## P3: absolute SpO2 on MTHS, leave-one-subject-out', '',
         f'{info["n_windows"]:,} windows (t >= 30 s) from {info["n_people"]} people (one of the 62 MTHS recordings lasts 26 s, so it has no window at t >= 30 s). **MTHS is normoxic**: reference '
         f'SpO2 in the scored windows has median {info["ref_median"]:.0f}, 5th percentile {info["ref_p5"]:.1f}, minimum {info["ref_min"]:.1f}; {info["share_below_95"] * 100:.1f}% of windows are below 95 and '
         f'{info["windows_below_90"]} window(s) below 90 (from {info["people_with_window_below_90"]} person). Predicting the training mean is therefore nearly optimal, and the AUROC columns '
         '(ranking by deviation from the fold training mean) rest on very few positives; the < 90 AUROC is not meaningful with that few windows. Nothing here measures hypoxemia detection. '
         'MAE in percentage points; intervals are 95% person-bootstrap intervals (B=1000); gap = held-out MAE minus training MAE (positive = over-fit).', '']
    rows = []
    for m in order:
        g = lambda k: get(S, m, 'full', k)
        imp = get(S, m, 'full', 'mae_improvement_vs_train_mean')
        rows.append([LABEL.get(m, m), ci(g('mae'), 2), msd(g('mae'), 2), ci(g('rmse'), 2), ci(g('within2')), ci(g('within5')), ci(g('auroc_lt95')), gap(g('mae'), 2),
                     '' if imp is None else f'{imp.pooled:+.3f} [{imp.pooled_lo:+.3f}, {imp.pooled_hi:+.3f}]'])
    L += [table(['model', 'MAE [95% CI]', 'MAE per person', 'RMSE [CI]', 'within 2 [CI]', 'within 5 [CI]', 'AUROC SpO2 < 95 [CI]', 'gap MAE', 'MAE improvement vs train-mean [CI]'], rows), '',
          'Positive improvement means lower error than the train-mean predictor. Train-mean predictions are constant within a fold, so they carry no ranking information (AUROC blank).', '']
    return L


def write(path=None):
    path = path or (RES / 'BASELINES.md')
    L = ['# Honest baselines under the common leave-one-subject-out harness', '',
         'Generated by `python -m oxydrop_ml.baselines.run_baselines --all` (from `python/`, ML venv); every number below is read from `results/ml/baselines_P{1,2,3}.csv`. '
         'Regenerate with `bash scripts/reproduce_ml.sh baselines`. Per-window scores: `results/ml/scores/baseline_P1_hoffman.csv`, `baseline_P2_butppg.csv` '
         '(the MTHS per-window scores are not committed: MTHS is CC BY-NC-ND, they stay in the cache).', '',
         '## Protocol (pre-registered, python/oxydrop_ml/PREREGISTRATION.md)', '',
         '- Leave-one-subject-out: the subject is the group (both Hoffman hands together). No window-level split exists anywhere.',
         '- Everything selected is selected by nested CV on the outer training people only: hyperparameters (XGBoost/LightGBM depth, leaves, trees; ridge alpha; chain C pairs), decision thresholds, '
         'abstention half-widths, the 90%-specificity threshold and the v2 SQI/ACF gate. `tests/test_baselines.py` checks that scrambling the held-out person changes none of them.',
         '- Calibration: first 60 s of each Hoffman recording (30 s MTHS, first 6 BUT clips), label-free, the same personal baseline the CNN uses.',
         '- Per row: pooled value with a 95% person-bootstrap interval, mean ± SD over held-out people, and the train-vs-held-out gap; long-format rows are in the CSVs.',
         '- CSV schema (long format, one row per task / dataset / model / variant / metric): `pooled`, `pooled_lo`, `pooled_hi` (all out-of-fold windows pooled, 95% person bootstrap), '
         '`fold_mean`, `fold_sd`, `fold_lo`, `fold_hi`, `fold_min`, `fold_max`, `n_folds` (per held-out person), `train_mean`, `test_mean`, `gap`, `gap_sd`, `gap_lo`, `gap_hi` (train minus held-out, '
         'positive = over-fit; undefined for counts and signed bias). Rows named `*_diff_vs_*` / `mae_improvement_vs_train_mean` are paired person-bootstrap differences. '
         '`baselines_P1_perperson.csv` (Hoffman, MIT) and `baselines_P2_perperson.csv` (BUT PPG, CC BY) hold every fold (selected hyperparameters, thresholds, abstain half-widths). MTHS is CC BY-NC-ND (aggregates only): its people appear only as the aggregate spread (people, mean, SD, 10/25/50/75/90th percentiles) in `baselines_P2_spread.csv` and `baselines_P3_spread.csv`.',
         '- Variants: `full` forced answer; `band_shipped` / `band_nested` abstention bands; `nolock_*` v2 abstains when the pulse is not locked; `spec90` threshold at 90% specificity on inner scores.',
         '- P2 estimators have no fitted parameters, so their train-vs-held-out gap is sampling noise around zero, not over-fitting.',
         '- The v1 chain refit reproduces `results/eval_v1_vs_v2.csv` (accuracy 0.8754, balanced accuracy 0.8733, AUROC 0.9475, TM accuracy 0.7962).', '']
    L += p1_section() + p2_section() + p3_section()
    L += ['## Caveats', '',
          '- Six Hoffman people (one dark-skinned) are the only open phone-camera recordings with real SpO2 below 90; all P1 intervals are wide and nothing here supports a claim about new people at scale.',
          '- Hoffman accuracy is largely the shared protocol timeline: the clock-only model reaches high plain accuracy but 0.5 on the time-matched metric. TM numbers are the honest ones.',
          '- Hyperparameter grids are small and fixed in code (`xgb_hand.py`, `chain_v1.py`, `p3.py`); selection optimism between the many variants is not removed by the intervals.',
          '- Tree models on six people select hyperparameters from five inner folds of one person each; their inner scores are noisy.',
          '- No frame timestamps exist in any dataset, so these baselines never see irregular sampling.',
          '- MTHS is non-commercial research data (CC BY-NC-ND): only aggregates are committed.', '']
    path.write_text('\n'.join(L))
    return path
