# Pre-registration of the evaluation protocol (written before any CNN result existed)

Committed on branch `ml-model` before any learned-model training or scoring was run. The git commit hash and timestamp of this file are the record. Any later change to this file must be a new commit with a dated "Amendment" section at the bottom. Rows in results tables that were not planned here must be labelled **exploratory**.

## Who it is for and what goes in and out

- User: a person with COVID, COPD, asthma, or at altitude who owns a phone but no pulse oximeter; also low-resource clinics.
- Input: 30 to 60 s of fingertip video from the phone camera (rear camera plus flash), reduced on the phone to per-frame mean R, G, B and frame timestamps. No pixels leave the device.
- Output, in one forward pass of an on-device 1D CNN: heart rate (bpm), an SpO2 estimate with its uncertainty, a signal-quality score, and a calibrated 3-state result (Normal vs your baseline / Possible low oxygen, re-check with a pulse oximeter / Unsure, re-measure).

## Primary metrics (fixed now)

| ID | Task | Population and split | Metric | Honest baselines (same folds, same windows, same calibration, same abstention rule) |
|---|---|---|---|---|
| **P1 (primary)** | Low-oxygen detection, label = window-mean reference SpO2 < 90 | UW Hoffman 2022, 6 people, 12 hands, **leave-one-person-out** (both hands held out together), windows with t >= 60 s, 10 s windows, 2 s hop (5,644 windows) | **Time-matched balanced accuracy at full coverage** (every 100 s of elapsed time has equal numbers of hypoxic and normal windows) | (a) v1 logistic chain refit LOSO, (b) v2 logistic chain refit LOSO, (c) XGBoost on handcrafted features, (d) time-only clock, (e) majority class |
| **P2** | Heart rate | BUT PPG v2.0 (38 people) plus MTHS (62 people), **leave-one-subject-out** | **% of windows within 5 bpm of the reference at full coverage**, and doubling rate | v2 spectral + autocorrelation estimator (`python/oxydrop/hr.py`), v1 peak-interval counter |
| **P3** | Absolute SpO2 | MTHS, 62 people, **leave-one-subject-out** (normoxic; reported as such) | **SpO2 MAE** | train-mean predictor, textbook ratio-of-ratios 110 - 25R, ridge on ratio features |

A "win" is claimed on a pre-registered metric only if the paired person-level bootstrap 95 % interval (B >= 1000, resampling people) of the difference to the best honest baseline excludes zero. If it does not, the result is reported as a tie or a loss, in the same table. Because three metrics are pre-registered, no other metric may be described as a win unless it is labelled exploratory.

## Secondary metrics (always reported, never used for selection)

HR MAE, doubling rate, halving rate, % within 5 bpm; SpO2 MAE and low-SpO2 AUROC; coverage of the abstain band and accuracy on answered windows; accuracy versus coverage curve; share of ALL hypoxic windows that receive a low-oxygen alert (not only answered ones); per-person results and worst person; per-skin-tone gaps on OpenOximetry (Monk and ITA); calibration (ECE, reliability); latency and model size on a phone-class CPU.

## Cross-validation rules

1. **All reported numbers are leave-one-subject-out** (LOSO) with the subject as the group (both hands, all recordings of one person together). Anything else (grouped k-fold used to save compute, a single fixed split) must be labelled in its row as "not LOSO" with the reason. No window-level split anywhere.
2. **No tuning on outer test folds.** Hyperparameters, thresholds, abstention cutoffs, ensemble weights, stacking meta-learners, conformal quantiles, label-lag offsets and the choice of which accuracy lever to keep are selected by **nested CV**: an inner subject-grouped CV on the outer training subjects only.
3. Lever keep/drop rule: a lever is kept if it improves the inner-CV score of the lever's own pre-declared selection metric (P1/P2/P3 as appropriate) on the training subjects. The outer-LOSO ablation table is then reported in full (kept and dropped levers), described as descriptive, and carries the caveat that comparing several variants on 6 people has selection optimism.
4. Scaling, normalisation, per-dataset standardisation, the SQI threshold and the personal baseline (first 60 s of each recording, label-free) are fit on training subjects or on the test person's own unlabelled first minute only.
5. Self-supervised pretraining may only use signals from subjects that are never an outer test subject for the task being evaluated (OpenOximetry contact PPG and other datasets disjoint from Hoffman, BUT and MTHS). Pretraining on a test subject's unlabelled signal is not allowed.
6. Report for every model: mean, SD across folds, 95 % person-bootstrap CI, per-fold values, and **train-fold score versus held-out score** (the train-vs-test gap).
7. Fixed seeds; at least 3 seeds for the final CNN rows; report seed SD.
8. The reference labels: for Hoffman use the same labelling as `python/oxydrop/hoffman.py`. For OpenOximetry use arterial blood gas SaO2 as the target (not the contact oximeter), with the lag between reference and tissue estimated on training subjects only.

## Declared limits (stated now so they are not discovered later)

- Only 6 people with real phone-camera SpO2 below 90 % exist in open data. P1 has wide intervals by construction.
- MTHS is almost entirely normoxic (about 1 % of seconds below 90). P3 measures normoxic accuracy only.
- OpenOximetry is contact red/infrared PPG, not a phone camera. It can support pretraining, auxiliary heads, HR, quality and fairness analysis. It cannot by itself validate phone SpO2.
- No real-phone live validation exists. This is a prototype, not a medical device.

## Amendments

### Amendment 1 (Nov 15 2026, written after the core CNN results existed; disclosed deviations of FEAT-006)

None of these changes a pre-registered metric, split or baseline. They are listed because the original text was not followed to the letter.

1. **Recipe selection (rule 2).** Four candidate training recipes were scored by inner folds before any outer comparison (`results/ml/nested_selection.csv`). The choice for each outer Hoffman person uses only its own inner jobs and is leakage free. The single recipe applied to every fold and ablation
   (`C_lowhi`) was however taken as the best mean over ALL outer folds' inner scores, which scores each of the six Hoffman people as an inner person, and the MTHS (8) and BUT (6) validation people of every selection job are people that are test subjects of P2/P3. This is a mild form of selection on
   evaluation people. `results/ml/nested_selection_sensitivity.csv` shows that dropping all selection jobs that touch any one Hoffman person never changes the winner (margins between 0.0006 and 0.013 of the composite score); a strictly per-outer choice would change one fold (hof_100002: `D_steps800`).
   The effect, if any, favours the CNN; the CNN's reported results are ties, not wins, so no conclusion depends on it.
2. **Seeds (rule 7).** At least 3 seeds exist for P1 and for the grouped P2/P3 protocol of the final model. For the true-LOSO P2/P3 run of the final model the number of seeds is the one stated in `results/ml/CORE_RESULTS.md` (seed 0 only until further seeds finished). Ablation rows have 1 seed for P2/P3.
3. **Cheaper variants (rule 1).** Ablation rows use 10 subject-grouped folds over the 100 MTHS+BUT people for P2/P3 ("not LOSO", labelled); the final model also has the true LOSO run.
4. **Input scale and early stopping (rule 4).** The scale-only standardisation is per domain (phone, contact), not per dataset, so it folds exactly into the exported weights. Early stopping uses an inner validation set (1 Hoffman, 6 MTHS, 4 BUT people, identical in every fold) removed from the training people of that fold.
5. **Gap definition (rule 6).** The train-vs-test gap compares the fold models' scores on their own training rows with the pooled held-out score (not a mean of per-person scores). The P3 training MAE of these runs did not apply the t >= 30 s filter, so no P3 gap is reported.
6. **Decision threshold.** The low-oxygen decision threshold is the fixed 0.5 of the class-balanced loss, not selected.
