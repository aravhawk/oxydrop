# SpO2 optics path: ratio-of-ratios on real phone video (negative result, not shipped)

**Outcome:** a separate optical SpO₂ estimator built on the red/green/blue ratio-of-ratios is **not shipped** in OxyDrop. It failed the decision rule below on Hoffman, and on MTHS it does not predict SpO2 better than predicting the training-set mean. **This does not mean ratio features are absent from the model:** the shipped fusion regression takes 19 features, including four ratio-of-ratios features with nonzero coefficients (`ror_RG`, `ror_RB`, `ror_GB`, `ror30_RG`), slopes and DC absorption, so it is not a heart-rate-only classifier (an earlier version of this page said the ratio was not a model input; that was wrong). No JavaScript optics function was added beyond those features because the debug view does not need one.

This page covers three things: the MTHS test (62 people), the Hoffman cross-check (6 people, already computed), and OpenOximetry (access now active; see the last section).

## Decision rule (fixed before the MTHS run)

Ship the optical ratio as a model input **only if**, in nested person-level leave-one-person-out (LOPO) on Hoffman, a model that uses it beats the shipped TINY v2 on accuracy **and** on time-matched accuracy, with the person-bootstrap 95% confidence interval of each difference excluding zero (lower bound above 0).

Result (`scripts/optics_decision.py`, `results/optics_experiment/decision_rule.csv`; this is arithmetic on the CSV, not an independent raw-data rerun, because the experiment scripts depend on external `pivot/kit.py` and host paths): 16 optics variants, all compared with TINY v2 on the same 4,594 windows (evaluated, optics gate passed, v2 pulse locked). **None passes.** Every point estimate for accuracy is below TINY v2 (0.873), and no accuracy lower bound is above zero. The best optics-bearing variants:

| Model (nested LOPO, Hoffman) | Accuracy | Time-matched acc | Difference in accuracy vs TINY v2 [95% CI] | Difference in TM acc [95% CI] |
|---|---|---|---|---|
| TINY v2 (shipped), same windows | 0.873 | 0.792 | | |
| GBM, optics + dHR | 0.836 | 0.810 | -0.037 [-0.104, +0.011] | +0.017 [-0.042, +0.077] |
| TINY + LR(optics + dHR), average | 0.842 | 0.815 | -0.031 [-0.077, +0.023] | +0.022 [-0.037, +0.060] |
| LR, DC + dHR | 0.842 | 0.782 | -0.031 [-0.068, +0.002] | -0.011 [-0.061, +0.062] |
| LR, optics + dHR | 0.805 | 0.789 | -0.068 [-0.151, +0.019] | -0.004 [-0.050, +0.038] |
| LR, ratio-of-ratios only | 0.673 | 0.663 | -0.199 [-0.291, -0.110] | -0.129 [-0.196, -0.036] |

Baselines on the 5,489 optics-gated Hoffman windows (`results/optics_experiment/main_table.csv`): the time-only clock reaches 0.838 accuracy but 0.519 time-matched; the shipped TINY chain 0.876 accuracy, 0.801 time-matched, worst person 0.778. The ratio-of-ratios alone (0.671 accuracy [0.557, 0.774]) is worse than the clock. Adding the optical features to the heart-rate change does not beat TINY. The features that help most are the DC (brightness) ones, which carry the same slow drift as the protocol clock, so they are not clean evidence of oxygen sensing.

## MTHS (62 people, phone fingertip, pulse-oximeter reference)

Script: `scripts/mths_spo2.py` (`results/mths_spo2/summary.csv`, `per_person.csv`, `label_range.csv`, `README.md`, `run_log.txt`).

### Label range first

| Quantity | Value |
|---|---|
| People / reference seconds | 62 / 4,646 |
| Reference SpO2 min / median / max | 86 / 97 / 100 |
| Seconds below 95% | 11.1% |
| Seconds below 90% | 1.0% (47 seconds, from 4 people) |
| People whose mean is below 95 / below 90 | 6 / 0 |
| People with any second below 95 / below 90 | 15 / 4 |
| Spread of person means (SD) / typical within-person SD | 1.68 / 0.65 points |

This is a normoxic cohort. Real hypoxemia detection cannot be tested here. What can be tested: (A) absolute-SpO2 error and a "below 95" screen, and (B) change from the person's own start. Accuracy for "below 90" is not reported because 47 seconds from 4 people cannot support it.

### Method

* Windows: 10 s, 2 s hop, the same causal band-pass and robust-HR lock as the app (`scripts/optics_feats.py` reusing `python/oxydrop/hr.py`).
* Features per window: per-beat AC/DC ratio-of-ratios (log R/G, R/B, G/B), perfusion index per channel, DC absorbance per channel, amplitude CV, heart rate.
* Quality gate: pulse locked, at most 10% saturated frames (any channel), amplitude CV at most 0.6, at least 4 beats, HR 30 to 200 bpm.
* Per-person calibration (no labels): median of the person's own gated windows ending in the first 30 s, subtracted. Persons with fewer than 3 such windows are excluded from every model. Evaluation windows start at 30 s.
* Learned correction: ridge regression (alpha by inner GroupKFold on training people) and a small gradient-boosted tree model as a sensitivity check.
* Validation: leave-one-person-out, 95% intervals by person bootstrap (1,000 draws). No window-level split anywhere (the script uses only `LeaveOneGroupOut` and `GroupKFold` by person).
* Every model and baseline is scored on the identical windows and people.
* **AUROC is not quoted for the learned models.** `results/mths_spo2/summary.csv` still has an AUROC column, but it is unreliable: even after centring each model on its fold's training mean, "ridge HR only" scores 0.05 [0.00, 0.21] and most learned rows fall below 0.5. In leave-one-person-out the training mean moves opposite to the held-out person's label, so the ranking is anti-correlated for a mechanical reason that says nothing about oxygen sensing. The MAE comparison is not affected by this. The one AUROC kept in the text below (textbook formula, 0.79) rests on 3 people and is not evidence of tracking.

### Gate coverage: the red channel is saturated

On this phone the red channel is at or above 250 of 255 in more than 10% of frames in 65.6% of labelled windows (green: 0%). Ratio-of-ratios needs an unclipped red AC amplitude. With the saturation criterion on, 26.6% of labelled windows pass (77.5% pass if saturation is ignored), and only **22 of 62 people** have the 3 gated calibration windows. This is itself a finding: with automatic exposure on this phone, the red light often clips. The app locks exposure and shows a saturation warning for the same reason.

### Results, primary gate (22 people calibrated, 220 windows from 20 people)

Labels in this set: 59 windows below 95 (from only 3 people), 2 below 90, 13 windows with a drop of at least 2 points from the person's start (from 2 people). The drop and "below 95" detection numbers therefore rest on 2 to 3 people and are not evidence.

Task A, absolute SpO2 from uncalibrated features (MAE in SpO2 points, 95% person-bootstrap interval; AUROC columns removed, see the method note above):

| Model | MAE [95% CI] | Within 2 / 3 / 5 |
|---|---|---|
| Baseline: training-set mean | 2.21 [1.66, 2.68] | 0.52 / 0.85 / 0.97 |
| Baseline: textbook 110 - 25 R (R = red/blue ratio-of-ratios, uncorrected, clipped to 70-100) | 3.49 [2.51, 4.63] | 0.43 / 0.53 / 0.81 |
| Baseline: textbook, R = red/green | 6.19 [4.02, 9.00] | 0.36 / 0.42 / 0.53 |
| HR only | 2.21 | 0.52 / 0.85 / 0.97 |
| Time in recording only | 2.36 | 0.52 / 0.74 / 0.95 |
| DC only | 2.34 | 0.44 / 0.79 / 0.98 |
| Ratio-of-ratios only | 2.27 | 0.51 / 0.75 / 0.97 |
| Ridge, optics (RoR + PI + DC + CV) | 2.29 | 0.48 / 0.85 / 0.97 |
| Ridge, optics + HR | 2.22 | 0.52 / 0.85 / 0.97 |
| Ridge, optics + HR + calibrated deltas | 2.21 | 0.52 / 0.85 / 0.97 |
| GBM, optics + HR | 2.27 [1.44, 3.04] | 0.55 / 0.69 / 0.97 |

No learned model has a lower MAE than the training-mean baseline (no paired difference to the mean baseline is negative; several intervals are wide because the evaluation set has 20 people). Ridge shrinks to almost the mean (alpha 10^4 to 10^5 chosen in most folds), which is the model saying the features carry no usable signal. The textbook map is worse than predicting the mean by 1.3 points (difference 1.28 [0.40, 2.28]). Its AUROC of 0.79 comes from 3 people and has a wide interval, while its within-person correlation with SpO2 is negative (-0.40), so it is not tracking oxygen inside a person.

Task B, change from the person's own start, from calibrated deltas (MAE vs the reference change; baseline is "no change"):

| Model | MAE [95% CI] | Difference vs "no change" [95% CI] |
|---|---|---|
| Baseline: no change | 0.77 [0.38, 1.12] | |
| Time in recording only | 0.73 [0.41, 1.13] | -0.03 [-0.15, 0.11] |
| HR delta only | 0.87 [0.48, 1.26] | +0.10 [0.04, 0.15] |
| DC delta only | 0.87 [0.44, 1.25] | +0.10 [0.04, 0.19] |
| Ratio-of-ratios delta only | 0.89 [0.42, 1.31] | +0.13 [0.02, 0.26] |
| Ridge, optics + HR deltas | 0.87 [0.43, 1.27] | +0.10 [0.03, 0.18] |
| GBM, optics + HR deltas | 0.97 [0.60, 1.33] | +0.20 [0.15, 0.27] |

Calibrated optical features predict the change worse than saying "no change". Within-person Spearman correlations of the learned predictions with SpO2 are between -0.14 and +0.12 across models and the signs differ between models (`summary.csv`).

### Sensitivity: saturation criterion ignored (54 people calibrated, 821 windows from 53 people)

Same conclusion with more people, but the red channel is clipped in many of these windows, so the ratio is not physically meaningful. Task A: training-mean MAE 1.39 [1.08, 1.70]; ridge optics + HR 1.41 [1.12, 1.73]; ridge optics 1.45; GBM 1.47; textbook (red/blue) 3.39 [2.90, 3.87]. Task B: no change 1.07 [0.68, 1.53]; ridge optics + HR deltas 1.14 [0.78, 1.59]. None beats its baseline. Windows below 95 come from 7 people; drops of at least 2 points from 4 people.

### Heart rate on the same clips

Reference HR comes from the same oximeter. For all windows with a locked pulse (1,642 of 2,065): MAE 6.2 bpm, 71.3% within 5 bpm. For windows that also pass the optics gate (saturation included, n = 541): MAE 4.6 bpm, 78.7% within 5 bpm. This matches `docs/HR_VALIDATION.md` (MTHS accepted windows 6.2 bpm there, a different window selection).

### Skin tone

MTHS has no Fitzpatrick or skin-tone labels (`gender.xlsx` holds only id and gender), and Hoffman has one Black participant (person 2). **No skin-tone claim is made.** The shipped chain's per-person results on Hoffman are in `results/table_b_per_person.csv`.

### Caveats

* One phone model in MTHS, auto exposure, a clipped red channel, and mostly healthy adults at 94 to 100%. These results say nothing about phones with locked exposure in the app.
* Per-recording lengths are 26 to 134 s (median 60 s). The calibration is 30 s here versus 60 s in the app, and 40 of 62 people lack 3 gated calibration windows under the strict gate.
* Reference labels are 1 Hz integer readings from a consumer oximeter with its own error (about 2 points).
* Few people have positives, so every detection number is fragile. A clean negative on 62 people is more informative than any detection number here.

## Hoffman cross-check (6 people, already computed, not rerun)

From `results/optics_experiment/main_table.csv` (`scripts/optics_experiment.py`), nested LOPO, same optics gate and per-recording calibration (first 60 s):

* Ratio-of-ratios only: 0.671 accuracy, AUROC 0.725, worst person 0.433. Logistic regression on RoR + PI + DC + CV: 0.755; with dHR: 0.802 [0.705, 0.885]; tiny GBM with dHR: 0.819. Shipped TINY on the same windows: 0.876 [0.834, 0.907].
* Average of TINY and the optics model: 0.835, below TINY alone (difference in accuracy -0.041).
* Time-only clock: 0.838 accuracy but 0.519 time-matched; optics models reach 0.51 to 0.79 time-matched versus TINY 0.80.

## OpenOximetry: now used (aggregates only)

OpenOximetry (PhysioNet credentialed access, healthy volunteers, staircase hypoxia, arterial-blood-gas SaO2 reference) is a **contact-oximeter** dataset (red/infrared finger-clip PPG at 86 Hz and 2 Hz tables). It is **not phone-camera video**, so it cannot by itself validate the phone ratio-of-ratios. An earlier version of this page said access was never activated and that no OpenOximetry data was used. That is out of date: access is active and the data are used for (1) validation of the heart-rate-rise stage, (2) a large labelled time-matched baseline and a fairness check by skin tone, and (3) later model pretraining (see `python/oxydrop_ml/`). Data governance: only derived aggregates are committed, raw files and per-window arrays are never committed or redistributed (see `DATA_LICENSES.md`). The aggregate tables are on the `openox-results` branch (`results/openoximetry.md`) and are not merged into this page; read them there. The synthetic stand-in files created while waiting were never used as evidence.

What it does not settle: whether a phone camera's optical ratio tracks desaturation inside a person, and error by skin tone for phone video. Until that is measured on phone video the optical estimator is not validated and not shipped.

## Reproduce

```
PY=python3   # or the path of a Python environment with numpy, scipy, scikit-learn, pandas
$PY -s scripts/mths_spo2.py        # needs OXY_MTHS (MTHS Data dir); about 1 minute, 172 MB peak memory; caches per-person features in $OXY_TMP/mths_feats
$PY -s scripts/optics_experiment.py  # Hoffman experiment (about 1 minute); writes scores.pkl used next
$PY -s scripts/optics_decision.py  # applies the decision rule against TINY v2
```

Measured on a 6.8 GB host: `mths_spo2.py` 60 s wall clock, 172 MB peak resident memory (`/usr/bin/time -v`), one person at a time, features cached outside the RAM-backed `/tmp`.
