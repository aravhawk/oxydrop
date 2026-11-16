<!-- GENERATED from docs/templates/WHITEPAPER.md.in by scripts/render_docs.py. Edit the template, not this file. -->
# OxyDrop: a personalised physiology-plus-CNN check for a drop in blood oxygen from phone-camera video

*Evaluated with an open, pre-registered, leave-one-person-out audit. Arav Jain and team, Veritas AI Hackathon 2026, Health track. Research prototype, not a medical device.*

## Abstract

A person with COVID-19, COPD, asthma or at altitude who owns a phone but no pulse oximeter has no way to notice a fall in blood oxygen. OxyDrop covers the rear camera and flash with a fingertip, learns that person's own normal from the first 60 seconds, and then answers every 2 seconds with one of three states: *Normal vs your baseline*, *Possible low oxygen: re-check with a pulse oximeter*, or *Unsure, re-measure*. It shows heart rate and a quality score and never shows an SpO2 number. The deployed model is a hybrid: a physiological logistic chain (heart rate, ratio-of-ratios and pulse-shape features relative to the person's baseline) plus a residual learned by a 167,588-parameter 1D CNN that runs in dependency-free JavaScript. We evaluate with leave-one-person-out (LOSO) cross-validation, nested selection and person-level bootstrap intervals, under a protocol committed before any CNN result existed (commit `3e708fe`). **The pre-registered primary metric, P1 (time-matched balanced accuracy for SpO2 < 90 on the six people of the UW Hoffman data), was not beaten:** the deployed hybrid (fixed pipeline R) scores 0.809 [0.644, 0.863], optimistic because R was chosen after its outer-fold scores were seen, and the nested per-person-selected hybrid, the unbiased estimate of the selection procedure, scores 0.779 [0.638, 0.831], both against 0.796 [0.635, 0.856] for the best baseline, the shipped logistic chain (a tie; nested difference -0.017 [-0.103, 0.069]). **The simpler chain has the higher point estimates:** a chain refit with the same nested levers and no CNN scores 0.816, and the chain-only conformal control answers 92.5% of windows at 90.3% accuracy against 74.5% at 89.5% for the deployed hybrid; all intervals overlap. The CNN alone also ties. What held: (i) an exploratory heart-rate gain (P2: within 5 bpm 0.480 against 0.419, +0.061 [0.013, 0.107]), which is post hoc, comes from BUT PPG and is not a clean win (the spectral estimator is nominally better on MTHS and better on the Hoffman people, 0.883 against 0.826); (ii) the CNN and hybrid beat the weaker baselines (clock, ratio-of-ratios); (iii) the model runs on the device as the app's default path with a calibrated three-state output; (iv) the open, pre-registered audit itself. Absolute SpO2 is not headlined: nothing beats the training mean on normoxic phone data (P3), and on a contact oximeter with blood-gas labels the CNN ties the textbook ratio-of-ratios (MAE 2.94 [2.45, 3.50] against 3.07 [2.60, 3.54]). **Round 2:** the app now runs PhotonDistill, one small network distilled from fold-internal teachers on clean and simulated distant-flash data. It **loses to the logistic chain on clean light by 0.038** (0.758 [0.603, 0.813] against 0.796 [0.636, 0.857], paired -0.038 [-0.103, -0.001]) and is more robust only under a simulated distant flash (X1 0.676 against 0.508). We state where the model ties or loses, and what only six hypoxic phone-camera people can and cannot show.

## 1. Problem and user

Low blood oxygen often feels like nothing, and the people most exposed (home COVID-19 or COPD patients, asthmatics, hikers and skiers at altitude, low-resource clinics) are often far from an oximeter while a phone is in their pocket. The design scenario is a person resting who wants to decide "should I re-check with a real oximeter, or go down?". **Input:** 30 to 60 s of fingertip video, reduced on the phone to per-frame mean R, G, B and frame timestamps (no pixels leave the device). **Output:** a three-state result every 2 s after a 60 s personal calibration (first answer at 70 s), plus heart rate and signal quality. Because the comparison is with the person's own first minute, the app cannot know whether they were already low during that minute ("oxygen status at start unknown"). This is a scenario, not evidence of efficacy: nothing here was tested at altitude or with patients.

## 2. Related work and what is new

Camera-based low-oxygen detection is established. Hoffman et al. [1] ran an induced-hypoxemia study with a smartphone and flash, released the data we use (MIT), and reported 81% sensitivity and 79% specificity for SpO2 < 90 with 5.00 SpO2 MAE using a deep model under their own protocol (not a head-to-head comparison with ours). MTHS (Samavati et al. [2]) provides 62 normoxic smartphone videos with oximeter references and an end-to-end network for vital signs. Sun et al. [3] and Tian et al. [4] developed ratio-of-ratios methods with calibration and heart-rate-adaptive filtering for camera oximetry. The ratio-of-ratios rule itself is classical pulse-oximetry optics. Pulse oximeters over-read oxygen in darker skin at low saturation (Bickler et al. [5]; Sjoding et al. [6]), which is why we report skin-tone analysis on a contact dataset and state that no phone-camera fairness claim is possible with the data that exist.

**What we claim.** (i) A personalised chain-plus-residual-CNN hybrid that runs on the device with abstention; (ii) an open, pre-registered, nested leave-one-person-out audit that reports ties and losses next to wins; (iii) OpenOximetry-based pretraining and a fairness analysis. **What we do not claim:** to be the first camera hypoxemia detector, a new sensing principle, accuracy beyond the engineered chain, a device-agnostic-augmentation benefit (the ablation does not support it), or any phone SpO2 accuracy.

## 3. Method

![Architecture and deployed hybrid.](docs/figures/ml_architecture.png)

**Data pipeline.** Per-frame mean RGB is resampled onto a causal 30 Hz grid (gaps are recorded, not hidden), band-passed causally (0.7 to 4 Hz, no zero-phase filtering), cut into 10 s windows with a 2 s hop, and compared with the person's own first minute (label-free). Training pools four sources: UW Hoffman (6 people, phone, SpO2 down to 70), MTHS (62, phone, normoxic), BUT PPG (38 smartphone PPG clips with ECG heart rate) and OpenOximetry (contact red/infrared PPG with arterial-blood-gas SaO2, Daytona only, aggregates only). Windows used: 5,644 (P1), 5,905 (P2), 1,141 (P3).

**Self-supervised pretraining and OpenOximetry warm-up.** SimCLR-style contrastive pretraining on 299,754 unlabeled windows from 212 people disjoint from every test person (loss 5.28 to 2.10, top-1 positive retrieval 0.83), followed by a supervised warm-up on OpenOximetry rows only (stage B). 34 OpenOximetry people are held out of everything for later tests. A Kaggle pretraining on HealthRing PPG tied training from scratch and could not initialise our network (SpO2 MAE 4.25 against 4.33).

**Physics-informed multi-task CNN (OxyNet).** Input: 10 channels by 300 samples (ac R, G, B, frame-time delta, per-beat ratio-of-ratios R/G and R/B relative to the person's baseline, beat mask, baseline-relative log DC), plus nine baseline-relative side features. A channel-shared stem feeds four stages of two residual blocks (widths 24, 40, 56, 80), global average and max pooling and a 64-d embedding. Heads: heart rate (141 bins, 40 to 180 bpm), SpO2 with log-variance (experimental, never shown), a low-oxygen logit on (embedding minus the person's baseline embedding), and a quality logit. Parameters: 167,588, under a 300,000 budget asserted in a test. JavaScript output matches PyTorch to better than 1e-4 on fixtures.

**Device augmentation.** Per-channel gain and gamma, colour-matrix jitter, DC steps, baseline wander, frame-rate jitter, dropped frames, saturation and motion bursts. No augmentation parameter and no threshold is fit to any particular phone. The ablation below shows the benefit is within noise, so we make no claim for it.

**Per-user calibration, distillation and the hybrid.** Per-person baselines (first minute) condition the network and are the comparison for the low-oxygen decision. Three ways of combining the chain and the CNN were pre-declared (Amendment 2): the plain CNN (P), a residual network that learns a correction to the chain logit (R), and a student distilled from the chain (D); stacking meta-learners combine their logits. All choices (source, per-user score calibration, number of windows averaged) are made by inner folds on the training people only. The deployed pipeline is chosen by the same rule on the six out-of-fold tables: source R, no score calibration, one window, with logit = 0.940 + 3.236 x (r - 1.486) / 4.275, where r is the chain logit plus the mean correction of three seeds (inner-CV score 0.818; this was selected with all six people in view, so its outer-fold score 0.809 is optimistic, and the honest estimate of the procedure is the nested 0.779).

**Conformal 3-state output.** The decision logit z becomes one of three states by class-conditional (Mondrian) split conformal prediction at the declared level alpha = 0.10 (Amendment 5, written before any conformal score existed). A prediction set {Normal} is *Normal vs your baseline*, {Low} is *Possible low oxygen*, and {both} or {} is *Unsure*. In the evaluation, the thresholds for each held-out person come only from the other five people's out-of-fold scores. The app ships thresholds fit on all six (`app/data/conformal.json`).

**What the app runs.** The hybrid is the on-device default path: the v1 physiological chain logit plus the residual of three CNN seeds, then the logistic meta-learner and the conformal three-state rule (`app/js/hybrid_engine.js`, `docs/APP_INTEGRATION.md`). The CNN runs in a Web Worker so a forward pass never delays camera frames, with logistic-chain fallback when module workers are unavailable or exceed the 2 s window deadline; the v2 logistic chain stays behind `?engine=logistic`. A generic low-amplitude guard (AC/DC below the 0.1th percentile of the Hoffman training windows, 0.24%) answers *Unsure* and shows the spectral heart rate only. With the shipped cut (`scripts/ood_range.py`) it would refuse 0.1% of Hoffman windows, 6.7% of MTHS windows and 8.8% of BUT PPG windows (other studies' phones), so a weak-flash phone will see more *Unsure*; the rate on any untested phone is unknown. Latency measured on a development laptop (`results/ml/DEPLOY.md`): 16.4 ms for one forward pass, 50 to 58 ms per 2 s window including all three seeds; phones are slower and unmeasured. The three seed files are 2.69 MB together. The replay demo is in-sample for the hybrid (the weights saw those two people) and is labelled demo only in the app.

## 4. Experiments

**Pre-registration and protocol.** `python/oxydrop_ml/PREREGISTRATION.md` was committed in `3e708fe` before any learned-model result. Primary metrics: **P1** Hoffman time-matched balanced accuracy at full coverage (6 people, 5,644 windows, majority class 57.48%); **P2** share of windows within 5 bpm of the reference on BUT PPG plus MTHS (100 people); **P3** SpO2 MAE on MTHS (61 people with a scored window). All numbers are LOSO with both hands and all recordings of a person held out together. Hyperparameters, thresholds, stacking and the choice of levers are selected by nested CV on training people only; pretraining uses only people who are never test people. We report mean, SD across folds, 95% person-bootstrap intervals (2000 resamples of people), seeds and a train-versus-test gap. A "win" requires the paired person-bootstrap interval of the difference to the *best* honest baseline to exclude zero. Five amendments are on file (`python/oxydrop_ml/amendments/`), each stating what had been seen when it was written: Amendment 1 discloses that the CNN recipe was chosen over all outer folds; Amendment 3 that the P2 rule was chosen after the pooled table was seen.

**Baselines (same folds, windows and calibration).** Majority class, clock-only (elapsed time), textbook and personally calibrated ratio-of-ratios, XGBoost and LightGBM on handcrafted features, the v1 and v2 logistic chains. Time-matched accuracy is used because the Hoffman protocol lowers oxygen steadily, so elapsed time alone predicts the label: the clock scores 0.828 accuracy but 0.495 time-matched.

## 5. Results

### 5.1 P1: low-oxygen detection (primary, pre-registered)

| Model (P1: Hoffman, 6 people, LOSO) | TM bal. acc. | 95% CI (person bootstrap) | Per person, mean ± SD | Plain AUROC | Train-vs-test gap | vs best baseline (pre-registered rule) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Majority class (always "hypoxic") | 0.500 | [0.500, 0.500] | 0.500 ± 0.000 | 0.500 | +0.000 | baseline |
| Clock only (elapsed seconds, not deployable) | 0.495 | [0.480, 0.514] | 0.509 ± 0.021 | 0.821 | +0.054 | baseline |
| Ratio-of-ratios, textbook 110-25R (red/blue) | 0.535 | [0.475, 0.594] | 0.528 ± 0.026 | 0.480 | +0.000 | baseline |
| Ratio-of-ratios, personally calibrated (red/green) | 0.612 | [0.493, 0.693] | 0.539 ± 0.089 | 0.651 | +0.020 | baseline |
| XGBoost, 26 handcrafted features | 0.655 | [0.528, 0.778] | 0.642 ± 0.096 | 0.899 | +0.170 | baseline |
| XGBoost, 19 chain features (v2 HR) | 0.763 | [0.613, 0.840] | 0.582 ± 0.115 | 0.854 | +0.015 | baseline |
| Logistic chain v2 (pulse-lock gate, forced answer) | 0.772 | [0.645, 0.828] | 0.609 ± 0.124 | 0.869 | +0.047 | baseline |
| Logistic chain v1 (best baseline) | 0.796 | [0.635, 0.856] | 0.666 ± 0.153 | 0.947 | +0.041 | baseline |
| **CNN alone** (167k parameters; mean of 3 seeds, own probability at 0.5) | 0.739 | [0.574, 0.811] | 0.618 ± 0.094 | 0.843 | -0.015 | tie with the best baseline (3-seed ensemble 0.754, paired diff -0.042 [-0.131, 0.005]) |
| CNN alone, 3-seed ensemble through the per-fold recalibration (`default_P`) | 0.781 | [0.632, 0.827] | 0.669 ± 0.081 | 0.852 | n/a | exploratory |
| **Deployed pipeline R** (the app; chain + CNN residual). Optimistic: chosen after the outer scores were seen | 0.809 | [0.644, 0.863] | 0.671 ± 0.088 | 0.918 | n/a | exploratory; interval overlaps the best baseline, no paired test run |
| **Hybrid, per-person selected** (nested: unbiased estimate of the selection procedure) | 0.779 | [0.638, 0.831] | 0.635 ± 0.102 | 0.891 | +0.076 | **tie** with the best baseline (-0.017 [-0.103, 0.069]) |
| Control: chain refit with the same nested levers (no CNN) | 0.816 | [0.674, 0.869] | 0.685 ± 0.142 | 0.947 | +0.019 | tie with the best baseline (+0.019 [-0.000, 0.049]) |

*TM = time-matched. Gap = mean over folds of the fold model's score on its own pooled training people minus the pooled held-out score (positive = optimistic). Per-person values are the mean and SD of the six held-out people.* **The CNN does not beat the chain.** The three CNN seeds score 0.752, 0.717 and 0.748 (seed SD 0.019); seed 1 alone loses to the chain (-0.079 [-0.217, -0.021]), the 3-seed ensemble ties (-0.042 [-0.131, 0.005]). The hybrid ties the chain (-0.017 [-0.103, 0.069]). A chain refit with the same nested levers and threshold calibration reaches 0.816, at least as good as every CNN candidate, and the hybrid minus this control is -0.037 [-0.119, 0.028]. Against the weak baselines (ratio-of-ratios, clock) the CNN and hybrid are clearly better; these are not pre-registered wins. The people are complementary (the CNN is better on person 6, clearly worse on person 1) but no combiner learned from five people exploits it:

(Per-person table and figure: `MODEL.md`, P1 section.)

The conformal three-state output and its comparison at matched coverage are in the conformal subsection below.

### 5.2 P2: heart rate and P3: absolute SpO2

| Model (P2: BUT PPG + MTHS, 100 people, LOSO, full coverage) | All: within 5 bpm [95% CI] | BUT PPG (38) | MTHS (62) | MAE (bpm) | Doubling | vs best baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| FFT peak, plain | 0.382 [0.329, 0.441] | 0.271 | 0.590 | 17.22 | 1.03% | baseline |
| v1 peak-interval counter | 0.397 [0.345, 0.450] | 0.315 | 0.549 | 14.53 | 2.44% | baseline |
| **v2 spectral + autocorrelation (best baseline)** | 0.419 [0.366, 0.481] | 0.304 | 0.633 | 13.68 | 0.24% | baseline |
| v2 with its shipped quality gate (abstains) | 0.312 [0.255, 0.378] | 0.167 | 0.582 | 8.85 | 0.00% | baseline |
| CNN alone (3-seed median) | 0.468 [0.419, 0.522] | 0.410 | 0.576 | 8.63 | 0.02% | tie (3-seed ensemble) |
| **Hybrid HR rule `agree`** (CNN, averaged with v2 spectral when within 5 bpm) | 0.480 [0.429, 0.534] | 0.419 | 0.594 | 8.55 | 0.02% | win by the rule (+0.061 [0.013, 0.107]); **post hoc, exploratory, driven by BUT PPG** |

The hybrid rule `agree` (mean of CNN and v2 spectral estimate if within 5 bpm, else CNN) shows a paired difference of +0.061 [0.013, 0.107] against the best baseline, which would be a win by the pre-registered rule, but it is **post hoc and exploratory, not a clean win**. It was chosen among CNN-containing rules on Hoffman inner folds after the pooled table had been seen (Amendment 3); the Hoffman inner folds alone (pooled out-of-fold windows of the six Hoffman people, the table that selected the rule, not an outer-fold test result) would choose the spectral estimator (0.883 against 0.826 for `agree`). The gain comes from BUT PPG (+0.115 [0.054, 0.177]; noisy ear and motion clips); on MTHS the spectral estimator is nominally better (-0.039 [-0.102, 0.017]). The CNN alone ties. A voting rule over autocorrelation, spectral and CNN estimates with octave check (Amendment 5, written before any result) does not help on phone-ear data:

**Voting rule on clean clips** (pass the shipped v2 quality gate; MAE in bpm; target stated before any result: MAE under 3 bpm and doubling under 1%). Voting of autocorrelation, spectral harmonic sum and the CNN, with octave check and a causal median of 5:

| Dataset | Clean windows | Voting MAE [95% CI] | Within 5 bpm | Doubling | CNN alone MAE | Spectral alone MAE | Target met |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| BUT PPG (38 people) | 1,379 (36%) | 10.94 [8.63, 13.44] | 49.2% [38.8%, 59.4%] | 0.00% | 8.05 | 11.85 | no |
| MTHS (62 people) | 1,701 (82%) | 5.22 [3.51, 7.34] | 77.4% [69.0%, 84.8%] | 0.00% | 6.30 | 6.42 | no |
| BUT PPG + MTHS (100 people) | 3,080 (52%) | 7.78 [6.29, 9.45] | 64.7% [57.8%, 71.4%] | 0.00% | 7.08 | 8.85 | no |
| Hoffman (6 people, extra) | 4,671 (83%) | 0.98 [0.78, 1.21] | 98.2% [95.7%, 99.7%] | 0.00% | 2.11 | 1.77 | yes |

**Slow pulse (Hoffman person 4, mean reference 48 bpm).** With the shipped spectral floor of 51 bpm the spectral estimate has MAE 14.3 bpm; widening the band to 40 bpm gives 1.9; the CNN does not track this person (21.0 bpm).

On BUT PPG the voting rule (10.94 bpm against 8.05) is worse than the CNN alone, and the target is not met on either phone dataset, so **it does not replace the P2 rule** and the app keeps `agree`. Source: `results/ml/hr_vote_summary.csv`, `hr_vote_paired.csv`, `hr_vote_slow_band.csv`; protocol in `results/ml/HR_VOTE.md`.

| Model (P3: MTHS SpO2, 61 people, LOSO) | MAE, SpO2 points | 95% CI | Per person, mean ± SD | Train-vs-test gap |
| :--- | ---: | ---: | ---: | ---: |
| Train-mean predictor (best baseline) | 1.354 | [1.097, 1.650] | 1.392 ± 1.040 | +0.065 |
| Ridge on six ratio features | 1.354 | [1.097, 1.649] | 1.393 ± 1.039 | +0.067 |
| LightGBM, 52 handcrafted features | 1.351 | [1.101, 1.649] | 1.351 ± 0.964 | +0.495 |
| XGBoost, 52 handcrafted features | 1.370 | [1.118, 1.673] | 1.364 ± 0.988 | +0.459 |
| Ratio-of-ratios 110-25R (red/blue) | 3.332 | [2.890, 3.787] | 3.276 ± 1.858 | -0.056 |
| Ratio-of-ratios 110-25R (red/green) | 6.004 | [4.521, 8.010] | 6.902 ± 8.167 | +0.897 |
| **CNN alone** (mean of 3 seeds) | 1.457 | [1.196, 1.787] | 1.524 ± 1.124 | not reported |
| CNN alone, 3-seed ensemble (interval column: paired difference in MAE points, train-mean MAE minus CNN MAE; a negative value means the CNN is worse) | 1.371 | -0.017 [-0.181, 0.148] |  | tie (CNN worse by 0.017, interval contains zero) |

MTHS is normoxic (median reference 97.0, minimum 88.1, 14.2% of windows below 95, 2 windows from 1 person below 90), so the train-mean predictor is near optimal and **no model beats it** (-0.017 [-0.181, 0.148] for the CNN). MTHS is not evidence about hypoxemia, so SpO2 is not headlined and is not shown to users.

### 5.3 Ablations and the lever ledger

Every lever with its effect on held-out people and the decision (six people, so effects carry selection optimism; single-seed rows compare with the seed-0 row; evidence column in `MODEL.md`).

| Lever | Measured effect on held-out people | Decision |
| :--- | :--- | :--- |
| OpenOximetry supervised warm-up (stage B) and OpenOx data | P1 -0.162 when removed; P2 -0.142 (grouped, vs seed 0) | KEEP |
| Self-supervised pretraining (SimCLR on 299,754 windows) | P1 -0.013; P2 -0.108 (seed 0) | KEEP (small) |
| Baseline conditioning (person-relative inputs and embedding subtraction) | P1 -0.028; gap +0.105 vs -0.015 | KEEP |
| Physics channels (ratio-of-ratios, beat mask, side features) | P1 -0.008; P2 -0.012 | NO MEASURED GAIN (still in the network, not claimed) |
| Device-agnostic augmentation | P1 -0.019; P2 -0.026 | NO MEASURED GAIN (still in the network, not claimed) |
| HR-only training (HR and quality heads) | P2 +0.010 vs full seed 0 | DROP |
| 20 s window | P1 -0.003; P2 on MTHS only 0.580 vs 0.588 for full seed 0 | DROP |
| Gradient-reversal domain head (exploratory) | P1 +0.010 | DROP |
| Residual on the chain logit (family R) | TM 0.809 vs plain CNN 0.781 (default levers, 3 seeds) | KEEP (selected in 3 of 6 folds) |
| Chain distillation (family D) | TM 0.744 | DROP |
| Frozen-backbone residual (family L, exploratory) | TM 0.754 | DROP |
| Per-user baseline calibration of the scores | TM 0.807 on vs 0.809 off (R, k = 1) | DROP (chosen in 1 of 6 folds) |
| Multi-window aggregation (k = 6 windows) | TM 0.810 vs 0.809 for k = 1 (R) | DROP (chosen in 1 of 6 folds) |
| Test-time BatchNorm adaptation on the first minute | TM 0.638 vs 0.752 without (seed 0) | DROP |
| Seed ensembling (3 seeds) | TM 0.754 vs 0.739 mean of single seeds | KEEP (R members shipped as 3 seeds) |
| Fold ensembling (extra members from the inner fits) | TM 0.769 vs 0.781 for the seed ensemble | DROP |
| Stacking: meta-learner on [chain logit, CNN logit] | TM 0.786 | KEEP (selected in 2 of 6 folds) |
| HR: average CNN and v2 spectral when they agree within 5 bpm | within 5 bpm 0.480 vs CNN alone 0.468 | KEEP |
| HealthRing PPG pretraining (Kaggle kernel, different CNN; predictions only) | SpO2 MAE 4.25 pretrained vs 4.33 scratch (6 Hoffman people) | DROP |
| Conformal 3-state abstention (Mondrian, alpha 0.10) | hybrid 85.9% at v1 coverage 84.4% against 93.9% for the hand-set v1 band (-8.0 points [-20.8, +1.2]); the same step on the chain: 92.0% | KEEP (output rule, not a win) |
| OpenOximetry lag-corrected SaO2 head (contact sensor) | MAE 2.94 [2.45, 3.50] against 3.07 for the textbook ratio-of-ratios (tie) | DROP as a claim (never shown) |
| HR voting with harmonic and octave correction | clean BUT PPG MAE 10.94 bpm against 8.05 for the CNN alone; target under 3 bpm not met | DROP |

**The OpenOximetry warm-up is the main source of skill in the CNN:** without it P1 falls from 0.739 to 0.577, and with no pretraining at all to 0.550. SSL on top of it is a small, uncertain gain.

### 5.4 Spread and train-versus-test gap

| Model | Train, pooled people | Held-out, pooled | Gap (train minus held-out) | Reading |
| :--- | ---: | ---: | ---: | :--- |
| Logistic chain v1 | 0.837 | 0.796 | +0.041 | baseline |
| Logistic chain v2 | 0.819 | 0.772 | +0.047 | baseline |
| XGBoost, 19 chain features | 0.778 | 0.763 | +0.015 | baseline |
| XGBoost, 26 handcrafted | 0.825 | 0.655 | +0.170 | baseline |
| LightGBM, 26 handcrafted | 0.815 | 0.647 | +0.168 | baseline |
| CNN alone (3 seeds) | 0.725 | 0.739 | -0.015 [-0.018, -0.011] | train score below held-out: no over-fitting signal at the fixed 0.5 threshold |
| Hybrid, selected pipeline | 0.855 | 0.779 | +0.076 | per person +0.093 (SD 0.121); selection optimism -0.000 |
| Chain, same nested levers | 0.834 | 0.816 | +0.019 | per person +0.004 (SD 0.172); selection optimism -0.038 |

Per-fold spread (mean ± SD over the six held-out people) is in the P1 table; the CNN's seed SD is 0.019. The plain CNN shows no over-fitting at its fixed 0.5 threshold, while the selected hybrid has a pooled gap of +0.076 (per person +0.093 ± 0.121 [0.009, 0.180]) against +0.019 for the chain: a likely reason (Amendment 4, not tested) is that fine-tuning a whole network on at most five hypoxic people costs generalisation, which favours the simple chain when the two tie.

### 5.5 Conformal three-state output (secondary, exploratory)

**Held-out conformal output at the declared level alpha = 0.10** (Mondrian, thresholds from the five other people only, 5,644 Hoffman windows after the first minute, 95% person-bootstrap CIs). The deployed pipeline R is the model the app runs; its row is exploratory and optimistic because R was chosen with all six people in view. The per-person selected row is the nested estimate of the selection procedure. The chain-only control has the higher point estimates:

| Pipeline | Coverage (answered) | Accuracy on answered | All hypoxic windows alerted | Hypoxic called Normal | Normal / Possible low / Unsure |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Deployed pipeline R** (the app; optimistic) | 74.5% [60.1%, 84.0%] | 89.5% [79.4%, 98.1%] | 70.0% [46.5%, 88.5%] | 8.7% | 31% / 43% / 26% |
| Hybrid, per-person selected (nested) | 79.1% [63.5%, 89.9%] | 87.0% [76.1%, 95.5%] | 70.1% [43.6%, 90.0%] | 11.4% | 35% / 44% / 21% |
| v1 chain alone (control) | 92.5% [86.9%, 98.2%] | 90.3% [86.1%, 94.4%] | 84.3% [78.9%, 88.2%] | 10.0% | 41% / 52% / 8% |

**At matched coverage** (each system is read at the coverage of the reference, by interpolation along alpha; the reference accuracy is the hand-set band of the shipped v1 or v2 chain):

| Reference at its own coverage | Pipeline (nested Mondrian conformal) | Matched coverage | Accuracy on answered | Reference accuracy | Difference, points | Verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| v1 shipped band | **Deployed pipeline R** (the app; optimistic) | 84.4% | 88.0% [78.5%, 96.2%] | 93.9% | -5.9 [-17.1, +2.3] | tie |
| v1 shipped band | Hybrid, per-person selected (nested) | 84.4% | 85.9% [75.6%, 94.9%] | 93.9% | -8.0 [-20.8, +1.2] | tie |
| v1 shipped band | v1 chain alone (control) | 84.4% | 92.0% [88.2%, 95.9%] | 93.9% | -1.9 [-3.7, +0.3] | tie |
| v2 gated band | **Deployed pipeline R** (the app; optimistic) | 68.0% | 90.8% [82.2%, 98.7%] | 94.7% | -3.9 [-13.9, +3.8] | tie |
| v2 gated band | Hybrid, per-person selected (nested) | 68.0% | 89.5% [78.9%, 98.2%] | 94.7% | -5.2 [-18.5, +3.3] | tie |
| v2 gated band | v1 chain alone (control) | 68.0% | 95.0% [91.0%, 98.2%] | 94.7% | +0.3 [-3.5, +3.1] | tie |

Source: `results/ml/conformal_curve.csv`, `conformal_matched.csv`, `conformal_per_person.csv`, `conformal_calibration.csv`; protocol and reading in `results/ml/CONFORMAL.md`.

**The deployed pipeline R does not beat v1 at matched coverage** (-5.9 [-17.1, +2.3] points at v1's coverage, -3.9 [-13.9, +3.8] at v2's; both intervals contain zero; the per-person-selected hybrid is at -8.0 [-20.8, +1.2] and -5.2 [-18.5, +3.3]). The same conformal step on the v1 chain reads 92.0% at v1's coverage and 95.0% at v2's, so the loss comes from the hybrid score and not from the conformal step; **the chain-only control, available in the app behind `?engine=logistic`, has the higher point estimates and answers more windows** (92.5% at 90.3%, 84.3% of hypoxic windows alerted, against 74.5%, 89.5% and 70.0%). These deployed-pipeline rows are optimistic: R was chosen with all six people in view. At alpha = 0.10 the deployed pipeline calls 8.7% of hypoxic windows Normal (the target was 10%). Set coverage holds approximately when pooled (91.3% of hypoxic and 93.2% of normal windows, against 90%) and **fails per person**: for person 1, 51.3% of hypoxic windows are called Normal, and for person 4 the output is Unsure for 58.2% of windows (slow pulse).

![Conformal output: accuracy versus coverage, and hypoxic windows called Normal per person.](docs/figures/ml_conformal.png)

### 5.6 Round 2: the shipped trained model under a simulated distant flash

**Model.** A multilayer perceptron on 39 causal baseline-relative window features (2,435 parameters per seed, 3 seeds; heads for P(SpO2 < 90), SpO2 and its own abstain logit), trained in each outer fold on the five training people with soft targets from logistic chain v1 (clean windows) and a DC-colour regression (physics-sampled distant-flash copies); all choices by inner CV. Only the student runs in the app.

**Results** (`results/photon/TABLE.csv`, paired to chain v1). Clean light: 0.758 [0.603, 0.813] against 0.796 [0.636, 0.857], paired -0.038 [-0.103, -0.001]: the shipped trained model **loses to the logistic chain on clean light by 0.038**. It was shipped because round 2 required a trained end-to-end model with in-browser inference and for its low-light robustness; no round-2 model beat or tied the chain, and the rule extension that picked it was written after the results were seen (Amendment 7). Simulated distant flash: L4 +0.083 [0.008, 0.178] (6/6 people), L5 +0.166 [0.035, 0.242], out-of-family X1 +0.169 [0.058, 0.285] (6/6); the chain keeps answering at about chance, PhotonDistill answers fewer windows (0.27 at X1). Null mean 0.492.

**Caveats.** Round-2 models were designed after round-1 results were seen, and PhotonDistill v2 was tuned after its v1 result (0.696). The simulation scales whole recordings, which baseline-relative features cancel by design; the DC-colour control is nearly as robust (0.646 at X1, a tie). Other candidates lose on clean light (OxySeq 0.708, PhotonNet 0.668, OxySSL 0.667; the last two fail their null controls). One dark-skinned person: no skin-tone claim. Details, figures and every disclosure: `MODEL.md` section 0.

## 6. Fairness

**Contact-sensor test, not a phone test.** OpenOximetry is a red/infrared finger clip with arterial-blood-gas SaO2. 34 people are held out of every training stage (24 with SaO2 labels, 25 scored); the label lag (19 s, bootstrap [15, 27]) was learned on 63 training people only. Windows: 33,001. Error in SpO2 points, 95% person-bootstrap CI:

| Predictor | MAE | Bias (prediction minus SaO2) | ARMS, SaO2 70 to 100 |
| :--- | ---: | ---: | ---: |
| CNN, hybrid R members (3 seeds, shipped family) | 2.94 [2.45, 3.50] | 1.22 [0.45, 2.02] | 4.19 [3.65, 4.69] |
| Textbook ratio-of-ratios (110 - 25 R) | 3.07 [2.60, 3.54] | -0.18 [-0.90, 0.63] | 4.21 [3.77, 4.61] |
| Training-mean predictor | 7.38 [6.95, 7.82] | -0.01 [-0.56, 0.62] | 8.19 [7.83, 8.52] |

**By skin tone** (Monk scale from the dataset; ITA from the image-derived tertiles; MAE in SpO2 points):

| Group | People | CNN MAE | Textbook ratio-of-ratios MAE | Training-mean MAE |
| :--- | ---: | ---: | ---: | ---: |
| Monk 1-3 | 8 | 3.25 [2.03, 4.04] | 3.44 [2.15, 4.16] | 7.11 [5.95, 7.85] |
| Monk 4-6 | 14 | 2.66 [2.14, 3.40] | 2.88 [2.54, 3.29] | 7.60 [7.17, 8.16] |
| Monk 7+ | 2 | suppressed (under 5 people) | suppressed (under 5 people) | suppressed (under 5 people) |
| ITA lightest tertile | 6 | 2.71 [1.89, 3.64] | 3.49 [2.54, 4.47] | 7.70 [6.99, 8.77] |
| ITA mid tertile | 7 | 2.97 [1.94, 3.96] | 3.11 [1.95, 3.93] | 7.27 [6.05, 7.91] |
| ITA darkest tertile | 7 | 3.12 [2.50, 4.09] | 2.87 [2.35, 3.66] | 7.26 [6.57, 8.40] |

CNN MAE dark minus light, ITA tertiles (7 versus 6 people): +0.41 [-0.75, +1.63], and the interval contains 0. Monk 7 and above has only 2 held-out people, so the gap by Monk group is **not estimable**. Minimum group size: every committed skin-tone cell needs at least 5 people, smaller groups are suppressed.

**The reference itself is biased.** Contact-oximeter display minus blood-gas SaO2 (all 137 people with draws), dark minus light: Monk 7+ versus 1-3 +1.77 [+0.64, +2.61] points (7 versus 55 people), ITA darkest versus lightest tertile +1.10 [+0.65, +1.55] points. The device over-reads more in darker skin, so any label from a pulse oximeter inherits this; the analysis above is anchored to blood gas.

Source: `results/ml/openox_spo2_overall.csv`, `openox_spo2_skin.csv`, `openox_spo2_gaps.csv`, `openox_spo2_paired.csv`, `openox_spo2_contact_bias_gaps.csv`; protocol in `results/ml/OPENOX_SPO2.md` (Daytona only, aggregates only).

![OpenOximetry SaO2 error by skin tone, and bias of the contact reference itself.](docs/figures/ml_openox_skin.png)

On Hoffman there is one dark-skinned participant (logistic chain v1 accuracy 88.6% for that person against 87.4% for the 5 others, gap -0.012; anecdote only, n = 1), which supports no fairness conclusion. BUT PPG and MTHS publish no skin tone. The reference oximeters themselves are biased in darker skin at low saturation [5, 6], so any label derived from a pulse oximeter inherits that bias: we therefore anchor the OpenOximetry analysis to arterial blood gas SaO2, and we do not claim the phone result is fair or unfair.

## 7. Failure modes

* **Motion, cold fingers and poor perfusion, saturated red channel:** the pulse lock, quality score or saturation gate fails and the app answers *Unsure*. The gates are heuristics checked only on a synthetic camera and on recorded data, not on live phones.
* **Far-flash and weak-flash phones** (flash several centimetres from the lens): weaker, diffuse signal. A runtime lens probe and adaptive exposure try to compensate; such phones are absent from every training and evaluation set. The generic low-amplitude guard (section 3) answers *Unsure* and would refuse 6.7% of MTHS and 8.8% of BUT PPG windows, against 0.1% of Hoffman.
* **Dark skin:** pulse-oximeter references over-read at low saturation [5, 6]; our evidence is the contact dataset only (section 6).
* **Already low at the start:** the output is relative to the first minute, so low oxygen present during calibration is invisible.
* **Untested low SpO2 in the wild:** only **six hypoxic phone-camera people** exist in open data, all one phone (Nexus 6P), induced desaturation at rest. The model has never been evaluated on a person who is hypoxic outside that protocol.
* **Slow pulse and exercise:** Hoffman person 4 (reference about 48.2 bpm) is below the spectral floor of 51 bpm (MAE 14.3 bpm; 1.9 with a 40 bpm floor) and the CNN does not track the person (21.0 bpm), so the app answers Unsure. Exercise raises heart rate without low oxygen and the heart-rate-rise signal then false-alarms (see `docs/ENGINEERED_CHAIN.md`).
* **Dicrotic-notch doubling (iPhone-class cameras):** with a strong second pulse peak the spectral estimate can lock to twice the heart rate; a team iPhone 13 double-counted at 150 bpm seated in an earlier test (`docs/PHONE_RETEST.md`). The octave check and the CNN vote guard against it (doubling 0.00% on clean clips of the public datasets), but no iPhone recording is in any evaluation set.
* **Seed instability:** one of three CNN seeds loses to the chain on P1, and that seed almost never alerts for person 1.

![Failure modes: synthetic schematics of the mechanisms and real out-of-fold scores.](docs/figures/ml_failure_modes.png)

## 8. Limitations and not a medical device

OxyDrop is a research prototype and **not a medical device**. It does not diagnose, must not replace a pulse oximeter or a clinician, and "Normal" is never reassurance. P1 has six people, so intervals are wide and a "tie" mostly means "cannot distinguish". The weights shipped in the app are fit on all people, so every reported number comes from held-out folds and no in-sample number is quoted as a result (the replay demo in the app is in-sample and labelled demo only). The shipped weights are derived from MTHS (CC BY-NC-ND) and credentialed OpenOximetry data; whether releasing them is permitted is for the data owners to confirm (`DATA_LICENSES.md`). **The app is built to work on every phone and is not tuned to any specific phone**: no threshold, gain or model behaviour is fit to any device, and the lens probe, adaptive gain, quality gating and augmentation are device-agnostic. The team's iPhone 13, iPhone 17 and iPhone 18 Pro serve only as a held-out sanity check, never as a tuning set. **Untested Android phones and other phones are a limitation**: no phone other than the Hoffman devices contributes to P1, no live-phone accuracy is claimed, and the weak-pulse guard would refuse 6.7% of MTHS and 8.8% of BUT PPG windows, so weak-flash phones will see more *Unsure*. Further limits: OpenOximetry is contact red/infrared PPG and cannot validate phone SpO2, the three-seed members cost about three times one (phone latency is unmeasured), and several pre-declared levers were exploratory.

## 9. Ethics

No video, pixel or identifier leaves the device. The datasets are public (MIT, CC BY 4.0), non-commercial (MTHS) or credentialed (OpenOximetry: aggregates only, never uploaded to Kaggle). A false "Normal" (delayed care) costs more than a false alarm, so the interface abstains rather than guessing. Skin-tone gaps are reported where data allow and withheld where they do not. AI tools (Claude) assisted with code, analysis and drafting; the team reviewed the work.

## 10. Reproducibility

Every number is generated from `results/ml/*.csv` and `results/photon/*.csv` (`scripts/render_docs.py`) and checked by `scripts/check_claims.py`. Stages: `bash scripts/reproduce_ml.sh <stage>`. Tests: `python -m pytest python/oxydrop_ml/tests`, `npm test`.

## References

1. Hoffman JS, Viswanath VK, Tian C, et al. Smartphone camera oximetry in an induced hypoxemia study. *npj Digital Medicine* 5:146 (2022). doi:10.1038/s41746-022-00665-y
2. Samavati et al. Efficient deep learning-based estimation of the vital signs on smartphones (MTHS dataset). arXiv:2204.08989 (2022).
3. Sun Z, He Q, Li Y, Wang W, Wang RK. Robust non-contact peripheral oxygenation saturation measurement using smartphone-enabled imaging photoplethysmography. *Biomedical Optics Express* 12:1746 (2021). PMC7984770.
4. Tian X, Wong C-W, Ranadive SM, Wu M. A multi-channel ratio-of-ratios method for noncontact hand video based SpO2 monitoring using smartphone cameras. *IEEE J. Sel. Topics Signal Process.* 16(2):197-207 (2022). doi:10.1109/JSTSP.2022.3152352
5. Bickler PE, Feiner JR, Severinghaus JW. Effects of skin pigmentation on pulse oximeter accuracy at low saturation. *Anesthesiology* 102:715-719 (2005).
6. Sjoding MW, Dickson RP, Iwashyna TJ, Gay SE, Valley TS. Racial bias in pulse oximetry measurement. *N Engl J Med* 383:2477-2478 (2020).
7. Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations (SimCLR). *ICML* (2020).
8. Wolpert DH. Stacked generalization. *Neural Networks* 5(2) (1992). Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 7:91 (2006).
