# OpenOximetry v1.1.1 results for OxyDrop (run 6)

Derived aggregates only. No row-level PhysioNet data, no credentials and no per-window files are in this repository. OpenOximetry is credentialed-access data under the PhysioNet data use agreement (DUA). Raw files and per-window tables stayed on the analysis sandbox and are not redistributed. Every number here is an aggregate that can be recomputed from the dataset by anyone with access.

Source: OpenOximetry Repository v1.1.1 on PhysioNet (healthy volunteers, staircase hypoxia protocol, reference SaO2 from arterial blood gas, finger-clip contact oximeters). It is not a phone camera dataset.

## Bottom line

- OpenOximetry confirms the dHR stage zero-shot. A logistic model trained only on OpenOx reference-HR deltas scores the 6-person Hoffman camera data at AUROC .880 and time-matched (TM) AUROC .865. The time-only baseline has TM AUROC .404 on the same data.
- It gives a large labelled time-matched baseline: 186 people, 581k windows. dHR+trend features reach accuracy .786, balanced accuracy .785, AUROC .856, TM accuracy .772. The time-only baseline reaches accuracy .656, TM accuracy .491.
- It shows no skin-tone gap for the HR-trend features (dark minus light AUROC gap +0.019 Monk with 9 dark people, -0.004 ITA). Reference pulse-oximeter bias by skin tone is the larger fairness issue.
- It does not improve the 6-person Hoffman chain, at full coverage or with abstention. Use it as validation and fairness evidence in the submission, not as extra training data. Do not claim a balanced accuracy improvement.

## Acquisition and cohort

- The Daytona egress policy blocks physionet.org (TLS reset, re-checked). The download therefore ran with `wget` basic auth to a host disk, every file was verified against the published SHA256 list (0 failures), and the verified subset was uploaded to the Daytona sandbox. All analysis ran on the sandbox.
- Subset: 1.6 GB, 6 tables, 751 `*_2hz.csv` files, 309 PPG pairs. The 150 Hz files were skipped.
- 237 patients, 907 encounters. 751 encounters (218 people) have a 2 Hz CSV. 307 encounters (139 people, 192 h) have raw red/IR PPG. 199 people and 631 encounters have reference SpO2.
- 39% of reference minutes are below SpO2 90 (9,601 of 24,708 min).
- Monk scale (people): 1-3 77, 4-6 107, 7+ 10, unknown 24.
- PPG and reference clocks agree after HR cross-correlation in 164 of 307 PPG encounters (93 people). PPG window labels carry about 5 to 10 s timing uncertainty. Use only aligned encounters for per-window PPG labels.

## (c) Time-matched baseline, 20-fold person-grouped CV, 186 people, 581k windows

TM columns are class-balanced inside 100 s bins, so the time-only baseline sits near chance by construction. Hyperparameters were chosen in nested 3-fold grouped CV.

| model | acc | bacc | AUROC | TM acc | TM AUROC | worst person | p10 person |
|---|---|---|---|---|---|---|---|
| time-only (100 s bins) | .656 | .639 | .722 | .491 | .487 | .34 | .51 |
| LR dHR | .778 | .774 | .843 | .763 | .832 | .45 | .63 |
| LR dHR+trend | .786 | .782 | .853 | .770 | .840 | .45 | .63 |
| LightGBM dHR+trend | .786 | .785 | .856 | .772 | .844 | .47 | .64 |
| LightGBM dHR+trend+dPI | .787 | .785 | .856 | .772 | .843 | .46 | .64 |
| control: LightGBM + elapsed time | .810 | .812 | .892 | .759 | .836 | .52 | .67 |
| control: label-scrambled LR | .486 | .486 | .477 | .487 | .480 | .16 | .30 |
| control: window-level random split, LightGBM | .790 | .788 | .865 | .776 | .853 | .47 | .64 |

- Perfusion-index change (dPI) adds nothing over dHR.
- A window-level random split gains only +.004 accuracy over the person-grouped split, so the low-capacity features show no leakage inflation. The scrambled-label control sits at chance.
- Adding elapsed time raises accuracy to .810 but lowers TM accuracy to .759, which is the time shortcut the TM metric is built to expose.
- PPG subset (76 aligned people): reference dHR accuracy .782 (bacc .780, AUROC .847); PPG-derived dHR from the FFT extractor accuracy .747 (bacc .748, AUROC .795, TM AUROC .786). The time-only baseline on the same people: accuracy .657, TM accuracy .490.

## (a) Zero-shot OpenOx dHR to Hoffman (6 people, leave-one-person-out, windows t >= 60 s)

Time-only baseline on Hoffman: accuracy .835, TM accuracy .519, TM AUROC .404. It is repeated beside each row.

| row | cov | acc | bacc | AUROC | TM acc | TM AUROC | worst person | time-only acc / TM acc |
|---|---|---|---|---|---|---|---|---|
| Hoffman TINY chain refit (baseline) | 1.00 | .875 | .873 | .948 | .796 | .887 | .81 | .835 / .519 |
| Hoffman TINY chain saved (baseline) | 1.00 | .870 | .869 | .942 | .795 | .885 | .78 | .835 / .519 |
| zero-shot OpenOx dHR LR, threshold 0.5 | 1.00 | .731 | .756 | .880 | .755 | .865 | .50 | .835 / .519 |
| zero-shot OpenOx dHR30 LR, threshold 0.5 | 1.00 | .753 | .772 | .880 | .775 | .865 | .50 | .835 / .519 |
| zero-shot, one-point calibration | 1.00 | .749 | .717 | .880 | .687 | .865 | .55 | .835 / .519 |
| chain + OpenOx rows (lambda 1, dHR30) | 1.00 | .876 | .876 | .948 | .795 | .884 | .81 | .835 / .519 |
| chain + OpenOx rows (lambda .25) | 1.00 | .875 | .874 | .948 | .793 | .886 | .81 | .835 / .519 |
| chain, OpenOx-only dHR stage | 1.00 | .874 | .874 | .947 | .791 | .881 | .81 | .835 / .519 |
| control: chain + scrambled OpenOx labels | 1.00 | .867 | .864 | .946 | .775 | .874 | .80 | .835 / .519 |
| baseline chain + nested abstention (target .90) | .85 | .914 | .913 | .962 | .827 | n/a | .85 | .833 / .510 |
| chain + OpenOx rows + abstention | .86 | .908 | .909 | .962 | .818 | n/a | .85 | .836 / .513 |
| OpenOx-only stage + abstention | .86 | .910 | .913 | .963 | .821 | n/a | .86 | .840 / .517 |

- Per-person zero-shot AUROC (people 1 to 6): .987, .952, .799, .892, .994, .885.
- Adding OpenOx rows to the chain gives .876 / .876 (accuracy / balanced accuracy) against the refit baseline .875 / .873. That is no real gain. The scrambled-label control reaches .867 / .864, so real labels beat it by .009, inside the confidence interval.
- With abstention the OpenOx variants land at .908 to .910 accuracy and .909 to .913 balanced accuracy at 85 to 86% coverage, against the baseline .914 / .913 at 85%. No gain.
- The time-only baseline scores high accuracy on Hoffman (.835) because of class imbalance across time, but its TM accuracy is .519. The answered-subset TM accuracy of the abstaining chains is .816 to .827.

### dHR threshold-rule guidance

- Ranking transfers. The OpenOx dHR score gives AUROC .880 on camera HR from a different sensor, population and protocol.
- The probability threshold does not transfer. At a fixed 0.5 the model gives sensitivity .59 to .65 and specificity .90 to .92 on Hoffman, with balanced accuracy .756 to .772.
- One-point calibration did not rescue it. Setting the threshold from the 95th percentile of the calibration minute gave accuracy .749 and balanced accuracy .717 (sensitivity .93, specificity .50). The per-person threshold must be calibrated against labelled data or fixed with a rule validated on the camera sensor. Do not ship a fixed OpenOx-derived cutoff.
- A rest/motion gate is still open.

## (b) Skin-tone fairness, 187 people, 612k windows, leave-one-person-out

Features are causal reference-HR deltas only (dHR, 20 s slope, 20 s SD, dHR minus 60 s median). No SpO2 feature, no elapsed time. One feature set, two models, one pooled Youden threshold per model. Bootstrap resamples people, B = 400.

| group | people | LR AUROC | LightGBM AUROC | LR TM AUROC | time-only AUROC | time-only TM AUROC |
|---|---|---|---|---|---|---|
| all | 187 | .843 | .848 | .830 | .712 | .480 |
| Monk 1-3 | 67 | .844 | .849 | .834 | .730 | .484 |
| Monk 4-6 | 93 | .840 | .846 | .825 | .696 | .479 |
| Monk 7+ (low confidence, 9 people) | 9 | .863 | .861 | .870 | .752 | .485 |
| ITA darkest tertile | 56 | .844 | .849 | .837 | .696 | .481 |
| ITA mid tertile | 55 | .842 | .847 | .808 | .733 | .484 |
| ITA lightest tertile | 55 | .848 | .855 | .847 | .709 | .482 |

Dark minus light gaps (independent person bootstraps):

| contrast | metric | gap | 95% CI |
|---|---|---|---|
| Monk 7+ vs 1-3 (9 vs 67 people, low confidence) | LR AUROC | +0.019 | -0.045 to 0.078 |
| ITA darkest vs lightest tertile (56 vs 55) | LR AUROC | -0.004 | -0.048 to 0.035 |
| ITA darkest vs lightest tertile | LR TM AUROC | -0.010 | -0.053 to 0.030 |

Reference pulse-oximeter bias (SpO2 minus SaO2 from arterial blood gas, 83,357 matched readings, 218 people):

| contrast | SpO2 minus SaO2 bias gap | occult hypoxemia gap (SpO2 >= 92 when SaO2 < 88) |
|---|---|---|
| Monk 7+ minus 1-3 | +1.23 (0.57 to 1.96) | +4.1 points (1.4 to 6.9) |
| ITA darkest minus lightest | +0.73 (0.37 to 1.16) | +2.3 points (1.1 to 3.4) |

The HR-trend features show no detectable skin-tone gap. The contact oximeter that supplies the reference SpO2 overreads on darker skin and misses hypoxemia more often. A camera app trained on pulse-oximeter labels inherits that bias. Evaluating against blood-gas SaO2 is the fair target.

## HR-extractor findings for the OxyDrop HR-capture fix (lead thread mcp:33e7ad2b)

Setting: 307 encounters, 139 people, 86 Hz red and IR contact PPG against a 2 Hz reference HR. Extractors were causal (forward-only 0.7 to 4 Hz band-pass, 10 s window, 2 s step). This is a contact sensor, so absolute numbers are an upper bound for a phone. Full tables are in `HR_EXTRACTOR_FINDINGS.md`.

| scope | extractor (IR) | MAE bpm | within 5 bpm |
|---|---|---|---|
| nominal clock | FFT peak | 7.74 | 72.0% |
| nominal clock | time-domain peak counting | 9.73 | 66.4% |
| nominal clock | FFT + 2x check + causal median of 5 | 5.84 | 77.1% |
| aligned (93 people) | FFT peak | 2.26 | 92.9% |
| aligned | time-domain peak counting | 3.61 | 88.8% |
| aligned | FFT + 2x check + median of 5 | 2.03 | 94.1% |

- Time-domain peak counting is the worst extractor. 3.4% of resting windows (reference 55 to 95 bpm) read 130 to 170 bpm, against 0.9% for the plain FFT.
- Harmonic locking is real but per-window noise. 1.3% of rest-like FFT windows read 1.8 to 2.2x the reference. Their half-frequency power is 0.26 of the peak, against about 0 when correct. No encounter had more than 50% locked windows.
- Stuck FFT peaks are a separate artifact. 18 of 592 encounter-channels (3%) have the FFT peak fixed at one value in at least half of the windows: 120 bpm (11), 110 bpm (6), 240 bpm (1). The spectral fraction is high (median .73), so the signal looks clean. A constant-frequency interferer such as LED or PWM flicker, or frame-rate aliasing on a phone, can produce the same pattern. Suggested detector: the same bpm in at least 60% of the last 30 windows.
- Recommended pipeline: band-limited FFT (or autocorrelation) on the IR-like channel, the 2x check (report f/2 when power at f/2 is at least 0.3 of the power at f and f/2 is at least 0.7 Hz), and a causal median of the last 5 estimates. Reject low-SNR windows. Log peak frequency, power at f/2 and 2f, and peak fraction next to the reported HR on the phone.
- Lowering the search ceiling from 4 Hz to 2.5 Hz reduces nominal FFT MAE from 7.74 to 6.67, but would fail on exercise. Use it only for rest-only capture.
- No HR penalty for darker skin on this sensor: dark minus light MAE -0.68 bpm (CI -1.31 to -0.19, aligned, 10 dark people).
- The red channel is slightly worse than IR (aligned FFT MAE 2.56 vs 2.26). A phone camera red channel is closest to red here.

## Selection optimism

- (c): 16 rows in `timematched.md` (the time-only baseline plus 15 model, control and PPG-subset rows; this page shows 8 of them). Only hyperparameters were tuned, inside nested CV.
- (a): 17 rows in the table above. Hoffman has 6 people, so confidence intervals are wide and small differences (.876 vs .875) are noise.
- (b): 1 feature set, 2 models, 2 group definitions, no hyperparameter search. One pooled Youden threshold per scorer.
- HR extractor: 8 extractors x 2 channels x 3 scopes were computed. The aligned lag is one parameter per encounter chosen from the data, so aligned numbers are optimistic.

## Limitations

- Contact finger-clip red/IR sensors, not phone cameras.
- Healthy volunteers on a staircase hypoxia protocol. Monk labels are sparse at the dark end (10 people at Monk 7+, 9 evaluated, none above 8).
- 143 of 307 PPG encounters (47%) are not clock-aligned to the reference, so PPG-based results use the aligned subset and carry 5 to 10 s label uncertainty.
- The Hoffman evaluation has 6 people.
- dHR labels from reference oximeters inherit pulse-oximeter skin-tone bias.
- 94 encounters (12 people) had no reference HR in the first 60 s and are excluded from the fairness set.

## Reproduction

Scripts and logs are in the research workspace (`pivot/openox/scripts`). Credentialed users can rerun them against their own copy of the dataset. Do not commit dataset files.
