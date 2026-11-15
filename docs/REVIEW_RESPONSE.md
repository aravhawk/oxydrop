# Response to the independent review (Codex, PR #3, `REVIEW.md`)

The review audited commit `aa9fe1f` on 14 November 2026. Its verdict: the fixed-protocol model numbers reproduce, the live app failed its own browser test, and the write-up described v1 as if it were the shipped v2. This page lists every item the review marked WRONG or UNSUPPORTED, plus the missing baselines, the novelty point and the five wrong-answer paths, with what was done.

Dispositions: **fixed** (code or number corrected), **rewritten** (claim reworded or removed), **kept** (claim stands, reason given), **deferred** (needs later work; where).

Every number added in response to the review comes from `scripts/review_checks.py` (arithmetic on committed files, writes `results/review_checks.csv`). It reproduced all of the review's own figures, which is an independent confirmation of them.

## Release blocker and live-flow failure

| Review item | Disposition | What changed |
|---|---|---|
| `app.js` read `gen = camGen` before the probe, `startCamera()` then did `++camGen`, and `if (gen !== camGen) return` fired before `#setup-live` was shown | **fixed** | Two counters. `sessionGen` changes only in `cleanup()` and identifies one setup attempt; the probe, its factory, `shouldAbort` and the checks after `startCamera()` use it. `camGen` identifies one camera instance and only drops stale frames. A cancelled attempt no longer tears down the session that replaced it, and a camera that finished opening after cancel is released. `app/js/app.js` |
| The default synthetic-camera run showed "Requested device not found" and timed out waiting for setup | **fixed** (the generation guard was the reproducible cause; the literal text did not appear with the installed Chrome 158) | `CameraSource.start()` retries once by facing mode when an exact `deviceId` fails with `NotFoundError`, `OverconstrainedError` or `DevicesNotFoundError`, and records the reason (`info.fallback`). The probe records the lens that really opened, the probe table and the debug view show the fallback. `app/js/camera.js`, `app/js/lensprobe.js` |
| "Unit lens/stall tests miss this integration bug" | **fixed** | `tests/flow.mjs` (in `npm test` as `test:flow`) loads the real `app/js/app.js` against a fake DOM and fake `mediaDevices` with synthetic fingertip frames and checks probe -> start -> setup UI, cancelling during the countdown, a stale `deviceId`, calibration start and the finger-lift restart. Run against the old `app.js` it fails at "setup UI must be shown after the probe and camera start" |
| "Browser e2e passes: 72 bpm, calibration, grey rejection, person-4 replay, 4 s stall, 3 s calibration stall" is WRONG as a reproduced result | **fixed** and re-run | `tests/e2e/fake_camera.mjs` cases good, bad, replay (two recordings), stall and calstall were rerun on the fixed build (results below) |

E2E rerun on the fixed build (synthetic Chromium camera, Chrome for Testing 158): **E2E PASS**, 39 of 39 checks, cases good (calibration took 70 s, heart rate 72 bpm, "Normal vs your baseline"), bad ("Unsure, re-measure", no heart rate), replay 100001L (324 of 324 answered readings correct) and 100004R (473 of 473 "unsure"), stall (flips to "Unsure" within 2.6 s and recovers) and calstall (calibration rejected, "stalled during calibration"). Before the fix, `CASES=good` failed after 21 s waiting for `#setup-live`. The app was committed at `64d3dd0` when this ran. Command and log: see the rerun checklist.

## Section 1: numbers, leakage and missing baselines

| Review item | Verdict | Disposition |
|---|---|---|
| Hoffman fixed-protocol table (v1, v2 forced, v2 gated, v1 band, v2 gated + band) | VERIFIED | **kept**. Labelled **engine-only** (no camera, placement, motion or calibration-plausibility gates). Added: 857 of 3,244 hypoxic windows unanswered, only 71.79% receive an alert, person 4 gets 0 answers in 947 windows (the worst-person figure excludes him) |
| 6 people, 12 hands, 5,644 labelled vs 5,646 parity windows, 57.5% hypoxic | VERIFIED | **kept** (README already states the 2 extra parity windows) |
| First answer "after 60 seconds" | WRONG (it is 70 s) | **fixed** in README, WRITEUP and the app home screen: calibration 60 s plus the first 10 s window, not counting probe and setup |
| Forced v2 scores no-lock windows as "no change" | WRONG | **fixed**: `chain_v2.deltas` forward-fills the last valid features; only a recording with no baseline gets zero baseline-relative features. README table reworded. The label string inside `scripts/eval_v2.py` and `results/eval_v1_vs_v2.csv` is corrected the same way |
| Two regressions, 62 numbers; C = .01/.1; band .10; 51 to 110 bpm | VERIFIED | **kept**; "62 numbers include the scaling statistics" added |
| "Ratio is not a model input", "optical oxygen sensing is not implemented" | WRONG about inputs | **rewritten** in README, `docs/SPO2_OPTICS.md`, WRITEUP and the app: the fusion regression has 19 features including four ratio features with nonzero coefficients (`ror_RG`, `ror_RB`, `ror_GB`, `ror30_RG`). Only a **separate optical SpO₂ estimator** is absent |
| Research rows: clock, ΔHR, TINY, abstention | VERIFIED (saved-score arithmetic only) | **kept**, labelled as saved-score arithmetic, not a rerun of the nested training; the abstention row is labelled v1 (shipped v2 is 67.97% coverage / 94.68%) |
| Nested-cutoff row (target .90, .903 at 86.7%) | UNSUPPORTED | **rewritten**: struck through in the README table and marked UNSUPPORTED in `results/RESULTS_full_audit.md` (`make_eval.py` copies constants from an external experiment) |
| "+3.7 points over the clock, CI [-7, +16]" | WRONG | **fixed**: +3.5 points (+3.47 on the saved scores), paired person bootstrap [-6.8, +16.7] (500 draws, seed 1). README, WRITEUP, `results/TABLES.md`, `scripts/make_eval.py`, `results/RESULTS_full_audit.md`. The "not statistically resolved" conclusion stands |
| Clock wins 3 of 6 people (1, 2, 4); person accuracies | VERIFIED (saved scores) | **kept** |
| Person 4 left hand "212 to 400 s" | WRONG | **fixed**: 79 alarms span 212 to 888 s, 13 after 400 s |
| Demographics (ages 20 to 34, 3 women and 3 men, 1 Black and 5 White, person 2 .886) | VERIFIED | **kept**; "no fairness claim" kept |
| Exercise 63% over 14, 68% over 13 | VERIFIED (saved-summary arithmetic only) | **rewritten** to 63.4% and 68.4%, with the note that raw exercise extraction is not reproducible here and that counting the person with no heart rate as never flagged lowers the headline |
| BUT: 3,840 clips, 38 people, full HR table | VERIFIED | **kept**. Fixed-gate MAE printed as 11.87 is **11.86** (1,351 clips, within-5 47.15%, doubling 0.074%, halving 4.515%): README, `docs/HR_VALIDATION.md` |
| BUT old and ungated v2 rows | VERIFIED | **kept** |
| BUT held-out 1,359 accepted, 11.89 MAE; good 70.6% / 3.74 / 84.5% | VERIFIED | **kept**; added that cross-fit and the final fixed gate are different estimators and their denominators must not be interchanged |
| Hoffman HR 1.2 / 1.8 bpm, MTHS HR 11.2 / 6.2 bpm | VERIFIED for the 10 s HR validation configuration | **rewritten**: stated as 10 s configuration results; the engine uses 30 s, so they are not exact shipped-engine errors |
| Simulator: 52 to 75 bpm old 90% doubled, v2 0%; 55 to 100 old 28.8%; 110 to 160 old 0% | VERIFIED on generated signals only | **kept** as a code-path check; the explanation ("second harmonic above the 4 Hz band-pass") was imprecise (110 bpm gives 3.67 Hz, still inside the filter), so it now says the cause is not established (README, `docs/HR_VALIDATION.md`) |
| MTHS: 62 people, 22 calibratable, 220 windows from 20 people, 66% red-clipped, about 1% below 90 | VERIFIED | **kept**; exact values now quoted: red-clipped 65.6%, hypoxic seconds 47 of 4,646 (1.012%) |
| MTHS mean 2.21, ridge 2.22, textbook 3.49, change .87 vs .77 | VERIFIED | **kept**; "implementation checks, not a universal camera calibration" added |
| MTHS GBM 2.28, time-only 2.37 | WRONG rounding | **fixed**: 2.27 and 2.36 (unrounded 2.274567 and 2.364612). README, `docs/SPO2_OPTICS.md` |
| 16 Hoffman optics variants, none meets the rule | VERIFIED as CSV arithmetic; UNSUPPORTED as a raw rerun | **rewritten**: stated as arithmetic on `decision_rule.csv`; the scripts need external `pivot/kit.py` |
| 32 candidate datasets; only 6 people anywhere; about 10 chain variants | UNSUPPORTED as exhaustive counts | **rewritten**: "the only one we found", "not an exhaustive or dated search"; variant counts marked approximate |
| Altitude cohort 20 people, zero-shot AUROC .88; absorbance rho +.62 to +.96 | UNSUPPORTED | **rewritten**: marked UNSUPPORTED and no longer relied on (README, WRITEUP, `DATA_LICENSES.md`, audit). OpenOximetry gives a better-documented zero-shot check (aggregates on branch `openox-results`) |
| Old leakage numbers (.772 to .751, label swap .74 to .83, time-added .854, whole-record z .890, random window .972 / .939, filtfilt +2, oracle .879, pooled .911, CNN +1.4) | UNSUPPORTED | **rewritten**: numbers removed from the README and WRITEUP, marked UNSUPPORTED in `results/RESULTS_full_audit.md`. The causal filling is inspectable in code; the controls are described without numbers |
| Parity 12 recordings, 5,646 windows, v2 2.1e-13, v1 3.2e-13 | VERIFIED (2.09e-13, 3.18e-13) | **fixed**: precise values; "3e-13" removed; stated as implementation agreement |
| HR parity 240 windows about 7e-15; capture 48 trials; stall 16 checks | VERIFIED | **fixed** to 417 calls, max relative difference 6.94e-15, and 138 capture trials including the 48-trial resting subset |
| Gap and stall thresholds | VERIFIED as code thresholds and Node tests | **kept** (already described as untuned heuristics, not validated real-phone limits) |
| 3,000 m persona, "2-minute phone retest", hackathon labels | UNSUPPORTED as efficacy and as measured completion time | **rewritten**: persona is a design scenario, not evidence; "2-minute" removed (`docs/PHONE_RETEST.md` is now "Phone retest" with the timeline stated as a plan) |
| Leakage verdict: "every setting chosen on the other five" unsupported and false for v2 | UNSUPPORTED | **rewritten**: README and WRITEUP now say v2 hardcodes C and the band with no independent provenance, that many variants were compared on the same six people (test-set reuse), and that the BUT final gate uses acceptance from both halves |

### Missing baselines and controls

| Review item | Disposition |
|---|---|
| Majority class gives 57.48% | **added** to README, WRITEUP (`results/review_checks.csv`) |
| On the same 3,836 windows v1 scores 94.60% against v2's 94.68%: the selective gain is mostly selection | **added** to the README reading notes and WRITEUP |
| Person-level alert yield | **added**: answered windows per person in `results/review_checks.csv`; 71.79% of hypoxic windows alerted; person 4 has 0 of 947 |
| 240 evaluation windows below 70% | **added** as a stated difference from the original paper's lower-range exclusion. A sensitivity analysis excluding them is **deferred** to the ML evaluation (`python/oxydrop_ml/PREREGISTRATION.md`) |
| Matched-coverage curves, HR-only rules, a reproduced Hoffman CNN or ratio baseline under identical labels, calibration and abstention | **deferred** to the ML baselines and evaluation features (logistic chain, XGBoost, ratio-of-ratios, CNN, all under the pre-registered LOSO protocol). The README says these are still missing |
| Coarse 100 s time balancing does not remove within-bin confounding; "clock-free" and clinical generalisation unsupported | **kept** as a stated limitation; the wording "clock-free" is used only for the time-matched control and is not claimed as proof |
| OpenOximetry "not used" is out of date (credentialed access is now active) | **fixed**: `DATA_LICENSES.md`, README, `docs/SPO2_OPTICS.md`, WRITEUP updated: used for validation, fairness and later pretraining, derived aggregates only, never redistributed, never uploaded to Kaggle |

## Section 2: novelty

| Review item | Disposition |
|---|---|
| "Our own idea" and the implication of a new detector | **rewritten**: WRITEUP "What we added" is now "Prior work, and what we added"; README gains "Prior work: what is and is not new". Hoffman 2022 (81% sensitivity, 79% specificity, 5.00 SpO₂ MAE; protocol differs), MTHS (Samavati 2022), Sun 2021 and Tian 2022 are cited; the claim is limited to engineering and reproducibility work |

## Section 3: measurement logic

| Review item | Disposition |
|---|---|
| Release blocker (generation guard) | **fixed**, see the first table |
| 1. Already hypoxic at calibration reads "Normal" | **partly fixed, partly kept**. The app and README now say "oxygen status at start unknown" (home screen, setup step 1, measure screen note, the Normal sub-text, README). An oxygen interpretation needs an independently established baseline, which a phone-only app cannot supply, so the limitation stays and is stated next to every answer |
| 2. HR changes for the wrong reason (exertion recovery, anxiety, caffeine, beta-blockers, arrhythmia) | **kept** as a documented limitation (exercise false-alarm rate 63.4%). An exertion detector is **deferred** (already listed under "what we'd do next") |
| 3. Rejected live samples poison calibration | **fixed in the app glue**: calibration starts only after exposure, white balance and focus have been settled (previously the lock could be taken after the minute began); a fingertip off the lens or flash for more than 2 s, or a camera settings change, voids the minute and restarts it once the finger is back; `checkCalibration` now requires the fingertip to cover the lens and flash for at least 90% of the minute. Stalls remain rejected by the existing calibration stall rule. `app/js/app.js`, `app/js/decision.js`, tests in `tests/decision.mjs` and `tests/flow.mjs`. **Left for the engine owner (FEAT-009):** `engine.js` still builds its baseline from every window with a valid pulse (`calRows`) and cannot see placement, motion or timing masks; a fully engine-side contiguous-clean-sample baseline needs an engine change |
| 4. Different camera, light or pressure gives wrong deltas | **partly fixed**: a settings change during calibration restarts it, and the fallback modes are labelled unvalidated. Restricting answers to validated acquisition modes and per-device paired validation are **deferred** (needs real phones: `docs/PHONE_RETEST.md`) |
| 5. Confident wrong pulse gives a confident wrong class | **deferred**: testing the full engine against ECG, morphology and cross-channel harmonic checks and withholding oxygen interpretation under material pulse uncertainty belong to the engine and model features |

## Rerun checklist

```bash
npm test                                   # includes tests/flow.mjs
node tests/docs_claims.mjs                 # the review's wrong phrases must not return (not part of npm test)
python scripts/review_checks.py            # every number added in response to the review
CASES=good,bad,replay,stall,calstall CHROME=<chromium> node tests/e2e/fake_camera.mjs   # see tests/e2e/README.md
```
