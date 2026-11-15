<!-- GENERATED from docs/templates/README.md.in by scripts/render_docs.py. Edit the template, not this file. -->
# OxyDrop

**A phone-camera check for a drop in your blood oxygen from your own baseline. No extra hardware, no install, no video leaves the phone.**

**Live app:** https://aravhawk.github.io/oxydrop/ (open on a phone; "Watch a recorded demo" works with no camera) · **Paper-style write-up:** [WHITEPAPER.md](WHITEPAPER.md) ([PDF](WHITEPAPER.pdf)) · **Model card:** [MODEL.md](MODEL.md)

> **This project is a prototype and is not medical advice.** It does not measure SpO2, is not a medical device, and the key low-oxygen result rests on only six people. Never use it instead of a pulse oximeter or a clinician. Built for the Veritas AI International High School Hackathon 2026, Health track.

## Who it is for, and input to output

**For** people with COVID-19, COPD, asthma or at altitude who own a phone but no pulse oximeter, and for low-resource clinics. **Input:** a fingertip over the rear camera and flash for 30 to 60 s, reduced on the phone to per-frame mean R, G, B and frame timestamps. **Output:** after a 60 s calibration of *your own* normal (first answer at 70 s), every 2 s one of three results, with exactly these words:

| State | Text on screen | Meaning |
|---|---|---|
| Normal | **Normal vs your baseline** | Heart-rate, pulse-shape and light-absorption changes from your own first minute look like the normal group. It cannot tell whether you were already low during that first minute (**oxygen status at start unknown**) |
| Possible low oxygen | **Possible low oxygen: re-check with a pulse oximeter** | Looks like the low-oxygen group (SpO2 < 90% in the study). Not a diagnosis |
| Unsure | **Unsure, re-measure** | Bad placement, motion, no stable pulse, a poor calibration, or a score near the boundary. No number is shown |

The app never shows an SpO2 value. It also shows heart rate and a quality score.

![From fingertip video to three states](docs/figures/ml_flow.png)

## The model in one paragraph

A **hybrid**: a physiological logistic chain (heart-rate rise, pulse shape and ratio-of-ratios features, each relative to the person's own first minute) gives a low-oxygen logit, and a **167,588-parameter 1D CNN** (`OxyNet`, three seeds, dependency-free JavaScript) learns a residual correction to it, with heart-rate, quality and experimental SpO2 heads. The network was pretrained self-supervised and warmed up on contact-oximeter data (OpenOximetry) before fine-tuning on phone video. Details: [MODEL.md](MODEL.md).

![Architecture](docs/figures/ml_architecture.png)

## Results, honestly (leave-one-person-out, nested selection, 95% person-bootstrap intervals)

The evaluation protocol was **pre-registered** (commit `3e708fe`, [PREREGISTRATION.md](python/oxydrop_ml/PREREGISTRATION.md)) before any CNN result existed. A win needs the paired interval of the difference to the **best** honest baseline to exclude zero.

**P1 (primary, pre-registered): detect SpO2 < 90 on the six people of the UW Hoffman data, time-matched balanced accuracy.**

| Model (P1: Hoffman, 6 people, LOSO) | TM bal. acc. | 95% CI (person bootstrap) | Per person, mean ± SD | AUROC | Train-vs-test gap | vs best baseline (pre-registered rule) |
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
| **Hybrid, selected pipeline** (chain logit + CNN residual, nested selection) | 0.779 | [0.638, 0.831] | 0.635 ± 0.102 | 0.891 | +0.076 | **tie** with the best baseline (-0.017 [-0.103, 0.069]) |
| Control: chain refit with the same nested levers (no CNN) | 0.816 | [0.674, 0.869] | 0.685 ± 0.142 | 0.947 | +0.019 | tie with the best baseline (+0.019 [-0.000, 0.049]) |

**The primary metric was not beaten.** The hybrid (0.779 [0.638, 0.831]) ties the shipped chain (0.796 [0.635, 0.856]), the CNN alone ties it, and a chain refit with the same levers and no CNN scores 0.816. The CNN beats only weak baselines (ratio-of-ratios, clock), which is not a pre-registered win. One of three CNN seeds loses to the chain by itself.

**P2 (pre-registered): heart rate within 5 bpm, BUT PPG plus MTHS, 100 people.** Hybrid rule 0.480 [0.429, 0.534] against 0.419 [0.366, 0.481] for the v2 spectral estimator: paired difference +0.061 [0.013, 0.107], a win by the rule, from a design choice made after the pooled table had been seen (Amendment 3). BUT PPG gains, MTHS ties or loses (-0.039 [-0.102, 0.017]).

**P3 (pre-registered): absolute SpO2 on MTHS, MAE.** CNN 1.457 against 1.354 for predicting the training mean: no skill. MTHS is normoxic, so SpO2 is not headlined and not shown to users.

What the experiments say about the network: **the OpenOximetry warm-up is the main source of its skill** (P1 0.739 with it, 0.577 without); a device-agnostic-augmentation benefit is **not supported**; physics channels change nothing measurable; test-time BatchNorm adaptation hurts. Every lever, kept or dropped, is in the ledger in [MODEL.md](MODEL.md) and [WHITEPAPER.md](WHITEPAPER.md).

**PENDING (generated blocks, filled automatically when the files exist):** conformal 3-state output and matched-coverage comparison; OpenOximetry SpO2 by skin tone; heart-rate voting. **PENDING:** the app in this tree still runs the logistic chain v2 (the hybrid export is ready, wiring is the app-integration branch).

![Ablation](docs/figures/ml_ablation.png)

## Where it fails

* **Only six hypoxic phone-camera people exist in open data**, one phone model, induced desaturation at rest: wide intervals, and "tie" mostly means "cannot distinguish". Nothing is known about low oxygen in the wild.
* **Person-level instability:** the chain is far better on person 1 and the CNN on person 6; no combiner learned from five people exploits it (per-person table in [MODEL.md](MODEL.md)).
* **Motion, cold fingers, saturated red channel, far-flash phones:** the app answers *Unsure* rather than guess ([figure](docs/figures/ml_failure_modes.png)).
* **Dark skin:** pulse-oximeter references are biased in darker skin at low saturation (Bickler 2005, Sjoding 2020). Phone-camera fairness cannot be assessed (one dark-skinned participant); the contact-oximeter analysis is in the whitepaper.
* **Already low at the start**, exercise, arrhythmia and a resting pulse under 51 bpm break the signal.
* **Every phone:** the app is built to work on every phone and is **not tuned to any specific phone**. The iPhone 13, 17 and 18 Pro the team owns are only a held-out sanity check, never a tuning set. **Untested Android phones and other phones are a limitation.**

![Failure modes](docs/figures/ml_failure_modes.png)

## Prior work: what is and is not new

Phone-camera low-oxygen detection is **not new**: [Hoffman et al., 2022](https://www.nature.com/articles/s41746-022-00665-y) (81% sensitivity and 79% specificity below 90% SpO2, 5.00 SpO2 MAE; a different protocol, so not a head-to-head comparison), [MTHS (Samavati et al., 2022)](https://arxiv.org/abs/2204.08989), [Sun et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7984770/), [Tian et al., 2022](https://doi.org/10.1109/JSTSP.2022.3152352). **What this project adds is limited to** a personalised chain-plus-residual-CNN hybrid on the device with abstention, an open pre-registered leave-one-person-out audit that reports ties and losses, and OpenOximetry-based pretraining and fairness analysis. It is not a first detector, not a new sensing principle and not evidence of superiority.

## Quick start

```bash
npm run serve            # http://localhost:8080 (camera needs HTTPS or localhost)
npm test                 # decision logic, JS-vs-Python parity, HR parity, capture simulation, stall, lens probe, CNN parity
python scripts/check_claims.py   # every number in README, MODEL.md and WHITEPAPER.md against results/ml/*.csv
```

## Repository map

```
app/        static PWA, no build step: engine.js (chain v2), engine_v1.js (v1 chain), cnn.js + cnn_features.js (CNN), data/ (weights, hybrid members), sw.js
python/     oxydrop/ (chain, HR, evaluation kit) and oxydrop_ml/ (data, CNN, SSL, training, baselines, hybrid, tests, PREREGISTRATION.md and amendments)
results/    ml/ (all CSVs behind every number, scores/, CORE_RESULTS.md, BASELINES.md, HYBRID.md), older chain tables
scripts/    reproduce_ml.sh (stages), ml_numbers.py, render_docs.py, make_ml_figs.py, check_claims.py, build_whitepaper_pdf.sh
docs/       figures/, templates/ (sources of the generated docs), ENGINEERED_CHAIN.md (the v1/v2 chain, camera flow, older audit), WRITEUP.md, HR_VALIDATION.md, SPO2_OPTICS.md, PHONE_RETEST.md, REVIEW_RESPONSE.md
tests/      JS parity, flow and e2e tests; docs_claims.mjs guards wording
```

## Reproduce

`bash scripts/reproduce_ml.sh <stage>` (stages in the script header: `baselines`, `core-*`, `hybrid-*`, `docs`). The README, MODEL.md and WHITEPAPER.md are generated from `results/ml/*.csv` with one command: `bash scripts/reproduce_ml.sh docs` (figures, templates, PDF, then `check_claims.py`). Heavy training ran on a Daytona sandbox; checkpoints stay outside git (SHA-256 in `results/ml/ssl_manifest.json`).

## Data and licences

Hoffman (MIT), BUT PPG (CC BY 4.0), MTHS (CC BY-NC-ND 4.0, non-commercial research; weights trained on it are flagged), OpenOximetry (PhysioNet credentialed, derived aggregates only, never redistributed), plus pretraining sources. See [DATA_LICENSES.md](DATA_LICENSES.md). No data from users is stored. Code is MIT. AI tools (Claude) assisted with code, analysis and writing; the team reviewed and can explain every component.

## Older documentation

The v1/v2 logistic chain, the heart-rate validation, the camera flow with its lens probe, the leakage audit and the untested-phone table are in [docs/ENGINEERED_CHAIN.md](docs/ENGINEERED_CHAIN.md). The independent review and how each item was handled: [docs/REVIEW_RESPONSE.md](docs/REVIEW_RESPONSE.md).
