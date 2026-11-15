<!-- GENERATED from docs/templates/WRITEUP.md.in by scripts/render_docs.py. Edit the template, not this file. -->
# OxyDrop: a phone-camera check for a drop in blood oxygen

**Track:** Health
**Team:** Arav Jain, [TEAMMATE 2 NAME], [TEAMMATE 3 NAME]  *(Arav: fill in or delete before submitting; see CHECKLIST.md)*
**Live app:** https://aravhawk.github.io/oxydrop/ · **Code:** https://github.com/aravhawk/oxydrop · **Paper-style write-up:** [WHITEPAPER.md](../WHITEPAPER.md) · **Model card:** [MODEL.md](../MODEL.md)

> **This project is a prototype and is not medical advice.**

## The problem
Low blood oxygen (SpO2 below 90%) is easy to miss because it often feels like nothing, and the people most exposed to it (people with COVID-19, COPD or asthma at home, hikers and skiers at altitude, low-resource clinics) are often far from a pulse oximeter. A phone is already in their pocket. Our design scenario is **a person resting, without a pulse oximeter, who wants to know "should I re-check with a real oximeter, or go down?"** This is a scenario, not evidence: the app has not been tested at altitude, on patients, or on a live phone by anyone outside the team.

## What we built
A web app (works in a phone browser, nothing to install, video never leaves the phone). The user sits at rest and covers the rear camera and flash with a fingertip. After a 60-second calibration of *their own* normal (the first answer arrives at 70 seconds), every 2 seconds it says **Normal vs your baseline**, **Possible low oxygen: re-check with a pulse oximeter**, or **Unsure, re-measure**, and shows heart rate and a quality score. It does not output an SpO2 number, and because it compares you with your own first minute it cannot know whether you were already low at the start (oxygen status at start unknown). A "recorded demo" mode replays real study recordings so anyone can try it without a camera.

![From video to three states](figures/ml_flow.png)

## How it works
**Model.** A hybrid: the v1 logistic chain (heart-rate rise, pulse shape and ratio-of-ratios features relative to your first minute) gives a low-oxygen logit, and a 167,588-parameter 1D CNN (three seeds, plain JavaScript, offline) learns a residual correction. The CNN was pretrained self-supervised on 299,754 unlabeled windows and warmed up on contact-oximeter data (OpenOximetry) before fine-tuning on phone video. It also outputs heart rate and a quality score. Details and every lever, kept or dropped: [MODEL.md](../MODEL.md).

![Architecture](figures/ml_architecture.png)

**Testing.** Leave-one-person-out: every result is scored on a person the model never saw, with nested selection on the training people only, three seeds and 95% person-bootstrap intervals, under a protocol committed before any CNN result existed (commit `3e708fe`). A "win" requires the paired interval against the **best** baseline to exclude zero.

## Results (all leave-one-person-out)
**Who it is for, input to output, baseline beaten on held-out data, and where it fails:**

* **Primary pre-registered metric (P1, six Hoffman people, time-matched balanced accuracy): not beaten.** The hybrid scores 0.779 [0.638, 0.831] against 0.796 [0.635, 0.856] for the best baseline (the shipped logistic chain): a tie (-0.017 [-0.103, 0.069]). The CNN alone ties too, and a chain refit with the same levers and no CNN scores 0.816. The hybrid and CNN are clearly better than the clock-only baseline (0.495) and the ratio-of-ratios rules (0.535), which are weak baselines.
* **Heart rate (P2, 100 people): a win with a caveat.** 0.480 [0.429, 0.534] within 5 bpm against 0.419 [0.366, 0.481] for the best baseline (+0.061 [0.013, 0.107]); the rule was chosen after the pooled table had been seen.
* **Absolute SpO2 (P3, MTHS):** no model beats the training mean (1.457 CNN against 1.354); MTHS is normoxic, so SpO2 is not shown to users.
* **What drives the CNN:** the OpenOximetry warm-up (P1 0.577 without it against 0.739 with it). A device-agnostic-augmentation benefit is not supported.
* **Where it fails:** only six hypoxic phone-camera people exist in open data, so intervals are wide; motion, cold fingers, saturated red channel and far-flash phones produce *Unsure*; skin-tone fairness cannot be shown on phones; low oxygen already present during the first minute is invisible; person 4's 43 bpm resting pulse is below the heart-rate floor of the v2 estimator. The app is built to work on every phone and is not tuned to any specific phone; untested Android and other phones are a limitation.

**PENDING (generated blocks, filled automatically when the files exist):** conformal 3-state output and matched-coverage comparison; OpenOximetry SpO2 by skin tone; heart-rate voting. **PENDING:** the app in this tree still runs the logistic chain v2 (the hybrid export is ready, wiring is the app-integration branch).

The v1/v2 chain tables, the heart-rate validation on BUT PPG and the earlier audit are in [ENGINEERED_CHAIN.md](ENGINEERED_CHAIN.md).

## Prior work, and what we added
Phone-camera low-oxygen detection is **not new**. Hoffman et al. (2022) demonstrated fingertip camera and flash hypoxemia screening with induced desaturation and held-out testing (81% sensitivity, 79% specificity below 90% SpO2, 5.00-point SpO2 MAE; their protocol differs from ours, so this is not a direct comparison). MTHS (Samavati et al., 2022), Sun et al. (2021) and Tian et al. (2022) studied smartphone and camera vital signs and oximetry with ratios, calibration and heart-rate-adaptive filtering. We add: a personalised chain-plus-residual-CNN hybrid on the device with abstention, an open pre-registered leave-one-person-out audit that reports ties and losses, and OpenOximetry-based pretraining and fairness analysis. We do not claim a new sensing principle, a first detector, or superiority over earlier work.

## What we'd do next
1. Collect consented phone-camera data from a broader, deliberately diverse group in a safe, supervised hypoxia protocol (altitude or clinical breath-holding studies), including many phone models, to test fairness and per-phone behaviour.
2. Replace the first-minute baseline assumption with an independently established baseline, and add a motion and exertion detector so the app refuses to answer after activity.

## Sources
- Hoffman et al., "Smartphone camera oximetry in an induced hypoxemia study", npj Digital Medicine 2022; data https://github.com/ubicomplab/oximetry-phone-cam-data (MIT)
- BUT PPG v2.0, PhysioNet (CC BY 4.0): https://physionet.org/content/butppg/2.0.0/
- MTHS (CC BY-NC-ND 4.0, non-commercial research), arXiv:2204.08989
- OpenOximetry (PhysioNet, credentialed; derived aggregates only, never redistributed); moshekam PPG-during-exercise data (MIT); HealthRing (experiment, no gain). Full list and licences: https://github.com/aravhawk/oxydrop/blob/main/DATA_LICENSES.md
- Libraries: numpy, scipy, scikit-learn, PyTorch, XGBoost, LightGBM, matplotlib, Node.js.
- AI tools: Claude (Anthropic) was used for code, analysis and drafting. The team reviewed the work and can explain every part.
