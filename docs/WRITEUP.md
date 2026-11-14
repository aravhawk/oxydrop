# OxyDrop: a phone-camera check for a drop in blood oxygen

**Track:** Health
**Team:** Arav Jain, [TEAMMATE 2 NAME], [TEAMMATE 3 NAME]  *(Arav: fill in or delete before submitting; see CHECKLIST.md)*
**Live app:** https://aravhawk.github.io/oxydrop/ · **Code:** https://github.com/aravhawk/oxydrop

> **This project is a prototype and is not medical advice.**

## The problem
Low blood oxygen (SpO₂ below 90%) is easy to miss because it often feels like nothing, and the people most exposed to it are often far from a pulse oximeter: a hiker or skier at 3,000 m, someone recovering at home, a parent at night. Our user is **a hiker resting in a lodge at altitude without a pulse oximeter**, who wants to know "should I re-check with a real oximeter, or go down?" A phone is already in their pocket.

## What we built
A web app (works in a phone browser, nothing to install, video never leaves the phone). The user sits at rest and covers the rear camera and flash with a fingertip. After a 60-second calibration of *their own* normal, every 2 seconds it says **normal**, **possible low-oxygen episode**, or **unsure, re-measure**, and shows heart rate, signal quality and a motion check. It does not output an SpO₂ number. A "recorded demo" mode replays real study recordings so anyone can try it without a camera.

## What we added
Our own idea. We also built the tool end to end: a JavaScript port of the model (numerically identical to the Python version on all 5,646 test windows, max difference 3×10⁻¹³) so the model runs live in the phone browser.

## How it works
**Data.** Only one open dataset has phone-camera video of people whose SpO₂ really fell below 90%: UW Hoffman 2022 (6 people, 12 hands, MIT licence). We searched 32 candidates; the rest are normoxic, gated, or have no optical signal. With so little hypoxemia data, a big neural network overfits, so we built a **chain of small stages** where the data-rich parts of the problem do the heavy lifting.
**Model.** Mean colour per camera frame → causal band-pass filter → beat detection (heart rate, variability, breathing rate, pulse strength) and light-absorption features → subtract the person's own first-minute median → two logistic regressions averaged (62 numbers in total) → decision, or "unsure" when the probability is near 0.5. Heart-rate extraction was checked on a separate smartphone-PPG dataset with ECG (BUT PPG), and the "heart rate rises when oxygen falls" link was reproduced on a separate 20-person altitude-chamber dataset (zero-shot AUROC 0.88). These datasets guided and validated the design; only the final 62 numbers are fit, on Hoffman.
**Testing.** Leave-one-person-out: the model is always scored on a person it never saw (both hands held out), with every setting chosen on the other five people. Calibration uses only the person's first 60 s. We audited for leakage and found and fixed one bug in our own code (full audit in the repo).

## Results
Baseline = a clock-only model that knows only how many seconds have passed (the study lowers oxygen steadily, so time alone predicts the label).

| | Accuracy | Missed hypoxemia (false negatives) | False alarms (false positives) | Accuracy with time controlled |
|---|---|---|---|---|
| Clock-only baseline | 83.5% | 13.6% | 20.3% | **51.9%** |
| **OxyDrop, answers every window** | **87.0%** (95% CI 83–91) | **12.1%** | **14.2%** | **79.5%** |
| **OxyDrop, may say "unsure"** (answers 85% of windows) | **92.5%** | **6.6%** | **8.7%** | 85.2% |

- The raw gain over the clock (+3.7 points) is not statistically resolved with 6 people (CI −7 to +16); the time-controlled score (79.5% vs 51.9%) and the abstention result are the solid claims. 90% accuracy on every window was **not** reached.
- **Where it fails.** Person 4 (20-year-old man, resting heart rate 43 bpm) is the worst case at 77.5%: on one hand the model misses hypoxemia for 107 windows, on the other it raises 79 false alarms. We have not established why; one untested hypothesis is that his very low resting heart rate (43 bpm) makes the heart-rate-rise signal unstable. Exercise is the biggest risk: the same rule flags **63%** of normoxic exercise windows as low-oxygen (68% counting only people with a usable heart rate), so the app requires seated rest and pauses when the phone moves.
- **Groups.** The 6 participants are adults aged 20–34, 3 women, 3 men, 1 Black participant (accuracy 88.6%, in line with the others, range 77.5–91.7%). One person per group cannot show fairness; the model is untested on children, older adults and darker skin tones beyond that one person.
- **Not tested:** any phone other than the study's Nexus 6P. iPhone Safari cannot control the flash, so the live camera path there is unvalidated.

## What we'd do next
1. Collect consented phone-camera data from a broader, deliberately diverse group in a safe, supervised hypoxia protocol (e.g. altitude or clinical breath-holding studies), including phones other than the study phone, to test fairness and calibrate per phone.
2. Add a motion/exertion detector trained on exercise data so the app refuses to answer after activity, rather than relying on the user to rest.

## Sources
- Hoffman et al., "Smartphone Camera Oximetry in an Induced Hypoxemia Study", npj Digital Medicine 2022; data https://github.com/ubicomplab/oximetry-phone-cam-data (MIT)
- BUT PPG v2.0, PhysioNet (CC BY 4.0), for heart-rate validation: https://physionet.org/content/butppg/2.0.0/
- 20-person altitude-chamber pilot (figshare, 2025); moshekam PPG-during-exercise data https://github.com/moshekam (MIT); MTHS and HealthRing (experiments with no gain). Full list and licences: https://github.com/aravhawk/oxydrop/blob/main/DATA_LICENSES.md
- Libraries: numpy, scipy, scikit-learn, matplotlib, Node.js.
- AI tools: Claude (Anthropic) was used for code, analysis and drafting. The team reviewed the work and can explain every part.
