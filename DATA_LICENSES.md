# Data sources and licences

| Dataset | Used for | Licence | Redistributed here? |
|---|---|---|---|
| **UW Hoffman 2022**, "Smartphone Camera Oximetry in an Induced Hypoxemia Study" (npj Digital Medicine 2022), https://github.com/ubicomplab/oximetry-phone-cam-data | The only open phone-camera data with real SpO2 < 90. Training and every reported result (6 people, 12 hands). | **MIT**, Copyright (c) 2020 UW Ubicomp Lab | Yes: two recordings' per-frame mean RGB as parity fixtures (`tests/fixtures/`), two recordings as the in-app replay demo (`app/data/replay/`), and per-window scores. MIT notice retained. |
| **BUT PPG v2.0** (Brno University of Technology smartphone PPG database), PhysioNet | Validating the heart-rate extractor against ECG (finger at rest: MAE 11.1 bpm; 4.5 bpm on expert-good clips). Not used to fit the shipped weights. | **CC BY 4.0** | No |
| **Altitude-chamber pilot (2025)**, 20 people, 1 Hz HR and SpO2, no optical signal (figshare) | Reproducing the "heart rate rises as SpO2 falls" link (zero-shot AUROC 0.88 on Hoffman windows). Not used to fit the shipped weights. | Licence not recorded in our notes; confirm at the source before reuse | No |
| **moshekam PPG exercise** (GitHub, Matlab/text files) | Exercise false-alarm test on normoxic people (63% of exercise windows flagged). Evaluation only. | **MIT**, Copyright (c) 2021 moshekam | No (only the per-window false-alarm counts, `results/scores/exercise_fp.csv`) |
| **MTHS** (smartphone PPG with SpO2, normoxic) | Pooling and synthetic-desaturation experiments that gave no gain. | **CC BY-NC-ND 4.0** | No |
| **HealthRing** (Zenodo, red/IR contact ring) | Source-domain experiment that was dropped (direction did not transfer). | **CC BY** | No |

All datasets are public. No real patient records were used. The model does not train on, and the app does not store, any data from its users: camera frames are averaged in the browser and discarded.
