# OxySSL (FEAT-010, round-2 candidate R2C) log

Code: `python/oxydrop_ml/photon/oxyssl.py` (docstring has the full protocol), `python/oxydrop_ml/photon/oxyssl_export.py`, tests `python/oxydrop_ml/tests/test_photon_oxyssl.py` (6 passed on Daytona).
Run log: `results/photon/oxyssl_run.txt`. Nested choices per fold: `results/photon/oxyssl_raw/<variant>_selection.json`. Scores: `results/photon/scores/oxyssl*__*.csv`. Numbers below are from `python -m oxydrop_ml.photon.evalkit` (TABLE.csv, 19:27 CT).

## Verdict
OxySSL **loses to chain_v1 at L0** and **fails the null control**, so under the round-2 winner rule it is not eligible to beat chain.
- L0, 3-seed ensemble: TM bal. acc. 0.667 [0.498, 0.741] vs chain_v1 0.796. Paired diff -0.130 [-0.274, -0.044] (loss), 1/6 people improved, worst person 0.400.
- Seeds: s0 0.661, s1 0.697, s2 0.651 (seed mean 0.670). Coverage of the model's own `answer` at L0: 0.80; answered bal. acc. 0.70. SpO2 MAE 7.1 points.
- Nulls (labels permuted globally across training windows, same nested pipeline): 0.654, 0.660, 0.614, **mean 0.643 > 0.55 (FLAG)**. The inner-CV objective of every null fold is about 0.50 (0.500-0.512), as expected with permuted labels. The outer score is not.

## Why the null is above chance (diagnosis, not a fix)
The trainable heads start from the stage-B heads (trained on OpenOx contact PPG, no Hoffman person). The hypoxia head already weights the baseline-relative side features (d_dc_R/G/B, d_ror). Those are the strongest per-person cues on Hoffman (dc_R -0.84, dc_B -0.83).
Training on permuted labels with a balanced class weight moves the weights little: the nested choice is mostly 2 epochs at lr 1e-3. It does re-centre the bias, and the per-fold side standardisation (label-free) rescales the pretrained side weights. So a "null" model keeps a pretrained, label-free ranking of hypoxia and gets a calibrated threshold.
- The pretrained net with no Hoffman training and unit side scale (`oxyssl_init`) scores TM 0.506 and AUROC 0.561 at L0. Its threshold is wrong (1.00 coverage, one class almost everywhere).
- The null's AUROC is 0.782, which is higher than the real model's 0.735.

No Hoffman label reaches the null's training. The permuted-label fit plus the pretrained prior is about as good as the true-label fit, so fine-tuning on Hoffman labels adds little here (+0.02 TM at L0 over the null mean).

A null that removes the pretrained prior (re-initialised heads) would be a different pipeline. It was not run.

## Stress (3-seed ensemble, held-out person degraded, not refit)
| level | OxySSL TM [95% CI] | chain_v1 | paired diff [CI] | people improved | coverage |
|---|---|---|---|---|---|
| L0 | 0.667 [0.498, 0.741] | 0.796 | -0.130 [-0.274, -0.044] loss | 1/6 | 0.80 |
| L1 | 0.670 [0.499, 0.739] | 0.805 | -0.135 [-0.289, -0.048] loss | 1/6 | 0.78 |
| L2 | 0.706 [0.588, 0.737] | 0.742 | -0.037 [-0.136, 0.037] tie | 1/6 | 0.72 |
| L3 | 0.697 [0.596, 0.733] | 0.565 | +0.132 [0.024, 0.171] | 1/6 | 0.59 |
| L4 | 0.548 [0.503, 0.601] | 0.598 | -0.050 [-0.123, 0.045] tie | 3/6 | 0.10 |
| L5 | 0.522 [0.500, 0.552] | 0.535 | -0.013 [-0.098, 0.068] tie | 2/5 | 0.14 |
| X1 | 0.686 [0.567, 0.732] | 0.508 | +0.178 [0.043, 0.214] | 5/6 | 0.51 |

Abstention responds to the degradation: the quality head stops answering at L4/L5 (10-14 % coverage). The L3 and X1 gains over chain are real paired differences. Because the null fails, they should be read as "the pretrained prior on baseline-relative colour survives stress", not as learned from Hoffman labels.

Ablation `oxyssl_noaug` (no distant-flash copies, seed 0): L0 0.652, L3 0.701, L4 0.502, L5 0.500, X1 0.672. Augmentation helps at L4 (0.557 vs 0.502 for seed 0) and keeps the model answering under stress.

## Method as run
- Init: `ml_cache/ckpt/stageB_full.pt` (SimCLR `ssl_full.pt` + OpenOx-only supervised stage B). It was reused, not re-pretrained. `assert_pretrain_disjoint` passed: there are 346,298 pretraining rows and no Hoffman subject; stage B `p_ds == {'openox': 1.0}`; no OpenOx subject is a Hoffman person.
- **Frozen encoder (disclosure).** OxyNet.encode (slot conv, stem, 8 residual blocks, avg+max pool, 160 features) stays at its pretrained weights in eval mode. Trained per fold, end to end, from the pretrained head weights: fc_emb (160->64), the HR head (CE, Gaussian soft labels), SpO2 mean + log-variance (Gaussian NLL), the hypoxia head on (emb - emb_base, side) (BCE, balanced pos_weight) and the quality head (BCE on "the hypoxia head is currently right", detached). Loss weights are 1 / 0.5 / 0.3 / 0.3 (low / SpO2 / HR / quality). The encoder was frozen because Daytona capacity was full when the run was planned (see compute). The exported network is still a single full OxyNet.
- Inputs: `features.windows_to_tensor` on each (degraded or augmented) recording itself, with the label-free first-minute baseline recomputed on that version. emb_base is the mean embedding of the recording's first 8 baseline windows. Side features are scaled by the training-people std (scale only, folded into the weights at export).
- Augmentation (own stream): K=3 copies per training recording drawn from the FlashGap continuous prior (`flashgap.sample_params` / `optics_gain` / `flash_augment`). The RNG is `SeedSequence([0x4F585353 'OXSS', 0, rec_idx, k])`. No `stress.degrade` call, no (level, seed) pair, X1 is never imported. The drawn parameters are checked against the stress levels' (d_rho, leak) pairs and saved in ml_cache/photon_oxyssl/aug_params_seed0.json. The same 3 copies serve every seed.
- Nesting, per outer fold: inner leave-one-person-out over the 5 training people. It selects LR in {3e-3, 1e-3} x epochs in {2, 4, 8, 16} on 0.5 x TM(clean inner OOF) + 0.5 x TM(augmented inner OOF). Ties go to fewer epochs, then the higher LR. The abstain threshold on sigmoid(quality) is the inner-OOF quantile q in {0, .1, .2, .3} with the best answered bal. acc. (ties go to higher coverage). pred = score > 0.5. Final fit on the 5 people. The held-out person is never used for a choice.
- Choices, seed 0 (lr, epochs, q) per held-out person 100001..100006: (1e-3,2,.3) (3e-3,2,.3) (3e-3,16,.1) (1e-3,2,.3) (1e-3,2,.3) (1e-3,2,.3). Seeds 1 and 2 are nearly identical (see the selection JSONs). 2 epochs is the smallest grid value and was chosen in 15 of the 18 fold-seed fits, so the grid edge binds: the heads overfit after very little training.
- Seeds 0, 1, 2 change head dropout and batch order. Ensemble (`oxyssl__*`): mean score, mean quality; answer = mean quality >= mean of that fold's per-seed nested thresholds. The 19:10 provisional `oxyssl__*` was a copy of seed 0, overwritten at 19:21 by the ensemble.
- Diagnostic `oxyssl_init__*` is the pretrained net with no Hoffman training (not a candidate).
- Export: `oxyssl_export.py` fits a deployment top on all 6 people (modal choice lr 1e-3, 2 epochs; abstain threshold = median nested threshold 0.543). It writes `results/photon/oxyssl_oxynet.json` (export_js `oxynet-v1` graph that app/js/cnn.js reads, 896 KB, plus an `oxyssl` block with the decision and abstain rule) and `results/photon/oxyssl_ref.json` (2 windows of tests/fixtures/ppg-csv/Left/100001.csv: x, side, emb_base, torch outputs; the frozen-encoder top and the assembled OxyNet agree to 1e-4). A JS parity run in node was not done.

## Compute
- The local box only ran feature prep (~90 s, 536 MB RSS) and a seed-0 fit until the planner stopped it at 18:58 (OOM risk). That fit was killed after 2 of 6 folds, and no file from it was used.
- Daytona D10 (`efc17856`, 2 vCPU / 2 GiB / 3 GiB) was created at 19:05 by `ml_cache/dt/d10_go.py`, a watcher that waited for capacity. The free tier was full before that: memory 10/10 GiB, then the CPU limit, then the disk limit. The watcher uses `scripts/ml_daytona_put.py uplite`, created by FEAT-009 and reused unchanged.
- Upload bundle: Hoffman shard (no tensor caches), stageB_full.pt, pretrain meta, OpenOx subject list, windows.csv, optics_data, 1 fixture CSV, the test file. The sandbox ran prep, seed 0, 3 nulls, seeds 1-2, ensemble, export, pytest, noaug and init, all done by 19:21.
- Results were pulled with `ml_daytona.py pull`, and the sandbox was deleted with `ml_daytona.py down --tag D10` (state DESTROYING at 19:23).
- No OpenOx-derived data or checkpoint went to Kaggle.

## Disclosures
- Round-2 design after round-1 outer results were seen (Amendment 7, planner).
- Encoder frozen (compute). This is a reduced version of the FEAT-010 description, which asked for full fine-tuning.
- Hoffman RGB for degradation is exp(series) - 1 from the float32 shard (the 30 Hz shard equals the raw 30 fps CSVs up to float32 rounding). chain_v1 used the raw CSVs.
- Stress seed 0 for every level, as in baselines_stress.py.
- `oxyssl_init` uses unit side scale, while the trained variants standardise the side features. It therefore understates the prior the trained heads start from.
