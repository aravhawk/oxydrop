# ens3 (round-3 R3-ENSEMBLE): nested stacking ensemble, log

Status: candidate only, stopped by the planner at 20:45 CT (ensemble handed to `ens3_gpu`). What exists: ens3__L0..L5 and ens3__X1 score files, landed 20:25 CT. The members' inner folds were retrained on a Kaggle CPU kernel, not a GPU. There are NO null files. L0 and stress files were finished at 20:25 CT, after the 20:15 CT L0 deadline, and the planner locked round 3 (winner distill2) at 20:26 CT. ens3 cannot replace the shipped model tonight. Not committed.

Code: [ens3.py](python/oxydrop_ml/photon/ens3.py). Inner-OOF member predictions: `results/photon/ens3_inner/*.npz`. Per-fold choices: `results/photon/ens3_fold_info.json`. Evaluation: `python -m oxydrop_ml.photon.evalkit --only chain_v1,distill,ens3,oxyseq,distill_nokd --out results/photon/ens3_eval`. This writes to its own directory; the shared TABLE was not rewritten.

## Results (3-seed members; the combiner is deterministic)

| level | ens3 TM [95 % CI] | vs chain_v1 [95 % CI] | people better vs chain | vs distill [95 % CI], better | vs distill2 [95 % CI] | coverage / answered acc. |
|---|---|---|---|---|---|---|
| L0 | 0.789 [0.621, 0.842] | -0.007 [-0.092, +0.043] tie | 4/6 | +0.031 [-0.012, +0.078], 3/6 | +0.005 [-0.028, +0.047] | 0.61 / 0.915 |
| L1 | 0.743 [0.588, 0.820] | -0.062 [-0.174, +0.043] tie | 3/6 | +0.037 [+0.002, +0.069], 4/6 | +0.031 [-0.002, +0.062] | 0.62 / 0.897 |
| L2 | 0.729 [0.597, 0.818] | -0.013 [-0.115, +0.101] tie | 3/6 | +0.025 [-0.007, +0.066], 4/6 | +0.017 [-0.023, +0.067] | 0.62 / 0.860 |
| L3 | 0.713 [0.540, 0.792] | +0.149 [-0.007, +0.255] tie | 4/6 | +0.033 [-0.034, +0.063], 5/6 | +0.015 [-0.036, +0.040] | 0.54 / 0.885 |
| L4 | 0.683 [0.573, 0.774] | +0.085 [+0.003, +0.193] win | 6/6 | +0.002 [-0.067, +0.062], 3/6 | -0.012 [-0.090, +0.048] | 0.51 / 0.890 |
| L5 | 0.675 [0.588, 0.764] | +0.141 [+0.043, +0.248] win | 5/6 | -0.025 [-0.119, +0.043], 3/6 | -0.029 [-0.141, +0.047] | 0.48 / 0.844 |
| X1 | 0.704 [0.589, 0.779] | +0.197 [+0.074, +0.280] win | 6/6 | +0.028 [-0.030, +0.068], 4/6 | +0.022 [-0.022, +0.058] | 0.49 / 0.861 |

chain_v1 L0: 0.796 [0.636, 0.857]. distill L0: 0.758 [0.603, 0.813]. Worst person at L0: ens3 0.585, chain_v1 0.458, distill 0.516.
The mean paired difference vs chain_v1 over L3-L5 and X1 is +0.143.
The CIs vs chain_v1 come from evalkit. The CIs vs distill and distill2 come from `oxytrack.paired_vs`, a paired person bootstrap with B = 2000 on the same windows.
Under the ROUND3 rule, ens3 would have qualified on numbers. Its L0 point estimate of 0.789 is above 0.758, the lower CI bound vs distill is -0.012 (above -0.02), and the stress mean vs chain is >= 0. Two things still rule it out: it finished late, and it fails rule 3 (swappability, see below). It ties distill2 at L0 (+0.005 [-0.028, +0.047]). It does not beat chain_v1.

## Null controls: NOT available

ens3 has no null score files (`ens3_null__L0` etc. were not written), so it has not passed the round-3 null requirement. The null-pipeline kernel `aravhawk/ens3-r3-null` finished on Kaggle at about 20:42 CT. The planner stopped work at 20:45 CT before its outputs were pulled or combined, so they sit unprocessed on Kaggle. Any null claim for ens3 must come from the `ens3_gpu` takeover.

## Members (pool, nested choice of top 3)

The pool is the trained round-2 models whose nulls pass and whose inner folds can be recomputed from data on disk:
- `distill`: KD student, 3 seeds. Outer predictions come from the shipped `distill__*.csv`.
- `oxyseq`: GRU. Outer predictions come from `oxyseq__*.csv`, a 3-seed ensemble. Inner predictions use seed 0, as in oxyseq's own nested selection.
- `distill_nokd`: label-only student, seed 0.

Excluded models:
- photonnet (null 0.618) and oxyssl (null 0.643) fail their nulls.
- distill_v1 is a superseded variant of distill.
- Round-3 models (distill2, multida, ssl3, photonnet_xl) had no inner folds on disk before the deadline. distill2 would be the obvious first member.

With a pool of 3 and K = 3, the nested "top 3 by inner-OOF TM" step keeps every member in all 6 folds. Only the member ordering is fold-dependent.

## Nesting (what the combiner of held-out person p sees)

For each outer person p, the job retrained every member on Kaggle CPU kernel `aravhawk/ens3-r3-real`. Each inner person ip of p's 5 training people was predicted by a member trained on the other 4 people.
- distill and distill_nokd used the lam / wd / epochs that their own nested selection had chosen for outer fold p, read from the stored fold pickles. Teachers were refit on the 4 people.
- oxyseq re-ran its own inner selection of steps. The inner TM traces reproduced its original run exactly (0.770, 0.858, 0.800, 0.799, 0.755, 0.723).

The following were all fit on these inner-OOF clean predictions only:
- member ranking;
- combiner C from {0.01, 0.1, 1}, chosen by leave-one-inner-person-out of the combiner;
- abstain tau, from `flashgap.select_tau` on the combiner's leave-one-inner-person-out predictions, with conf = |p - 0.5|.

No outer-OOF score of another person was used to fit anything. The held-out person's outer member predictions are only passed through the fitted combiner.

The combiner is a class-balanced logistic regression on standardised [logit(score), logit(conf)] of each member. distill's conf is its abstain-head probability; oxyseq's conf is its quality probability. The combiner is fit once per fold on clean inner OOF and applied unchanged to L0-L5 and X1.

## Per-fold combiner (coefficients on raw logit inputs)

| held-out | member rank order | C | inner LOPO TM | tau (cert, inner cov) | weights | intercept | inner 1-d TM per member |
|---|---|---|---|---|---|---|---|
| 100001 | oxyseq, distill, distill_nokd | 0.01 | 0.730 | 0.284 (cert, 0.63) | oxyseq logit +0.43, conf -0.28; distill logit +0.68, conf +0.02; nokd logit +0.10, conf +0.15 | -1.01 | distill 0.743, oxyseq 0.764, nokd 0.692 |
| 100002 | oxyseq, distill, distill_nokd | 0.1 | 0.802 | 0.287 (cert, 0.75) | oxyseq +1.15 / -0.83; distill +0.52 / -0.46; nokd +0.20 / +0.69 | -0.42 | 0.742, 0.858, 0.677 |
| 100003 | oxyseq, distill, distill_nokd | 1.0 | 0.750 | 0.496 (cert, 0.26) | oxyseq -0.09 / +0.54; distill +2.20 / -0.60; nokd -0.20 / +1.48 | -3.80 | 0.747, 0.759, 0.663 |
| 100004 | distill, oxyseq, distill_nokd | 1.0 | 0.756 | 0.405 (cert, 0.65) | distill +2.09 / -0.14; oxyseq -0.21 / +0.67; nokd -0.07 / +0.11 | -1.40 | 0.790, 0.768, 0.720 |
| 100005 | oxyseq, distill, distill_nokd | 0.01 | 0.629 | 0.293 (cert, 0.68) | oxyseq +0.28 / +0.01; distill +0.68 / -0.11; nokd +0.15 / +0.42 | -1.46 | 0.678, 0.732, 0.644 |
| 100006 | oxyseq, distill, distill_nokd | 1.0 | 0.729 | 0.354 (cert, 0.66) | oxyseq -0.16 / -0.02; distill +2.59 / -0.28; nokd -0.24 / +1.08 | -1.96 | 0.707, 0.720, 0.608 |

distill's logit carries most of the weight in every fold. oxyseq's logit gets a positive weight in 3 of the 6 folds.

## Disclosures

- The combiner was designed after the round-1 and round-2 outer results were seen (Amendment 8).
- Before the real run, a code smoke test fed the combiner the members' OUTER-OOF scores of the other 5 people as a stand-in for inner OOF. That input is leaky and not allowed; the test only checked the code and its printed mock numbers were never used. Its mock L0 was 0.717. The design (features, grid, K) was not changed after it.
- distill's inner OOF is computed at the config that its own selection picked on those same inner folds. That is mildly optimistic for the distill input to the combiner.
- For oxyseq, the inner predictions come from seed 0 while the outer predictions are the 3-seed ensemble. The combiner therefore sees a slightly noisier oxyseq at fit time than at test time.
- The combiner is fit on clean inner OOF only. No augmentation enters the combiner.
- Compute: Daytona was full (CPU 10/10, disk 30/30 GiB). A 1 GiB sandbox could not install torch and was deleted at 19:57 CT. Kaggle CPU sessions were full until 20:17 CT. That delay is why the files are late. Kaggle (private, Hoffman-derived features only): dataset `aravhawk/ens3-r3-data`, kernels `aravhawk/ens3-r3-real` and `aravhawk/ens3-r3-null`.

## Browser feasibility (ROUND3 rule 3)

- distill and distill_nokd already run in JS. Both use the 39-feature front end (app/js/distill_features.js) and app/js/nn_runtime.js through app/js/distill_engine.js; distill_nokd only needs its own weights.
- oxyseq has a JSON export (results/photon/oxyseq_model.json) and its GRU op exists in nn_runtime.js. Its front end has no JS port: per-window chain FUSION deltas / sd with clipping, log SNR inputs, and the causal gated-EMA test-time adaptation. Its 19 deltas overlap the ones distill_features.js already computes. The estimated port is about 1-2 hours plus a parity test.
- The combiner is a 6-input logistic regression per fold, a few lines of JS.
- So ens3 is NOT swappable today under rule 3. A distill-family-only ensemble (distill, distill2, distill_nokd) would be swappable; it was not built.
