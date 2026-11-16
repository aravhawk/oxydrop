# PhotonDistill2 (round-3 candidate R3-DISTILL+): log

Code: `python/oxydrop_ml/photon/distill2.py` (imports `distill_student.py`, which is unchanged). Scores: `results/photon/scores/distill2*__*.csv`.
Evaluation: `python -m oxydrop_ml.photon.evalkit --only chain_v1,distill,distill2,distill2_s0,distill2_s1,distill2_s2,distill2_null,distill2_d1_null,distill2_d2_null --out results/photon/distill2_eval`
(own output dir; the shared `results/photon/TABLE.*` was not rewritten). Paired differences vs `distill`: `python -m oxydrop_ml.photon.distill2 --compare` ->
`results/photon/distill2_compare.csv` (evalkit's `Boot`, same 2000 resampled people for every model, so the differences are paired).
Per-fold choices and full inner grids: `results/photon/distill2_fold_info.json`; remote run log: `results/photon/distill2_run.log`.
Timeline: seeds launched 19:54 CT, L0 ensemble at 20:08 CT, nulls (3 draws) and all stress levels at 20:16 CT.

## Verdict

At L0 the `distill2` 3-seed ensemble ties chain_v1, where distill lost to it, and its point estimate beats distill's. Under stress it beats chain_v1 at L3, L4, L5 and X1.

| level | distill2 TM [95 % CI] | vs chain_v1 [95 % CI] | better than chain | vs distill [95 % CI] | coverage / answered acc. |
|---|---|---|---|---|---|
| L0 | **0.784 [0.596, 0.834]** | -0.012 [-0.091, +0.012] (tie) | 4/6 | **+0.026 [-0.010, +0.040]** (4/6) | 0.69 / 0.933 |
| L1 | 0.711 [0.574, 0.790] | -0.093 [-0.197, +0.008] (tie) | 2/6 | +0.006 [-0.004, +0.014] | 0.61 / 0.963 |
| L2 | 0.712 [0.557, 0.791] | -0.031 [-0.132, +0.100] (tie) | 2/6 | +0.008 [-0.013, +0.025] | 0.67 / 0.923 |
| L3 | 0.698 [0.555, 0.789] | +0.133 [+0.001, +0.248] (win) | 4/6 | +0.018 [-0.001, +0.033] | 0.58 / 0.918 |
| L4 | 0.695 [0.558, 0.785] | +0.097 [+0.011, +0.195] (win) | 6/6 | +0.014 [-0.008, +0.029] | 0.53 / 0.926 |
| L5 | 0.704 [0.562, 0.779] | +0.170 [+0.027, +0.249] (win) | 5/6 | +0.004 [-0.010, +0.023] | 0.35 / 0.910 |
| X1 | 0.682 [0.547, 0.748] | +0.174 [+0.044, +0.284] (win) | 6/6 | +0.006 [-0.017, +0.023] | 0.40 / 0.861 |

References: chain_v1 L0 0.796 [0.636, 0.857]; distill L0 0.758 [0.603, 0.813] (vs chain -0.038 [-0.103, -0.001], loss).
Seeds at L0: s0 0.772, s1 0.806, s2 0.760; seed mean 0.780; ensemble 0.784. Seed vs distill at L0: s0 +0.014 [-0.019, +0.031], s1 +0.048 [+0.011, +0.078],
s2 +0.002 [-0.054, +0.036]. Worst person at L0: 0.500 (chain_v1 0.458, distill 0.516). SpO2 head MAE (ensemble): 4.3 points at L0 (distill 4.5), 5.2 at X1.
Under stress the ensemble answers more often than distill (coverage L3 0.58 vs 0.49, L5 0.35 vs 0.21, X1 0.40 vs 0.27). At X1 its answered accuracy is
lower (0.861 vs 0.964).

**ROUND3 winner rule (stated 19:55 CT):**
- L0 point estimate above distill's: 0.784 > 0.758 (pass).
- Paired difference vs distill has lower CI bound above -0.02: the bound is -0.010 (pass).
- Mean paired difference vs chain_v1 over L3, L4, L5 and X1 is >= 0: (0.133 + 0.097 + 0.170 + 0.174) / 4 = +0.144 (pass).
- Nulls: mean 0.504 (pass).
- Rule 3: same 39-feature front end and a 2-hidden-layer nn-runtime-v1 MLP (pass).

distill2 therefore meets the conditions to replace distill. It does **not** "beat chain" at L0: the CI includes 0, only 4/6 people are better, and the
worst person (0.500) is above chain's (0.458).

## Null controls

Labels and SpO2 are permuted globally across the training windows of each outer fold, with the same permutation function as distill
(`distill_student.null_perm`). The teachers see the permuted labels as well. The pipeline is the seed-0 pipeline, including the full two-stage nested grid.

| file | TM |
|---|---|
| `distill2_null__L0` (draw 0) | 0.543 |
| `distill2_d1_null__L0` (draw 1) | 0.525 |
| `distill2_d2_null__L0` (draw 2) | 0.444 |
| mean | **0.504** (pass, <= 0.55; every draw <= 0.55) |

No inner tau was certified in any null fold, so the fallback coverage of 0.20 applied there. The student is a per-window MLP and not temporal, so there is no blank control.

## Model (the only thing deployed)

Same form as distill: `flashgap.Prep` on the 39 causal features (`flashgap.COLS`), then 39 -> H -> ReLU -> H -> ReLU -> 3 heads (hyp logit,
(SpO2 - 90) / 5, abstain logit), with answer = sigmoid(abstain) >= tau. H is nested in {16, 32, 64}. Depth stays at 2 hidden layers because
`distill_app_export.graph` (the app format) is a 2-hidden-layer graph, so a depth sweep would have broken rule 3. There is no teacher, gate or chain_v1 at inference.

## What changed vs distill (all selection nested; teachers refit per outer and per inner fold on training people only)

- **Teachers.** `base` is distill's set: T_chain on clean rows and the DC-colour LR on augmented rows. `gbdt` averages each base teacher with a
  gradient-boosted tree model on all 39 features: GBDT_clean is fit on clean rows, and GBDT_all on clean + augmented rows. The GBDT is sklearn
  `HistGradientBoostingClassifier` (150 iterations, lr 0.05, 15 leaves, min leaf 40, l2 1, class balanced). That is the LightGBM histogram algorithm;
  the lightgbm package itself was not installed on the sandbox.
- **KD weight and temperature.** lam in {8, 16, 32, pure KD} and T in {2, 4}. distill used lam in {0.5, 2, 8} and fixed T = 2. "Pure KD" drops the label BCE.
- **Width and weight decay.** H in {16, 32, 64} and weight decay in {1e-4, 1e-2}. Epochs are {15, 30}, as before.
- **Search.** Two stages, both scored on leave-one-inner-person-out OOF over the 5 training people, with distill's criterion
  (0.5 TM clean + 0.5 mean TM of the augmented copies):
  - Stage 1: teacher x lam x T at H 32 and weight decay 1e-4, giving 16 configurations x 2 epoch checkpoints.
  - Stage 2: the stage-1 winner with H x weight decay, giving 5 new configurations x 2 epoch checkpoints.
  - Ties go to fewer epochs, then smaller H, then more decay.
  - tau = `flashgap.select_tau` on the clean inner OOF abstain output.
- Loss, optimiser (Adam lr 3e-3, batch 512), row weights (1 clean / 0.25 augmented), class balancing, augmentation prior (FlashGap, K = 4,
  augmentation seed = student seed) and stress evaluation (held-out people degraded by stress.py seed 0, whole recording, no refit) are as in distill.

Nested choices over the 18 seed folds (full grids in `distill2_fold_info.json`):

| item | counts over 18 folds |
|---|---|
| teacher | base 18, gbdt 0 |
| lam | 8: 2, 16: 7, 32: 9, pure: 0 |
| T | 2: 11, 4: 7 |
| H | 16: 1, 32: 10, 64: 7 |
| weight decay | 1e-4: 16, 1e-2: 2 |
| epochs | 15: 3, 30: 15 |
| tau certified | 17/18 (s0 fold 100003 not certified, fallback coverage 0.20) |

The lam grid edge that distill hit (8 in 18/18 folds) is now inside the grid. Values 16 and 32 were picked most often, and pure KD, the lam -> infinity
end, was never picked, so the optimum is bracketed. The GBDT-augmented teacher never won on the inner criterion. The L0 gain over distill therefore
comes from the larger KD weight, the temperature and the width, not from the new teacher.

## Teachers requested but not used (disclosed)

- **twin_inverse physics path.** It needs the raw per-frame RGB and the optics renderer. Neither is in the 39-feature pack, and recomputing the
  physics inversion per fold on the sandbox did not fit the 20:15 CT deadline.
- **oxyseq / oxytrack outputs as per-window teachers.** The saved score files are LOSO outputs. The prediction for training person p in outer fold k
  came from a model trained on the held-out person of fold k, so using them as teachers would leak. A leak-free version would require refitting oxyseq
  and oxytrack inside every inner fold, which was out of time.

## Export

`results/photon/distill2_weights.json` uses the same format as `distill_weights.json` (`oxydrop-photon-distill-v1`; written by `distill2.export`, which
reuses `distill_export`'s helpers). It contains the 3 students refit on all 6 people with the modal nested recipe of each seed. tau is the median fold tau,
and the deployed tau is the mean of the 3 seed taus, 0.916.

| seed | teacher | lam | T | H | weight decay | epochs | tau |
|---|---|---|---|---|---|---|---|
| s0 | base | 16 | 2 | 32 | 1e-4 | 30 | 0.949 |
| s1 | base | 32 | 2 | 32 | 1e-4 | 30 | 0.914 |
| s2 | base | 32 | 4 | 64 | 1e-4 | 30 | 0.885 |

Seed 2's student has width 64. The graph shapes come from the tensors, so `distill_app_export.app_json` / `forward` accept the file unchanged. This was
checked locally: the app forward reproduces the reference scores of the 2 fixture windows to 2.4e-9, and the answer flags match. Shipping needs only
`distill_app_export.SRC` pointed at `distill2_weights.json`. That file is not this agent's, so it was not edited, and `app/data/distill.json` was not
regenerated. The quoted numbers are the LOSO ones above, never the refit's.

## Compute

Everything ran on one Daytona sandbox (tag R3D2, 4 vCPU, 2 GiB, python:3.12-slim), with up to 4 single-thread fold jobs in parallel at about 2 minutes
per fold, using distill's 13 MB feature pack (`ml_cache/photon_distill/distill_pack.npz`). No OOM kills. A second sandbox for the nulls was refused
(org CPU limit 10). The sandbox was deleted at 20:19 CT. Locally only evalkit and the compare step were run (about 120 MB RSS).

## Disclosures

- The grid design (lam above 8, T, width) was chosen after distill's outer results were seen, in particular the 18/18 edge hit (Amendment 8).
  All selection inside it is nested.
- L1-L5 lie inside FlashGap's augmentation prior, so those numbers are in-simulator and optimistic. X1 is out of family.
- No test-person labels were used in any teacher, selection, tau or refit. Test labels were read only to score.
