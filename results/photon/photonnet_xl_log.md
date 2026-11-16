# PhotonNet-XL (round-3 candidate R3-PHOTONNET-XL) log

Code: `python/oxydrop_ml/photon/photonnet_xl.py` (model, nested sweep, nulls, ensemble), `python/oxydrop_ml/photon/photonnet_xl_export.py` (nn-runtime-v1 export
+ reference). `photonnet.py` is imported, not edited. Designed 19:52-19:58 CT on Sun Nov 15 2026, after the round-1 and round-2 outer results were seen
(Amendment 8). One run of the configuration below; nothing was changed after any outer score was seen.

**Timeline:** the L0 score files and the plain nulls were on disk at 20:17 CT, two minutes after the 20:15 CT pick deadline (ROUND3 rule 4). Stress levels
L1-L5 and X1 were written by the same run (20:17 CT), the physics-scrambled null at 20:19 CT. Being late does not change the outcome: the model loses to distill
at L0 (below).

## Verdict

**photonnet_xl does not replace distill and does not beat chain_v1.** At L0 it loses to both. Under the simulated distant flash it ties distill and beats chain_v1 at L5
and X1. The plain null passes (0.412), so it is eligible, but it fails the round-3 replacement rule at the first condition (L0 0.660 < 0.758).

| level | TM bal. acc. [95 % CI] | diff vs chain_v1 [95 % CI] | people better (vs chain) | diff vs distill [95 % CI] | coverage | answered acc |
|---|---|---|---|---|---|---|
| L0 | 0.660 [0.545, 0.762] | -0.136 [-0.210, -0.033] loss | 2/6 | -0.098 [-0.161, -0.011] loss | 0.92 | 0.795 |
| L1 | 0.667 [0.541, 0.767] | -0.138 [-0.237, -0.021] loss | 2/6 | -0.039 [-0.072, +0.001] tie | 0.91 | 0.812 |
| L2 | 0.662 [0.546, 0.754] | -0.080 [-0.176, +0.069] tie | 3/6 | -0.042 [-0.067, +0.014] tie | 0.91 | 0.787 |
| L3 | 0.680 [0.520, 0.772] | +0.115 [-0.028, +0.229] tie | 3/6 | -0.000 [-0.040, +0.036] tie | 0.86 | 0.844 |
| L4 | 0.684 [0.522, 0.794] | +0.086 [-0.020, +0.207] tie | 5/6 | +0.004 [-0.046, +0.040] tie | 0.92 | 0.815 |
| L5 | 0.704 [0.563, 0.796] | +0.169 [+0.018, +0.270] win | 5/6 | +0.003 [-0.037, +0.037] tie | 0.87 | 0.814 |
| X1 | 0.691 [0.597, 0.746] | +0.184 [+0.072, +0.290] win | 6/6 | +0.015 [-0.033, +0.064] tie | 0.82 | 0.783 |

References: chain_v1 L0 0.796 [0.636, 0.857]; distill L0 0.758 [0.603, 0.813]; round-2 photonnet L0 0.668.
Stress mean paired diff vs chain_v1 over L3, L4, L5, X1: +0.139. Worst person at L0: 0.50 (100004); chain_v1's worst is 0.458.
Seeds at L0: s0 0.639 [0.524, 0.723], s1 0.671 [0.501, 0.778], s2 0.668 [0.563, 0.767]; seed mean 0.659, ensemble 0.660. SpO2 MAE 5.99 points at L0.
Sources: `results/photon/photonnet_xl_eval/TABLE.csv` (evalkit, base chain_v1) and `results/photon/photonnet_xl_eval_vs_distill/TABLE.csv` (evalkit's code
with `BASE = 'distill'`, same 2000-resample person bootstrap). Neither run rewrote the shared `results/photon/TABLE.*`.

Scaling up did not help. The larger encoder (15.4 k inference parameters vs 4.9 k) and the sweep land at the same clean-light level as round-2 PhotonNet
(0.660 vs 0.668). The inner CV predicted this: the nested inner TM on clean validation people is 0.63-0.74 (table below). The abstain head is weak. Three of
the 18 nested c* came out 0 (answer everything), so the ensemble answers 92 % of the windows at 0.795 answered accuracy. distill answers 61 % at 0.922.

## Null controls (whole nested pipeline including the sweep, seed 0, 3 draws, labels and sat_b permuted globally across each outer fold's training windows)

| control | draw 0 | draw 1 | draw 2 | mean | rule (<= 0.55) |
|---|---|---|---|---|---|
| `photonnet_xl_null` (plain, physics decoder intact) | 0.391 | 0.445 | 0.399 | **0.412** | passes |
| `photonnet_xl_nullscr` (also the decoder's 6 optical targets + SNR permuted, independent permutation) | 0.356 | 0.493 | 0.434 | **0.428** | passes |

Nested choices of the null runs: only draw 2 of the plain null and the 3 scrambled draws kept a choices file (`photonnet_xl_raw/choices_null.csv`, `choices_nullscr.csv`); draws 0 and 1 overwrote the same name in separate kernels, their TM is from the kernel logs and the score files.
Files: `scores/photonnet_xl_null__L0.csv` (= draw 0), `scores/photonnet_xl_null_d{0,1,2}__L0.csv`, `scores/photonnet_xl_nullscr_d{0,1,2}__L0.csv`.
evalkit `null_check` on `photonnet_xl_null`: ok.
Round-2 PhotonNet's null scored 0.618 because the SpO2 head was hard-wired to the physical latent (mu = 100 (sat_b + d_sat) + r), so the label-free
reconstruction loss alone made a colour-to-saturation inversion. Here the decoder is training-only and the gain beta on d_sat is learned from the labels
(start 0), so permuted labels leave nothing to transfer. The physics prior W_PHYS (beta - 1)^2 could put that route back. The sweep could pick W_PHYS = 1
in the null runs, and it did in 2 of the 6 folds of draw 2, with beta reaching only 0.31-0.71. Even so, the plain null stays below 0.55.
Both null means sit below 0.5 (draws as low as 0.36), with wide CIs. This is the usual LOSO anti-correlation under a permuted-label null: each fold's
training class balance excludes the held-out person, which pushes predictions the other way. It is not a sign of signal in the null.

## Model as run

Input: the 39 FlashGap / distill window features (`flashgap.COLS`). These are the features `app/js/distill_features.js` computes, with 1e-12 parity per
distill's export. They are winsorised and standardised with training-row statistics (`photonnet.Prep`).
- Encoder 39 -> 96 -> 96 (tanh). Physical latents z = 2 P_SD tanh(W_z h) (d_sat, d_ln_fb, d_ln_alpha, d_ln_rho). Physiological embedding e = tanh(W_e h) [16],
  learned jointly.
- Multi-task heads: SpO2 mu = 100 sat_b + beta 100 d_sat + w_r.e + b_r and sigma = softplus + 0.5 (Gaussian NLL). Hypoxia logit = w_y.[h, e] + b_y +
  1.6 (90 - mu) / sigma (class-balanced BCE; score = sigmoid). Abstain head sigmoid(w_a.[h, e] + b_a) = P(hyp head correct) (BCE, input detached).
- Training-only physics decoder: the PhotonTwin renderer (`photonnet.Decoder`, Farrell sum, band-integrated channels, learned spectral scalars and tissue
  fractions) renders d ln DC_RGB and d ln |AC|_RGB from z. The loss on the six observed optical changes is a heteroscedastic Gaussian NLL with weight
  W_REC (128 rows per step).
- Physics-prior strength W_PHYS (beta - 1)^2; z prior 1e-2, residual 1e-3; Adam lr 3e-3, wd 1e-4, batch 512, snapshots at 200 and 400 steps.
- 15,433 inference parameters (15,452 with the training-only decoder, noise model and spectral scalars).
- Augmentation (declared, as round-2 PhotonNet): FlashGap's distant-flash prior, AUG_TAG 'FGAP', 4 copies per training recording with aug seed = model seed,
  weight 0.25 each. `flashgap.assert_disjoint` runs on bundle load. No stress.py (level, seed) pair and no X1 effect is sampled.

## Sweep and protocol

LOSO by person (both hands). The held-out person is asserted absent from every fit. For each outer fold and seed, 5 inner person-grouped fits run per grid
point. The grid is W_REC in {0.3, 3} x W_PHYS in {0, 1} x snapshot in {200, 400} steps, 4 x 6 x 3 x 6 = 432 fits for the main model. The configuration is
chosen by 0.5 TM(clean inner OOF) + 0.5 mean TM(augmented inner OOF copies). c* comes from `twin_inverse.choose_conf` on the pooled clean + augmented inner
OOF windows. Then one outer fit per grid point is trained on the 5 training people, and the chosen one is scored on the held-out person at L0-L5
(`stress.degrade(., level, 0)`) and X1 from FlashGap's cached features. The weights are not refit per level. Ensemble: mean score, mean SpO2, mixture sigma,
mean abstain probability vs mean c*.

Nested choices (`results/photon/photonnet_xl_raw/choices_main_s{0,1,2}.csv`):

| held out | seed | W_REC | W_PHYS | steps | c* | inner crit | inner TM clean | beta |
|---|---|---|---|---|---|---|---|---|
| 100001 | 0 / 1 / 2 | 3 / 3 / 3 | 0 / 0 / 0 | 200 / 200 / 400 | 0.600 / 0.000 / 0.850 | 0.630 / 0.658 / 0.649 | 0.634 / 0.656 / 0.642 | 0.36 / 0.39 / 0.53 |
| 100002 | 0 / 1 / 2 | 3 / 3 / 0.3 | 0 / 1 / 1 | 400 / 200 / 400 | 0.825 / 0.000 / 0.925 | 0.654 / 0.679 / 0.682 | 0.659 / 0.673 / 0.673 | 0.48 / 0.51 / 0.77 |
| 100003 | 0 / 1 / 2 | 3 / 3 / 3 | 1 / 1 / 1 | 200 / 200 / 200 | 0.800 / 0.725 / 0.700 | 0.662 / 0.713 / 0.692 | 0.694 / 0.734 / 0.707 | 0.48 / 0.53 / 0.51 |
| 100004 | 0 / 1 / 2 | 0.3 / 3 / 0.3 | 0 / 0 / 1 | 200 / 200 / 200 | 0.750 / 0.725 / 0.875 | 0.694 / 0.719 / 0.708 | 0.717 / 0.737 / 0.725 | 0.22 / 0.32 / 0.47 |
| 100005 | 0 / 1 / 2 | 3 / 3 / 3 | 0 / 0 / 0 | 400 / 200 / 200 | 0.900 / 0.000 / 0.850 | 0.637 / 0.666 / 0.643 | 0.656 / 0.667 / 0.659 | 0.41 / 0.33 / 0.32 |
| 100006 | 0 / 1 / 2 | 3 / 3 / 3 | 0 / 0 / 1 | 200 / 200 / 400 | 0.775 / 0.825 / 0.875 | 0.618 / 0.634 / 0.634 | 0.644 / 0.640 / 0.631 | 0.29 / 0.28 / 0.76 |

The strong reconstruction weight (W_REC = 3) was chosen in 15 of 18 fold-seeds, and W_PHYS = 0 in 11 of 18. Learned beta is 0.2-0.8: the label data use
d_sat at less than its physical value.

## Export (ROUND3 rule 3)

`python -m oxydrop_ml.photon.photonnet_xl_export` ran on Kaggle (kernel aravhawk/oxy-pnxl-export). It refits the 3 seeds on all 6 people with each seed's
modal nested recipe: s0 and s1 use W_REC 3, W_PHYS 0, 200 steps; s2 uses W_REC 3, W_PHYS 0, 400 steps. c* = median of the 18 nested c* values.
- `results/photon/photonnet_xl_weights.json` (1.0 MB): per seed, the input prep (lo, hi, mu, sd), an **nn-runtime-v1** graph (linear, tanh, concat; outputs
  zt, r, s, y, a), sat_b and beta. The PhotonTwin decoder is not in the file. Inference needs five lines of head arithmetic after the graph:
  mu = 100 sat_b + beta 100 (2 P_SD0) zt + r, sigma = softplus(s) + 0.5, score = sigmoid(y + 1.6 (90 - mu)/sigma), conf = sigmoid(a).
- `results/photon/photonnet_xl_ref.json`: 2 windows (t = 60 s and 200 s of tests/fixtures/ppg-csv/Left/100001.csv, features from
  `results/photon/photonnet_ref_features.json`). Expected values: mu 97.21 / 94.39, score 0.0126 / 0.2310, conf 0.838 / 0.889.
- Parity: torch vs numpy is asserted in the export. Running the exported graphs through `app/js/nn_runtime.js` (`loadNN`) in Node reproduces the Python
  reference to 1.4e-16 (ad-hoc check, not committed as a test). Not done: wiring into app/ (not this task's files).
- The LOSO numbers above are the ones to quote, not numbers from this all-people refit.

## Compute disclosure (ROUND3 rule 2)

All training ran on Kaggle (private kernels, Hoffman-derived data only). The private dataset aravhawk/oxy-photonnet-bundle (round 2, the photonnet feature
bundle + code tarball) was reused unchanged; current photonnet.py and photonnet_xl*.py were embedded in each kernel.
- Kernels created: aravhawk/oxy-pnxl-s0, -s1, -s2 (main sweep per seed, CPU), -n0, -n1 (plain null draws 0 and 1, CPU), -n2 (plain null draw 2) and -ns
  (physics-scrambled null, 3 draws), and -export (CPU). Each version 1 failed at bundle load (the dataset's old photonnet.py lacks allow_pickle); version 2
  is the run. All are private.
- **GPU:** -n2 and -ns ran in GPU sessions, because Kaggle's limit of 5 concurrent CPU sessions was reached. The work itself ran on the session's 4 CPU
  workers. The model is a float64 MLP plus a renderer with many small ops, and none of the code moves tensors to CUDA. So the sweep was effectively CPU on
  Kaggle, not GPU-accelerated.
- No Daytona sandbox was created. Local runs: one 6-fit smoke test (`--quick`, 433 MB RSS for 7 s, above the ~300 MB guide; free memory was 2.9 GB),
  `--part combine`, evalkit (119 MB), a Node parity check.
