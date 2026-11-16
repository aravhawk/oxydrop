# distill2_gpu: Kaggle GPU rerun of distill2 (round 3, GPU addendum)

Status: candidate only. The planner made the final lock at 21:46 CT (`ens3_gpu_s0` ships). These files arrived after the 21:00 / 21:05 CT GPU deadlines:
the L0-L5 and X1 score files were pulled at 21:28 CT and the null files at 21:45 CT.

## What ran
- Code: [distill2_gpu.py](../../python/oxydrop_ml/photon/distill2_gpu.py). It runs the same `distill2` pipeline: same teachers (base / gbdt), the same
  two-stage nested inner-LOSO grid, epochs {15, 30}, seeds 0-2 and null draws 0-2, all from distill's feature pack (Hoffman only). The one change
  replaces `distill2.train` with a version that puts the student network and all tensors on cuda and asserts `torch.cuda.is_available()`. Batch
  order uses the same CPU `torch.Generator`, and initialisation uses `torch.manual_seed` before `.to(cuda)`, so the batches and initial weights match
  the CPU run. The sklearn teachers stay on CPU, as in distill2.
- Dataset: `aravhawk/oxy-distill2-gpu` (private; code.tgz + distill_pack.npz + distill_fixture.npz).
- Kernel A, `aravhawk/oxy-distill2-gpu-a` v1 (private, internet on, accelerator T4): seeds s0-s2, 18 outer folds with 4 fold processes in parallel,
  then collect and export. **Tesla T4 (x2), torch 2.11.0+cu130, CUDA 13.0, 4 CPUs; wall time 2539.5 s** (pushed 20:44 CT). Each fold took 372-556 s.
- Kernel B, `aravhawk/oxy-distill2-gpu-b` v1 (same settings): null draws 0-2, 18 outer folds. **Tesla T4 (x2), same torch; wall time 2349.4 s.** Push
  was refused with "Maximum batch GPU session count of 2 reached" from 20:44 to 21:03 CT, because another candidate's kernel held the second slot.
  All 36 fold jobs exited with code 0.
- Device, torch version and wall time are recorded per fold in [distill2_gpu_fold_info.json](distill2_gpu_fold_info.json) (`device`,
  `fold_wall_s`) and per score file in [distill2_gpu_run.json](distill2_gpu_run.json). The raw kernel logs are
  [distill2_gpu_kernel_a.log](distill2_gpu_kernel_a.log) and [distill2_gpu_kernel_b.log](distill2_gpu_kernel_b.log).
- Speed: the GPU was slower than the CPU here, about 9 min per fold on the T4 against about 2 min on the Daytona CPU. The 39 -> H -> H -> 3 MLP with
  batch 512 is limited by kernel-launch overhead, and 4 processes shared one T4 and 4 vCPUs.

## Results (evalkit, 95% person bootstrap, paired at the same level)
| model | L0 TM [CI] | vs chain_v1 | vs distill2 (CPU) | vs distill |
|---|---|---|---|---|
| distill2_gpu (3-seed ens.) | 0.761 [0.561, 0.830] | -0.035 [-0.127, -0.006], 3/6 | -0.023 [-0.044, -0.004], 1/4 | +0.003 [-0.044, 0.024], 3/6 |
| distill2_gpu_s0 | 0.771 [0.583, 0.846] | -0.025 [-0.108, 0.009] | -0.013 [-0.036, 0.009] | +0.013 [-0.028, 0.037] |
| distill2_gpu_s1 | 0.760 [0.628, 0.819] | -0.036 [-0.082, 0.008] | -0.024 [-0.042, 0.035] | +0.002 [-0.015, 0.038] |
| distill2_gpu_s2 | 0.750 [0.539, 0.822] | -0.047 [-0.152, -0.007] | -0.034 [-0.079, -0.002] | -0.009 [-0.073, 0.037] |

The ensemble's worst person at L0 is 0.480. Against chain_v1 at L0 it is a loss, because the CI excludes 0 on the negative side.

Stress, ensemble TM [CI], paired vs chain_v1:
L1 0.708 [0.575, 0.786] -0.097 [-0.195, 0.003]; L2 0.708 [0.558, 0.788] -0.034 [-0.134, 0.097]; L3 0.693 [0.561, 0.784] +0.129 [0.007, 0.245] 4/6;
L4 0.692 [0.569, 0.781] +0.094 [0.018, 0.191] 6/6; L5 0.700 [0.558, 0.777] +0.166 [0.022, 0.247] 5/6; X1 0.685 [0.550, 0.747] +0.177 [0.044, 0.287] 6/6.
The mean paired difference vs chain_v1 over L3-L5 and X1 is +0.141.

Nulls (seed-0 pipeline, global label permutation): draw 0 0.458 [0.372, 0.535], draw 1 0.556 [0.407, 0.695] (evalkit FLAG > 0.55), draw 2 0.509
[0.321, 0.663]. The mean is 0.508, which passes the round-3 rule (mean <= 0.55), but one draw is individually flagged.

ROUND3 winner rule vs distill: the point estimate 0.761 exceeds 0.758, but the paired lower bound vs distill is -0.044, below -0.02, so it does not qualify.

## The CPU 0.784 did not reproduce
distill2 (Daytona CPU) scored 0.784 [0.596, 0.834] at L0. The same code with the student on cuda scored 0.761, and the paired difference of -0.023
[-0.044, -0.004] excludes 0. Seed by seed, the GPU scored 0.771 / 0.760 / 0.750. Batches and initialisation are the same, so the cause is
floating-point: cuBLAS/cuDNN accumulate in a different order from CPU float32, and the difference grows over up to 30 epochs of Adam. That changes
which grid point wins on the inner folds: 5 of 18 seed-fold recipes differ from the CPU run (s0/100005 lam 16 -> 32; s1/100002 H 32 ep 15 -> H 16 ep 30;
s1/100003 lam 16 T 2 -> lam 8 T 4; s1/100004 lam 8 T 4 wd 1e-2 -> lam 32 T 2 wd 1e-4; s2/100004 lam 8 H 64 -> pure KD H 32). The other 13 match exactly,
but their trained weights still differ slightly. At L1-L5 and X1 the GPU and CPU runs agree to within 0.015, with every paired CI vs distill2 containing 0. The CPU L0 lead of about
0.02-0.03 over distill therefore depends on numerics and is not a stable property of the method. The honest GPU-logged figure is 0.761, a tie with
distill (+0.003 [-0.044, 0.024]).

## Files
- Scores: `results/photon/scores/distill2_gpu__{L0..L5,X1}.csv`, `distill2_gpu_s{0,1,2}__*.csv`, `distill2_gpu_null__L0.csv`,
  `distill2_gpu_d1_null__L0.csv`, `distill2_gpu_d2_null__L0.csv`.
- [distill2_gpu_run.json](distill2_gpu_run.json) (device, wall time, kernel slug and version, per score file), [distill2_gpu_fold_info.json](distill2_gpu_fold_info.json),
  [distill2_gpu_compare.csv](distill2_gpu_compare.csv) (paired vs chain_v1 / distill2 / distill).
- [distill2_gpu_weights.json](distill2_gpu_weights.json): format `oxydrop-photon-distill-v1` (the distill_weights.json format plus `model` and
  `trained_on`). It holds the refit on all 6 people on the T4 in kernel A with the modal nested recipe per seed (H 32/32/64, base teacher, lam
  16/32/32, T 2/2/4, 30 epochs), plus 2-window reference outputs. It was not swapped into the app.
- Fold models: `ml_cache/photon_distill2_gpu/models/*.pkl`. The Kaggle staging directory is `ml_cache/distill2_gpu_kaggle/`.
- Nothing was trained locally. Locally I ran only the kaggle CLI, evalkit, and the paired compare. Nothing was committed and no app files were touched.
