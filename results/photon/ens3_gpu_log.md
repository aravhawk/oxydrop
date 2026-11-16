# ens3_gpu (round-3 GPU relaunch of the nested stacking ensemble), log

Status: candidate, not committed. The kernel was pushed at 20:48:51 CT, after a wait for a free GPU slot (the account's 2 batch GPU sessions were in use). It completed at about 21:02 CT. The L0 file and the nulls therefore missed the 21:00 CT deadline by about 2 minutes. The stress files met the 21:05 CT deadline.

Code: [ens3_gpu.py](../../python/oxydrop_ml/photon/ens3_gpu.py) (members and combiner) and [ens3_gpu_export.py](../../python/oxydrop_ml/photon/ens3_gpu_export.py) (single-graph fold and check). Run metadata: [ens3_gpu_run.json](ens3_gpu_run.json). Per-fold combiner: [ens3_gpu_fold_info.json](ens3_gpu_fold_info.json). Paired comparisons vs distill, distill2 and ens3: [ens3_gpu_paired.json](ens3_gpu_paired.json). Kernel logs: `ens3_gpu_logs/`.

## Compute
- Kaggle kernel `aravhawk/ens3-gpu-all`, version 1: private, GPU, internet on. `torch.cuda.get_device_name()` = **Tesla T4** in every fold job (2 GPUs visible, torch 2.11.0+cu130).
- Wall time: 763 s for the 24 fold jobs (6 real + 18 null, 12 in parallel); 770 s for the whole kernel. Per real fold: 573-661 s; per null fold: about 250-305 s.
- Dataset: the existing private `aravhawk/ens3-r3-data` (Hoffman-only 39-feature pack and code). The new code was embedded in the kernel script.
- Nothing was trained locally. The local box ran only evalkit, the paired bootstrap and the numpy graph check (peak RSS 264 MB).

## Design
- Members: `distill` (seeds 0-2), `distill2` (seeds 0-2) and `distill_nokd` (seed 0). All three use the 39-feature front end. All student networks were trained on cuda.
- multida was left out because it trains a numpy network with no cuda path. oxyseq was left out because it uses a different front end that has no JS port.
- Hyperparameters: each member and outer fold reuses the configuration that the member's own nested inner-LOPO selection chose in the earlier Daytona CPU runs (`distill_fold_info.json`, `distill2_fold_info.json`). The selection grid was not re-run on GPU; only training at those configurations was. Every real-label distill2 fold picked the `base` teacher, so no GBDT teacher appears in the real run.
- Nesting: for each outer person p, every member was retrained on cuda, with teachers refit on the same rows:
  - on p's 5 training people, giving p's predictions at L0-L5 and X1;
  - for each inner person ip, on the other 4 people, giving ip's clean inner-OOF prediction.
- Combiner: `ens3.fit_outer`, fit on those inner-OOF rows only. It ranks members (top K = 3 of 3), picks C from {0.01, 0.1, 1} by leave-one-inner-person-out, and sets tau with `flashgap.select_tau`. Outer OOF predictions of other people are never used.
- Teachers (LR / chain expert) are sklearn fits on the Kaggle CPU, inside the GPU kernel.
- Nulls: in draw d, labels and SpO2 are permuted globally within each outer fold's training rows (`distill_student.null_perm`, the same permutation for every member). Then the full pipeline runs: inner, outer and combiner, with seed 0 per member.
  - distill and distill_nokd use their real-label configurations; they were not re-selected under permuted labels.
  - distill2 uses the configurations its own null runs selected.
- `ens3_gpu_s0`: the same combiner fit on seed-0 members only. This is the form that folds exactly into one graph, and it matches the null pipeline.

## Results (evalkit TABLE; CIs are 95 % person bootstrap)

| level | ens3_gpu TM [CI] | vs chain_v1 | people better vs chain | vs distill | vs distill2 | coverage / answered acc. |
|---|---|---|---|---|---|---|
| L0 | **0.823 [0.656, 0.877]** | +0.026 [-0.011, +0.057] tie | 3/6 | +0.065 [+0.006, +0.118], 4/6 | +0.039 [-0.014, +0.107], 4/6 | 0.69 / 0.951 |
| L1 | 0.723 [0.596, 0.804] | -0.082 [-0.191, +0.034] tie | 2/6 | +0.017 [-0.017, +0.060] | +0.011 [-0.017, +0.050] | 0.73 / 0.922 |
| L2 | 0.707 [0.556, 0.803] | -0.035 [-0.161, +0.091] tie | 2/6 | +0.003 [-0.030, +0.041] | -0.005 [-0.034, +0.028] | 0.71 / 0.892 |
| L3 | 0.723 [0.585, 0.795] | +0.158 [+0.030, +0.254] win | 4/5 | +0.043 [+0.000, +0.068] | +0.025 [-0.005, +0.050] | 0.72 / 0.919 |
| L4 | 0.713 [0.555, 0.793] | +0.115 [+0.012, +0.212] win | 6/6 | +0.032 [-0.014, +0.072] | +0.018 [-0.010, +0.045] | 0.71 / 0.884 |
| L5 | 0.689 [0.533, 0.759] | +0.154 [-0.005, +0.254] tie | 4/6 | -0.012 [-0.045, +0.032] | -0.016 [-0.044, +0.024] | 0.66 / 0.832 |
| X1 | 0.742 [0.575, 0.809] | +0.235 [+0.067, +0.326] win | 6/6 | +0.066 [+0.004, +0.106] | +0.060 [+0.015, +0.098] | 0.65 / 0.864 |

Reference L0 values: chain_v1 0.796 [0.636, 0.857], distill 0.758 [0.603, 0.813], distill2 (CPU) 0.784 [0.596, 0.834], ens3 (CPU) 0.789 [0.621, 0.842].
- Worst person at L0: ens3_gpu 0.534, chain_v1 0.458.
- The mean paired difference vs chain_v1 over L3-L5 and X1 is +0.165.
- ens3_gpu_s0 (single graph): L0 0.761 [0.534, 0.847], vs chain -0.035 [-0.147, +0.021].

Nulls (L0 TM): draw 0 0.572 (evalkit FLAG, > 0.55), draw 1 0.477, draw 2 0.447. The mean is 0.499, which passes the ROUND3 rule (mean <= 0.55), but draw 0 on its own is above 0.55. Null coverage is 0.10-0.27.

Under the ROUND3 winner rule, ens3_gpu passes on the numbers:
- L0 0.823 > 0.758;
- the lower CI bound vs distill is +0.006 > -0.02;
- the stress mean vs chain is >= 0.

It does not beat chain_v1 (L0 CI includes 0; 3/6 people better). The blockers are listed in the deployability section below.

## Single network (deployability)
- **ens3_gpu_s0 folds exactly into one nn-runtime-v1 graph.** The three seed-0 members train on the same stacked design, so their `flashgap.Prep` statistics are identical; this was checked in all 6 folds. They therefore share the 39-feature front end z.
- The combiner reads `logit(sigmoid(o)) = o` of each member's hyp and abstain heads, clipped to ±C = ±logit(1 - 1e-4). The clip is written as clamp(x) = relu(x + C) - relu(x - C) - C.
- Resulting graph: z[39] → Linear(96 or 128) → ReLU → block-diagonal Linear → ReLU → Linear(12) → ReLU → Linear(1) → sigmoid. It uses only the existing linear and act ops. The wider first layer comes from the fold-3 distill2 member with H = 64.
- Measured max |score difference| vs the stack computed as in `combine`: 1.1e-7 over the held-out windows of each fold (`ens3_gpu_graph_check.json`, graphs `ens3_gpu_graph_f<k>.json`). The fold is exact in real arithmetic. I did not trace why the float64 residual is 1e-7 rather than about 1e-12.
- The scored 3-seed **ens3_gpu is not exactly foldable**. It averages three seeds' probabilities, has three Prep fits, and then takes a logit; nn-runtime-v1 has no log/logit op.
- No all-6-people deployment refit (members + combiner) was built. The per-fold graphs exist only to show the fold.
