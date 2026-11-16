# ssl3_gpu: SSL pretraining + unfrozen fine-tuning on Kaggle GPU

Run: private Kaggle kernel `aravhawk/oxy-ssl3-remote` (2 x Tesla T4, `torch.cuda.get_device_name()` logged; internet off), dataset `aravhawk/oxy-ssl3-bundle` (Hoffman, BUT PPG, MTHS tensors, PTT-PPG/moshekam/apnoea contact subset; no OpenOximetry, MIMIC or BIDMC rows). Started ~20:17 CT, finished ~20:48 CT, wall time 1861 s. Code: `python/oxydrop_ml/photon/ssl3.py`, byte-identical to the copy embedded in the kernel script. Score files were written as `ssl3*` and renamed to `ssl3_gpu*` on pull with contents unchanged. Run metadata: `ssl3_gpu_run.json`; logs, inner-CV selections and pretraining curve: `ssl3_gpu_raw/`.

- Pretraining: 300 s (3174 steps), NT-Xent + masked reconstruction, 92,477 contact windows + 5,905 phone windows, 156 subjects. `assert_pretrain_disjoint` passed (0 overlap with the 6 Hoffman people). NT-Xent plateaued at ~4.95 (top-1 pair accuracy ~0.25) from step 1000.
- Fine-tuning: unfrozen encoder, fresh multi-task heads, FlashGap augmentation prior (3 copies per recording), nested inner-LOSO selection of epochs {1,2,4,6} and abstain quantile, LOSO by person, seeds 0/1/2, 3 permuted-label null draws through the same pipeline.

| | TM bal. acc. [95% CI] | paired vs chain_v1 | vs distill | vs distill2 |
|---|---|---|---|---|
| L0 3-seed ensemble | 0.596 [0.514, 0.665] | -0.201 [-0.286, -0.099], 2/6 | -0.162 [-0.232, -0.085] | -0.188 [-0.251, -0.080] |
| L3 | 0.603 [0.519, 0.662] | +0.038 [-0.056, +0.137] | -0.077 [-0.124, -0.024] | -0.095 [-0.140, -0.033] |
| L4 | 0.631 [0.535, 0.716] | +0.033 [-0.030, +0.126] | -0.050 [-0.086, -0.015] | -0.064 [-0.112, -0.012] |
| L5 | 0.646 [0.539, 0.706] | +0.112 [-0.001, +0.165] | -0.054 [-0.090, -0.016] | -0.058 [-0.104, -0.016] |
| X1 | 0.640 [0.532, 0.715] | +0.132 [+0.006, +0.236], 6/6 | -0.036 [-0.088, +0.012] | -0.042 [-0.086, +0.010] |

Per-seed L0: 0.524 / 0.639 / 0.559. Nulls L0: 0.465, 0.510, 0.455 (mean 0.477, all pass <= 0.55). Verdict: a clear loss at L0 versus chain_v1, distill and distill2; not eligible to replace the shipped model. Paired comparisons vs distill/distill2: `ssl3_gpu_eval_vs_distill*/` (their `diff_vs_chain_v1` column holds the difference vs the named reference).
