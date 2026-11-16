# SSL3 (round-3 candidate R3-SSL) log: stopped by the planner at 20:45 CT, no scores

## Status
SSL3 produced **no score files, no nulls, no stress results and no export**. Nothing can be compared with chain_v1 or distill. The planner locked round 3 at 20:26 CT (winner distill2) and stopped this agent at 20:45 CT. The work continues under another agent as `ssl3_gpu`, which may edit ssl3.py.

## What exists
- Code: `python/oxydrop_ml/photon/ssl3.py` and `python/oxydrop_ml/photon/ssl3_export.py`. The ssl3.py docstring describes the full protocol:
  - pretraining: NT-Xent plus masked reconstruction on the public non-OpenOx, non-MIMIC corpus (PTT-PPG, moshekam, apnoea, BUT PPG, MTHS);
  - fine-tuning: encoder unfrozen at LR 2e-4, heads re-initialised at LR 1e-3, nested epochs in {1, 2, 4, 6}, nested abstain threshold, FlashGap prior with its own 'OXS3' RNG stream;
  - controls: 3 seeds and 3 permutation nulls.
- Pretraining subset: `ml_cache/ssl3/pub_ac.f16.npy` (92,477 windows: moshekam 54,509, ptt_ppg 31,481, apnoea 6,487), `pub_meta.parquet` and `pub_info.json`. BIDMC (a MIMIC-II derivative) and OpenOx are excluded. `assert_pretrain_disjoint` passed: 156 pretraining subjects, none of them a Hoffman person.
- Kaggle bundle and kernel generator: `ml_cache/ssl3_k/` (`ds/`, `kern/`, `make_kernel.py`).
- A CPU smoke test (SSL3_SMOKE=1, one outer fold, 6 pretraining steps) ran the code paths end to end. Its numbers mean nothing.

## Kaggle resources created (all private)
- Dataset `aravhawk/oxy-ssl3-bundle` contains code.tgz and data.tar. The data is the Hoffman shard, the BUT PPG and MTHS tensors, and the public pretraining subset. It holds no OpenOx- or MIMIC-derived data.
- Kernel `aravhawk/oxy-ssl3-remote` (GPU T4):
  - v1 (20:08) ended in ERROR, most likely because Kaggle auto-extracts data.tar. The log was not retrieved.
  - v2 (pushed 20:17) was still RUNNING at 20:43, so nothing has been pulled. The Kaggle CLI has no cancel command. v2 is left to finish or be deleted by the planner (`kaggle kernels delete aravhawk/oxy-ssl3-remote`). Its outputs are not part of any result.
- No Daytona sandbox was created.

## Disclosures
- Both deadlines were missed: L0 + nulls (20:15) and stress (20:30).
- The local box exceeded the 300 MB RSS rule several times:
  - bundle: 529 MB;
  - pretraining smoke test: 616 MB;
  - fine-tuning smoke test: 1.66 GB, 600 s, stopped by a timeout.

  This should not have happened.
- One known train/app mismatch, left as in oxyssl: the baseline windows used for emb_base are built with the baseline, while the app builds its calibration windows without one.
