# Round 3 status 2100 CT

Generated Sun Nov 15 21:01:18 CST 2026 by `python -m oxydrop_ml.photon.status3` (R3-EVAL). Every number is recomputed from `results/photon/scores/*.csv` with the evalkit bootstrap (2000 person resamples, same people for every model). Reference: chain_v1 L0 TM 0.796 [0.636, 0.857]; distill L0 TM 0.758 [0.603, 0.813].

## Plain statement

* No model beats chain_v1 (all of: paired CI above 0, >= 5/6 people better, worst person not below chain_v1, nulls passed with >= 3 distinct draws).
* Round-3 candidates with an L0 file at this moment: distill2, multida, photonnet_xl, ens3, ssl3_gpu.
* Replacement rule (ROUND3.md): **REPLACE `distill` with `distill2`** by the planner's ROUND3 LOCK: planner ROUND3 LOCK, 20:22 CT (task.json): winner = distill2; its files landed 20:16 CT per its author log, one minute after the 20:15 CT cutoff, and the planner accepted that with disclosure (file mtime on this box 20:18, copied after the sandbox run). Rule 3 is author-confirmed (same 39-feature front end and an nn-runtime-v1 MLP), not verified by R3-EVAL. `distill2` passes every numeric condition (L0 0.784 > 0.758; paired vs distill CI lower bound -0.010 > -0.02; stress mean vs chain_v1 +0.144 >= 0; nulls passed); the mtime flag above marks it late (distill2_d1_null, distill2_d2_null, distill2_null, distill2__L0), and the lock waives that by disclosure, not silently.

**Context for this 21:00 report.** (1) GPU addendum (user 20:40 CT): a run without a torch.cuda.get_device_name() log does not count. Of the round-3 models, only ssl3_gpu has such a log (private Kaggle kernel, 2 x Tesla T4, wall time 1861 s, finished about 20:48 CT; its log states no OpenOximetry, MIMIC or BIDMC rows and an assert_pretrain_disjoint check with 0 overlap with the six Hoffman people). distill2, multida, photonnet_xl and ens3 files on disk are CPU/Daytona runs with no GPU log found by the probe. No distill2_gpu, multida_gpu or ens3_gpu score file exists yet at 21:00. (2) The 20:22 CT lock (distill2) is provisional per the planner (task.json 20:40 and 20:45 notes) until a GPU-logged distill2 rerun exists, so the 'REPLACE distill with distill2' line below is the CPU-run lock, not a GPU-confirmed result. (3) ssl3_gpu arrived 20:49 (after the 20:15 and 20:30 deadlines); L0 0.596 [0.514, 0.665], paired vs chain_v1 -0.201 [-0.286, -0.099] 2/6, vs distill -0.162 [-0.232, -0.085]; null mean 0.476 over 3 draws passes; stress mean +0.079 (X1 +0.132 [+0.006, +0.236] 6/6 only). It is a loss at L0 and fails g1 and g2. (4) No model beats chain_v1; distill2 and ens3 tie it at L0. The 90% target is not reached (best L0 point estimate 0.789, ens3, ineligible; 0.784 distill2).

## L0 (clean light), 3-seed ensemble (primary file); people better = of the people with a usable comparison

| model | round | L0 TM [95 % CI] | paired vs chain_v1 (people better) | paired vs distill (people better) | null mean over distinct draws (<= 0.55) | worst person | levels with files |
|---|---|---|---|---|---|---|---|
| photonnet | 2 | 0.668 [0.508, 0.785] | -0.128 [-0.208, -0.047] (2/6) | -0.090 [-0.157, -0.023] (2/6) | 0.618 over 3 **FAILS** | 0.500 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| distill | 2 | 0.758 [0.603, 0.813] | -0.038 [-0.103, -0.001] (2/6) | incumbent | 0.492 over 3 | 0.516 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| oxyssl | 2 | 0.667 [0.498, 0.741] | -0.130 [-0.274, -0.044] (1/6) | -0.091 [-0.211, -0.020] (1/6) | 0.643 over 3 **FAILS** | 0.400 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| oxyseq | 2 | 0.708 [0.452, 0.790] | -0.088 [-0.221, -0.023] (2/6) | -0.050 [-0.167, +0.030] (2/6) | 0.475 over 1, blank 0.500 **FAILS** | 0.516 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| distill2 | 3 | 0.784 [0.596, 0.834] | -0.012 [-0.091, +0.012] (4/6) | +0.026 [-0.010, +0.040] (4/6) | 0.504 over 3 | 0.500 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| multida | 3 | 0.744 [0.553, 0.807] | -0.053 [-0.142, -0.010] (3/6) | -0.014 [-0.055, +0.023] (3/6) | 0.522 over 3 | 0.500 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| ssl3 | 3 | no L0 file yet |  |  |  |  | none |
| photonnet_xl | 3 | 0.660 [0.545, 0.762] | -0.136 [-0.210, -0.033] (2/6) | -0.098 [-0.161, -0.011] (2/6) | 0.412 over 3 | 0.500 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| ens3 | 3 | 0.789 [0.621, 0.842] | -0.007 [-0.092, +0.043] (4/6) | +0.031 [-0.012, +0.078] (3/6) | none **FAILS** | 0.585 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| distill2_gpu | 3 | no L0 file yet |  |  |  |  | none |
| ssl3_gpu | 3 | 0.596 [0.514, 0.665] | -0.201 [-0.286, -0.099] (2/6) | -0.162 [-0.232, -0.085] (1/6) | 0.476 over 3 | 0.500 (chain 0.458) | L0, L1, L2, L3, L4, L5, X1 |
| multida_gpu | 3 | no L0 file yet |  |  |  |  | none |
| ens3_gpu | 3 | no L0 file yet |  |  |  |  | none |

## Stress: paired TM difference to chain_v1 at the same level [95 % CI] people better (simulated distant flash L3-L5, out-of-family X1)

| model | L3 | L4 | L5 | X1 | mean over the four |
|---|---|---|---|---|---|
| photonnet | +0.125 [-0.019, +0.267] 4/6 | +0.105 [-0.025, +0.245] 5/6 | +0.185 [+0.023, +0.290] 5/6 | +0.165 [+0.020, +0.316] 6/6 | 0.145 |
| distill | +0.116 [-0.002, +0.226] 4/6 | +0.083 [+0.008, +0.178] 6/6 | +0.166 [+0.035, +0.242] 5/6 | +0.169 [+0.058, +0.285] 6/6 | 0.133 |
| oxyssl | +0.132 [+0.024, +0.171] 1/6 | -0.050 [-0.123, +0.045] 3/6 | -0.013 [-0.098, +0.068] 2/5 | +0.178 [+0.043, +0.214] 5/6 | 0.062 |
| oxyseq | -0.032 [-0.133, +0.046] 3/6 | -0.094 [-0.137, -0.015] 3/6 | -0.036 [-0.098, +0.017] 2/6 | +0.001 [-0.047, +0.059] 3/6 | -0.040 |
| distill2 | +0.133 [+0.001, +0.248] 4/6 | +0.097 [+0.011, +0.195] 6/6 | +0.170 [+0.027, +0.249] 5/6 | +0.174 [+0.044, +0.284] 6/6 | 0.144 |
| multida | +0.115 [+0.003, +0.216] 4/6 | +0.093 [+0.002, +0.189] 6/6 | +0.161 [+0.026, +0.232] 5/6 | +0.170 [+0.027, +0.328] 6/6 | 0.135 |
| photonnet_xl | +0.115 [-0.028, +0.229] 3/6 | +0.086 [-0.020, +0.207] 5/6 | +0.169 [+0.018, +0.269] 5/6 | +0.184 [+0.072, +0.290] 6/6 | 0.139 |
| ens3 | +0.149 [-0.007, +0.255] 4/6 | +0.085 [+0.003, +0.193] 6/6 | +0.141 [+0.043, +0.248] 5/6 | +0.197 [+0.074, +0.280] 6/6 | 0.143 |
| ssl3_gpu | +0.038 [-0.056, +0.137] 2/6 | +0.033 [-0.030, +0.126] 4/6 | +0.112 [-0.001, +0.165] 4/6 | +0.132 [+0.006, +0.236] 6/6 | 0.079 |

## ROUND3 winner script verdict

* Incumbent `distill`: L0 TM 0.758 [0.603, 0.813]; nulls passed: null mean 0.492 over 3 distinct draw(s).
* `photonnet` does NOT replace distill: nulls: NOT PASSED (null mean 0.618 > 0.55): null mean 0.618 over 3 distinct draw(s); g1 fails: L0 0.668 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.157 <= -0.02.
* `oxyssl` does NOT replace distill: nulls: NOT PASSED (null mean 0.643 > 0.55): null mean 0.643 over 3 distinct draw(s); g1 fails: L0 0.667 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.211 <= -0.02.
* `oxyseq` does NOT replace distill: nulls: NOT PASSED (only 1 distinct null draw(s), need >= 3): null mean 0.475 over 1 distinct draw(s), blank mean 0.500 over 1 distinct draw(s); g1 fails: L0 0.708 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.167 <= -0.02; g3 fails: mean stress paired diff vs chain_v1 -0.040 < 0.
* `distill2` passes the numeric rule (g1, g2, g3, nulls): L0 0.784, vs distill +0.026 [-0.010, +0.040], stress mean vs chain +0.144. LATE (cannot replace the shipped model): distill2_d1_null, distill2_d2_null, distill2_null, distill2__L0.
* `multida` does NOT replace distill: g1 fails: L0 0.744 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.055 <= -0.02.
* `ssl3`: no L0 score file yet.
* `photonnet_xl` does NOT replace distill: g1 fails: L0 0.660 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.161 <= -0.02.
* `ens3` does NOT replace distill: nulls: NOT PASSED (missing null control).
* `distill2_gpu`: no L0 score file yet.
* `ssl3_gpu` does NOT replace distill: g1 fails: L0 0.596 does not exceed distill 0.758; g2 fails: paired CI lower bound vs distill -0.232 <= -0.02.
* `multida_gpu`: no L0 score file yet.
* `ens3_gpu`: no L0 score file yet.
* **REPLACE `distill` with `distill2`** by the planner's ROUND3 LOCK: planner ROUND3 LOCK, 20:22 CT (task.json): winner = distill2; its files landed 20:16 CT per its author log, one minute after the 20:15 CT cutoff, and the planner accepted that with disclosure (file mtime on this box 20:18, copied after the sandbox run). Rule 3 is author-confirmed (same 39-feature front end and an nn-runtime-v1 MLP), not verified by R3-EVAL. `distill2` passes every numeric condition (L0 0.784 > 0.758; paired vs distill CI lower bound -0.010 > -0.02; stress mean vs chain_v1 +0.144 >= 0; nulls passed); the mtime flag above marks it late (distill2_d1_null, distill2_d2_null, distill2_null, distill2__L0), and the lock waives that by disclosure, not silently.
* At L0 `distill2` TIES the logistic chain_v1: paired -0.012 [-0.091, +0.012], 4/6 people better, worst person 0.500. It does not beat chain_v1.

## Spot checks (two score files per new model) and leakage probe

### distill2

`distill2__L0.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.021, 0.995]
* OK not a byte copy of another file: unique

`distill2__X1.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.021, 0.993]
* OK not a byte copy of another file: unique

Seed-ensemble check of `distill2__L0.csv`: OK: 3 seed file(s); max |primary - mean(seeds)| = 0.0000, median 0.0000; side of 0.5 agrees on 1.000 of windows

Leakage probe (grep of 6 own log/code/result files): 1 matching line(s)
* `python/oxydrop_ml/photon/distill2.py:202`: SP.assert_no_leak(sid, tr, te)

GPU-log probe (grep of 5 own log/run files for torch.cuda.get_device_name / a GPU name): **no GPU device log found (does not count under the 20:40 CT GPU addendum)**

### multida

`multida__L0.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.008, 0.994]
* OK not a byte copy of another file: unique

`multida__X1.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.016, 0.982]
* OK not a byte copy of another file: unique

Seed-ensemble check of `multida__L0.csv`: OK: 3 seed file(s); max |primary - mean(seeds)| = 0.0000, median 0.0000; side of 0.5 agrees on 1.000 of windows

Leakage probe (grep of 6 own log/code/result files): 3 matching line(s)
* `python/oxydrop_ml/photon/multida.py:33`: The Hoffman held-out person is never seen. Grid values and GRL were fixed before any multida score, after distill's round-2 outer results
* `python/oxydrop_ml/photon/multida.py:328`: SP.assert_no_leak(sid, tr, te)
* `python/oxydrop_ml/photon/multida_gpu.py:189`: SP.assert_no_leak(sid, tr, te)

GPU-log probe (grep of 3 own log/run files for torch.cuda.get_device_name / a GPU name): **no GPU device log found (does not count under the 20:40 CT GPU addendum)**

### photonnet_xl

`photonnet_xl__L0.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.000, 0.992]
* OK not a byte copy of another file: unique

`photonnet_xl__X1.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.000, 0.988]
* OK not a byte copy of another file: unique

Seed-ensemble check of `photonnet_xl__L0.csv`: OK: 3 seed file(s); max |primary - mean(seeds)| = 0.0000, median 0.0000; side of 0.5 agrees on 1.000 of windows

Leakage probe (grep of 5 own log/code/result files): 2 matching line(s)
* `python/oxydrop_ml/photon/photonnet_xl.py:110`: assert not set(map(str, held_out)) & set(map(str, tr_sids)), 'held-out person among the training people'
* `results/photon/photonnet_xl_log.md:67`: weight 0.25 each. `flashgap.assert_disjoint` runs on bundle load. No stress.py (level, seed) pair and no X1 effect is sampled.

GPU-log probe (grep of 11 own log/run files for torch.cuda.get_device_name / a GPU name): **no GPU device log found (does not count under the 20:40 CT GPU addendum)**

### ens3

`ens3__L0.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.000, 1.000]
* OK not a byte copy of another file: unique

`ens3__X1.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.000, 0.999]
* OK not a byte copy of another file: unique

Seed-ensemble check of `ens3__L0.csv`: **NOT an exact mean of the seed scores**: no seed files with an L0 file found

Leakage probe (grep of 5 own log/code/result files): 5 matching line(s)
* `python/oxydrop_ml/photon/ens3.py:16`: NESTING (the held-out person never enters any step that produces its own prediction):
* `python/oxydrop_ml/photon/ens3.py:89`: assert str(folds[fi]['sid']) == str(f.test_group)
* `python/oxydrop_ml/photon/ens3.py:282`: assert not (tab.sid.values == p).any()
* `python/oxydrop_ml/photon/ens3_gpu.py:142`: SP.assert_no_leak(sid, tr, te)
* `python/oxydrop_ml/photon/ens3_gpu.py:213`: assert not (tab.sid.values == str(z['outer_sid'])).any()

GPU-log probe (grep of 3 own log/run files for torch.cuda.get_device_name / a GPU name): **no GPU device log found (does not count under the 20:40 CT GPU addendum)**

### ssl3_gpu

`ssl3_gpu__L0.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.208, 0.991]
* OK not a byte copy of another file: unique

`ssl3_gpu__X1.csv`:

* OK row count: 5644 rows (expected 5644)
* OK no NaN/inf in score, pred: NaN score 0, NaN pred 0, inf score 0
* OK no duplicate (rec, t): 0 duplicates
* OK all evaluated windows present: 0 of 5644 evaluated windows missing
* OK sid matches windows.csv (person labels for OOF grouping): 0 windows with a different sid
* OK six people, non-constant score within each: 6 people; constant-score people: none
* OK score range: [0.268, 0.997]
* OK not a byte copy of another file: unique

Seed-ensemble check of `ssl3_gpu__L0.csv`: OK: 3 seed file(s); max |primary - mean(seeds)| = 0.0000, median 0.0000; side of 0.5 agrees on 1.000 of windows

Leakage probe (grep of 7 own log/code/result files): 8 matching line(s)
* `results/photon/ssl3_gpu_log.md:5`: - Pretraining: 300 s (3174 steps), NT-Xent + masked reconstruction, 92,477 contact windows + 5,905 phone windows, 156 subjects. `assert_pretrain_disjoint` passe
* `python/oxydrop_ml/photon/ssl3.py:19`: derivative) are excluded, so the corpus may run on Kaggle. assert_pretrain_disjoint checks that no pretraining subject is a Hoffman person.
* `python/oxydrop_ml/photon/ssl3.py:28`: on sigmoid(quality) (inner-OOF quantile in Q_GRID with the best answered bal. acc., ties to higher coverage). The held-out person is never used.
* `python/oxydrop_ml/photon/ssl3.py:96`: def assert_pretrain_disjoint():
* `python/oxydrop_ml/photon/ssl3.py:123`: print('bundle', len(rows), meta.iloc[rows].src.value_counts().to_dict(), assert_pretrain_disjoint())

GPU-log probe (grep of 22 own log/run files for torch.cuda.get_device_name / a GPU name): 5 matching line(s)
* `results/photon/ssl3_gpu_log.md:3`: Run: private Kaggle kernel `aravhawk/oxy-ssl3-remote` (2 x Tesla T4, `torch.cuda.get_device_name()` logged; internet off), dataset `aravhawk/oxy-ssl3-bundle` (H
* `results/photon/ssl3_gpu_run.json:11`: "Tesla T4",
* `results/photon/ssl3_gpu_run.json:12`: "Tesla T4"
* `results/photon/ssl3_gpu_run.json:55`: "accelerator": "NvidiaTeslaT4 (2 x Tesla T4 visible)",

## Problems found

* distill2: no GPU device log found (GPU addendum 20:40 CT: a run without a GPU log does not count)
* distill2: late files distill2_d1_null, distill2_d2_null, distill2_null, distill2__L0
* multida: no GPU device log found (GPU addendum 20:40 CT: a run without a GPU log does not count)
* multida: late files multida_null, multida_null_d1, multida_null_d2, multida__L0
* photonnet_xl: no GPU device log found (GPU addendum 20:40 CT: a run without a GPU log does not count)
* ens3: no GPU device log found (GPU addendum 20:40 CT: a run without a GPU log does not count)
* ens3: nulls NOT PASSED (missing null control)
* ens3: late files ens3__L0
* ssl3_gpu: late files ssl3_gpu_d1_null, ssl3_gpu_d2_null, ssl3_gpu_null, ssl3_gpu__L0, ssl3_gpu__L3, ssl3_gpu__L4, ssl3_gpu__L5, ssl3_gpu__X1

## Skipped files

* null file photonnet_null_d0__L0.csv is a byte-identical copy of photonnet_null__L0.csv; counted once
* null file photonnet_xl_null_d0__L0.csv is a byte-identical copy of photonnet_xl_null__L0.csv; counted once

Notes: the stress levels are simulations; every round-3 model was designed after rounds 1-2 outer results were seen (Amendment 8), so L0 numbers are optimistic. Nothing here is a medical-device claim.
