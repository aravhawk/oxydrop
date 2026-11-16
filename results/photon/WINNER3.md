# Round 3 winner (mechanical application of ROUND3.md, Amendment 8)

Written by `python -m oxydrop_ml.photon.winner3` (Sun Nov 15 21:01:16 CST 2026) from the score files that existed at that moment. Rule, verbatim from `.agents/tasks/task-oxydrop-photon-model/ROUND3.md` (stated 19:55 CT, before any round-3 score):

> Eligible: trained models from rounds 2 and 3 that pass nulls and rule 3. Replace `distill` only if a candidate's L0 3-seed ensemble TM point estimate exceeds distill's (0.758) AND its paired difference vs distill has a lower CI bound > -0.02 AND its stress mean paired diff vs chain_v1 over L3-L5 and X1 is >= 0. "Beats chain" still requires the paired CI vs chain_v1 excluding 0, >= 5/6 people better, worst person not below chain's. All round-3 models are designed after rounds 1-2 outer results were seen (Amendment 8).

## Verdict

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

## Beats chain_v1?

* No model beats chain_v1 (all of: paired CI above 0, >= 5/6 people better, worst person not below chain_v1, nulls passed with >= 3 distinct draws).

Reference `chain_v1`: L0 TM 0.796 [0.636, 0.857], worst person 0.458.

## Candidates at L0 (3-seed ensemble = primary file)

| model | round | L0 TM [95 % CI] | paired vs distill | g1 TM > distill | g2 CI lo > -0.02 | g3 stress mean vs chain >= 0 | paired vs chain_v1 (people better) | beats chain | nulls (>= 3 distinct draws, mean <= 0.55) | missing levels |
|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 2 | 0.668 [0.508, 0.785] | -0.090 [-0.157, -0.023] | no | no | yes (+0.145) | -0.128 [-0.208, -0.047] 2/6 | no | NOT PASSED (null mean 0.618 > 0.55): null mean 0.618 over 3 distinct draw(s) | none |
| distill | 2 | 0.758 [0.603, 0.813] | incumbent | n/a | n/a | yes (+0.133) | -0.038 [-0.103, -0.001] 2/6 | no | passed: null mean 0.492 over 3 distinct draw(s) | none |
| oxyssl | 2 | 0.667 [0.498, 0.741] | -0.091 [-0.211, -0.020] | no | no | yes (+0.062) | -0.130 [-0.274, -0.044] 1/6 | no | NOT PASSED (null mean 0.643 > 0.55): null mean 0.643 over 3 distinct draw(s) | none |
| oxyseq | 2 | 0.708 [0.452, 0.790] | -0.050 [-0.167, +0.030] | no | no | no (-0.040) | -0.088 [-0.221, -0.023] 2/6 | no | NOT PASSED (only 1 distinct null draw(s), need >= 3): null mean 0.475 over 1 distinct draw(s), blank mean 0.500 over 1 distinct draw(s) | none |
| distill2 | 3 | 0.784 [0.596, 0.834] | +0.026 [-0.010, +0.040] | yes | yes | yes (+0.144) | -0.012 [-0.091, +0.012] 4/6 | no | passed: null mean 0.504 over 3 distinct draw(s) | none |
| multida | 3 | 0.744 [0.553, 0.807] | -0.014 [-0.055, +0.023] | no | no | yes (+0.135) | -0.053 [-0.142, -0.010] 3/6 | no | passed: null mean 0.522 over 3 distinct draw(s) | none |
| ssl3 | 3 | no L0 score file yet |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| photonnet_xl | 3 | 0.660 [0.545, 0.762] | -0.098 [-0.161, -0.011] | no | no | yes (+0.139) | -0.136 [-0.210, -0.033] 2/6 | no | passed: null mean 0.412 over 3 distinct draw(s) | none |
| ens3 | 3 | 0.789 [0.621, 0.842] | +0.031 [-0.012, +0.078] | yes | yes | yes (+0.143) | -0.007 [-0.092, +0.043] 4/6 | no | NOT PASSED (missing null control) | none |
| distill2_gpu | 3 | no L0 score file yet |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| ssl3_gpu | 3 | 0.596 [0.514, 0.665] | -0.162 [-0.232, -0.085] | no | no | yes (+0.079) | -0.201 [-0.286, -0.099] 2/6 | no | passed: null mean 0.476 over 3 distinct draw(s) | none |
| multida_gpu | 3 | no L0 score file yet |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| ens3_gpu | 3 | no L0 score file yet |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |

## Stress: paired difference to chain_v1 [95 % CI] people better

| model | L3 | L4 | L5 | X1 | mean (g3) |
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

## Seeds, timeline, rule 3

| model | seed files with L0 | seed files | late files (after 20:15 L0/nulls, 20:30 stress) | rule 3 |
|---|---|---|---|---|
| photonnet | 3 | photonnet_seed0, photonnet_seed1, photonnet_seed2 | none | unconfirmed (manual) |
| distill | 3 | distill_s0, distill_s1, distill_s2 | none | unconfirmed (manual) |
| oxyssl | 3 | oxyssl_s0, oxyssl_s1, oxyssl_s2 | none | unconfirmed (manual) |
| oxyseq | 0 | none | none | unconfirmed (manual) |
| distill2 | 3 | distill2_s0, distill2_s1, distill2_s2 | distill2_d1_null, distill2_d2_null, distill2_null, distill2__L0 | confirmed |
| multida | 3 | multida_s0, multida_s1, multida_s2 | multida_null, multida_null_d1, multida_null_d2, multida__L0 | unconfirmed (manual) |
| ssl3 | 0 | none | none | unconfirmed (manual) |
| photonnet_xl | 3 | photonnet_xl_seed0, photonnet_xl_seed1, photonnet_xl_seed2 | none | unconfirmed (manual) |
| ens3 | 0 | none | ens3__L0 | unconfirmed (manual) |
| distill2_gpu | 0 | none | none | unconfirmed (manual) |
| ssl3_gpu | 3 | ssl3_gpu_s0, ssl3_gpu_s1, ssl3_gpu_s2 | ssl3_gpu_d1_null, ssl3_gpu_d2_null, ssl3_gpu_null, ssl3_gpu__L0, ssl3_gpu__L3, ssl3_gpu__L4, ssl3_gpu__L5, ssl3_gpu__X1 | unconfirmed (manual) |
| multida_gpu | 0 | none | none | unconfirmed (manual) |
| ens3_gpu | 0 | none | none | unconfirmed (manual) |

## Skipped files

* null file photonnet_null_d0__L0.csv is a byte-identical copy of photonnet_null__L0.csv; counted once
* null file photonnet_xl_null_d0__L0.csv is a byte-identical copy of photonnet_xl_null__L0.csv; counted once

The 3-seed ensemble composition of a primary file cannot be verified from the score file; the seed files found are listed above. Nothing here is a medical-device claim.
