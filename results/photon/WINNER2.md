# Round 2 winner (mechanical application of ROUND2.md, Amendment 7)

Written by `python -m oxydrop_ml.photon.winner2` (Sun Nov 15 19:33:34 CST 2026) from the score files that existed at that moment. Rerun it after every new file: it ignores missing and partial files and says so. The rule is quoted verbatim from `.agents/tasks/task-oxydrop-photon-model/ROUND2.md` (stated 18:25 CT, before any round-2 score):

> Only round-2 trained models are eligible. Primary: L0 TM bal. acc. (3-seed ensemble) paired vs chain_v1. "Beats chain" = paired CI excluding 0, >= 5/6 people improved, worst person not below chain's, nulls passed. If none beats chain: the eligible model with the highest L0 point estimate whose paired CI includes 0 (a tie), breaking near-ties (within 0.01) by mean paired diff over L3..L5 and X1, then by answered accuracy at its own coverage under stress. All round-2 models are designed after round-1 outer results were seen; this is disclosed as Amendment 7.

## Verdict

* Rule step 1: no round-2 model beats chain_v1 (all of: paired CI above 0, >= 5/6 people better, worst person not below chain_v1, nulls passed).
* `photonnet` loses to chain_v1 at L0 (paired -0.128 [-0.208, -0.047]); not eligible for the tie rule.
* `distill` loses to chain_v1 at L0 (paired -0.038 [-0.103, -0.001]); not eligible for the tie rule.
* `oxyssl` loses to chain_v1 at L0 (paired -0.130 [-0.274, -0.044]); not eligible for the tie rule.
* `oxyseq` loses to chain_v1 at L0 (paired -0.088 [-0.221, -0.023]); not eligible for the tie rule.
* Rule step 2: no round-2 model is a tie with passed nulls; the pre-stated rule (18:25 CT) has no branch for this case.
* Extension: `photonnet` is excluded because its null controls are NOT PASSED: null mean 0.618 > 0.55.
* Extension: `oxyssl` is excluded because its null controls are NOT PASSED: null mean 0.643 > 0.55.
* Rule step 3 (planner extension, task.json 19:15/19:32 CT, stated after the round-2 outer results were seen, disclosed in Amendment 7): when no eligible model beats or ties chain_v1, ship the eligible model with passed nulls and the highest L0 point estimate, and say that it loses. Candidates: distill 0.758, oxyseq 0.708.
* **SHIPPED (extension): `distill`, which LOSES to chain_v1 on clean light**: L0 TM 0.758 [0.603, 0.813] vs chain_v1, paired -0.038 [-0.103, -0.001], 2/6 people better. It is not a win and not a tie; mean paired difference over L3+L4+L5+X1 is +0.133 (simulated distant-flash stress).

Reference: `chain_v1` L0 TM 0.796 [0.636, 0.857], worst person 0.458.

## Eligible models at L0 (primary file = the 3-seed ensemble)

| model | status | L0 TM [95 % CI] | paired diff vs chain_v1 | people better (sign p) | worst person vs chain_v1 | (a) CI > 0 | (b) >= 5/6 | (c) worst ok | (d) nulls | missing levels |
|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | loss | 0.668 [0.508, 0.785] | -0.128 [-0.208, -0.047] | 2/6 (p = 0.891) | 0.500 vs 0.458 | no | no | yes | NOT PASSED: null mean 0.618 > 0.55 | none |
| distill | loss | 0.758 [0.603, 0.813] | -0.038 [-0.103, -0.001] | 2/6 (p = 0.891) | 0.516 vs 0.458 | no | no | yes | passed: null mean 0.492 over 3 draw(s) | none |
| oxyssl | loss | 0.667 [0.498, 0.741] | -0.130 [-0.274, -0.044] | 1/6 (p = 0.984) | 0.400 vs 0.458 | no | no | no | NOT PASSED: null mean 0.643 > 0.55 | none |
| oxyseq | loss | 0.708 [0.452, 0.790] | -0.088 [-0.221, -0.023] | 2/6 (p = 0.891) | 0.516 vs 0.458 | no | no | yes | passed: null mean 0.475 over 1 draw(s), blank mean 0.500 over 1 draw(s) | none |

## Tie-break inputs (only used among near-ties, within 0.01 of the top tie)

| model | L0 TM | mean paired diff over L3..L5, X1 | mean answered acc over the same levels | levels used |
|---|---|---|---|---|
| photonnet | 0.668 | 0.145 | 0.917 | L3+L4+L5+X1 |
| distill | 0.758 | 0.133 | 0.947 | L3+L4+L5+X1 |
| oxyssl | 0.667 | 0.062 | 0.679 | L3+L4+L5+X1 |
| oxyseq | 0.708 | -0.040 | 0.555 | L3+L4+L5+X1 |

## Seeds and ablations found

| model | seed files with L0 | seed-mean L0 TM | ablations (not eligible, not promotable) |
|---|---|---|---|
| photonnet | 3 | 0.664 | photonnet_nophys |
| distill | 3 | 0.753 | distill_nokd, distill_v1, distill_v1_nokd, distill_v1_s0, distill_v1_s1, distill_v1_s2 |
| oxyssl | 3 | 0.670 | oxyssl_init, oxyssl_noaug |
| oxyseq | 0 | n/a | oxyseq_notta |

ROUND2.md rule 6 asks for three seeds of the final configuration and reports the seed mean next to the ensemble; the rule above uses the ensemble (primary file).

## Missing models and files

* `photonnet`: has L0, L1, L2, L3, L4, L5, X1; missing none
* `distill`: has L0, L1, L2, L3, L4, L5, X1; missing none
* `oxyssl`: has L0, L1, L2, L3, L4, L5, X1; missing none
* `oxyseq`: has L0, L1, L2, L3, L4, L5, X1; missing none

## Skipped files (partial, unreadable or in progress)

* null file photonnet_null_d0__L0.csv is a byte-identical copy of photonnet_null__L0.csv; counted once

See `SIDE_BY_SIDE.md` for the clean and stress tables of every round-2 model, ablation and baseline. Nothing here is a medical-device claim.
