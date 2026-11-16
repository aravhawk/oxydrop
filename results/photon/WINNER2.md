# Round 2 winner (mechanical application of ROUND2.md, Amendment 7)

Written by `python -m oxydrop_ml.photon.winner2` (Sun Nov 15 18:36:22 CST 2026) from the score files that existed at that moment. Rerun it after every new file: it ignores missing and partial files and says so. The rule is quoted verbatim from `.agents/tasks/task-oxydrop-photon-model/ROUND2.md` (stated 18:25 CT, before any round-2 score):

> Only round-2 trained models are eligible. Primary: L0 TM bal. acc. (3-seed ensemble) paired vs chain_v1. "Beats chain" = paired CI excluding 0, >= 5/6 people improved, worst person not below chain's, nulls passed. If none beats chain: the eligible model with the highest L0 point estimate whose paired CI includes 0 (a tie), breaking near-ties (within 0.01) by mean paired diff over L3..L5 and X1, then by answered accuracy at its own coverage under stress. All round-2 models are designed after round-1 outer results were seen; this is disclosed as Amendment 7.

## Verdict

* No round-2 model has an L0 score file yet; no verdict is possible.

Reference: `chain_v1` L0 TM 0.796 [0.636, 0.857], worst person 0.458.

## Eligible models at L0 (primary file = the 3-seed ensemble)

| model | status | L0 TM [95 % CI] | paired diff vs chain_v1 | people better (sign p) | worst person vs chain_v1 | (a) CI > 0 | (b) >= 5/6 | (c) worst ok | (d) nulls | missing levels |
|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | no L0 score file yet |  |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| distill | no L0 score file yet |  |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| oxyssl | no L0 score file yet |  |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |
| oxyseq | no L0 score file yet |  |  |  |  |  |  |  | no null files yet | L0, L1, L2, L3, L4, L5, X1 |

## Tie-break inputs (only used among near-ties, within 0.01 of the top tie)

No round-2 L0 file yet.

## Seeds and ablations found

| model | seed files with L0 | seed-mean L0 TM | ablations (not eligible, not promotable) |
|---|---|---|---|
| photonnet | 0 | n/a | none |
| distill | 0 | n/a | none |
| oxyssl | 0 | n/a | none |
| oxyseq | 0 | n/a | none |

ROUND2.md rule 6 asks for three seeds of the final configuration and reports the seed mean next to the ensemble; the rule above uses the ensemble (primary file).

## Missing models and files

* `photonnet`: no score file yet
* `distill`: no score file yet
* `oxyssl`: no score file yet
* `oxyseq`: no score file yet

See `SIDE_BY_SIDE.md` for the clean and stress tables of every round-2 model, ablation and baseline. Nothing here is a medical-device claim.
