### Amendment 8 (Nov 15 2026: round 3, scale-up with cloud compute; written at 19:52 CST, BEFORE any round-3 score file existed)

Time stamp. `date` at the start of writing: Sun Nov 15 07:52:26 PM CST 2026. The winner rule below was fixed by the planner at 19:55 CT in `.agents/tasks/task-oxydrop-photon-model/ROUND3.md` (copied verbatim in section D) before any round-3 score. This file is written by the round-3 evaluation child (R3-EVAL), which has not opened or produced any round-3 score.

Disclosure about files on disk. At the time of writing, `ls results/photon/scores` (377 files) showed **no** file whose name contains `distill2`, `multida`, `ssl3`, `photonnet_xl` or `ens3` (`ls | grep -Ei 'distill2|multida|ssl3|photonnet_xl|ens3'` returned nothing). The newest file in that directory was dated 19:22 CT (`photonnet_nophys_null*__L0.csv`), i.e. all round-2 files. No round-3 candidate directory or log was found under `python/` or `scripts/` either. The candidates below are therefore named from the evaluation brief only; their authors own the design details and must record any deviation in their own logs.

Nothing here changes Amendment 6 or Amendment 7. Amendment 7 stays as written, including its decision (`distill` shipped; it loses to `chain_v1` at clean L0 by 0.038). Round 3 is a **third look at the same six people**.

#### A. What had been seen before the round-3 candidates were designed

* **All of round 1 and round 2**, on the six held-out Hoffman people: every outer-fold L0 to L5 and X1 score in `results/photon/TABLE.csv`, `STRESS.csv`, `NULLS.csv`, `WINNER.md` and `WINNER2.md`, and all of Amendment 6 and Amendment 7 including their outcome sections.
* The numbers that motivate round 3 (ROUND3.md): `chain_v1` L0 TM 0.796 [0.636, 0.857]. The shipped `distill` L0 0.758 [0.603, 0.813], paired vs chain_v1 -0.038 [-0.103, -0.001] (loss, 2/6 people better); L4 +0.083 [0.008, 0.178] 6/6, L5 +0.166 [0.035, 0.242], X1 +0.169 [0.058, 0.285] 6/6; nulls mean 0.492. `distill` teacher weight hit the grid edge (8) in 18/18 folds. `oxyseq` 0.708 (loss), `photonnet` 0.668 (loss; null 0.618 fails), `oxyssl` 0.667 (loss; null 0.643 fails).
* `distill` v2 was tuned after its v1 outer results were seen (Amendment 7, disclosure 2).
* The user's round-3 steer (19:50 CT): go all in until 9:00 PM CT, no scope cuts, offload heavy training to Daytona/Kaggle GPUs, winner must be a genuinely trained end-to-end model with real in-browser inference; target 90%+ TM balanced accuracy under honest LOSO, stated as a target and not a promise.

**Consequence.** Every round-3 model is designed after the outer results of rounds 1 and 2 were seen, on the same six people, so every round-3 L0 number is optimistic and not a blind test. Selection among round-3 candidates by the L0 point estimate adds a further, unquantified selection effect.

**Not seen by the author of this amendment**: any score of the round-3 candidates (or any seed, ablation, null, blank or stress file of them).

#### B. Round-3 candidates (named; details are the candidate authors')

1. `distill2`: second-generation PhotonDistill (same 39-feature causal front end, `app/js/distill_features.js`, and an `nn_runtime-v1` network).
2. `multida`: a multi-augmentation-domain variant (distant-flash and other declared augmentation priors).
3. `ssl3`: third-generation self-supervised pretraining then LOSO fine-tuning (OxyNet-shaped or `nn_runtime-v1`).
4. `photonnet_xl`: a larger PhotonNet (differentiable physics layer inside the network).
5. `ens3`: a 3-member ensemble (its members and the combination rule are fixed by its author; whether it counts as one trained model under ROUND2 rule 3 is not settled by this amendment, and the winner script only checks the numeric rule).

Common rules (ROUND2.md rules 1 to 6 plus ROUND3.md rules 1 to 5): same harness and score-file format (`results/photon/CONTRACT.md`), `python -m oxydrop_ml.photon.evalkit`, LOSO by person, all selection nested on inner person-grouped CV of the five training people, stress test on held-out people only, nulls with at least 3 global permutation draws (mean <= 0.55; temporal models also blank), no test-subject leakage, no tuning on test folds, no app-side tricks. L0 3-seed ensemble files and nulls by 20:15 CT, stress levels by 20:30 CT; later files are candidates only and cannot replace the shipped model. Shippability (rule 3): distill's 39-feature front end plus an `nn_runtime-v1` network, or the OxyNet tensor plus an OxyNet-shaped network.

#### C. Mechanical application

`python/oxydrop_ml/photon/winner3.py` (written by R3-EVAL before any round-3 score) reads the score files through `evalkit` and applies section D with no discretion. Byte-identical null files (same SHA-256) count once. A candidate with fewer than 3 distinct null draws fails the null gate. The "shippable by rule 3" condition is not computable from scores and is reported as a manual column (default: not assumed). Output: `results/photon/WINNER3.md`.

#### D. Winner rule for round 3 (verbatim from ROUND3.md, stated 19:55 CT before any round-3 score)

> Eligible: trained models from rounds 2 and 3 that pass nulls and rule 3. Replace `distill` only if a candidate's L0 3-seed ensemble TM point estimate exceeds distill's (0.758) AND its paired difference vs distill has a lower CI bound > -0.02 AND its stress mean paired diff vs chain_v1 over L3-L5 and X1 is >= 0. "Beats chain" still requires the paired CI vs chain_v1 excluding 0, >= 5/6 people better, worst person not below chain's. All round-3 models are designed after rounds 1-2 outer results were seen (Amendment 8).

#### E. Outcome

First filled at about 20:28 CT (file mtime 20:27; an earlier draft of this line said 20:35, which was wrong) and re-verified by R3-EVAL at 20:31 CT against a regenerated `WINNER3.md` (numbers, the three distill2 null draws 0.543, 0.525, 0.444 and the multida and photonnet_xl null means rechecked from the score files). Source:  `results/photon/WINNER3.md` / `WINNER3.csv` (regenerated by `python -m oxydrop_ml.photon.winner3 --shippable distill2 --lock distill2 --lock-note ...`) and `results/photon/TABLE.csv` / `STRESS.csv`. No number below is typed from memory.

**Decision (planner ROUND3 LOCK, 20:22 CT): `distill2` (PhotonDistill v2) replaces `distill` as the shipped model. It TIES the logistic chain_v1 on clean light and does not beat it.**

* `distill2` L0 3-seed ensemble TM 0.784 [0.596, 0.834]; paired vs chain_v1 -0.012 [-0.091, +0.012] (tie: the CI includes 0), 4/6 people better, worst person 0.500 (chain_v1 0.458). "Beats chain" fails on (a) the CI and (b) people better (4/6 < 5/6).
* Replacement conditions of section D: g1 0.784 > distill 0.758 (pass); g2 paired vs distill +0.026 [-0.010, +0.040], lower bound -0.010 > -0.02 (pass); g3 mean stress paired difference vs chain_v1 over L3, L4, L5, X1 = +0.144 (pass; L3 +0.133 [+0.001, +0.248] 4/6, L4 +0.097 [+0.011, +0.195] 6/6, L5 +0.170 [+0.027, +0.249] 5/6, X1 +0.174 [+0.044, +0.284] 6/6); nulls 0.543, 0.525, 0.444 over 3 distinct draws, mean 0.504 (pass); rule 3: same 39-feature front end and a 2-hidden-layer nn-runtime-v1 MLP (author-confirmed; app export by R3-EXPORT).
* **Late by one minute (disclosed, waived by the planner).** distill2's L0 ensemble landed at 20:08 CT; its three null files and all stress files landed at 20:16 CT, one minute after the stated 20:15 CT cutoff (`distill2_log.md`; the mtime on this box is 20:18 because the files were copied after the sandbox run). The winner script flags them LATE mechanically; the lock waives that flag in writing, next to the flag, in WINNER3.md.
* Teachers (`distill2_log.md`): the GBDT-augmented teacher was never chosen by the nested inner search (base teacher in 18/18 folds); the physics (twin_inverse) teacher was not used, and the oxyseq/oxytrack outputs were not used as teachers (they would leak). The L0 gain over distill comes from the larger KD weight (lam 16 or 32 in 16/18 folds), the temperature and the width.
* Other round-3 candidates: `multida` L0 0.744 [0.553, 0.807], paired vs chain_v1 -0.053 [-0.142, -0.010] (loss; fails g1 and g2). `photonnet_xl` L0 0.660 [0.545, 0.762], -0.136 [-0.210, -0.033] (loss; nulls mean 0.412 pass). `ssl3`: no score file by the lock or by this outcome. `ens3` (stacking of distill, oxyseq and distill_nokd, not one network): score files landed at 20:25 CT, after the lock, with no null files; L0 0.789 [0.621, 0.842], paired vs chain_v1 -0.007 [-0.092, +0.043] (tie, 4/6). It cannot replace the shipped model (late, nulls missing, and a three-model stack is not shippable under rule 3); reported as a candidate only.
* The 90% TM balanced-accuracy target of the 19:50 CT steer was **not reached**: the best eligible L0 point estimate is 0.784.
* As section A says, every round-3 L0 number is a third look at the same six people and is optimistic; picking the round-3 winner by its L0 point estimate adds a selection effect that is not quantified.
