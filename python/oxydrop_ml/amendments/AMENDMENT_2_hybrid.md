### Amendment 2 (Nov 15 2026, written by FEAT-008 BEFORE any outer-fold score of the new hybrid models was produced or read)

Context. The core CNN (FEAT-006) tied the best honest P1 baseline (v1 chain, time-matched balanced accuracy 0.796) and one seed lost to it. FEAT-008 therefore builds a physics-informed hybrid whose headline score is still the **unchanged primary metric P1** (Hoffman, leave-one-person-out, windows with t >= 60 s, time-matched balanced accuracy at full coverage, threshold at logit 0, same 5,644 windows). Splits, labels, baselines, the win rule (paired person-bootstrap 95 % interval of the difference to the **best** honest baseline excludes zero) and the bootstrap (people resampled, B = 2000, seed 0) are unchanged. P2 and P3 definitions are unchanged.

What was already seen before this amendment (so these rows are not blind): the plain CNN (`P` below, seed 0) and the chain + CNN stack of `stack_run.py` on the outer folds (`results/ml/hybrid_p1.csv`, 0.752 and 0.796), and test-time BatchNorm adaptation (0.638 to 0.682, it hurts). Test-time BatchNorm adaptation is not repeated in FEAT-008.

#### Model families (every one is a full LOSO + nested run)

All families use the FEAT-006 recipe (SSL + OpenOximetry warm-up init `stageB_full.pt`, 400 fine-tuning steps, AdamW lr 1e-3, `w_low` 3.0, early stop on inner validation people, student of 167,588 parameters, no change to the exported graph). Per family and seed there are 6 outer fits (one Hoffman person held out) and 30 inner fits (outer person o and inner person q held out, for every ordered pair, q is scored out of fold).

* `P` plain CNN (the FEAT-006 student).
* `R` **residual hybrid**. The low-oxygen logit of a window is `prior + net`, where `prior` is the v1 chain logit (two class-balanced ridge-logistic models on dHR and the 19 calibration-relative FUSION features, shipped C) and the network output `net` is a learned correction. Training rows get an out-of-person chain prior (chain fit on the training people without the row's own person); the held-out person gets the chain fit on all training people. Rows without a chain feature (OpenOximetry contact rows) use prior 0 in their own domain; MTHS low-oxygen loss terms are switched off in `R` (normoxic rows have no chain prior and would bias the correction). Early stopping also adds the prior. Export: the network is unchanged, the app adds the chain logit to the network output.
* `D` **chain distillation**. As `P`, plus a soft-target term `0.5 * BCE(sigmoid(net logit), sigmoid(prior))` on Hoffman training rows, with the same out-of-person chain prior as teacher. At inference the student runs alone (no chain input).

Seeds 0, 1, 2 for every family; the seed ensemble is the mean of the seed logits.

#### Selection rule (inner folds only, fixed now)

For each outer person o, the only data used to select are the inner out-of-fold predictions of the 5 other people (each inner person q scored by the fits that exclude {o, q}). The score of a candidate is the leave-one-inner-person-out CV time-matched balanced accuracy of its meta-learner on the pooled inner predictions. The outer person's windows are only scored after the choice is frozen. The meta-learner is a standardised, class-balanced logistic regression (C = 1) on the feature vector below, decision threshold 0. Feature logits are seed means.

1. **Source** (the CNN must stay in the inference path, so the headline is chosen among CNN candidates only; the two chain rows are controls and are always reported):
   control `chain_shipped` (raw chain logit, thr 0), control `chain_platt` `[c]`;
   CNN candidates in this order of simplicity: `P` `[p]`, `D` `[d]`, `R` `[prior + net]`, `c+p` `[c, p]`, `c+d` `[c, d]`, `c+corr` `[c, net_R]`, `all` `[c, p, d, net_R]`.
2. **Per-user baseline calibration** off or on: each feature minus the median of that feature over the person's own label-free first minute (windows ending before 60 s of the same recording). On only if its inner CV score beats off by at least 0.01.
3. **Multi-window aggregation** k in {1, 6, 11, 21} trailing windows (2 s hop, so 10, 20, 30, 50 s of signal; causal mean of the features within the recording). The smallest k whose inner CV score is within 0.01 of the best is taken.

Ties in step 1: the earliest candidate in the order above whose inner CV score is within 0.01 of the best CNN candidate. The same procedure restricted to the chain (`chain_platt` + steps 2 and 3) is reported as the matched-lever control, so any gain from steps 2 and 3 is not credited to the CNN.

Exploratory rows (reported, never used for selection, labelled exploratory): every candidate at default levers, single seeds, fold ensembling (the 5 inner fits that exclude o also score o; their mean logit with the outer fit), the train-vs-test gap of the whole pipeline (pipeline score on the outer training people, using the outer fit's in-sample logits, versus the held-out score), and the inner-CV minus outer-test optimism of the selected pipeline.

#### P2 final HR choice (selection population disjoint from the P2 test people)

Candidates: `cnn` (seed median), `spec_v2` (the v2 spectral + autocorrelation baseline), `guard` (CNN unless it is within 8 bpm of 2x or 0.5x the spectral estimate, then the spectral estimate), `agree` (mean of the two if they are within 5 bpm of each other, else CNN). The rule is selected by HR within-5 on the Hoffman inner out-of-fold windows (6 Hoffman people with pulse-oximeter HR, none of which is a BUT or MTHS test person), then applied unchanged to the true-LOSO BUT PPG + MTHS predictions of the final model (3 seeds, 100 people) and compared with the best P2 baseline by the paired person bootstrap. P2 CNN predictions are the FEAT-006 LOSO runs (the recipe `P`); the HR heads of `R` and `D` are not evaluated for P2.

#### Device agnosticism

Nothing here is fit to a particular phone. The three iPhones are a held-out sanity check after the fact, never a tuning set. Per-user calibration uses only the person's own first minute.

#### Limits stated now

Six Hoffman people; every interval is wide; with a handful of candidates a selection optimism remains (inner CV on 5 people is noisy); the OpenOximetry warm-up and SSL were fit on people disjoint from Hoffman, MTHS and BUT; no phone other than the Hoffman devices contributes to P1. A tie with the chain is reported as a tie.
