### Amendment 3 (Nov 15 2026, FEAT-008, P2 final HR rule; written after the Hoffman rule table of seed 0 and the pooled P2 table of `cnn` vs `spec_v2` were seen, before any other rule was applied to BUT PPG or MTHS)

What was seen first, so the reader can judge the forking path. The Hoffman inner out-of-fold HR table of the plain CNN (seed 0) gave within-5 `cnn` 0.803, `spec` 0.883, `guard` 0.817, `agree` 0.823, so the argmax rule of Amendment 2 would choose the spectral estimator alone. The pooled true-LOSO P2 table of the CNN (0.468) against the v2 spectral estimator (0.419) had also been printed.

Changes to Amendment 2 (P2 part only; P1 is untouched):

1. The headline HR output must contain the CNN, the same principle as P1 (the CNN stays in the inference path). The headline rule is therefore the best rule by Hoffman inner within-5 **among the rules that use the CNN**: `cnn`, `guard`, `agree` and the new rule `gate`. `spec` alone is still computed, reported and compared as a control (it is the v2 baseline).
2. New rule `gate`: the v2 spectral estimate when the shipped v2 quality gate passes (the estimate of `best_channel_estimate`: SQI >= 0.30, ACF >= 0.30, harmonic agreement 15 %), otherwise the CNN. Device agnostic: the gate is the shipped one and is not fit to any phone.
3. Selection population unchanged: Hoffman inner out-of-fold windows (6 people, disjoint from every BUT and MTHS test person). Ties inside 0.005 within-5 go to the earlier rule in the order `cnn`, `gate`, `guard`, `agree`.
4. The P2 test people (BUT PPG + MTHS, 100 people) never choose a rule. The comparison to the best P2 baseline uses the same paired person bootstrap as before; a win needs the interval to exclude zero.
