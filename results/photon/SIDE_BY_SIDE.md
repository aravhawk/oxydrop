# Side by side: round-2 trained models against the round-1 baselines

Written by `python -m oxydrop_ml.photon.winner2` from `results/photon/scores/*.csv` (the same bootstrap as `TABLE.csv`: TM = Hoffman time-matched balanced accuracy at the forced decision `pred`, 95 % person-bootstrap CI, paired differences to `chain_v1` at the same level, same resampled people). Clean light (L0) next to the simulated distant-flash levels L3 to L5 and the out-of-family stress X1. All stress levels are simulations, not measurements of any phone. A difference is a **win** only when its paired CI excludes 0; otherwise it is a tie. Missing files show `n/a`. Models are not refit per level. Round-2 models were designed after the round-1 outer results were seen (Amendment 7), so their L0 numbers are optimistic.

## A. TM balanced accuracy [95 % CI]

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| **Baselines (round 1, for reference)** |  |  |  |  |  |
| chain_v1 | 0.796 [0.636, 0.857] | 0.565 [0.503, 0.625] | 0.598 [0.526, 0.640] | 0.535 [0.477, 0.598] | 0.508 [0.439, 0.554] |
| hybrid_final | 0.779 [0.638, 0.831] | n/a | n/a | n/a | n/a |
| cnn_full | 0.753 [0.573, 0.817] | n/a | n/a | n/a | n/a |
| ctl_dc_lr | 0.744 [0.543, 0.848] | 0.692 [0.529, 0.769] | 0.658 [0.511, 0.748] | 0.598 [0.536, 0.652] | 0.646 [0.511, 0.735] |
| twin_inverse | 0.752 [0.577, 0.810] | 0.652 [0.524, 0.770] | 0.624 [0.550, 0.702] | 0.625 [0.537, 0.724] | 0.615 [0.530, 0.683] |
| twin_gate | 0.796 [0.636, 0.857] | 0.677 [0.600, 0.752] | 0.715 [0.621, 0.751] | 0.647 [0.561, 0.717] | 0.514 [0.454, 0.557] |

## B. Paired difference to chain_v1 at the same level [95 % CI], people better of compared

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| hybrid_final | -0.017 [-0.103, +0.069] 2/5 tie | n/a | n/a | n/a | n/a |
| cnn_full | -0.043 [-0.140, +0.005] 2/6 tie | n/a | n/a | n/a | n/a |
| ctl_dc_lr | -0.052 [-0.155, +0.043] 2/6 tie | +0.127 [-0.015, +0.232] 4/6 tie | +0.060 [-0.051, +0.163] 5/6 tie | +0.064 [-0.026, +0.150] 4/6 tie | +0.139 [-0.014, +0.225] 5/5 tie |
| twin_inverse | -0.044 [-0.127, -0.000] 2/6 loss | +0.087 [+0.000, +0.170] 4/6 win | +0.026 [-0.020, +0.087] 4/6 tie | +0.090 [+0.020, +0.154] 5/5 win | +0.107 [+0.023, +0.213] 6/6 win |
| twin_gate | +0.000 [+0.000, +0.000] 0/0 tie | +0.112 [+0.052, +0.172] 5/6 win | +0.117 [+0.061, +0.178] 6/6 win | +0.112 [+0.022, +0.207] 5/6 win | +0.007 [+0.000, +0.026] 1/1 tie |

## C. Worst person TM (the weakest of the six held-out people)

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| chain_v1 | 0.458 | 0.502 | 0.445 | 0.393 | 0.393 |
| hybrid_final | 0.517 | n/a | n/a | n/a | n/a |
| cnn_full | 0.500 | n/a | n/a | n/a | n/a |
| ctl_dc_lr | 0.514 | 0.548 | 0.500 | 0.500 | 0.500 |
| twin_inverse | 0.500 | 0.531 | 0.385 | 0.500 | 0.455 |
| twin_gate | 0.458 | 0.537 | 0.500 | 0.537 | 0.393 |

## D. Own abstention: coverage, accuracy on answered windows, accuracy of `pred` on all windows (the random-abstention reference)

Random abstention at the same coverage keeps the all-window accuracy in expectation, so answered accuracy only means something when it exceeds the all-window column.

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| chain_v1 | cov 0.84 / ans 0.939 / all 0.875 | cov 0.92 / ans 0.534 / all 0.538 | cov 0.86 / ans 0.575 / all 0.572 | cov 0.91 / ans 0.491 / all 0.492 | cov 0.92 / ans 0.507 / all 0.505 |
| hybrid_final | cov 1.00 / ans 0.813 / all 0.813 | n/a | n/a | n/a | n/a |
| cnn_full | cov 1.00 / ans 0.798 / all 0.798 | n/a | n/a | n/a | n/a |
| ctl_dc_lr | cov 0.57 / ans 0.917 / all 0.771 | cov 0.53 / ans 0.909 / all 0.735 | cov 0.45 / ans 0.882 / all 0.693 | cov 0.45 / ans 0.733 / all 0.578 | cov 0.47 / ans 0.861 / all 0.682 |
| twin_inverse | cov 0.35 / ans 0.915 / all 0.821 | cov 0.23 / ans 0.697 / all 0.653 | cov 0.01 / ans 0.568 / all 0.646 | cov 0.00 / ans n/a / all 0.586 | cov 0.18 / ans 0.637 / all 0.605 |
| twin_gate | cov 0.84 / ans 0.939 / all 0.875 | cov 0.62 / ans 0.650 / all 0.668 | cov 0.01 / ans 0.967 / all 0.775 | cov 0.00 / ans n/a / all 0.673 | cov 0.82 / ans 0.523 / all 0.513 |

## E. Per-person TM at L0

| model | person 100001 | person 100002 | person 100003 | person 100004 | person 100005 | person 100006 |
|---|---|---|---|---|---|---|
| chain_v1 | 0.889 | 0.458 | 0.660 | 0.711 | 0.737 | 0.539 |
| hybrid_final | 0.594 | 0.517 | 0.656 | 0.766 | 0.735 | 0.539 |
| cnn_full | 0.633 | 0.500 | 0.659 | 0.624 | 0.766 | 0.509 |
| ctl_dc_lr | 0.659 | 0.514 | 0.626 | 0.625 | 0.693 | 0.869 |
| twin_inverse | 0.662 | 0.500 | 0.654 | 0.656 | 0.689 | 0.632 |
| twin_gate | 0.889 | 0.458 | 0.660 | 0.711 | 0.737 | 0.539 |

## F. Null and blank controls at L0 (mean over draws; must be <= 0.55)

| model | kind | draws | mean TM | max TM | check | files |
|---|---|---|---|---|---|---|
| ctl_dc_lr (round 1) | null | 1 | 0.500 | 0.500 | ok | ctl_dc_lr_null |
| twin_inverse (round 1) | null | 1 | 0.406 | 0.406 | ok | twin_inverse_null |
| twin_gate (round 1) | null | 1 | 0.578 | 0.578 | flag (single draw) | twin_gate_gperm_null |
| twin_gate (round 1) | null | 1 | 0.623 | 0.623 | flag (single draw) | twin_gate_null |
| twin_gate (round 1) | null | 1 | 0.550 | 0.550 | flag (single draw) | twin_gate_perm_null |

