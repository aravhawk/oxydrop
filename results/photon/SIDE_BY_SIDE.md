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
| **Round 2: photonnet** |  |  |  |  |  |
| photonnet | 0.668 [0.508, 0.785] | 0.689 [0.526, 0.812] | 0.703 [0.516, 0.829] | 0.720 [0.564, 0.818] | 0.672 [0.537, 0.775] |
| photonnet_nophys | 0.642 [0.528, 0.749] | 0.650 [0.549, 0.765] | 0.657 [0.544, 0.756] | 0.662 [0.555, 0.753] | 0.663 [0.507, 0.777] |
| photonnet seed mean over 3 seed file(s) | 0.664 (n=3) | n/a | n/a | n/a | n/a |
| **Round 2: distill** |  |  |  |  |  |
| distill | 0.758 [0.603, 0.813] | 0.680 [0.552, 0.772] | 0.681 [0.566, 0.772] | 0.701 [0.570, 0.770] | 0.676 [0.564, 0.747] |
| distill_nokd | 0.660 [0.529, 0.760] | 0.633 [0.503, 0.771] | 0.654 [0.504, 0.750] | 0.705 [0.548, 0.767] | 0.626 [0.483, 0.750] |
| distill_v1 | 0.696 [0.524, 0.795] | 0.651 [0.516, 0.783] | 0.667 [0.536, 0.760] | 0.706 [0.581, 0.776] | 0.658 [0.523, 0.782] |
| distill_v1_nokd | 0.670 [0.548, 0.769] | 0.624 [0.503, 0.763] | 0.653 [0.495, 0.762] | 0.695 [0.551, 0.752] | 0.615 [0.477, 0.742] |
| distill_v1_s0 | 0.705 [0.554, 0.785] | 0.625 [0.481, 0.770] | 0.676 [0.553, 0.765] | 0.720 [0.574, 0.776] | 0.631 [0.500, 0.754] |
| distill_v1_s1 | 0.699 [0.564, 0.818] | 0.649 [0.528, 0.774] | 0.655 [0.511, 0.752] | 0.691 [0.594, 0.762] | 0.662 [0.533, 0.781] |
| distill_v1_s2 | 0.709 [0.515, 0.821] | 0.686 [0.544, 0.802] | 0.687 [0.539, 0.782] | 0.712 [0.569, 0.789] | 0.670 [0.540, 0.776] |
| distill seed mean over 3 seed file(s) | 0.753 (n=3) | 0.680 (n=3) | 0.684 (n=3) | 0.704 (n=3) | 0.671 (n=3) |
| **Round 2: oxyssl** |  |  |  |  |  |
| oxyssl | 0.667 [0.498, 0.741] | 0.697 [0.596, 0.733] | 0.548 [0.503, 0.601] | 0.522 [0.500, 0.552] | 0.686 [0.567, 0.732] |
| oxyssl_init | 0.506 [0.410, 0.581] | 0.540 [0.478, 0.593] | 0.500 [0.499, 0.502] | 0.500 [0.500, 0.500] | 0.524 [0.488, 0.556] |
| oxyssl_noaug | 0.652 [0.455, 0.779] | 0.701 [0.578, 0.781] | 0.502 [0.500, 0.507] | 0.500 [0.500, 0.500] | 0.672 [0.536, 0.751] |
| oxyssl seed mean over 3 seed file(s) | 0.670 (n=3) | 0.707 (n=3) | 0.543 (n=3) | 0.532 (n=3) | 0.687 (n=3) |
| **Round 2: oxyseq** |  |  |  |  |  |
| oxyseq | 0.708 [0.452, 0.790] | 0.533 [0.399, 0.620] | 0.503 [0.493, 0.515] | 0.498 [0.490, 0.505] | 0.508 [0.441, 0.587] |
| oxyseq_notta | 0.639 [0.437, 0.764] | 0.559 [0.445, 0.641] | 0.446 [0.404, 0.517] | 0.468 [0.438, 0.507] | 0.513 [0.444, 0.587] |

## B. Paired difference to chain_v1 at the same level [95 % CI], people better of compared

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| hybrid_final | -0.017 [-0.103, +0.069] 2/5 tie | n/a | n/a | n/a | n/a |
| cnn_full | -0.043 [-0.140, +0.005] 2/6 tie | n/a | n/a | n/a | n/a |
| ctl_dc_lr | -0.052 [-0.155, +0.043] 2/6 tie | +0.127 [-0.015, +0.232] 4/6 tie | +0.060 [-0.051, +0.163] 5/6 tie | +0.064 [-0.026, +0.150] 4/6 tie | +0.139 [-0.014, +0.225] 5/5 tie |
| twin_inverse | -0.044 [-0.127, -0.000] 2/6 loss | +0.087 [+0.000, +0.170] 4/6 win | +0.026 [-0.020, +0.087] 4/6 tie | +0.090 [+0.020, +0.154] 5/5 win | +0.107 [+0.023, +0.213] 6/6 win |
| twin_gate | +0.000 [+0.000, +0.000] 0/0 tie | +0.112 [+0.052, +0.172] 5/6 win | +0.117 [+0.061, +0.178] 6/6 win | +0.112 [+0.022, +0.207] 5/6 win | +0.007 [+0.000, +0.026] 1/1 tie |
| photonnet | -0.128 [-0.208, -0.047] 2/6 loss | +0.125 [-0.019, +0.267] 4/6 tie | +0.105 [-0.025, +0.245] 5/6 tie | +0.185 [+0.023, +0.290] 5/6 win | +0.165 [+0.020, +0.316] 6/6 win |
| photonnet_nophys | -0.154 [-0.218, -0.061] 2/6 loss | +0.086 [+0.003, +0.195] 4/6 win | +0.059 [-0.018, +0.168] 5/6 tie | +0.127 [+0.016, +0.217] 5/6 win | +0.155 [-0.003, +0.305] 5/6 tie |
| distill | -0.038 [-0.103, -0.001] 2/6 loss | +0.116 [-0.002, +0.226] 4/6 tie | +0.083 [+0.008, +0.178] 6/6 win | +0.166 [+0.035, +0.242] 5/6 win | +0.169 [+0.058, +0.285] 6/6 win |
| distill_nokd | -0.136 [-0.216, -0.038] 2/6 loss | +0.068 [-0.044, +0.192] 3/6 tie | +0.056 [-0.049, +0.150] 5/6 tie | +0.171 [+0.022, +0.253] 5/6 win | +0.119 [-0.019, +0.291] 6/6 tie |
| distill_v1 | -0.100 [-0.201, -0.022] 2/6 loss | +0.086 [-0.032, +0.205] 4/6 tie | +0.069 [-0.025, +0.163] 6/6 tie | +0.171 [+0.041, +0.260] 5/6 win | +0.150 [+0.007, +0.318] 6/6 win |
| distill_v1_nokd | -0.126 [-0.211, -0.013] 2/6 loss | +0.059 [-0.043, +0.188] 3/6 tie | +0.055 [-0.054, +0.153] 5/6 tie | +0.161 [+0.024, +0.233] 6/6 win | +0.108 [-0.026, +0.283] 6/6 tie |
| distill_v1_s0 | -0.091 [-0.159, -0.026] 2/6 loss | +0.060 [-0.053, +0.195] 3/6 tie | +0.079 [+0.001, +0.170] 6/6 win | +0.186 [+0.049, +0.256] 5/6 win | +0.124 [-0.010, +0.293] 6/6 tie |
| distill_v1_s1 | -0.097 [-0.187, +0.012] 3/6 tie | +0.084 [-0.014, +0.198] 4/6 tie | +0.057 [-0.041, +0.145] 5/6 tie | +0.156 [+0.039, +0.242] 5/6 win | +0.154 [+0.017, +0.316] 6/6 win |
| distill_v1_s2 | -0.087 [-0.194, -0.011] 1/6 loss | +0.121 [-0.002, +0.232] 4/6 tie | +0.089 [-0.020, +0.194] 5/6 tie | +0.178 [+0.030, +0.267] 5/6 win | +0.162 [+0.026, +0.314] 6/6 win |
| oxyssl | -0.130 [-0.274, -0.044] 1/6 loss | +0.132 [+0.024, +0.171] 1/6 win | -0.050 [-0.123, +0.045] 3/6 tie | -0.013 [-0.098, +0.068] 2/5 tie | +0.178 [+0.043, +0.214] 5/6 win |
| oxyssl_init | -0.290 [-0.397, -0.156] 1/6 loss | -0.025 [-0.119, +0.067] 2/6 tie | -0.098 [-0.141, -0.026] 2/6 loss | -0.035 [-0.098, +0.023] 2/5 tie | +0.017 [-0.034, +0.101] 2/6 tie |
| oxyssl_noaug | -0.144 [-0.327, -0.011] 1/6 loss | +0.136 [+0.014, +0.197] 3/6 win | -0.096 [-0.137, -0.020] 2/6 loss | -0.035 [-0.098, +0.023] 2/5 tie | +0.164 [+0.018, +0.215] 4/6 win |
| oxyseq | -0.088 [-0.221, -0.023] 2/6 loss | -0.032 [-0.133, +0.046] 3/6 tie | -0.094 [-0.137, -0.015] 3/6 loss | -0.036 [-0.098, +0.017] 2/6 tie | +0.001 [-0.047, +0.059] 3/6 tie |
| oxyseq_notta | -0.158 [-0.265, -0.058] 2/6 loss | -0.006 [-0.101, +0.071] 5/6 tie | -0.152 [-0.190, -0.015] 5/6 loss | -0.067 [-0.102, +0.002] 2/6 tie | +0.005 [-0.046, +0.064] 4/6 tie |

## C. Worst person TM (the weakest of the six held-out people)

| model | L0 | L3 | L4 | L5 | X1 |
|---|---|---|---|---|---|
| chain_v1 | 0.458 | 0.502 | 0.445 | 0.393 | 0.393 |
| hybrid_final | 0.517 | n/a | n/a | n/a | n/a |
| cnn_full | 0.500 | n/a | n/a | n/a | n/a |
| ctl_dc_lr | 0.514 | 0.548 | 0.500 | 0.500 | 0.500 |
| twin_inverse | 0.500 | 0.531 | 0.385 | 0.500 | 0.455 |
| twin_gate | 0.458 | 0.537 | 0.500 | 0.537 | 0.393 |
| photonnet | 0.500 | 0.500 | 0.492 | 0.539 | 0.500 |
| photonnet_nophys | 0.500 | 0.500 | 0.500 | 0.569 | 0.416 |
| distill | 0.516 | 0.500 | 0.516 | 0.516 | 0.500 |
| distill_nokd | 0.505 | 0.508 | 0.492 | 0.547 | 0.447 |
| distill_v1 | 0.500 | 0.516 | 0.500 | 0.572 | 0.500 |
| distill_v1_nokd | 0.505 | 0.508 | 0.492 | 0.521 | 0.442 |
| distill_v1_s0 | 0.500 | 0.517 | 0.492 | 0.541 | 0.461 |
| distill_v1_s1 | 0.500 | 0.516 | 0.492 | 0.584 | 0.502 |
| distill_v1_s2 | 0.500 | 0.558 | 0.500 | 0.495 | 0.500 |
| oxyssl | 0.400 | 0.458 | 0.500 | 0.500 | 0.444 |
| oxyssl_init | 0.399 | 0.461 | 0.500 | 0.500 | 0.475 |
| oxyssl_noaug | 0.351 | 0.500 | 0.500 | 0.500 | 0.466 |
| oxyseq | 0.516 | 0.438 | 0.483 | 0.357 | 0.470 |
| oxyseq_notta | 0.500 | 0.463 | 0.500 | 0.459 | 0.474 |

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
| photonnet | cov 0.35 / ans 0.924 / all 0.785 | cov 0.33 / ans 0.948 / all 0.796 | cov 0.32 / ans 0.922 / all 0.799 | cov 0.16 / ans 0.933 / all 0.811 | cov 0.34 / ans 0.865 / all 0.754 |
| photonnet_nophys | cov 0.47 / ans 0.922 / all 0.790 | cov 0.29 / ans 0.914 / all 0.812 | cov 0.12 / ans 0.977 / all 0.805 | cov 0.04 / ans 0.987 / all 0.803 | cov 0.23 / ans 0.831 / all 0.767 |
| distill | cov 0.61 / ans 0.922 / all 0.863 | cov 0.49 / ans 0.937 / all 0.822 | cov 0.42 / ans 0.946 / all 0.823 | cov 0.21 / ans 0.943 / all 0.818 | cov 0.27 / ans 0.964 / all 0.800 |
| distill_nokd | cov 0.49 / ans 0.940 / all 0.800 | cov 0.42 / ans 0.909 / all 0.767 | cov 0.22 / ans 0.981 / all 0.801 | cov 0.03 / ans 0.990 / all 0.773 | cov 0.28 / ans 0.883 / all 0.745 |
| distill_v1 | cov 0.59 / ans 0.949 / all 0.835 | cov 0.52 / ans 0.925 / all 0.803 | cov 0.46 / ans 0.939 / all 0.822 | cov 0.21 / ans 0.965 / all 0.832 | cov 0.35 / ans 0.906 / all 0.778 |
| distill_v1_nokd | cov 0.55 / ans 0.940 / all 0.802 | cov 0.46 / ans 0.887 / all 0.751 | cov 0.24 / ans 0.978 / all 0.799 | cov 0.06 / ans 0.980 / all 0.769 | cov 0.30 / ans 0.862 / all 0.720 |
| distill_v1_s0 | cov 0.54 / ans 0.972 / all 0.832 | cov 0.54 / ans 0.857 / all 0.771 | cov 0.42 / ans 0.932 / all 0.816 | cov 0.17 / ans 0.985 / all 0.810 | cov 0.42 / ans 0.830 / all 0.748 |
| distill_v1_s1 | cov 0.56 / ans 0.957 / all 0.839 | cov 0.52 / ans 0.911 / all 0.806 | cov 0.44 / ans 0.970 / all 0.819 | cov 0.28 / ans 0.921 / all 0.825 | cov 0.35 / ans 0.880 / all 0.767 |
| distill_v1_s2 | cov 0.72 / ans 0.915 / all 0.834 | cov 0.64 / ans 0.892 / all 0.818 | cov 0.57 / ans 0.900 / all 0.826 | cov 0.41 / ans 0.917 / all 0.832 | cov 0.46 / ans 0.885 / all 0.779 |
| oxyssl | cov 0.80 / ans 0.721 / all 0.683 | cov 0.59 / ans 0.793 / all 0.695 | cov 0.10 / ans 0.618 / all 0.487 | cov 0.14 / ans 0.528 / all 0.446 | cov 0.51 / ans 0.776 / all 0.691 |
| oxyssl_init | cov 1.00 / ans 0.450 / all 0.450 | cov 1.00 / ans 0.483 / all 0.483 | cov 1.00 / ans 0.426 / all 0.426 | cov 1.00 / ans 0.425 / all 0.425 | cov 1.00 / ans 0.458 / all 0.458 |
| oxyssl_noaug | cov 0.58 / ans 0.754 / all 0.657 | cov 0.22 / ans 0.715 / all 0.673 | cov 0.19 / ans 0.477 / all 0.429 | cov 0.19 / ans 0.464 / all 0.425 | cov 0.19 / ans 0.652 / all 0.650 |
| oxyseq | cov 0.79 / ans 0.769 / all 0.779 | cov 0.71 / ans 0.556 / all 0.551 | cov 0.71 / ans 0.543 / all 0.585 | cov 0.71 / ans 0.539 / all 0.565 | cov 0.71 / ans 0.583 / all 0.553 |
| oxyseq_notta | cov 0.79 / ans 0.773 / all 0.758 | cov 0.72 / ans 0.573 / all 0.560 | cov 0.71 / ans 0.552 / all 0.577 | cov 0.71 / ans 0.539 / all 0.551 | cov 0.71 / ans 0.566 / all 0.536 |

## E. Per-person TM at L0

| model | person 100001 | person 100002 | person 100003 | person 100004 | person 100005 | person 100006 |
|---|---|---|---|---|---|---|
| chain_v1 | 0.889 | 0.458 | 0.660 | 0.711 | 0.737 | 0.539 |
| hybrid_final | 0.594 | 0.517 | 0.656 | 0.766 | 0.735 | 0.539 |
| cnn_full | 0.633 | 0.500 | 0.659 | 0.624 | 0.766 | 0.509 |
| ctl_dc_lr | 0.659 | 0.514 | 0.626 | 0.625 | 0.693 | 0.869 |
| twin_inverse | 0.662 | 0.500 | 0.654 | 0.656 | 0.689 | 0.632 |
| twin_gate | 0.889 | 0.458 | 0.660 | 0.711 | 0.737 | 0.539 |
| photonnet | 0.634 | 0.532 | 0.602 | 0.500 | 0.692 | 0.676 |
| photonnet_nophys | 0.629 | 0.526 | 0.573 | 0.500 | 0.686 | 0.703 |
| distill | 0.670 | 0.516 | 0.649 | 0.721 | 0.700 | 0.536 |
| distill_nokd | 0.594 | 0.548 | 0.635 | 0.505 | 0.686 | 0.625 |
| distill_v1 | 0.664 | 0.500 | 0.663 | 0.500 | 0.704 | 0.518 |
| distill_v1_nokd | 0.594 | 0.645 | 0.655 | 0.505 | 0.704 | 0.607 |
| distill_v1_s0 | 0.662 | 0.500 | 0.668 | 0.585 | 0.701 | 0.518 |
| distill_v1_s1 | 0.835 | 0.500 | 0.674 | 0.500 | 0.715 | 0.649 |
| distill_v1_s2 | 0.659 | 0.500 | 0.651 | 0.517 | 0.710 | 0.508 |
| oxyssl | 0.500 | 0.500 | 0.638 | 0.400 | 0.736 | 0.503 |
| oxyssl_init | 0.500 | 0.500 | 0.526 | 0.399 | 0.525 | 0.500 |
| oxyssl_noaug | 0.500 | 0.500 | 0.643 | 0.351 | 0.670 | 0.488 |
| oxyseq | 0.645 | 0.645 | 0.550 | 0.516 | 0.696 | 0.542 |
| oxyseq_notta | 0.659 | 0.500 | 0.507 | 0.519 | 0.809 | 0.500 |

## F. Null and blank controls at L0 (mean over draws; must be <= 0.55)

| model | kind | draws | mean TM | max TM | check | files |
|---|---|---|---|---|---|---|
| photonnet | null | 3 | 0.618 | 0.624 | **FLAG** | photonnet_null 0.614, photonnet_null_d1 0.617, photonnet_null_d2 0.624 |
| distill | null | 3 | 0.492 | 0.560 | ok | distill_d1_null 0.560, distill_d2_null 0.480, distill_null 0.436 |
| oxyssl | null | 3 | 0.643 | 0.660 | **FLAG** | oxyssl_d1_null 0.660, oxyssl_d2_null 0.614, oxyssl_null 0.654 |
| oxyseq | null | 1 | 0.475 | 0.475 | ok | oxyseq_null 0.475 |
| oxyseq | blank | 1 | 0.500 | 0.500 | ok | oxyseq_blank 0.500 |
| ctl_dc_lr (round 1) | null | 1 | 0.500 | 0.500 | ok | ctl_dc_lr_null |
| twin_inverse (round 1) | null | 1 | 0.406 | 0.406 | ok | twin_inverse_null |
| twin_gate (round 1) | null | 1 | 0.578 | 0.578 | flag (single draw) | twin_gate_gperm_null |
| twin_gate (round 1) | null | 1 | 0.623 | 0.623 | flag (single draw) | twin_gate_null |
| twin_gate (round 1) | null | 1 | 0.550 | 0.550 | flag (single draw) | twin_gate_perm_null |

## Skipped files

* null file photonnet_null_d0__L0.csv is a byte-identical copy of photonnet_null__L0.csv; counted once

