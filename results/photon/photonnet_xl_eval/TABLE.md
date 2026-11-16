# Photon-stress table

Hoffman, leave-one-person-out (both hands together), 5644 evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision (clock-only = 0.5) with the 2000-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `chain_v1` at the same level: **win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats `chain_v1` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements of any phone. Full columns: `TABLE.csv`.

## Null controls (level 0, must be ~0.50 TM; flagged above 0.55)

| model | TM bal. acc. [95 % CI] | check |
|---|---|---|
| photonnet_xl_null | 0.391 [0.240, 0.558] | ok |

## Level 0 (flash +0 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.796 [0.636, 0.857] | 0.873 | 0.947 |  | reference |  | 0.458 | 0.84 | 0.939 | 0.037 |  |
| distill | 0.758 [0.603, 0.813] | 0.860 | 0.944 | -0.038 [-0.103, -0.001] | loss | 2/6 (p = 0.891) | 0.516 | 0.61 | 0.922 | 0.064 | 4.49 |
| photonnet_xl_seed1 | 0.671 [0.501, 0.778] | 0.745 | 0.858 | -0.125 [-0.210, -0.051] | loss | 2/6 (p = 0.891) | 0.500 | 0.94 | 0.759 | 0.154 | 6.29 |
| photonnet | 0.668 [0.508, 0.785] | 0.775 | 0.865 | -0.128 [-0.208, -0.047] | loss | 2/6 (p = 0.891) | 0.500 | 0.35 | 0.924 | 0.120 | 4.98 |
| photonnet_xl_seed2 | 0.668 [0.563, 0.767] | 0.805 | 0.889 | -0.129 [-0.204, -0.002] | loss | 2/6 (p = 0.891) | 0.575 | 0.56 | 0.865 | 0.084 | 5.85 |
| photonnet_xl | 0.660 [0.545, 0.762] | 0.763 | 0.865 | -0.136 [-0.210, -0.033] | loss | 2/6 (p = 0.891) | 0.500 | 0.92 | 0.795 | 0.119 | 5.99 |
| photonnet_xl_seed0 | 0.639 [0.524, 0.723] | 0.729 | 0.826 | -0.157 [-0.230, -0.063] | loss | 2/6 (p = 0.891) | 0.500 | 0.81 | 0.729 | 0.155 | 6.27 |
| photonnet_xl_null_d1 | 0.445 [0.392, 0.525] | 0.486 | 0.488 | -0.351 [-0.429, -0.172] | loss | 1/6 (p = 0.984) | 0.376 | 0.56 | 0.546 | 0.536 | 7.18 |
| photonnet_xl_null_d2 | 0.399 [0.314, 0.530] | 0.416 | 0.425 | -0.398 [-0.461, -0.253] | loss | 1/6 (p = 0.984) | 0.357 | 0.53 | 0.409 | 0.603 | 7.36 |
| photonnet_xl_null | 0.391 [0.240, 0.558] | 0.485 | 0.516 | -0.406 [-0.556, -0.186] | loss | 1/6 (p = 0.984) | 0.215 | 0.35 | 0.440 | 0.541 | 6.89 |
| photonnet_xl_null_d0 | 0.391 [0.240, 0.558] | 0.485 | 0.516 | -0.406 [-0.556, -0.186] | loss | 1/6 (p = 0.984) | 0.215 | 0.35 | 0.440 | 0.541 | 6.89 |

## Level 1 (flash +2 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.805 [0.646, 0.862] | 0.871 | 0.945 |  | reference |  | 0.431 | 0.84 | 0.929 | 0.040 |  |
| distill | 0.706 [0.564, 0.788] | 0.838 | 0.934 | -0.099 [-0.201, +0.007] | tie | 2/6 (p = 0.891) | 0.500 | 0.59 | 0.926 | 0.060 | 4.45 |
| photonnet | 0.678 [0.518, 0.794] | 0.767 | 0.861 | -0.126 [-0.232, -0.024] | loss | 2/6 (p = 0.891) | 0.500 | 0.37 | 0.918 | 0.124 | 5.08 |
| photonnet_xl | 0.667 [0.541, 0.767] | 0.777 | 0.877 | -0.138 [-0.237, -0.021] | loss | 2/6 (p = 0.891) | 0.500 | 0.91 | 0.812 | 0.111 | 5.85 |

## Level 2 (flash +4 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.742 [0.585, 0.820] | 0.806 | 0.814 |  | reference |  | 0.500 | 0.88 | 0.822 | 0.250 |  |
| distill | 0.704 [0.561, 0.791] | 0.844 | 0.928 | -0.038 [-0.129, +0.078] | tie | 2/6 (p = 0.891) | 0.500 | 0.60 | 0.920 | 0.065 | 4.89 |
| photonnet | 0.677 [0.534, 0.783] | 0.758 | 0.841 | -0.066 [-0.176, +0.097] | tie | 3/6 (p = 0.656) | 0.500 | 0.37 | 0.878 | 0.156 | 5.35 |
| photonnet_xl | 0.662 [0.546, 0.754] | 0.753 | 0.857 | -0.080 [-0.176, +0.069] | tie | 3/6 (p = 0.656) | 0.500 | 0.91 | 0.787 | 0.117 | 6.06 |

## Level 3 (flash +6 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.689 [0.526, 0.812] | 0.789 | 0.879 | +0.125 [-0.019, +0.267] | tie | 4/6 (p = 0.344) | 0.500 | 0.33 | 0.948 | 0.099 | 4.82 |
| distill | 0.680 [0.552, 0.772] | 0.817 | 0.912 | +0.116 [-0.002, +0.226] | tie | 4/6 (p = 0.344) | 0.500 | 0.49 | 0.937 | 0.079 | 5.24 |
| photonnet_xl | 0.680 [0.520, 0.772] | 0.815 | 0.903 | +0.115 [-0.028, +0.229] | tie | 3/6 (p = 0.656) | 0.500 | 0.86 | 0.844 | 0.106 | 5.49 |
| chain_v1 | 0.565 [0.503, 0.625] | 0.584 | 0.557 |  | reference |  | 0.502 | 0.92 | 0.534 | 0.503 |  |

## Level 4 (flash +8 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.703 [0.516, 0.829] | 0.786 | 0.859 | +0.105 [-0.025, +0.245] | tie | 5/6 (p = 0.109) | 0.492 | 0.32 | 0.922 | 0.109 | 5.02 |
| photonnet_xl | 0.684 [0.522, 0.794] | 0.796 | 0.895 | +0.086 [-0.020, +0.207] | tie | 5/6 (p = 0.109) | 0.500 | 0.92 | 0.815 | 0.119 | 5.44 |
| distill | 0.681 [0.566, 0.772] | 0.817 | 0.915 | +0.083 [+0.008, +0.178] | win | 6/6 (p = 0.016) | 0.516 | 0.42 | 0.946 | 0.066 | 4.48 |
| chain_v1 | 0.598 [0.526, 0.640] | 0.603 | 0.621 |  | reference |  | 0.445 | 0.86 | 0.575 | 0.434 |  |

## Level 5 (flash +10 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.720 [0.564, 0.818] | 0.802 | 0.871 | +0.185 [+0.023, +0.290] | win | 5/6 (p = 0.109) | 0.539 | 0.16 | 0.933 | 0.139 | 5.15 |
| photonnet_xl | 0.704 [0.563, 0.796] | 0.812 | 0.888 | +0.169 [+0.018, +0.269] | win | 5/6 (p = 0.109) | 0.583 | 0.87 | 0.814 | 0.123 | 5.63 |
| distill | 0.701 [0.570, 0.770] | 0.822 | 0.912 | +0.166 [+0.035, +0.242] | win | 5/6 (p = 0.109) | 0.516 | 0.21 | 0.943 | 0.072 | 4.51 |
| chain_v1 | 0.535 [0.477, 0.598] | 0.529 | 0.542 |  | reference |  | 0.393 | 0.91 | 0.491 | 0.516 |  |

## Stress X1: out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet_xl | 0.691 [0.597, 0.746] | 0.782 | 0.871 | +0.184 [+0.072, +0.290] | win | 6/6 (p = 0.016) | 0.500 | 0.82 | 0.783 | 0.162 | 6.18 |
| distill | 0.676 [0.564, 0.747] | 0.812 | 0.908 | +0.169 [+0.058, +0.285] | win | 6/6 (p = 0.016) | 0.500 | 0.27 | 0.964 | 0.088 | 5.09 |
| photonnet | 0.672 [0.537, 0.775] | 0.772 | 0.863 | +0.165 [+0.020, +0.316] | win | 6/6 (p = 0.016) | 0.500 | 0.34 | 0.865 | 0.157 | 5.50 |
| chain_v1 | 0.508 [0.439, 0.554] | 0.548 | 0.533 |  | reference |  | 0.393 | 0.92 | 0.507 | 0.536 |  |

## Accuracy at coverage (by `conf`)

| model | level | 100 % | 90 % | 80 % | 70 % | 60 % | 50 % |
|---|---|---|---|---|---|---|---|
| chain_v1 | 0 | 0.875 | 0.918 | 0.945 | 0.957 | 0.970 | 0.976 |
| chain_v1 | 1 | 0.872 | 0.912 | 0.935 | 0.950 | 0.966 | 0.975 |
| chain_v1 | 2 | 0.794 | 0.818 | 0.835 | 0.834 | 0.821 | 0.791 |
| chain_v1 | 3 | 0.538 | 0.534 | 0.526 | 0.516 | 0.506 | 0.469 |
| chain_v1 | 4 | 0.572 | 0.573 | 0.575 | 0.566 | 0.567 | 0.572 |
| chain_v1 | 5 | 0.492 | 0.492 | 0.489 | 0.489 | 0.493 | 0.493 |
| chain_v1 | X1 | 0.505 | 0.507 | 0.507 | 0.500 | 0.488 | 0.463 |
| distill | 0 | 0.863 | 0.879 | 0.890 | 0.900 | 0.909 | 0.933 |
| distill | 1 | 0.842 | 0.868 | 0.894 | 0.915 | 0.925 | 0.948 |
| distill | 2 | 0.844 | 0.871 | 0.890 | 0.909 | 0.921 | 0.933 |
| distill | 3 | 0.822 | 0.839 | 0.864 | 0.889 | 0.911 | 0.931 |
| distill | 4 | 0.823 | 0.857 | 0.889 | 0.912 | 0.934 | 0.948 |
| distill | 5 | 0.818 | 0.841 | 0.866 | 0.889 | 0.912 | 0.933 |
| distill | X1 | 0.800 | 0.827 | 0.854 | 0.866 | 0.878 | 0.897 |
| photonnet | 0 | 0.785 | 0.793 | 0.814 | 0.821 | 0.859 | 0.900 |
| photonnet | 1 | 0.777 | 0.791 | 0.802 | 0.809 | 0.866 | 0.896 |
| photonnet | 2 | 0.766 | 0.787 | 0.789 | 0.799 | 0.848 | 0.875 |
| photonnet | 3 | 0.796 | 0.820 | 0.830 | 0.841 | 0.876 | 0.919 |
| photonnet | 4 | 0.799 | 0.802 | 0.831 | 0.844 | 0.883 | 0.901 |
| photonnet | 5 | 0.811 | 0.811 | 0.817 | 0.819 | 0.820 | 0.828 |
| photonnet | X1 | 0.754 | 0.780 | 0.783 | 0.802 | 0.824 | 0.825 |
| photonnet_xl | 0 | 0.772 | 0.780 | 0.821 | 0.839 | 0.842 | 0.848 |
| photonnet_xl | 1 | 0.787 | 0.793 | 0.835 | 0.857 | 0.865 | 0.872 |
| photonnet_xl | 2 | 0.762 | 0.765 | 0.805 | 0.842 | 0.856 | 0.865 |
| photonnet_xl | 3 | 0.821 | 0.819 | 0.843 | 0.863 | 0.875 | 0.881 |
| photonnet_xl | 4 | 0.803 | 0.801 | 0.808 | 0.827 | 0.848 | 0.870 |
| photonnet_xl | 5 | 0.813 | 0.798 | 0.794 | 0.810 | 0.840 | 0.870 |
| photonnet_xl | X1 | 0.760 | 0.755 | 0.763 | 0.784 | 0.808 | 0.835 |
| photonnet_xl_null | 0 | 0.466 | 0.461 | 0.480 | 0.465 | 0.454 | 0.475 |
| photonnet_xl_null_d0 | 0 | 0.466 | 0.461 | 0.480 | 0.465 | 0.454 | 0.475 |
| photonnet_xl_null_d1 | 0 | 0.494 | 0.524 | 0.550 | 0.552 | 0.533 | 0.543 |
| photonnet_xl_null_d2 | 0 | 0.400 | 0.380 | 0.387 | 0.423 | 0.445 | 0.425 |
| photonnet_xl_seed0 | 0 | 0.741 | 0.753 | 0.795 | 0.816 | 0.812 | 0.838 |
| photonnet_xl_seed1 | 0 | 0.756 | 0.761 | 0.783 | 0.807 | 0.823 | 0.828 |
| photonnet_xl_seed2 | 0 | 0.816 | 0.831 | 0.840 | 0.859 | 0.882 | 0.923 |
