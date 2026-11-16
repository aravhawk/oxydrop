# Photon-stress table

Hoffman, leave-one-person-out (both hands together), 5644 evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision (clock-only = 0.5) with the 2000-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `distill` at the same level: **win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats `distill` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements of any phone. Full columns: `TABLE.csv`.

## Level 0 (flash +0 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.758 [0.603, 0.813] | 0.860 | 0.944 |  | reference |  | 0.516 | 0.61 | 0.922 | 0.064 | 4.49 |
| photonnet | 0.668 [0.508, 0.785] | 0.775 | 0.865 | -0.090 [-0.157, -0.023] | loss | 2/6 (p = 0.891) | 0.500 | 0.35 | 0.924 | 0.120 | 4.98 |
| photonnet_xl | 0.660 [0.545, 0.762] | 0.763 | 0.865 | -0.098 [-0.161, -0.011] | loss | 2/6 (p = 0.891) | 0.500 | 0.92 | 0.795 | 0.119 | 5.99 |

## Level 1 (flash +2 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.706 [0.564, 0.788] | 0.838 | 0.934 |  | reference |  | 0.500 | 0.59 | 0.926 | 0.060 | 4.45 |
| photonnet | 0.678 [0.518, 0.794] | 0.767 | 0.861 | -0.027 [-0.085, +0.027] | tie | 2/6 (p = 0.891) | 0.500 | 0.37 | 0.918 | 0.124 | 5.08 |
| photonnet_xl | 0.667 [0.541, 0.767] | 0.777 | 0.877 | -0.039 [-0.072, +0.001] | tie | 3/6 (p = 0.656) | 0.500 | 0.91 | 0.812 | 0.111 | 5.85 |

## Level 2 (flash +4 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.704 [0.561, 0.791] | 0.844 | 0.928 |  | reference |  | 0.500 | 0.60 | 0.920 | 0.065 | 4.89 |
| photonnet | 0.677 [0.534, 0.783] | 0.758 | 0.841 | -0.027 [-0.069, +0.020] | tie | 4/6 (p = 0.344) | 0.500 | 0.37 | 0.878 | 0.156 | 5.35 |
| photonnet_xl | 0.662 [0.546, 0.754] | 0.753 | 0.857 | -0.042 [-0.067, +0.014] | tie | 4/6 (p = 0.344) | 0.500 | 0.91 | 0.787 | 0.117 | 6.06 |

## Level 3 (flash +6 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.689 [0.526, 0.812] | 0.789 | 0.879 | +0.009 [-0.036, +0.051] | tie | 3/5 (p = 0.500) | 0.500 | 0.33 | 0.948 | 0.099 | 4.82 |
| distill | 0.680 [0.552, 0.772] | 0.817 | 0.912 |  | reference |  | 0.500 | 0.49 | 0.937 | 0.079 | 5.24 |
| photonnet_xl | 0.680 [0.520, 0.772] | 0.815 | 0.903 | -0.000 [-0.040, +0.036] | tie | 4/6 (p = 0.344) | 0.500 | 0.86 | 0.844 | 0.106 | 5.49 |

## Level 4 (flash +8 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.703 [0.516, 0.829] | 0.786 | 0.859 | +0.022 [-0.053, +0.089] | tie | 3/6 (p = 0.656) | 0.492 | 0.32 | 0.922 | 0.109 | 5.02 |
| photonnet_xl | 0.684 [0.522, 0.794] | 0.796 | 0.895 | +0.004 [-0.046, +0.040] | tie | 2/6 (p = 0.891) | 0.500 | 0.92 | 0.815 | 0.119 | 5.44 |
| distill | 0.681 [0.566, 0.772] | 0.817 | 0.915 |  | reference |  | 0.516 | 0.42 | 0.946 | 0.066 | 4.48 |

## Level 5 (flash +10 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet | 0.720 [0.564, 0.818] | 0.802 | 0.871 | +0.019 [-0.027, +0.058] | tie | 4/6 (p = 0.344) | 0.539 | 0.16 | 0.933 | 0.139 | 5.15 |
| photonnet_xl | 0.704 [0.563, 0.796] | 0.812 | 0.888 | +0.003 [-0.037, +0.037] | tie | 2/5 (p = 0.812) | 0.583 | 0.87 | 0.814 | 0.123 | 5.63 |
| distill | 0.701 [0.570, 0.770] | 0.822 | 0.912 |  | reference |  | 0.516 | 0.21 | 0.943 | 0.072 | 4.51 |

## Stress X1: out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| photonnet_xl | 0.691 [0.597, 0.746] | 0.782 | 0.871 | +0.015 [-0.033, +0.064] | tie | 1/5 (p = 0.969) | 0.500 | 0.82 | 0.783 | 0.162 | 6.18 |
| distill | 0.676 [0.564, 0.747] | 0.812 | 0.908 |  | reference |  | 0.500 | 0.27 | 0.964 | 0.088 | 5.09 |
| photonnet | 0.672 [0.537, 0.775] | 0.772 | 0.863 | -0.004 [-0.049, +0.031] | tie | 2/5 (p = 0.812) | 0.500 | 0.34 | 0.865 | 0.157 | 5.50 |

## Accuracy at coverage (by `conf`)

| model | level | 100 % | 90 % | 80 % | 70 % | 60 % | 50 % |
|---|---|---|---|---|---|---|---|
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
