# Photon-stress table

Hoffman, leave-one-person-out (both hands together), 5644 evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision (clock-only = 0.5) with the 2000-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `distill` at the same level: **win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats `distill` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements of any phone. Full columns: `TABLE.csv`.

## Level 0 (flash +0 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.758 [0.603, 0.813] | 0.860 | 0.944 |  | reference |  | 0.516 | 0.61 | 0.922 | 0.064 | 4.49 |
| ssl3_gpu | 0.596 [0.514, 0.665] | 0.617 | 0.857 | -0.162 [-0.232, -0.085] | loss | 1/6 (p = 0.984) | 0.500 | 0.77 | 0.763 | 0.144 | 5.32 |

## Level 1 (flash +2 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.706 [0.564, 0.788] | 0.838 | 0.934 |  | reference |  | 0.500 | 0.59 | 0.926 | 0.060 | 4.45 |
| ssl3_gpu | 0.593 [0.511, 0.672] | 0.605 | 0.865 | -0.113 [-0.233, -0.015] | loss | 1/5 (p = 0.969) | 0.500 | 0.73 | 0.769 | 0.147 | 5.24 |

## Level 2 (flash +4 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.704 [0.561, 0.791] | 0.844 | 0.928 |  | reference |  | 0.500 | 0.60 | 0.920 | 0.065 | 4.89 |
| ssl3_gpu | 0.574 [0.506, 0.638] | 0.595 | 0.873 | -0.130 [-0.237, -0.033] | loss | 0/5 (p = 1.000) | 0.486 | 0.71 | 0.781 | 0.143 | 5.10 |

## Level 3 (flash +6 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.680 [0.552, 0.772] | 0.817 | 0.912 |  | reference |  | 0.500 | 0.49 | 0.937 | 0.079 | 5.24 |
| ssl3_gpu | 0.603 [0.519, 0.662] | 0.660 | 0.861 | -0.077 [-0.124, -0.024] | loss | 1/5 (p = 0.969) | 0.500 | 0.78 | 0.787 | 0.142 | 5.64 |

## Level 4 (flash +8 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.681 [0.566, 0.772] | 0.817 | 0.915 |  | reference |  | 0.516 | 0.42 | 0.946 | 0.066 | 4.48 |
| ssl3_gpu | 0.631 [0.535, 0.716] | 0.652 | 0.890 | -0.050 [-0.086, -0.015] | loss | 0/6 (p = 1.000) | 0.500 | 0.62 | 0.829 | 0.135 | 5.40 |

## Level 5 (flash +10 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.701 [0.570, 0.770] | 0.822 | 0.912 |  | reference |  | 0.516 | 0.21 | 0.943 | 0.072 | 4.51 |
| ssl3_gpu | 0.646 [0.539, 0.706] | 0.661 | 0.892 | -0.054 [-0.090, -0.016] | loss | 2/6 (p = 0.891) | 0.500 | 0.48 | 0.888 | 0.125 | 5.52 |

## Stress X1: out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill | 0.676 [0.564, 0.747] | 0.812 | 0.908 |  | reference |  | 0.500 | 0.27 | 0.964 | 0.088 | 5.09 |
| ssl3_gpu | 0.640 [0.532, 0.715] | 0.663 | 0.884 | -0.036 [-0.088, +0.012] | tie | 4/6 (p = 0.344) | 0.500 | 0.77 | 0.777 | 0.129 | 5.22 |

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
| ssl3_gpu | 0 | 0.674 | 0.710 | 0.751 | 0.797 | 0.851 | 0.890 |
| ssl3_gpu | 1 | 0.664 | 0.698 | 0.744 | 0.785 | 0.843 | 0.883 |
| ssl3_gpu | 2 | 0.655 | 0.699 | 0.746 | 0.788 | 0.838 | 0.885 |
| ssl3_gpu | 3 | 0.708 | 0.747 | 0.777 | 0.808 | 0.840 | 0.861 |
| ssl3_gpu | 4 | 0.701 | 0.738 | 0.777 | 0.809 | 0.843 | 0.877 |
| ssl3_gpu | 5 | 0.708 | 0.735 | 0.768 | 0.803 | 0.842 | 0.887 |
| ssl3_gpu | X1 | 0.711 | 0.742 | 0.770 | 0.804 | 0.842 | 0.883 |
