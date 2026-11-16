# Photon-stress table

Hoffman, leave-one-person-out (both hands together), 5644 evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision (clock-only = 0.5) with the 2000-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `distill2` at the same level: **win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats `distill2` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements of any phone. Full columns: `TABLE.csv`.

## Level 0 (flash +0 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.784 [0.596, 0.834] | 0.866 | 0.948 |  | reference |  | 0.500 | 0.69 | 0.933 | 0.038 | 4.31 |
| ssl3_gpu | 0.596 [0.514, 0.665] | 0.617 | 0.857 | -0.188 [-0.251, -0.080] | loss | 1/5 (p = 0.969) | 0.500 | 0.77 | 0.763 | 0.144 | 5.32 |

## Level 1 (flash +2 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.711 [0.574, 0.790] | 0.851 | 0.942 |  | reference |  | 0.500 | 0.61 | 0.963 | 0.041 | 3.96 |
| ssl3_gpu | 0.593 [0.511, 0.672] | 0.605 | 0.865 | -0.119 [-0.234, -0.021] | loss | 1/5 (p = 0.969) | 0.500 | 0.73 | 0.769 | 0.147 | 5.24 |

## Level 2 (flash +4 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.712 [0.557, 0.791] | 0.845 | 0.928 |  | reference |  | 0.500 | 0.67 | 0.923 | 0.054 | 4.44 |
| ssl3_gpu | 0.574 [0.506, 0.638] | 0.595 | 0.873 | -0.137 [-0.239, -0.044] | loss | 0/5 (p = 1.000) | 0.486 | 0.71 | 0.781 | 0.143 | 5.10 |

## Level 3 (flash +6 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.698 [0.555, 0.789] | 0.824 | 0.914 |  | reference |  | 0.500 | 0.58 | 0.918 | 0.071 | 5.11 |
| ssl3_gpu | 0.603 [0.519, 0.662] | 0.660 | 0.861 | -0.095 [-0.140, -0.033] | loss | 2/6 (p = 0.891) | 0.500 | 0.78 | 0.787 | 0.142 | 5.64 |

## Level 4 (flash +8 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.695 [0.558, 0.785] | 0.822 | 0.913 |  | reference |  | 0.500 | 0.53 | 0.926 | 0.062 | 4.71 |
| ssl3_gpu | 0.631 [0.535, 0.716] | 0.652 | 0.890 | -0.064 [-0.112, -0.012] | loss | 1/6 (p = 0.984) | 0.500 | 0.62 | 0.829 | 0.135 | 5.40 |

## Level 5 (flash +10 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.704 [0.562, 0.779] | 0.820 | 0.910 |  | reference |  | 0.548 | 0.35 | 0.910 | 0.077 | 4.48 |
| ssl3_gpu | 0.646 [0.539, 0.706] | 0.661 | 0.892 | -0.058 [-0.104, -0.016] | loss | 1/6 (p = 0.984) | 0.500 | 0.48 | 0.888 | 0.125 | 5.52 |

## Stress X1: out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| distill2 | 0.682 [0.547, 0.748] | 0.797 | 0.909 |  | reference |  | 0.484 | 0.40 | 0.861 | 0.117 | 5.18 |
| ssl3_gpu | 0.640 [0.532, 0.715] | 0.663 | 0.884 | -0.042 [-0.086, +0.010] | tie | 3/6 (p = 0.656) | 0.500 | 0.77 | 0.777 | 0.129 | 5.22 |

## Accuracy at coverage (by `conf`)

| model | level | 100 % | 90 % | 80 % | 70 % | 60 % | 50 % |
|---|---|---|---|---|---|---|---|
| distill2 | 0 | 0.866 | 0.883 | 0.916 | 0.945 | 0.972 | 0.982 |
| distill2 | 1 | 0.856 | 0.877 | 0.914 | 0.957 | 0.972 | 0.977 |
| distill2 | 2 | 0.847 | 0.869 | 0.903 | 0.919 | 0.934 | 0.950 |
| distill2 | 3 | 0.828 | 0.842 | 0.869 | 0.892 | 0.913 | 0.936 |
| distill2 | 4 | 0.826 | 0.854 | 0.885 | 0.912 | 0.934 | 0.951 |
| distill2 | 5 | 0.814 | 0.842 | 0.864 | 0.892 | 0.916 | 0.931 |
| distill2 | X1 | 0.780 | 0.804 | 0.825 | 0.847 | 0.866 | 0.887 |
| ssl3_gpu | 0 | 0.674 | 0.710 | 0.751 | 0.797 | 0.851 | 0.890 |
| ssl3_gpu | 1 | 0.664 | 0.698 | 0.744 | 0.785 | 0.843 | 0.883 |
| ssl3_gpu | 2 | 0.655 | 0.699 | 0.746 | 0.788 | 0.838 | 0.885 |
| ssl3_gpu | 3 | 0.708 | 0.747 | 0.777 | 0.808 | 0.840 | 0.861 |
| ssl3_gpu | 4 | 0.701 | 0.738 | 0.777 | 0.809 | 0.843 | 0.877 |
| ssl3_gpu | 5 | 0.708 | 0.735 | 0.768 | 0.803 | 0.842 | 0.887 |
| ssl3_gpu | X1 | 0.711 | 0.742 | 0.770 | 0.804 | 0.842 | 0.883 |
