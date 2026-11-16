# Photon-stress table

Hoffman, leave-one-person-out (both hands together), 5644 evaluated windows (t >= 60 s). TM bal. acc. = Hoffman time-matched balanced accuracy at the forced decision (clock-only = 0.5) with the 2000-resample person-bootstrap 95 % CI. Difference = paired person bootstrap against `chain_v1` at the same level: **win** only when the interval excludes 0 on the positive side, otherwise **tie** (or loss). "People better" = held-out people whose own TM balanced accuracy beats `chain_v1` (ties and people without a usable 100 s bin dropped) with the exact one-sided sign test (6/6: p = 0.016, 5/6: p = 0.109). Levels 0-5 are the simulated distant-flash stress of `stress.py`; X1 is the out-of-family stress (motion steps, frame drops, tone curve, AR(1) noise). Both are simulations, not measurements of any phone. Full columns: `TABLE.csv`.

## Level 0 (flash +0 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.796 [0.636, 0.857] | 0.873 | 0.947 |  | reference |  | 0.458 | 0.84 | 0.939 | 0.037 |  |
| photonnet_xl_nullscr_d1 | 0.493 [0.433, 0.553] | 0.492 | 0.492 | -0.303 [-0.373, -0.165] | loss | 1/6 (p = 0.984) | 0.397 | 0.63 | 0.518 | 0.510 | 7.77 |
| photonnet_xl_nullscr_d2 | 0.434 [0.398, 0.523] | 0.420 | 0.426 | -0.362 [-0.427, -0.166] | loss | 2/6 (p = 0.891) | 0.440 | 0.49 | 0.396 | 0.507 | 7.74 |
| photonnet_xl_nullscr_d0 | 0.356 [0.283, 0.472] | 0.409 | 0.423 | -0.440 [-0.525, -0.180] | loss | 1/6 (p = 0.984) | 0.346 | 0.57 | 0.466 | 0.542 | 7.41 |

## Level 1 (flash +2 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.805 [0.646, 0.862] | 0.871 | 0.945 |  | reference |  | 0.431 | 0.84 | 0.929 | 0.040 |  |

## Level 2 (flash +4 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.742 [0.585, 0.820] | 0.806 | 0.814 |  | reference |  | 0.500 | 0.88 | 0.822 | 0.250 |  |

## Level 3 (flash +6 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.565 [0.503, 0.625] | 0.584 | 0.557 |  | reference |  | 0.502 | 0.92 | 0.534 | 0.503 |  |

## Level 4 (flash +8 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.598 [0.526, 0.640] | 0.603 | 0.621 |  | reference |  | 0.445 | 0.86 | 0.575 | 0.434 |  |

## Level 5 (flash +10 mm)

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chain_v1 | 0.535 [0.477, 0.598] | 0.529 | 0.542 |  | reference |  | 0.393 | 0.91 | 0.491 | 0.516 |  |

## Stress X1: out-of-family: level-3 optics (+6 mm), motion-like DC steps, sample-and-hold frame drops, per-channel gamma, AR(1) correlated noise

| model | TM bal. acc. [95 % CI] | bal. acc. | AUROC | diff vs chain_v1 [95 % CI] | verdict | people better (sign p) | worst person TM | coverage | acc. answered | AURC | SpO2 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
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
| photonnet_xl_nullscr_d0 | 0 | 0.414 | 0.442 | 0.418 | 0.409 | 0.427 | 0.484 |
| photonnet_xl_nullscr_d1 | 0 | 0.468 | 0.474 | 0.494 | 0.532 | 0.529 | 0.560 |
| photonnet_xl_nullscr_d2 | 0 | 0.421 | 0.426 | 0.426 | 0.389 | 0.398 | 0.425 |
