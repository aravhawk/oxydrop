# Photon-stress harness

Shared code: `python/oxydrop_ml/photon/stress.py` (degradation), `evalkit.py` (scoring), `baselines_stress.py` (chain v1 under stress). Binding interface: the task's `CONTRACT.md`.

## What the stress test is

A **simulation** of a phone whose flash sits farther from the lens than on the recording phone (iPhone 18 Pro: little light crosses the finger; iPhone 13 Pro: marginal). It is not a measurement of any phone. Only the held-out person's recordings (baseline minute included) are degraded. Training people stay clean unless a candidate adds its own disclosed augmentation with an independent seed stream. Models are not refit per level.

Pipeline of `stress.degrade(rgb, level, seed)` on the per-frame ROI-mean RGB (linear 0-255):

1. split each channel into DC (centred 5 s moving average) and the pulsatile part (ln x / DC);
2. PhotonTwin band-integrated render at the recording geometry (rho0 = 3 mm) and at rho0 + 0.25 d_rho for a nominal finger (sat 0.97, fb 0.02, fm 0.03): DC attenuation a_c and pulsatile gain m_c per channel (blue dies first, red last);
3. auto-exposure: gain g = min(1 / a_R, 1024) restores the red DC and multiplies shot and read noise;
4. oblique-illumination leak: an additive non-pulsatile fraction f of the red light (ambient / shunt light), with the white balance of the recording;
5. noise: shot sd = sqrt(g (signal + leak) / K), K = 2e5 e-/DN, plus read noise g x 0.002 DN;
6. quantisation of the ROI mean to 1/32 DN and clipping to [0, 255].

Deterministic in (level, seed, recording). Level 0 is the identity.

| level | d_rho (mm) | red light kept | green | blue | m_c R / G / B | gain | leak | read noise after gain (DN) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 1 | 1 | 1.00 / 1.00 / 1.00 | 1 | 0 % | 0.002 |
| 1 | 2 | 0.171 | 0.14 | 0.103 | 1.01 / 1.00 / 0.77 | 6 | 2 % | 0.0117 |
| 2 | 4 | 0.0309 | 0.0207 | 0.012 | 1.03 / 1.00 / 0.74 | 32 | 5 % | 0.0646 |
| 3 | 6 | 0.00586 | 0.00322 | 0.00147 | 1.04 / 0.99 / 0.74 | 171 | 10 % | 0.341 |
| 4 | 8 | 0.00116 | 0.000522 | 0.000188 | 1.05 / 0.98 / 0.74 | 862 | 20 % | 1.72 |
| 5 | 10 | 0.000238 | 8.8e-05 | 2.49e-05 | 1.07 / 0.96 / 0.74 | 1024 (cap) | 35 % | 2.05 |

Median pulsatile-band (0.7-3.5 Hz) SNR of the red channel over the 12 recordings: 32, 24, 13, 0 and -13 dB at levels 1-5 (`stress.snr_db`, no labels used).

**Out-of-family stress `X1`** (`stress.degrade_x(rgb, 'X1', seed)`, files `<model>__X1.csv`): level-3 optics and leak without the white noise, then motion-like DC steps (Poisson, one per 20 s, relaxing over 3 s), AR(1) correlated sensor noise (phi 0.7, half shared across channels, level-3 marginal sd), sample-and-hold frame drops (a few percent of frames) and a per-channel tone curve (gamma 0.75 / 0.80 / 0.85). The L-family is white noise + attenuation + leak, so X1 tests whether a candidate's augmentation only learned that family.

### Deviations from the contract text, stated openly

* **optics.py sign.** `PhotonTwin.ln_reflectance` subtracts the image-source term of the Farrell et al. (1992) dipole; Eq. 10 of that paper adds it. With the subtraction the reflectance decays 10-20 orders of magnitude too fast with rho (red x5e-18 at +2 mm). `stress.py` recomputes the reflectance with the correct sign (log-sum-exp) from the twin's own mu_a, mu_s', camera bands and flash spectrum. `optics.py` itself is untouched; models that train against it inherit the subtraction.
* **Effective separation.** Even with the right sign, a semi-infinite medium over-attenuates a finger slab (red x1e-11 at +10 mm), which would make levels 2-5 a cliff. The renderer is evaluated at rho0 + 0.25 d_rho (`PATH_SCALE`), a constant fixed from the input-side SNR of the recordings before any model was scored on a degraded recording. The shape (colour order, modulation, band shift) is the renderer's; the absolute scale is a design choice.
* **Quantisation** is 1/32 DN of the ROI mean instead of 8-bit steps: a mean over thousands of dithered 8-bit pixels has sub-DN resolution, and a 1 DN step erases the 0.1-1 DN pulse of these recordings even at level 1.

## Score files and the evaluator

`results/photon/scores/<model>__L<0-5>.csv` and `<model>__X1.csv`: one row per evaluated Hoffman window (t >= 60 s, 5,644 rows; windows.csv rows with t < 60 are ignored), columns `rec, t, sid, y_spo2, y, score, pred, answer` (+ optional `spo2_hat, spo2_lo, spo2_hi, conf`). All predictions out-of-fold: leave-one-person-out, both hands together, every selection nested on training people.

`cd python && python -m oxydrop_ml.photon.evalkit` writes `TABLE.csv`, `TABLE.md`, `STRESS.csv`:

* TM bal. acc. (Hoffman time-matched, clock-only = 0.5), bal. acc., AUROC, each with a 2000-resample person-bootstrap 95 % CI (same resampled people for every model, closed-form TM per resample);
* paired person-bootstrap difference to `chain_v1` at the same level, verdict win / tie / loss;
* people better than `chain_v1` (per-person TM) and the exact one-sided sign test (6/6 p = 0.016, 5/6 p = 0.109);
* worst person TM, coverage and accuracy of the model's own `answer`, accuracy at 100-50 % coverage by `conf`, AURC, SpO2 MAE when `spo2_hat` is present;
* null controls: a file named `<model>_null__L0` or `<model>_blank__L0` must score about 0.50 TM; the table flags it when TM > 0.55.

## Adding a model

1. Write `python/oxydrop_ml/photon/<name>.py` with `run(level=0, seed=0)` returning the score frame, and a `__main__` that writes `results/photon/scores/<name>__L0..L5.csv` and `<name>__X1.csv` (use `stress.degrade_x` for X1).
2. Degrade only the held-out person's recordings, with `stress.degrade(rgb, level, seed)` applied to each full recording (raw per-frame RGB from `oxydrop.hoffman.load`); training people stay clean.
3. Add the null controls (`<name>_null__L0.csv`: the same pipeline fed constant or other-person observations).
4. Re-run the evaluator. A win needs the paired CI to exclude 0; otherwise write tie.

For chain-type feature paths, `baselines_stress.degraded_records(level)` replaces `chain_v1.hoffman_records` with the degraded recordings (hook used for `chain_v1`).

## Baselines under stress (this directory)

`chain_v1` (shipped C) and `chain_v1_nestedC` (C on a 5 x 5 grid by inner CV on the training people, clean) are scored at levels 0-5 and X1. `cnn_full` (3-seed mean p_low, threshold 0.5) and `hybrid_final` (prob, threshold 0.5) are converted at level 0 only: the repository has no leave-one-person-out checkpoints, only the 12 final all-people fits (`ml_cache/runs/hyb_final`), which have seen every test person; scoring degraded recordings would need retraining. See `TABLE.md` for the numbers.
