# OxyDrop ML data pipeline

Builds windowed, causal, label-aligned shards for every dataset into the cache directory (`OXY_CACHE`, default `/home/aravhawk/.t3/scratch/veritas/ml_cache/data/<dataset>/`).
Nothing row-level is ever written inside the repository. Everything below runs from the repo root with the ML venv:

```bash
source scripts/ml_env.sh                       # TMPDIR on disk, venv on PATH
cd python                                      # oxydrop_ml is imported from python/
python -m oxydrop_ml.data.hoffman  --build --check     # 3 s
python -m oxydrop_ml.data.mths     --build --check     # 3 s
python -m oxydrop_ml.data.butppg   --build --check     # 25 s
python -m oxydrop_ml.data.openox   --build --check     # 2 min, peak RAM < 1 GB
python -m oxydrop_ml.data.pretrain --build --check     # 25 s, writes ml_cache/DATA_MANIFEST.md
python -m oxydrop_ml.data.pretrain --manifest          # rewrite only DATA_MANIFEST.md
cd .. && python -m pytest python/oxydrop_ml/tests -q
```

`--check` exits non-zero if a shard does not match the documented counts (Hoffman: 5,644 evaluated windows, 6 people, 12 recordings, labels equal to
`results/loPO_v1_windows.csv`; MTHS 62 people; BUT 38 people; OpenOx >= 130 people with HR, ABG and skin-tone labels; pretraining corpus >= 300k windows).
Data locations can be overridden with `OXY_HOF`, `OXY_MTHS`, `OXY_BUT`, `OXY_OPENOX`, `OXY_OPENOX_ALIGN`, `OXY_DATA`.

## Window counts (built Nov 15 2026)

| shard | windows | evaluated windows | people (groups) | recordings | domain |
|---|---|---|---|---|---|
| hoffman | 6,004 | 5,644 (t >= 60 s) | 6 | 12 (2 hands each) | phone |
| mths | 2,065 | 1,141 (t >= 30 s) | 62 | 62 | phone |
| butppg | 3,840 (one per 10 s clip) | 3,612 (clips after a person's first 6) | 38 | 3,840 | phone |
| openox | 344,354 (2 s hop) | 335,373 (t >= 60 s) | 137 (of 139; 6 recordings are shorter than one window) | 301 | contact red/IR |
| pretrain | 346,298 (unlabeled) | - | 246 | - | contact + exercise |

OpenOx labels: HR 170,086 windows, contact-oximeter SpO2 154,201, ABG-anchored SaO2 154,201, ABG-draw windows 19,581 (6,916 draws, 3,818 in aligned recordings), all on the 164
clock-aligned recordings (93 people). Skin tone (Monk, Fitzpatrick, ITA) for all 137 people. Pretraining sources: OpenOx 229,123, PhysioNet PTT-PPG 31,481, moshekam exercise 54,509, apnoea neck oximeter 6,487, BIDMC 24,698.

## Datasets

| dataset | licence | size | labels | time alignment and caveats |
|---|---|---|---|---|
| Hoffman 2022 (UW) | MIT | 6 people, 12 hands, Nexus 6P, ~1,100 s each | reference SpO2 (1 Hz, mean of the working oximeters), pulse (HR) | Same labelling as `python/oxydrop/hoffman.py`. Both hands of a person share one `subject`. No frame timestamps in the release: `dt` is nominal. |
| MTHS | CC BY-NC-ND 4.0 | 62 people, iPhone 5s, 26 to ~90 s | 1 Hz oximeter HR and SpO2 | Normoxic (about 1 % of seconds below 90). Never commit or redistribute raw data or adaptations; weights trained on it are non-commercial research. Baseline span is 30 s because recordings are short. Channel order as released (red is often saturated). |
| BUT PPG v2.0 | CC BY 4.0 | 3,840 clips from the 38 SpO2 people (3,888 clips / 50 subjects exist; clips whose header uses the 300-signal layout are skipped) | ECG HR per clip, expert quality (1 good, 0 bad), a **spot** SpO2 per person | Clips are 10 s and not contiguous, so the baseline of a person is its first 6 clips. The SpO2 is one value per person, not time aligned. No frame timestamps in the release (fs = 30 only): `dt` is nominal. Ear and finger clips are both present (`ear_finger`), `motion` codes kept. |
| OpenOximetry v1.1.1 | PhysioNet credentialed (DUA) | 237 people, 907 encounters; 307 with 86 Hz red/IR PPG | 2 Hz reference HR and SpO2, ABG SaO2 at draws, skin tone | Contact finger-clip PPG, **not a phone camera**. Reference trails the PPG; per-window labels only on the 164 clock-aligned recordings. Row-level arrays stay in the cache: never commit, never Kaggle (Daytona only). |
| pretrain corpus | mixed (see sources) | 346k windows, float16 | none | Disjoint from Hoffman / BUT / MTHS people. OpenOx windows keep `subject` so held-out OpenOx people can be dropped (`splits.pretrain_allowed`). |

Frame timestamps: no phone dataset here ships per-frame times. The `dt` channel is therefore all zeros in the cached data; train with
`windows.simulate_irregular` (jittered frames, drops, stalls, resampled to the grid exactly like real data) so the model sees the channel, and use `windows.resample_uniform`
for real timestamps from the app.

## Shard layout (see `python/oxydrop_ml/windows.py`)

`series.f32` (log-RGB at 30 Hz, all recordings back to back), `dt.f16`, `meta.parquet` (one row per window), `recs.parquet`, `info.json`; OpenOx adds `ref.parquet` (1 Hz reference HR,
SpO2 and bias-corrected SaO2 per recording on the PPG time axis) and `draws.parquet`. Windows are index ranges of `series`; `WindowSet.lrgb(idx)` returns `[B, 450, 3]` buffers
(5 s warm-up, odd-reflection padded at the start of a recording) and `WindowSet.tensors(idx)` returns the model input `x [B, 10, 300]` and side vector `s [B, 9]` from
`features.windows_to_tensor` (channel layout and definitions: top of `python/oxydrop_ml/features.py`).

Tensor cache: `WindowSet.build_tensor_cache()` writes `x.f16.npy` / `s.f32.npy` next to a shard (already built for all four shards; OpenOx is 2 GB); `ws.cached()` memory-maps them. Rebuild it after rebuilding a shard.

Per-window columns: `subject`, `rec`, `t_start`, `end_idx`, `domain`, `eval`, `in_baseline`, `hr_ref/has_hr`, `spo2_ref/has_spo2`, `low/has_low` (SpO2 < 90),
`sao2/has_sao2`, `sao2_draw/has_sao2_draw/draw_dt`, `q_expert/has_q`, `monk`, `ita`, `fitz`, `skin_group`, `abs_*` (window features), `base_*` (personal baseline of the recording), `has_base`.
`has_*` is the per-head label mask. Splits use `subject` only.

## OpenOx details

* PPG: red -> channel R, IR -> channel G, no B (`m2 = False`), resampled 86 -> 30 Hz with `resample_poly` (anti-alias FIR) before the log. Domain flag `domain = 1`.
* HR label: reference-device HR (30 to 220 bpm) window mean shifted by the encounter's HR lag `lstar` (5 to 10 s typical).
* SpO2 labels: `spo2_ref` is the display series; `sao2` maps it to ABG SaO2 with a per-encounter linear fit on the draws (median fit RMSE 0.76 points); `sao2_draw` is the ABG value on the windows whose interval contains the draw marker.
* Reference lag: stored labels use lag 0. `openox.relabel(ws, lag)` re-derives `spo2_ref`, `sao2`, `low` for a lag; `openox.estimate_lag(train_subjects)` learns it from the given subjects only
  (`method='ppg'`: correlation of the per-beat red/IR ratio-of-ratios with the lagged reference; `method='abg'`: ABG draws against the lagged display). Never pass a test subject.
  Example (diagnostic, first 80 of 137 subjects by sorted id): PPG method 25 s (bootstrap CI 20 to 30), ABG method 18 s (CI 15 to 19), both inside the 10 to 30 s expected range.
