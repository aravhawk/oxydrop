# OxyNet specification (the graph FEAT-006 trains and FEAT-009 ships)

Code: `python/oxydrop_ml/model.py` (torch), `python/oxydrop_ml/export_js.py` (BatchNorm folding, ONNX, fixtures), `app/js/cnn.js` (browser forward pass), `app/js/cnn_features.js` (streaming input builder). Input construction is defined by `python/oxydrop_ml/features.py` (`windows_to_tensor`, FEAT-003); the JS port follows it step for step.

This is a research prototype, not a medical device. The shipped weights are the three residual-CNN seeds of the final hybrid (`app/data/hybrid_R_s{0,1,2}.json`, see `results/ml/HYBRID.md`); `app/data/oxynet.onnx` is the ONNX export of seed 0 and `tests/fixtures/cnn_ref.json` its PyTorch reference.

## Input

`x` float32 `[B, 10, 300]` (10 s at 30 Hz, last 300 samples of a 15 s buffer), channel order:

| idx | name | meaning |
|---|---|---|
| 0, 1, 2 | `ac_R`, `ac_G`, `ac_B` | 100 x causal 0.7-4 Hz band-pass of ln(rgb + 1) (percent modulation in log units). **Waveform slots.** |
| 3 | `dt` | `clip(dt*30 - 1, -1, 8)`: 0 when frames arrive on time, positive for late frames and stalls |
| 4, 5 | `ror_RG_rel`, `ror_RB_rel` | per-beat log ratio-of-ratios ln(AC_R/AC_G), ln(AC_R/AC_B) (peak-to-trough in log units, i.e. AC/DC), held between beats, minus the person's baseline value; 0 where no beat in the last 3 s |
| 6 | `beat_ok` | 1 while a beat was seen within 3 s |
| 7, 8, 9 | `dc_R_rel`, `dc_G_rel`, `dc_B_rel` | slow ln DC (tau 3 s) minus the person's first-minute ln DC |

Side vector `side` float32 `[B, 9]`: `d_ror_RG, d_ror_RB, d_dc_R/G/B, d_lac_R/G/B, beat_frac` (window means minus the person's baseline; 0 where the baseline is missing).

`domain` int64 `[B]`: 0 phone camera (R, G, B), 1 contact oximeter. For domain 1 the data builders store `[red, IR, 0]`, so red goes to slot 0, infrared to slot 1, slot 2 is masked, and `ror_RG_rel` is ln(red/IR). The domain embedding tells the network which physics it is looking at.

`emb_base` float32 `[B, E]`: the person's baseline window embedding (see "Baseline subtraction").

## Graph (student, `student_cfg()`)

Every conv is `Conv1d` + folded BatchNorm. Padding is explicit zero padding `(left, right)` (same: `(d(k-1)//2, rest)`, causal option: `(d(k-1), 0)`). ReLU after every block.

1. **Channel-shared stem** (`stem='shared'`): one `Conv1d(1, 8, k=5)` + ReLU applied to each of the 3 waveform slots with the same weights, outputs of masked slots (per domain, `domain_slot_mask`) zeroed, concatenated to 24 channels, then the 7 auxiliary channels appended (31 channels), `Conv1d(31, 24, k=5)` (BN folded), **add `domain_emb[domain]` (24 values, per channel, broadcast over time)**, ReLU. `stem='joint'` instead runs a plain `Conv1d(10, 24, k=5)` + ReLU (no domain input) and is supported by the exporter and JS.
2. Four stages of 2 residual blocks, widths **24, 40, 56, 80**. Block: `conv(k=3, dilation d) -> BN -> ReLU -> conv(k=3, dilation d) -> BN`, plus skip (identity, or `1x1` strided conv + BN when width or stride changes), then ReLU. The first block of stages 2 to 4 has stride 2 (time 300 -> 150 -> 75 -> 38). Dilation of block j in a stage is `2**(j % 2)` (1, 2).
3. **Pooling**: global average and global max over time, concatenated (160 values).
4. `emb = ReLU(Linear(160, 64))`, the window embedding.
5. Heads:
   - `hr_logits [B,141]` = `Linear(64,64) -> ReLU -> Linear(64,141)`; bin i is (40 + i) bpm. Decode: argmax, or the probability-weighted mean within +-7 bins of the argmax (`hr_from_logits`, `decodeHr`).
   - `spo2`, `spo2_logvar`: `Linear(emb ++ side, 2)`; `spo2 = 93 + 5*raw` (absolute %), the second output is the Gaussian log-variance for the NLL loss.
   - `lowox_logit`: `Linear((emb - emb_base) ++ side, 1)`.
   - `sqi_logit`: `Linear(emb, 1)` (target: HR within 5 bpm of the reference).
   - `aux_spo2`: `Linear(emb, 1)`, contact-oximeter SpO2 for the OpenOximetry domain. **Training only**: returned by `forward(..., with_aux=True)`, never exported to JSON or ONNX.

Teacher (`teacher_cfg()`): slot width 16, stem 64, widths 64/128/192/256, depths 2/3/3/2, embedding 128, HR hidden 128, dropout 0.1. Same ops, same exporter.

### Op list (exactly what `cnn.js` implements)

`shared_stem_input` (shared slot conv + mask + aux concat), `conv1d` (stride, dilation, explicit pad, optional ReLU), `add_domain` (+ ReLU), `add` (residual, + ReLU), `pool` (avg + max), `dense` (concatenate inputs, optional ReLU), `sub`. ONNX contains Conv, Pad, Relu, Add, Sub, Gemm, Gather, Concat, ReduceMean, ReduceMax, Reshape and a few shape ops, dynamic batch.

### Baseline subtraction

The low-oxygen head sees `emb - emb_base`, where `emb_base` is the **mean of `emb` over the person's own calibration windows** (the windows ending in the first 60 s, label free; on device: run `forward` on each calibration window, keep `out.emb`, average with `meanEmbedding`). During training use the same recipe per recording (detach it or not, a FEAT-006 choice; the exporter does not care). With `emb_base = 0` the head sees the raw embedding. `side` carries the same baseline-relative idea for scalar physics features. The baseline never uses labels.

## Sizes (this repo state)

Student 167,588 parameters without the training-only head (167,653 with it), 11.6 M multiply-adds per window, JSON 874 KB (base64 float32), ONNX 711 KB. Teacher about 1.9 M parameters. JS forward about 17 ms median per window on the development laptop (Node 26, one thread).

## Files and how they agree

- `app/data/hybrid_R_s{0,1,2}.json` (one file per seed, exported by `export_js.py`/`export_hybrid.py`): `{format:'oxynet-v1', config, input, hr, outputs, params, tensors{name:{shape,b64}}, ops[]}`. BN is folded in float64 and stored as float32.
- `app/data/oxynet.onnx`: eval-mode graph, inputs `x, domain, side, emb_base`, outputs `hr_logits, spo2, spo2_logvar, lowox_logit, sqi_logit, emb`. `onnxruntime` agrees with torch to 1e-5 absolute plus 1e-6 relative (float32 spacing at spo2 near 93 is 7.6e-6).
- `tests/fixtures/cnn_ref.json`: random, contact-domain and recorded (Hoffman 100001L) inputs with torch outputs for every head, plus streaming feature checks (uniform dt, irregular dt, noise, contact mask). `cnn_parity.mjs` requires JS to match torch to 1e-4 (measured about 1e-5 or better).
- Regenerate everything from the repo root with `ml_venv/bin/python python/oxydrop_ml/export_js.py` (random init) or `--ckpt model.pt` (a `{cfg, state}` dict from `train.py`). The feature part of the fixture records the sha256 of `features.py` it was built from.

## Notes for training (FEAT-006)

- Build `OxyNet(OxyCfg(**cfg))`, call `net(x, domain, side, emb_base, with_aux=True)` in training. Loss terms: soft-label CE on `hr_logits`, Gaussian NLL on `spo2/spo2_logvar` (clamp the log-variance in the loss), BCE on `lowox_logit` and `sqi_logit`, MSE or Huber on `aux_spo2` for domain-1 batches only.
- Checkpoints for the exporter: `torch.save({'cfg': net.cfg.to_dict(), 'state': net.state_dict()}, path)`. `export_js.py` rebuilds the net from `cfg` and loads `state`. The exporter drops `aux_spo2.*` weights automatically.
- Keep the layout and the 10 s window fixed for the shipped model; changing channels or the padding mode requires re-exporting and regenerating fixtures. `causal=True` and `stem='joint'` are exercised by `tests/test_model.py` for JS parity.
- BatchNorm statistics must be in eval mode for export (`net.eval()` is applied by the exporter).
