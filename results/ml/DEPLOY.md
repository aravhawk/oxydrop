# On-device deployment: latency and size

Measured with `node tests/hybrid_engine.mjs` and `node tests/cnn_parity.mjs` (Node 26, one thread, the Surface laptop host, other jobs running). These are **laptop numbers**. Phones are slower by an unmeasured factor; the app reports the real latency per window in its debug view (`engine ... latency per 2 s window`), so every phone test gives one.

## Latency per window

| step | median | p95 / max | note |
|---|---|---|---|
| one OxyNet forward pass (10 s window) | 16.4 ms | 16.5 ms p95 | `cnn_parity.mjs`, 167,588 parameters, 11.6 M multiply-adds |
| feature tensor build (causal, 10 channels + side) | 0.2 to 1 ms | | `cnn_parity.mjs` |
| full hybrid window: v2 quality chain + v1 chain + features + 3 CNN forwards + combiner | 50 to 58 ms | 55 to 59 ms p95, 64 ms max | `hybrid_engine.mjs`, every 2 s |
| one-time baseline embedding after the first minute (8 windows x 3 seeds = 24 forwards, rebuilt with the finished baseline) | about 0.5 s | | once per recording |
| v2 quality chain alone (the old main-path cost) | a few ms | | main thread in the live app |

The hop is 2 s, so the model uses about 3 % of one core on this laptop. In the live app the three CNN forward passes and the v1 chain run in a Web Worker (`app/js/hybrid_worker.js`), and only the v2 quality chain runs on the page, so a slow phone delays the answer by the model time (50 ms here, several times that on a phone) but not the camera frames. Without module-worker support the same engine runs in the page thread (debug line `runs in the page thread`); then a window that costs more than about 50 ms can skip camera frames and the dropped-frame gate will answer *Unsure*. This has not been tested on a phone.

## Size

| asset | bytes | gzip |
|---|---|---|
| `hybrid_R_s0.json`, `hybrid_R_s1.json`, `hybrid_R_s2.json` (3 CNN seeds) | 895,238 each, 2,685,714 total | 668,260 each |
| `hybrid_weights.json` (chain, meta-learner, members) | 7,143 | 2,461 |
| `conformal.json` | 3,792 | 1,393 |
| `weights.json` (v2 quality chain) | 1,869 | 835 |
| `oxynet.onnx` (seed 0, not loaded by the app) | 727,702 | 625,873 |
| `js/hybrid_engine.js`, `hybrid_async.js`, `hybrid_worker.js`, `cnn.js`, `cnn_features.js` | 39,000 together | 15,200 |
| replay demo (2 recordings, `.bin` + `.json`) | 757,980 for the `.bin` files | |

First load downloads about 2.7 MB of model JSON (about 2.0 MB gzipped) once; the service worker (`app/sw.js`, cache `oxydrop-v3.0-hybrid`) keeps all three seed files, the engines and the replay demo for offline use. The model JSON size is below the 1.5 MB per-file target of the plan; the three seeds together are 2.7 MB.

## Reproduce
```bash
npm test                              # includes test:cnn, test:hybrid, test:sw
node tests/hybrid_engine.mjs          # prints LATENCY and ASSETS lines
python scripts/final_model_json.py    # results/ml/FINAL_MODEL.json (hashes, parameters, seeds, training subjects)
```
