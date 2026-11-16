# multida (R3-MULTI): one network, Hoffman + MTHS + BUT PPG, domain adaptation

**Licence notice.** multida was trained on MTHS (CC BY-NC-ND 4.0, non-commercial research only). Its weights (results/photon/multida_weights.json, results/photon/multida_app.json, ml_cache/photon_multida/) must carry the non-commercial notice (stored in both JSON files as `licence_notice`), are not shipped in the app, and are not committed. No raw MTHS or BUT PPG data is in the repo.

**Status.** Candidate only. ROUND3 lock (20:22 CT) picked distill2; multida L0 0.744 is reported as a candidate (it would not meet the replacement rule in any case). L0 + nulls landed 20:17 CT, stress 20:17 CT (2 min after the 20:15 L0 cutoff; disclosed). Training stopped; nothing running.

Module: `python/oxydrop_ml/photon/multida.py` (training, nested selection, scores), `python/oxydrop_ml/photon/multida_export.py` (export).
Run on Daytona sandbox R3M (2 CPU, 2 GiB, 1 GiB disk; numpy only), 20:03-20:17 CT; sandbox deleted at 20:27 CT. Scored locally with
`python -m oxydrop_ml.photon.evalkit` (results/photon/TABLE.csv); paired differences vs `distill` with the same person bootstrap (winner2.Evaluator, B = 2000).

## Design (fixed before any multida score; designed after round-1/2 outer results, Amendment 8)
- Front end: distill's 39 causal features (flashgap.rec_features), so the browser path is app/js/distill_features.js + nn_runtime.js.
- Data: Hoffman training people of the outer fold (clean + flashgap augmentation, seed = model seed); MTHS 1,141 windows (62 recordings, baseline = first
  30 s, windows t >= 30 s; SpO2 87.4-100, 19 windows < 90); BUT PPG 2,460 windows (38 subjects, motion-free clips concatenated per subject, baseline =
  first 6 clips, one window per clip, HR label). Aux rows enter every inner and outer fit; they contain no Hoffman person. OpenOximetry not used.
- Network: 39 -> 32 -> ReLU -> 32 -> ReLU -> 4 heads [hyp, SpO2, abstain, dHR]. Domain adaptation: per-domain normalisation (Prep per dataset; only the
  Hoffman Prep is deployed) + gradient-reversal domain head (GRL 0.1, 32 -> 16 -> 3, training only).
- Loss: distill's Hoffman loss (BCE + chain/DC teacher KD, Huber SpO2, abstain BCE) + w_aux x (MTHS: hypoxia BCE, Huber SpO2, Huber dHR; BUT: Huber dHR)
  + domain CE through the GRL.
- Nested selection (leave-one-inner-person-out on the 5 training people): lam_kd {2, 8} x w_aux {0.3, 1} x epochs {15, 30}; tau by flashgap.select_tau.
- Training is plain numpy (manual backprop, Adam with torch semantics): the only sandbox that fitted the shared Daytona quota had 1 GiB of disk, too small for torch.
- Disclosure: one smoke test of the numpy trainer (2 configs, no selection) printed the AUROC of fold 100001's held-out person; nothing was chosen from it.

## Results (TM bal. acc., 95% person bootstrap; paired person bootstrap)
| model | L0 | vs chain_v1 L0 | people > chain | vs distill L0 |
|---|---|---|---|---|
| multida (3-seed ensemble) | 0.744 [0.553, 0.807] | -0.053 [-0.142, -0.010] | 3/6 | -0.014 [-0.055, +0.023] |
| multida_s0 / s1 / s2 | 0.748 / 0.733 / 0.750 (mean 0.744) | -0.048 / -0.064 / -0.047 | 4 / 3 / 3 of 6 | -0.010 / -0.025 / -0.008 |
| multida_hofonly (ablation, seed 0, no aux, no GRL) | 0.730 [0.549, 0.804] | -0.066 [-0.151, -0.018] | 2/6 | -0.028 [-0.064, +0.004] |
| chain_v1 | 0.796 [0.636, 0.857] | | | |
| distill | 0.758 [0.603, 0.813] | | | |

Aux data effect, same seed: multida_s0 - multida_hofonly = +0.018 [-0.006, +0.056] (2/3 people differing); ensemble - hofonly = +0.014 [-0.002, +0.033].

Stress (ensemble), vs chain_v1 and vs distill:
| level | TM | vs chain_v1 | people | vs distill | coverage | answered acc |
|---|---|---|---|---|---|---|
| L1 | 0.722 | -0.082 [-0.188, -0.006] | 2/6 | +0.017 [-0.021, +0.039] | 0.66 | 0.940 |
| L2 | 0.722 | -0.021 [-0.119, +0.100] | 2/6 | +0.018 [-0.024, +0.046] | 0.65 | 0.933 |
| L3 | 0.680 | +0.115 [+0.003, +0.216] | 4/6 | -0.000 [-0.030, +0.020] | 0.61 | 0.919 |
| L4 | 0.691 | +0.093 [+0.002, +0.189] | 6/6 | +0.010 [-0.008, +0.023] | 0.58 | 0.918 |
| L5 | 0.696 | +0.161 [+0.026, +0.232] | 5/6 | -0.005 [-0.022, +0.010] | 0.35 | 0.914 |
| X1 | 0.678 | +0.170 [+0.027, +0.328] | 6/6 | +0.002 [-0.053, +0.044] | 0.45 | 0.924 |
Mean paired diff vs chain_v1 over L3-L5, X1: +0.135. L0 coverage 0.72, answered accuracy 0.907.

Per person L0 (100001..100006): multida 0.636 0.500 0.687 0.681 0.708 0.589; chain_v1 0.889 0.458 0.660 0.711 0.737 0.539; distill 0.670 0.516 0.649 0.721 0.700 0.536.

Nulls (Hoffman labels permuted globally, aux labels permuted within dataset): multida_null 0.492, multida_null_d1 0.534, multida_null_d2 0.541; mean 0.522 (<= 0.55, pass).

## Verdict under the round-3 winner rule
Not a replacement: L0 ensemble 0.744 is below distill's 0.758 (the rule needs a higher point estimate, and the paired lower bound vs distill, -0.055, is below
-0.02). Loses to chain_v1 at L0 (CI excludes 0). Stress is a tie with distill and a win over chain_v1 from L3 on. Adding MTHS + BUT PPG helps this pipeline by
about +0.014 to +0.018 L0 over the Hoffman-only ablation, CI includes 0. The Hoffman-only ablation (0.730) is itself below distill (0.758), because this
grid fixes weight decay at 1e-4 where distill searched {1e-4, 1e-2}.

## Seed recipes for the export (modal nested choice; tau = median fold tau)
s0: lam 8, w_aux 1.0, 15 epochs, tau 0.922; s1: lam 8, w_aux 0.3, 30 epochs, tau 0.911; s2: lam 8, w_aux 1.0, 30 epochs, tau 0.945. Refit on all 6 Hoffman
people + aux (on Daytona). Exports: results/photon/multida_weights.json (distill_export format, works with `distill_app_export.py --src`) and
results/photon/multida_app.json (nn-runtime-v1 graphs, model "multida"). Reference on the 2 fixture windows (100001L t = 60, 300 s): p = 0.0797, 0.1906;
SpO2 96.8, 92.8; conf 0.9992, 0.9631; both answered. The app-schema round trip matches within 9.4e-09. app/ not modified.

## Files
Scores: results/photon/scores/multida__L0..L5,X1, multida_s{0,1,2}__*, multida_hofonly__*, multida_null__L0, multida_null_d1__L0, multida_null_d2__L0.
Fold pickles + aux feature pack (derived MTHS/BUT features, not committed): ml_cache/photon_multida/. Remote logs: ml_cache/dt/R3M/x/runs/.
Cloud: Daytona sandbox R3M deleted. Kaggle: private dataset aravhawk/oxy-r3multi-data (code + Hoffman pack + MTHS/BUT cached series; created while
Daytona was full, never used because Kaggle CPU/GPU session limits were full); no Kaggle kernel was pushed.

## GPU relaunch `multida_gpu` (21:00 CT deadline): no result
Torch/CUDA port written (python/oxydrop_ml/photon/multida_gpu.py: same 39 features, per-domain Prep, GRL 0.1 domain head, distill KD loss; all 24 grid
configs of an inner fit trained as one stacked batch; grid lam {8, 16, 32} x wd {1e-4, 1e-2} x w_aux {0.3, 1} x epochs {15, 30}). Kaggle refused every push
from 20:45 to 21:02 CT (account limit: 2 batch GPU sessions, both held by other agents' kernels). The one run that started (aravhawk/oxy-r3mgpu-a, Tesla T4,
21:02 CT) crashed after 18 s, before any training: the module called multida._sig, which the older multida.py in the Kaggle dataset does not have. Fixed
locally; the re-push was refused again (same limit) until 21:07 CT and then abandoned. No multida_gpu score files exist; the CPU multida numbers above
do not count under the GPU addendum. Details: results/photon/multida_gpu_run.json.
