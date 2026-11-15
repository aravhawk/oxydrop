# Figures and generated documents (FEAT-010)

Every figure is drawn by `scripts/make_ml_figs.py` and every number in README.md, MODEL.md, WHITEPAPER.md and docs/WRITEUP.md is resolved by `scripts/ml_numbers.py` from the CSV and JSON files in this directory. The Markdown sources are `docs/templates/*.md.in`; `results/ml/CLAIMS_MANIFEST.csv` lists every token and the value it rendered to.

## One command to refresh everything

```bash
source scripts/ml_env.sh && bash scripts/reproduce_ml.sh docs
```

This runs, in order: `python scripts/make_ml_figs.py`, `python scripts/render_docs.py`, `bash scripts/build_whitepaper_pdf.sh` (pandoc to HTML with print CSS, then headless Chromium via playwright-core; no LaTeX), `python scripts/check_claims.py` (must print `CHECK CLAIMS PASS`; it also requires WHITEPAPER.pdf to be 4 to 6 pages and up to date). `bash scripts/reproduce_ml.sh docs-md` skips the PDF.

### Tables backed by the FEAT-007 files

Nothing is pending. The conformal, OpenOximetry and heart-rate-voting tables are typed tables built by `block_conformal`, `block_openox` and `block_hrvote` in `scripts/ml_numbers.py` from the files below; scalar tokens in the templates (`cf:`, `cm:`, `cpp:`, `ox:`, `oxp:`, `oxg:`, `oxb:`, `oxl:`, `hv:`, `hvs:`) read the same files.

| Section | Files | Figure |
|---|---|---|
| Conformal 3-state output and matched coverage with v1 and v2 | `conformal_curve.csv`, `conformal_matched.csv`, `conformal_per_person.csv`, `conformal_calibration.csv`, `conformal_reference_bands.csv` | `ml_conformal.png` |
| OpenOximetry SaO2 (blood-gas anchored, lag corrected) by skin tone | `openox_spo2_overall.csv`, `_skin.csv`, `_gaps.csv`, `_paired.csv`, `_lag.csv`, `_counts.csv`, `_contact_bias.csv`, `_contact_bias_gaps.csv` | `ml_openox_skin.png` |
| Heart-rate voting and slow pulse | `hr_vote_summary.csv`, `hr_vote_paired.csv`, `hr_vote_slow_band.csv` | (tables only) |
| App statement (hybrid is the default engine, latency, sizes) | `app/js/hybrid_engine.js` wired in `app/js/app.js`; `results/ml/DEPLOY.md` | none |

The generator detects whether the app is wired (`_app_state` in `scripts/ml_numbers.py`) and says so in the documents; if the files are missing, the block says PENDING instead of claiming it.

## Figure list

| File in `docs/figures/` | Shows | Source |
|---|---|---|
| `ml_architecture.png` | OxyNet and the deployed hybrid | `python/oxydrop_ml/model.py` (parameter count) |
| `ml_flow.png` | input to output, three states | none (diagram) |
| `ml_ssl_curves.png` | SSL loss and retrieval, with and without OpenOximetry | `ssl_loss_curve_full.csv`, `ssl_loss_curve_noox.csv` |
| `ml_ablation.png` | ablation with 95% intervals, P1 and P2 | `ablation_core.csv`, `baselines_P1.csv`, `hybrid_p2.csv` |
| `ml_coverage_accuracy.png` | accuracy versus coverage, share of hypoxic windows alerted: v1 chain, XGBoost, CNN, hybrid | `scores/hybrid_final_hoffman.csv`, `scores/baseline_P1_hoffman.csv`, `scores/cnn_full_hoffman.csv` |
| `ml_conformal.png` | held-out accuracy versus coverage (hybrid, deployed R, v1 chain, with the v1 and v2 bands) and hypoxic windows called Normal per person | `conformal_curve.csv`, `conformal_reference_bands.csv`, `conformal_per_person.csv` |
| `ml_hr_blandaltman.png` | HR Bland-Altman (v2 spectral and hybrid) and error by dataset | `scores/hybrid_final_butppg.csv`, `hybrid_p2.csv` |
| `ml_p1_per_person.png` | P1 per held-out person | `hybrid_p1_final.csv` |
| `ml_openox_skin.png` | OpenOximetry SaO2 MAE by skin tone (CNN vs textbook ratio-of-ratios) and contact-reference bias by group | `openox_spo2_skin.csv`, `openox_spo2_contact_bias.csv` |
| `ml_gap.png` | train-versus-test gap | `baselines_P1.csv`, `ablation_core.csv`, `hybrid_p1_final_gap.csv` |
| `ml_failure_modes.png` | failure modes: four synthetic schematics (labelled in the figure) and two real out-of-fold score timelines | `scores/hybrid_final_hoffman.csv`, `scores/cnn_full_hoffman.csv` |

The failure-mode schematics (motion burst, cold finger, far-flash phone, saturated channel) are synthetic signals that illustrate a mechanism; they are not recorded data and say so in the figure. The accuracy-versus-coverage curves rank windows by distance from the decision boundary and tune no cutoff: they are descriptive, and the nested conformal result of FEAT-007 supersedes them once available.
