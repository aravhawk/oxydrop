# Figures and generated documents (FEAT-010)

Every figure is drawn by `scripts/make_ml_figs.py` and every number in README.md, MODEL.md, WHITEPAPER.md and docs/WRITEUP.md is resolved by `scripts/ml_numbers.py` from the CSV and JSON files in this directory. The Markdown sources are `docs/templates/*.md.in`; `results/ml/CLAIMS_MANIFEST.csv` lists every token and the value it rendered to.

## One command to refresh everything

```bash
source scripts/ml_env.sh && bash scripts/reproduce_ml.sh docs
```

This runs, in order: `python scripts/make_ml_figs.py`, `python scripts/render_docs.py`, `bash scripts/build_whitepaper_pdf.sh` (pandoc to HTML with print CSS, then headless Chromium via playwright-core; no LaTeX), `python scripts/check_claims.py` (must print `CHECK CLAIMS PASS`; it also requires WHITEPAPER.pdf to be 4 to 6 pages and up to date). `bash scripts/reproduce_ml.sh docs-md` skips the PDF.

### Sections that wait for the FEAT-007 files

These render as a marked **PENDING** block (and as a PENDING panel in the figure) until the files exist, and render in full when they do. Nothing else in the documents depends on them.

| Section | Files that switch it on (any one matching file is enough) | Figure |
|---|---|---|
| Conformal 3-state output and matched-coverage comparison with the v1 chain | `results/ml/conformal*.csv`, `CONFORMAL.md` | `ml_conformal.png` |
| OpenOximetry SpO2 (ABG-anchored) by skin tone | `results/ml/openox_spo2*.csv`, `OPENOX_SPO2.md` | `ml_openox_skin.png` |
| Heart-rate voting | `results/ml/hr_vote*.csv`, `HR_VOTE.md` | (table only) |
| App wiring statement (hybrid is the default engine) | `app/js/hybrid_engine.js` or `app/js/cnn_engine.js` imported by `app/js/app.js` | none |

Once the files exist the documents embed them with a generic renderer (first columns of each CSV). If their column names differ from what `fig_conformal` and `fig_openox_skin` in `scripts/make_ml_figs.py` recognise, the figure shows a "no columns recognised" panel and the table is still rendered; adapt the figure function, then re-run the one command above.

## Figure list

| File in `docs/figures/` | Shows | Source |
|---|---|---|
| `ml_architecture.png` | OxyNet and the deployed hybrid | `python/oxydrop_ml/model.py` (parameter count) |
| `ml_flow.png` | input to output, three states | none (diagram) |
| `ml_ssl_curves.png` | SSL loss and retrieval, with and without OpenOximetry | `ssl_loss_curve_full.csv`, `ssl_loss_curve_noox.csv` |
| `ml_ablation.png` | ablation with 95% intervals, P1 and P2 | `ablation_core.csv`, `baselines_P1.csv`, `hybrid_p2.csv` |
| `ml_coverage_accuracy.png` | accuracy versus coverage, share of hypoxic windows alerted: v1 chain, XGBoost, CNN, hybrid | `scores/hybrid_final_hoffman.csv`, `scores/baseline_P1_hoffman.csv`, `scores/cnn_full_hoffman.csv` |
| `ml_conformal.png` | conformal coverage and reliability | PENDING (FEAT-007) |
| `ml_hr_blandaltman.png` | HR Bland-Altman (v2 spectral and hybrid) and error by dataset | `scores/hybrid_final_butppg.csv`, `hybrid_p2.csv` |
| `ml_p1_per_person.png` | P1 per held-out person | `hybrid_p1_final.csv` |
| `ml_openox_skin.png` | OpenOximetry SpO2 error by skin-tone group | PENDING (FEAT-007) |
| `ml_gap.png` | train-versus-test gap | `baselines_P1.csv`, `ablation_core.csv`, `hybrid_p1_final_gap.csv` |
| `ml_failure_modes.png` | failure modes: four synthetic schematics (labelled in the figure) and two real out-of-fold score timelines | `scores/hybrid_final_hoffman.csv`, `scores/cnn_full_hoffman.csv` |

The failure-mode schematics (motion burst, cold finger, far-flash phone, saturated channel) are synthetic signals that illustrate a mechanism; they are not recorded data and say so in the figure. The accuracy-versus-coverage curves rank windows by distance from the decision boundary and tune no cutoff: they are descriptive, and the nested conformal result of FEAT-007 supersedes them once available.
