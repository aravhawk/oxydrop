# Licence notice for the model weights in this folder

Applies to `hybrid_R_s0.json`, `hybrid_R_s1.json`, `hybrid_R_s2.json` and `oxynet.onnx`. These files are **not** under the MIT licence that covers the code.

**Non-commercial, research and education use only.** No commercial use, no clinical or diagnostic use. OxyDrop is a research prototype and not a medical device.

**Attribution required.** The weights were trained on data from:

* **MTHS** (Samavati et al., arXiv:2204.08989), licence CC BY-NC-ND 4.0
* **OpenOximetry** (PhysioNet, credentialed dataset, data use agreement)
* Hoffman et al. 2022 (MIT), BUT PPG v2.0 (CC BY 4.0) and PhysioNet pretraining sources

Credit MTHS and OpenOximetry whenever you use or redistribute these files. The weights contain no raw signal and no per-person record. Do not redistribute the MTHS or OpenOximetry data.

The project owner approved shipping these weights on 2026-11-15; this is not a legal opinion and the data owners have not been asked to confirm it. Full details: `DATA_LICENSES.md` at the repository root. `weights.json` (the logistic fallback chain) is fit on Hoffman data (MIT) only.
