# OxyDrop photon paper (arXiv LaTeX)

`oxydrop_photon.tex` is the paper. Every number comes from the harness: `make_numbers.py` reads
`results/photon/TABLE.csv`, `WINNER2.md` (shipped-model name) and `NULLS.csv` and writes `numbers.tex` (macros) and
`tables/*.tex`. `make_figures.py` draws `figures/fig_stress.pdf` and `fig_forest.pdf` from the same table.
Nothing in the prose is typed from a result file. Do not edit `numbers.tex` or `tables/`.

## Build

```bash
bash paper/build.sh                          # provisional numbers, DRAFT banner
bash paper/build.sh --winner KEY --locked    # after the planner's lock (KEY = row name in TABLE.csv, e.g. distill2)
```

`build.sh` needs `free -m` available >= 1.2 GB. It runs, in order:

1. `python3 make_numbers.py [--winner KEY] [--locked]` (stdlib only). Without `--winner` the shipped model is parsed from the `SHIPPED ... \`name\`` line of `results/photon/WINNER2.md`. `--locked` removes the DRAFT banner.
2. `~/.venvs/paper/bin/python make_figures.py` (matplotlib; setup: `python3 -m venv ~/.venvs/paper && ~/.venvs/paper/bin/pip install matplotlib`; override with `PAPER_PY`). Without it the existing PDFs in `figures/` are kept.
3. `pdflatex` + `bibtex` if installed, else `tectonic` (`~/.local/bin/tectonic` 0.17.0 is installed on this box; it downloads its TeX bundle on the first run, about 2.5 min), else only the `.tex` bundle.
4. `oxydrop_photon.pdf`, `oxydrop_photon.bbl` and `oxydrop_photon_arxiv.tar.gz` (tex, numbers, tables, figures, refs.bib, .bbl) are written next to the source. Upload the tarball to arXiv. The source uses only pdflatex-compatible standard packages (natbib, booktabs, longtable, adjustbox, tabularx, microtype, lmodern).

Manual build: `cd paper && pdflatex oxydrop_photon && bibtex oxydrop_photon && pdflatex oxydrop_photon && pdflatex oxydrop_photon`.

## Macro names

Macro names cannot contain digits, so levels are spelled out: `\WinLzero`, `\WinLzeroCI`, `\WinLzeroDiff`, `\WinLzeroDiffCI`, `\WinLzeroVerdict`, `\WinLzeroBetter`, `\WinLzeroWorst`, `\WinLzeroCov`, `\WinLzeroAnsAcc`, and likewise `Lone` ... `Lfive`, `Xone`. Prefix `Win` is the shipped model; other prefixes are fixed per model (`Chain`, `DcLr`, `Distill`, `Oxyseq`, `Photonnet`, `Oxyssl`, `DistillTwo`, `Multida`, `SslThree`, `PhotonnetXl`, `EnsThree`, ...; see `MODELS` in `make_numbers.py`). A missing model or level prints `n/a`. Switches for the shipped model: `\ifWinLossLzero`, `\ifWinTieLzero`, `\ifWinWinLzero` (and the same for each level), `\ifWinNinety`, `\ifWinIsDistill`, `\ifWinNullPass`, `\ifDraft`.

## Status (22:10 CT, final)

Built with `bash paper/build.sh --winner distill2 --locked`. The shipped model is PhotonDistill2 (`distill2`, locked 20:22 CT). The text calls it on par with the logistic chain on clean light (paired interval includes 0) and discloses the GPU rerun (`distill2_gpu`, `results/photon/distill2_gpu_log.md`) as seed variance: the rows come from `TABLE.csv` (macro prefix `DistillTwoGpu`). The paper has no accuracy-target claim. `WINNER2.md` still names `distill`, so always pass `--winner distill2`.

No em dashes in prose; `grep -n -- '---' oxydrop_photon.tex` should stay empty.
