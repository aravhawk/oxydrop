#!/usr/bin/env bash
# Build paper/oxydrop_photon.pdf and an arXiv source bundle.
#   bash paper/build.sh                          # provisional numbers (DRAFT banner)
#   bash paper/build.sh --winner KEY --locked    # after the planner's lock
# Needs free memory >= 1.2 GB (one process at a time). Figures need matplotlib (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
avail=$(free -m | awk '/Mem:/ {print $7}')
if [ "${avail:-0}" -lt 1200 ]; then echo "available memory ${avail} MB < 1200 MB, aborting" >&2; exit 1; fi
python3 make_numbers.py "$@"
PY=${PAPER_PY:-$HOME/.venvs/paper/bin/python}
if [ -x "$PY" ]; then "$PY" make_figures.py "$@" || true; else echo "no matplotlib venv at $PY: keeping existing figures/*.pdf"; fi
mkdir -p build
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -output-directory build oxydrop_photon.tex
  (cd build && cp ../refs.bib . && bibtex oxydrop_photon) ; 
  pdflatex -interaction=nonstopmode -output-directory build oxydrop_photon.tex
  pdflatex -interaction=nonstopmode -output-directory build oxydrop_photon.tex
elif command -v tectonic >/dev/null 2>&1 || [ -x "$HOME/.local/bin/tectonic" ]; then
  PATH="$HOME/.local/bin:$PATH" tectonic --keep-logs --keep-intermediates --outdir build oxydrop_photon.tex
else
  echo "neither pdflatex nor tectonic found: only the .tex bundle is produced" >&2
fi
[ -f build/oxydrop_photon.bbl ] && cp build/oxydrop_photon.bbl oxydrop_photon.bbl
[ -f build/oxydrop_photon.pdf ] && cp build/oxydrop_photon.pdf oxydrop_photon.pdf
tar czf oxydrop_photon_arxiv.tar.gz oxydrop_photon.tex numbers.tex shipped_model.tex refs.bib oxydrop_photon.bbl tables figures
echo "built: $(ls -la oxydrop_photon.pdf 2>/dev/null | awk '{print $5}') bytes pdf; bundle oxydrop_photon_arxiv.tar.gz"
