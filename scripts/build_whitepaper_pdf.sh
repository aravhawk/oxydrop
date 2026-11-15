#!/usr/bin/env bash
# WHITEPAPER.md -> WHITEPAPER.pdf. No LaTeX needed: pandoc -> self-contained HTML with print CSS -> headless Chromium print-to-PDF (playwright-core).
# Env: PANDOC (default pandoc or /home/linuxbrew/.linuxbrew/bin/pandoc), CHROME, PLAYWRIGHT_CORE, TMPDIR (use the disk, not the RAM /tmp).
set -euo pipefail
cd "$(dirname "$0")/.."
PANDOC=${PANDOC:-$(command -v pandoc || echo /home/linuxbrew/.linuxbrew/bin/pandoc)}
T=${TMPDIR:-/tmp}; mkdir -p "$T"
HTML="$T/whitepaper.$$.html"
# strip the generated-file comment, keep everything else
"$PANDOC" WHITEPAPER.md -f markdown+pipe_tables-yaml_metadata_block -t html5 --standalone --embed-resources --resource-path=. \
  --metadata pagetitle="OxyDrop whitepaper" --css docs/templates/whitepaper.css -o "$HTML"
node scripts/print_pdf.mjs "$HTML" WHITEPAPER.pdf
rm -f "$HTML"
pdfinfo WHITEPAPER.pdf | grep -E "Pages|Page size"
