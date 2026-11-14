#!/usr/bin/env bash
# Publish app/ to the gh-pages branch (orphan, force-updated). Run from anywhere inside the repo.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tmp=$(mktemp -d); cp -r app/. "$tmp"/; touch "$tmp/.nojekyll"
git -C "$tmp" init -q -b gh-pages; git -C "$tmp" add -A; git -C "$tmp" commit -q -m "Deploy app"
git -C "$tmp" push -q -f "$(git remote get-url origin)" gh-pages:gh-pages
rm -rf "$tmp"; echo "deployed gh-pages"
