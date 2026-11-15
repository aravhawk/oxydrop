#!/usr/bin/env bash
# Publish app/ to the gh-pages branch (orphan, force-updated). Run from anywhere inside the repo.
# DRY_RUN=1 bash scripts/deploy_pages.sh   does the same copy and commit in a temp dir, pushes only to a throwaway local
# bare repo (origin is never read or contacted), prints what would be published, and ends with "DRY RUN OK, nothing pushed".
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${DRY_RUN:-}" = "1" ]; then
  # Any failure in this block exits non-zero; nothing in it can reach the real push below.
  dry_tmp=$(mktemp -d); srv_pid=""
  trap '[ -n "$srv_pid" ] && kill "$srv_pid" 2>/dev/null || true; rm -rf "$dry_tmp"' EXIT
  trap 'echo "DRY RUN FAILED, nothing pushed" >&2; exit 1' ERR
  tmp="$dry_tmp/work"; bare="$dry_tmp/bare.git"
  mkdir "$tmp"; cp -r app/. "$tmp"/; touch "$tmp/.nojekyll"
  git init -q --bare "$bare"
  git -C "$tmp" init -q -b gh-pages; git -C "$tmp" add -A; git -C "$tmp" commit -q -m "Deploy app"
  git -C "$tmp" push -q -f "$bare" gh-pages:gh-pages
  files=$(git -C "$bare" ls-tree -r --name-only gh-pages)
  echo "== commit file list (gh-pages in the local bare repo) =="; echo "$files"
  n=$(echo "$files" | wc -l)
  bytes=$(git -C "$bare" ls-tree -r -l gh-pages | awk '{s+=$4} END {print s}')
  echo "== file count: $n; total bytes: $bytes =="
  cache=$(sed -n "s/^const V = '\([^']*\)'.*/\1/p" "$tmp/sw.js")
  echo "== service worker CACHE name: $cache =="
  [ -n "$cache" ] || { echo "no cache name found in sw.js" >&2; exit 1; }
  # every file in the SW precache list must exist in the temp tree (same parse as tests/sw_assets.mjs)
  node -e '
    const fs = require("fs"), path = require("path");
    const src = fs.readFileSync(process.argv[1] + "/sw.js", "utf8");
    const list = JSON.parse(src.match(/const ASSETS = (\[[\s\S]*?\]);/)[1].replace(/\x27/g, "\""));
    const miss = list.filter((a) => a !== "./" && !fs.existsSync(path.join(process.argv[1], a)));
    if (miss.length) { console.log("precache files MISSING from the temp tree: " + miss.join(", ")); process.exit(1); }
    console.log("== precache check: all " + list.length + " files in the SW precache list exist in the temp tree ==");
  ' "$tmp"
  # serve the temp tree on a random localhost port and fetch three files (localhost only)
  port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  python3 -m http.server "$port" --bind 127.0.0.1 --directory "$tmp" >/dev/null 2>&1 & srv_pid=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do curl -s -o /dev/null "http://127.0.0.1:$port/" && break; sleep 0.5; done
  for f in index.html sw.js data/hybrid_R_s0.json; do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/$f")
    echo "== GET /$f -> HTTP $code =="
    [ "$code" = "200" ] || { echo "unexpected status for $f" >&2; exit 1; }
  done
  echo "DRY RUN OK, nothing pushed"
  exit 0
fi

tmp=$(mktemp -d); cp -r app/. "$tmp"/; touch "$tmp/.nojekyll"
git -C "$tmp" init -q -b gh-pages; git -C "$tmp" add -A; git -C "$tmp" commit -q -m "Deploy app"
git -C "$tmp" push -q -f "$(git remote get-url origin)" gh-pages:gh-pages
rm -rf "$tmp"; echo "deployed gh-pages"
