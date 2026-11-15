"""Render README.md, MODEL.md, WHITEPAPER.md and docs/WRITEUP.md from docs/templates/*.md.in using scripts/ml_numbers.py.

    python scripts/render_docs.py            # write the four documents and results/ml/CLAIMS_MANIFEST.csv
    python scripts/render_docs.py --check    # exit 1 if a committed document differs from a fresh render (used by check_claims.py)

Template syntax: {{ns:arg:arg|mod}} scalar tokens and {{@block}} generated tables (see scripts/ml_numbers.py). An unknown token is an error.
"""
from __future__ import annotations

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_numbers as N  # noqa: E402

ROOT = N.ROOT
TEMPLATES = {'README.md.in': 'README.md', 'MODEL.md.in': 'MODEL.md', 'WHITEPAPER.md.in': 'WHITEPAPER.md', 'WRITEUP.md.in': 'docs/WRITEUP.md'}
TOKEN = re.compile(r'\{\{([^{}]+)\}\}')


def render_text(text: str, doc: str, manifest: list) -> str:
    def sub(m):
        tok = m.group(1).strip()
        if tok.startswith('@'):
            val = N.block(tok[1:])
            manifest.append((doc, tok, f'<block {len(val)} chars>'))
        else:
            val = N.resolve(tok)
            manifest.append((doc, tok, val))
        return val
    return TOKEN.sub(sub, text)


def render_all() -> tuple[dict, list]:
    out, manifest = {}, []
    for tpl, dst in TEMPLATES.items():
        path = os.path.join(ROOT, 'docs', 'templates', tpl)
        if not os.path.exists(path):
            continue
        out[dst] = render_text(open(path, encoding='utf-8').read(), dst, manifest)
    return out, manifest


def main(argv):
    out, manifest = render_all()
    if '--check' in argv:
        bad = [dst for dst, txt in out.items() if not os.path.exists(os.path.join(ROOT, dst)) or open(os.path.join(ROOT, dst), encoding='utf-8').read() != txt]
        if bad:
            print('STALE (re-run python scripts/render_docs.py):', ', '.join(bad))
            return 1
        print('rendered documents are up to date:', ', '.join(out))
        return 0
    for dst, txt in out.items():
        with open(os.path.join(ROOT, dst), 'w', encoding='utf-8') as fh:
            fh.write(txt)
        print('wrote', dst, f'({len(txt.split())} words)')
    with open(os.path.join(N.RES, 'CLAIMS_MANIFEST.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['document', 'token', 'rendered_value'])
        w.writerows(manifest)
    print('wrote results/ml/CLAIMS_MANIFEST.csv', len(manifest), 'tokens;', 'pending:', N.pending_report())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
