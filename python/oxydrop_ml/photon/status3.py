"""Round-3 status report: results/photon/STATUS_<HHMM>.md (honest, verified numbers; every number comes from the score files).

    cd python && python -m oxydrop_ml.photon.status3 --tag 2000 [--boot 2000] [--candidates distill2,multida,ssl3,photonnet_xl,ens3] [--note "free text"]

Contents: every trained model with an L0 file (rounds 2 and 3), L0 TM [CI], paired vs chain_v1 (0.796 [0.636, 0.857]) and vs distill, people better, null mean over
distinct draws, stress paired differences (L3, L4, L5, X1), the winner3 verdict, the plain statement whether anything beats the chain, then per new model: spot checks of
two score files (raw row count 5,644, no NaN in score/pred, one sid per recording and identical to windows.csv, all six people present with non-constant scores, no
duplicate (rec, t), not a byte copy of another model's file) and a leakage probe (the model's own logs / code must contain a held-out exclusion assertion; grep result shown).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import evalkit as E
from . import winner2 as W
from . import winner3 as W3

LEAK_RE = re.compile(r'assert_no_leak|held[- ]?out[^\n]{0,60}(never|excluded|not (seen|used|in))|(excluded|never)[^\n]{0,60}held[- ]?out|loso[^\n]{0,40}assert|assert[^\n]{0,60}(held|test|sid|person)', re.I)
RAW_ROWS = 5644


def spot_check(path, ref, others):
    """Checks of one score file. Returns (list of (check, ok, detail))."""
    res = []
    d = pd.read_csv(path)
    n = len(d)
    res.append(('row count', n == RAW_ROWS, f'{n} rows (expected {RAW_ROWS})'))
    nan_s, nan_p = int(d['score'].isna().sum()), int(d['pred'].isna().sum())
    inf_s = int(np.isinf(pd.to_numeric(d['score'], errors='coerce')).sum())
    res.append(('no NaN/inf in score, pred', nan_s == 0 and nan_p == 0 and inf_s == 0, f'NaN score {nan_s}, NaN pred {nan_p}, inf score {inf_s}'))
    d['rec'] = E._norm_rec(d['rec'])
    dup = int(d.duplicated(['rec', 't']).sum())
    res.append(('no duplicate (rec, t)', dup == 0, f'{dup} duplicates'))
    m = ref.merge(d[['rec', 't', 'sid', 'score']].rename(columns={'sid': 'sid_file'}), on=['rec', 't'], how='left')
    miss = int(m['score'].isna().sum())
    res.append(('all evaluated windows present', miss == 0, f'{miss} of {len(ref)} evaluated windows missing'))
    sid_bad = int((m['sid_file'].astype(str) != m['sid'].astype(str)).sum()) if 'sid_file' in m else -1
    res.append(('sid matches windows.csv (person labels for OOF grouping)', sid_bad == 0, f'{sid_bad} windows with a different sid'))
    per = m.groupby('sid')['score'].agg(['count', 'std'])
    flat = [s for s, r in per.iterrows() if not (r['std'] > 1e-9)]
    res.append(('six people, non-constant score within each', len(per) == 6 and not flat, f'{len(per)} people; constant-score people: {flat or "none"}'))
    rng = (float(d['score'].min()), float(d['score'].max()))
    res.append(('score range', True, f'[{rng[0]:.3f}, {rng[1]:.3f}]'))
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    same = [o for o, oh in others.items() if oh == h and o != Path(path).name]
    res.append(('not a byte copy of another file', not same, 'copies: ' + ', '.join(same) if same else 'unique'))
    return res


def ensemble_check(model, seeds, scores_dir):
    """Is the primary L0 score the mean of the seed files' scores? Returns (ok, detail). Informational: ensembling rules differ (mean prob, mean logit)."""
    if not seeds:
        return False, 'no seed files with an L0 file found'
    def rd(n):
        d = pd.read_csv(Path(scores_dir) / f'{n}__L0.csv')
        d['rec'] = E._norm_rec(d['rec'])
        return d.drop_duplicates(['rec', 't']).set_index(['rec', 't'])
    prim = rd(model)
    S = pd.concat([rd(n)['score'].rename(n) for n in seeds], axis=1).reindex(prim.index)
    mean = S.mean(axis=1)
    dif = (prim['score'] - mean).abs()
    agree = float(((prim['score'] > 0.5) == (mean > 0.5)).mean())
    return bool(dif.max() < 1e-3), f'{len(seeds)} seed file(s); max |primary - mean(seeds)| = {dif.max():.4f}, median {dif.median():.4f}; side of 0.5 agrees on {agree:.3f} of windows'


def leak_probe(model, repo):
    """Grep the model's own log / code for a held-out exclusion assertion."""
    RES = repo / 'results' / 'photon'
    pyd = repo / 'python' / 'oxydrop_ml' / 'photon'
    files = sorted({*RES.glob(f'{model}*log*.md'), *RES.glob(f'{model}*.log'), *RES.glob(f'{model}*run*'), *pyd.glob(f'{model}*.py'), *RES.glob(f'{model}_*')} )
    files = [f for f in files if (f.name.startswith(model + '_') or f.name.startswith(model + '.')) and f.is_file() and f.suffix in ('.md', '.log', '.py', '.txt', '.json') and f.stat().st_size < 3_000_000]
    hits = []
    for f in files:
        try:
            for i, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
                if LEAK_RE.search(line):
                    hits.append((f.relative_to(repo), i, line.strip()[:160]))
        except Exception:
            pass
    return files, hits


def fmt_ci(r):
    return 'n/a' if r is None else f'{r["tm"]:.3f} [{r["lo"]:.3f}, {r["hi"]:.3f}]'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='HHMM of the report, e.g. 2000')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--candidates', default=','.join(W3.ROUND3))
    ap.add_argument('--shippable', default='')
    ap.add_argument('--note', default='')
    a = ap.parse_args(argv)
    repo = E.REPO
    cands = [x for x in a.candidates.split(',') if x]
    names = list(dict.fromkeys(W3.ROUND2 + cands))
    models = sorted(W.discover(E.SCORES))
    ev = W.Evaluator(E.SCORES, a.boot)
    import datetime as dt
    today = dt.datetime.now()
    rows = W3.evaluate(ev, models, names, set(x for x in a.shippable.split(',') if x), today, True)
    inc = next(c for c in rows if c['model'] == W3.INCUMBENT)
    winner, vlines = W3.verdict(rows, inc, False)
    stamp = subprocess.run(['date', '+%a %b %d %H:%M:%S %Z %Y'], capture_output=True, text=True).stdout.strip()
    chain = ev.get(W3.REF, 0)
    inc_r = ev.get(W3.INCUMBENT, 0)
    out = [f'# Round 3 status {a.tag} CT', '', f'Generated {stamp} by `python -m oxydrop_ml.photon.status3` (R3-EVAL). Every number is recomputed from `results/photon/scores/*.csv` with the evalkit bootstrap '
           f'(2000 person resamples, same people for every model). Reference: chain_v1 L0 TM {fmt_ci(chain)}; distill L0 TM {fmt_ci(inc_r)}.', '']
    # plain statement
    out += ['## Plain statement', ''] + [f'* {l}' for l in W3.beats_chain_lines(rows)]
    rd3 = [c for c in rows if c['round'] == 3 and c['r0'] is not None]
    out += [f'* Round-3 candidates with an L0 file at this moment: {", ".join(c["model"] for c in rd3) or "none"}.',
            f'* Replacement rule (ROUND3.md): {[l for l in vlines if l.startswith("**")][0] if any(l.startswith("**") for l in vlines) else "see verdict"}', '']
    if a.note:
        out += [a.note, '']
    # main table
    t = []
    for c in rows:
        if c['r0'] is None:
            t.append([c['model'], c['round'], 'no L0 file yet', '', '', '', '', ', '.join(W.lname(l) for l in c['have']) or 'none'])
            continue
        vd = c['vd']
        t.append([c['model'], c['round'], fmt_ci(c['r0']), f'{c["chain_diff"]:+.3f} [{c["clo"]:+.3f}, {c["chi"]:+.3f}] ({c["nb"]}/{c["nc"]})',
                  'incumbent' if vd is None else f'{vd["diff"]:+.3f} [{vd["lo"]:+.3f}, {vd["hi"]:+.3f}] ({vd["nb"]}/{vd["nc"]})',
                  (f'{c["nulls"]["ns"]["null"]["mean"]:.3f} over {c["nulls"]["ns"]["null"]["n"]}' if 'null' in c['nulls']['ns'] else 'none')
                  + (f', blank {c["nulls"]["ns"]["blank"]["mean"]:.3f}' if 'blank' in c['nulls']['ns'] else '') + ('' if c['nulls']['ok'] else ' **FAILS**'),
                  f'{W.f3(c["worst"])} (chain {W.f3(chain["worst"])})', ', '.join(W.lname(l) for l in c['have'])])
    out += ['## L0 (clean light), 3-seed ensemble (primary file); people better = of the people with a usable comparison', '',
            W.table(['model', 'round', 'L0 TM [95 % CI]', 'paired vs chain_v1 (people better)', 'paired vs distill (people better)', 'null mean over distinct draws (<= 0.55)', 'worst person', 'levels with files'], t), '']
    st = []
    for c in rows:
        if c['r0'] is None:
            continue
        cells = []
        for l in W3.STRESS_LEVELS:
            q = c['stress'].get(W.lname(l))
            cells.append('n/a' if q is None else f'{q["diff"]:+.3f} [{q["lo"]:+.3f}, {q["hi"]:+.3f}] {q["nb"]}/{q["nc"]}')
        st.append([c['model']] + cells + [W.f3(c['g3_mean']) if np.isfinite(c['g3_mean']) else 'incomplete'])
    out += ['## Stress: paired TM difference to chain_v1 at the same level [95 % CI] people better (simulated distant flash L3-L5, out-of-family X1)', '',
            W.table(['model', 'L3', 'L4', 'L5', 'X1', 'mean over the four'], st), '']
    out += ['## ROUND3 winner script verdict', ''] + [f'* {l}' for l in vlines] + ['']
    # spot checks and leakage for round-3 models with files
    hashes = {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(E.SCORES.glob('*.csv'))}
    problems = []
    out += ['## Spot checks (two score files per new model) and leakage probe', '']
    new = [c for c in rows if c['round'] == 3 and c['have']]
    if not new:
        out += ['No round-3 score file exists yet; nothing to check.', '']
    for c in new:
        m = c['model']
        out += [f'### {m}', '']
        have = c['have']
        pick = [0] if 0 in have else []
        rest = [l for l in have if l != 0]
        if rest:
            pick.append(rest[-1])
        for l in pick[:2]:
            p = E.SCORES / f'{m}__{W.lname(l)}.csv'
            try:
                chk = spot_check(p, ev.ref, hashes)
            except Exception as e:
                chk = [('readable', False, f'{type(e).__name__}: {e}')]
            out += [f'`{p.name}`:', ''] + [f'* {"OK" if ok else "**PROBLEM**"} {name}: {det}' for name, ok, det in chk] + ['']
            problems += [f'{m}: {p.name} {name}: {det}' for name, ok, det in chk if not ok]
        ok_e, det_e = ensemble_check(m, c['seeds'], E.SCORES)
        out += [f'Seed-ensemble check of `{m}__L0.csv`: {"OK" if ok_e else "**NOT an exact mean of the seed scores**"}: {det_e}', '']
        if not ok_e and c['seeds']:
            problems.append(f'{m}: primary L0 score is not the plain mean of its seed files ({det_e}); may use another ensembling rule, ask its author')
        files, hits = leak_probe(m, repo)
        out += [f'Leakage probe (grep of {len(files)} own log/code/result files): ' + (f'{len(hits)} matching line(s)' if hits else '**no held-out exclusion assertion found**')]
        for f, i, line in hits[:5]:
            out.append(f'* `{f}:{i}`: {line}')
        out.append('')
        if not hits:
            problems.append(f'{m}: no held-out exclusion assertion found in its logs/code ({len(files)} files searched)')
        if c['r0'] is not None and not c['nulls']['ok']:
            problems.append(f'{m}: nulls {c["nulls"]["text"]}')
        if c['late']:
            problems.append(f'{m}: late files {", ".join(c["late"])}')
    out += ['## Problems found', ''] + ([f'* {p}' for p in problems] if problems else ['None found by the automated checks.']) + ['']
    if ev.notes:
        out += ['## Skipped files', ''] + [f'* {n}' for n in ev.notes] + ['']
    out += ['Notes: the stress levels are simulations; every round-3 model was designed after rounds 1-2 outer results were seen (Amendment 8), so L0 numbers are optimistic. Nothing here is a medical-device claim.']
    path = E.RES / f'STATUS_{a.tag}.md'
    path.write_text('\n'.join(out) + '\n')
    print(f'wrote {path}')
    print('\n'.join(['PLAIN: ' + l for l in W3.beats_chain_lines(rows)] + ['VERDICT: ' + l for l in vlines] + ['PROBLEMS: ' + p for p in problems]))


if __name__ == '__main__':
    main()
