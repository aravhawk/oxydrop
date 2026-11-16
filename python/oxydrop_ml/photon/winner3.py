"""Round-3 evaluation: results/photon/WINNER3.md (the mechanical winner rule of ROUND3.md / Amendment 8).

    cd python && python -m oxydrop_ml.photon.winner3 [--boot 2000] [--candidates distill2,multida,ssl3,photonnet_xl,ens3] [--scores DIR] [--out DIR]
                                                     [--no-write] [--ignore-deadline]

Written BEFORE any round-3 score file existed (see Amendment 8). It reads results/photon/scores/*.csv through the evalkit / winner2 machinery (same person
bootstrap, same closed-form TM, same 5,644 windows), so every number equals the TABLE.csv value. Missing or partial files are skipped with a note. No number is typed
by hand.

Rule (ROUND3.md, "Winner rule for round 3", stated 19:55 CT before any round-3 score), verbatim:
  Eligible: trained models from rounds 2 and 3 that pass nulls and rule 3. Replace `distill` only if a candidate's L0 3-seed ensemble TM point estimate exceeds
  distill's (0.758) AND its paired difference vs distill has a lower CI bound > -0.02 AND its stress mean paired diff vs chain_v1 over L3-L5 and X1 is >= 0.
  "Beats chain" still requires the paired CI vs chain_v1 excluding 0, >= 5/6 people better, worst person not below chain's.

Operational reading (no discretion):
  * "distill's 0.758" is the L0 TM of the primary file `distill__L0.csv` as computed now; the 0.758 is only printed for comparison.
  * g1  candidate L0 TM point estimate > distill L0 TM point estimate.
  * g2  lower bound of the 95 % person-bootstrap CI of (candidate TM - distill TM) at L0 (same resampled people) > -0.02.
  * g3  mean over L3, L4, L5, X1 of the paired difference to chain_v1 at the same level >= 0. All four stress files are required; with a missing one g3 is
        "incomplete" and the candidate cannot replace distill (a note says which file is missing).
  * nulls  the model's null files `<m>_null*__L0.csv` (also `_perm`, `_gperm`, `_d<k>`, `_s<k>`, ...; byte-identical files, by SHA-1, are counted ONCE) must number
        >= 3 distinct draws and have mean TM <= 0.55; temporal models (TEMPORAL) additionally need a blank control `<m>_blank*__L0.csv` with mean <= 0.55.
  * rule 3 (shippable through existing browser code) is not computable from scores: it is a manual column (`--shippable m1,m2` marks models whose author states a
        compatible front end + nn_runtime-v1 / OxyNet network and export). A model not listed is reported as "numeric pass, rule 3 unconfirmed" and is NOT announced
        as a replacement.
  * timeline (ROUND3.md rule 4): the L0 file and its null files are late if their mtime is after 20:15 CT, a stress file after 20:30 CT. A late candidate is reported
        and cannot replace the shipped model (`--ignore-deadline` turns this check off). mtime is the time the file arrived on this box.
  * Several replacements pass: the highest L0 point estimate wins.
  * Eligible set = rounds 2 and 3: photonnet, distill, oxyssl, oxyseq (round 2) and the round-3 candidates; ablation, seed and null files are never eligible.
  * "Beats chain" for each model as in winner2 (a: paired CI above 0, b: >= 5/6 people better, c: worst person not below chain's, d: nulls passed with >= 3 draws).
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

import numpy as np

from . import evalkit as E
from . import winner2 as W

REF = E.BASE
INCUMBENT = 'distill'
ROUND2 = ['photonnet', 'distill', 'oxyssl', 'oxyseq']
ROUND3 = ['distill2', 'multida', 'ssl3', 'photonnet_xl', 'ens3']
TEMPORAL = {'oxyseq'}
STRESS_LEVELS = [3, 4, 5, 'X1']
MIN_NULL_DRAWS = 3
G2_MIN = -0.02
DEADLINES = {'L0': (20, 15), 'null': (20, 15), 'stress': (20, 30)}       # CT, ROUND3.md rule 4
ALL_LEVELS = E.ALL_LEVELS

ROUND3_RULE = (
    'Eligible: trained models from rounds 2 and 3 that pass nulls and rule 3. Replace `distill` only if a candidate\'s L0 3-seed ensemble TM point estimate exceeds '
    'distill\'s (0.758) AND its paired difference vs distill has a lower CI bound > -0.02 AND its stress mean paired diff vs chain_v1 over L3-L5 and X1 is >= 0. '
    '"Beats chain" still requires the paired CI vs chain_v1 excluding 0, >= 5/6 people better, worst person not below chain\'s. '
    'All round-3 models are designed after rounds 1-2 outer results were seen (Amendment 8).')


def classify(name, primaries):
    """Like winner2.classify but the LONGEST primary wins (photonnet_xl_s0 belongs to photonnet_xl, not to photonnet)."""
    m = W.NULL_RE.match(name)
    if m:
        base = m.group('base')
        while W.BASE_TOKEN_RE.search(base):
            base = W.BASE_TOKEN_RE.sub('', base)
        return m.group('kind'), (base if base in primaries else None)
    if name in primaries:
        return 'primary', name
    for p in sorted(primaries, key=len, reverse=True):
        if name.startswith(p + '_'):
            return ('seed' if W.SEED_RE.search(name) and W.SEED_RE.sub('', name) == p else 'ablation'), p
    return 'other', None


def _ct(path):
    t = dt.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return t


def late(path, key, today):
    h, mi = DEADLINES[key]
    return _ct(path) > dt.datetime(today.year, today.month, today.day, h, mi)


def null_info(ev, models, m, today, check_deadline):
    """Distinct-draw null summary (winner2.null_summary dedups byte-identical files) + the >= 3 draws requirement + timeline."""
    files = [n for n in models if classify(n, [m])[1] == m]
    ns = W.null_summary(ev, files, m)
    need = ['null'] + (['blank'] if m in TEMPORAL else [])
    problems = []
    for k in need:
        if k not in ns:
            problems.append(f'missing {k} control')
        elif k == 'null' and ns[k]['n'] < MIN_NULL_DRAWS:
            problems.append(f'only {ns[k]["n"]} distinct null draw(s), need >= {MIN_NULL_DRAWS}')
    for k in ns:
        if ns[k]['mean'] > E.NULL_MAX:
            problems.append(f'{k} mean {ns[k]["mean"]:.3f} > {E.NULL_MAX}')
    late_files = []
    if check_deadline:
        for k in ns:
            for n, _ in ns[k]['tms']:
                if late(ev.dir / f'{n}__L0.csv', 'null', today):
                    late_files.append(n)
    ok = not problems
    text = ('passed: ' if ok else 'NOT PASSED (' + '; '.join(problems) + '): ') + ', '.join(f'{k} mean {ns[k]["mean"]:.3f} over {ns[k]["n"]} distinct draw(s)' for k in ns) if ns else \
        'NOT PASSED (' + '; '.join(problems) + ')'
    return dict(ns=ns, ok=ok, text=text, late=late_files)


def paired_vs(a, b):
    """Paired person-bootstrap difference a - b at the same level (same resampled people)."""
    lo, hi = E.ci(a['tm_b'] - b['tm_b'])
    x, y = a['per'], b['per']
    ok = np.isfinite(x) & np.isfinite(y) & (x != y)
    return dict(diff=a['tm'] - b['tm'], lo=lo, hi=hi, nb=int((x[ok] > y[ok]).sum()), nc=int(ok.sum()))


def evaluate(ev, models, names, shippable, today, check_deadline):
    chain0 = ev.get(REF, 0)
    inc0 = ev.get(INCUMBENT, 0)
    rows = []
    for m in names:
        c = dict(model=m, round=3 if m in ROUND3 else 2)
        c['have'] = [l for l in ALL_LEVELS if ev.get(m, l) is not None]
        c['missing'] = [W.lname(l) for l in ALL_LEVELS if l not in c['have']]
        c['seeds'] = sorted(n for n in models if classify(n, [m])[:2] == ('seed', m) and ev.get(n, 0) is not None)
        r0 = ev.get(m, 0)
        c['r0'] = r0
        c['nulls'] = null_info(ev, models, m, today, check_deadline)
        c['shippable'] = m in shippable
        c['late'] = list(c['nulls']['late'])
        if r0 is None:
            c['status'] = 'no L0 score file yet' if not c['have'] else 'no usable L0 file'
            rows.append(c)
            continue
        pc = ev.paired(r0, 0)
        c.update(tm=r0['tm'], lo=r0['lo'], hi=r0['hi'], worst=r0['worst'], chain_diff=pc['diff'], clo=pc['lo'], chi=pc['hi'], nb=pc['nb'], nc=pc['nc'])
        c['beats_a'] = bool(pc['lo'] > 0)
        c['beats_b'] = bool(pc['nb'] >= W.MIN_PEOPLE_BETTER)
        c['beats_c'] = bool(chain0 is not None and np.isfinite(r0['worst']) and r0['worst'] >= chain0['worst'])
        c['beats_d'] = c['nulls']['ok']
        c['beats'] = c['beats_a'] and c['beats_b'] and c['beats_c'] and c['beats_d']
        # g1 g2 vs the incumbent
        if m == INCUMBENT or inc0 is None:
            c.update(g1=None, g2=None, vd=None)
        else:
            vd = paired_vs(r0, inc0)
            c['vd'] = vd
            c['g1'] = bool(r0['tm'] > inc0['tm'])
            c['g2'] = bool(vd['lo'] > G2_MIN)
        # g3: stress
        diffs, miss = [], []
        c['stress'] = {}
        for l in STRESS_LEVELS:
            r = ev.get(m, l)
            if r is None:
                miss.append(W.lname(l))
                continue
            q = ev.paired(r, l)
            c['stress'][W.lname(l)] = q
            diffs.append(q['diff'])
        c['stress_missing'] = miss
        c['g3_mean'] = float(np.mean(diffs)) if len(diffs) == len(STRESS_LEVELS) else np.nan
        c['g3'] = bool(c['g3_mean'] >= 0) if not miss else None          # None = incomplete
        # timeline
        late_files = list(c['nulls']['late'])
        if check_deadline:
            if late(ev.dir / f'{m}__L0.csv', 'L0', today):
                late_files.append(f'{m}__L0')
            for l in STRESS_LEVELS:
                if ev.get(m, l) is not None and late(ev.dir / f'{m}__{W.lname(l)}.csv', 'stress', today):
                    late_files.append(f'{m}__{W.lname(l)}')
        c['late'] = late_files
        rows.append(c)
    return rows


def verdict(rows, inc, shippable_known):
    """-> (replacement model or None, lines)."""
    lines = []
    if inc is None or inc['r0'] is None:
        return None, ['`distill__L0.csv` is missing or unusable: no comparison is possible.']
    inc_ok = inc['nulls']['ok']
    lines.append(f'Incumbent `distill`: L0 TM {inc["tm"]:.3f} [{inc["lo"]:.3f}, {inc["hi"]:.3f}]; nulls {inc["nulls"]["text"]}.')
    if not inc_ok:
        lines.append('NOTE: the incumbent itself does not satisfy the null requirement of this script (>= 3 distinct draws, mean <= 0.55); reported, rule applied as written.')
    cands = [c for c in rows if c['model'] != INCUMBENT]
    passers = []
    for c in cands:
        if c['r0'] is None:
            lines.append(f'`{c["model"]}`: {c["status"]}.')
            continue
        why = []
        if not c['nulls']['ok']:
            why.append('nulls: ' + c['nulls']['text'])
        if c['g1'] is False:
            why.append(f'g1 fails: L0 {c["tm"]:.3f} does not exceed distill {inc["tm"]:.3f}')
        if c['g2'] is False:
            why.append(f'g2 fails: paired CI lower bound vs distill {c["vd"]["lo"]:+.3f} <= {G2_MIN}')
        if c['g3'] is None:
            why.append('g3 incomplete: no ' + ', '.join(c['stress_missing']) + ' file yet')
        elif c['g3'] is False:
            why.append(f'g3 fails: mean stress paired diff vs chain_v1 {c["g3_mean"]:+.3f} < 0')
        if not why:
            passers.append(c)
            extra = []
            if c['late']:
                extra.append('LATE (cannot replace the shipped model): ' + ', '.join(c['late']))
            if not c['shippable']:
                extra.append('rule 3 (shippable in browser) unconfirmed')
            lines.append(f'`{c["model"]}` passes the numeric rule (g1, g2, g3, nulls): L0 {c["tm"]:.3f}, vs distill {c["vd"]["diff"]:+.3f} [{c["vd"]["lo"]:+.3f}, {c["vd"]["hi"]:+.3f}], '
                         f'stress mean vs chain {c["g3_mean"]:+.3f}.' + (' ' + '; '.join(extra) + '.' if extra else ''))
        else:
            lines.append(f'`{c["model"]}` does NOT replace distill: ' + '; '.join(why) + '.')
    valid = [c for c in passers if not c['late'] and c['shippable']]
    if valid:
        valid.sort(key=lambda c: -c['tm'])
        w = valid[0]
        lines.append(f'**REPLACE `distill` with `{w["model"]}`** (highest L0 point estimate among {len(valid)} candidate(s) passing every condition).')
        return w['model'], lines
    numeric = [c for c in passers if not c['late']]
    if numeric:
        lines.append('**NO REPLACEMENT YET**: numeric rule passed by ' + ', '.join(c['model'] for c in numeric) + ' but rule 3 (browser-shippable) is not confirmed; confirm and rerun with `--shippable`.')
    elif passers:
        lines.append('**NO REPLACEMENT**: every candidate passing the numeric rule is late (ROUND3.md rule 4).')
    else:
        lines.append('**NO REPLACEMENT**: `distill` stays the shipped model.')
    return None, lines


def beats_chain_lines(rows):
    b = [c for c in rows if c['r0'] is not None and c['beats']]
    if b:
        return [f'`{c["model"]}` BEATS chain_v1 at L0: {c["tm"]:.3f}, paired {c["chain_diff"]:+.3f} [{c["clo"]:+.3f}, {c["chi"]:+.3f}], {c["nb"]}/{c["nc"]} people better, nulls passed.' for c in b]
    return ['No model beats chain_v1 (all of: paired CI above 0, >= 5/6 people better, worst person not below chain_v1, nulls passed with >= 3 distinct draws).']


def md(ev, rows, lines, winner, stamp, notes):
    cell = W.cell
    out = ['# Round 3 winner (mechanical application of ROUND3.md, Amendment 8)', '',
           f'Written by `python -m oxydrop_ml.photon.winner3` ({stamp}) from the score files that existed at that moment. Rule, verbatim from `.agents/tasks/task-oxydrop-photon-model/ROUND3.md` '
           '(stated 19:55 CT, before any round-3 score):', '', '> ' + ROUND3_RULE, '', '## Verdict', '']
    out += [f'* {l}' for l in lines] + ['']
    out += ['## Beats chain_v1?', ''] + [f'* {l}' for l in beats_chain_lines(rows)] + ['']
    c0 = ev.get(REF, 0)
    if c0:
        out += [f'Reference `{REF}`: L0 TM {cell(c0)}, worst person {W.f3(c0["worst"])}.', '']
    t = []
    for c in rows:
        if c['r0'] is None:
            t.append([c['model'], c['round'], c['status'], '', '', '', '', '', '', c['nulls']['text'] if c['nulls']['ns'] else 'no null files yet', ', '.join(c['missing'])])
            continue
        vd = c['vd']
        g = lambda v: 'n/a' if v is None else ('yes' if v else 'no')
        t.append([c['model'], c['round'], cell(c['r0']),
                  'incumbent' if vd is None else f'{vd["diff"]:+.3f} [{vd["lo"]:+.3f}, {vd["hi"]:+.3f}]',
                  g(c['g1']), g(c['g2']), ('incomplete' if c['g3'] is None else g(c['g3'])) + ('' if not np.isfinite(c['g3_mean']) else f' ({c["g3_mean"]:+.3f})'),
                  f'{c["chain_diff"]:+.3f} [{c["clo"]:+.3f}, {c["chi"]:+.3f}] {c["nb"]}/{c["nc"]}', 'yes' if c['beats'] else 'no',
                  c['nulls']['text'], ', '.join(c['missing']) or 'none'])
    out += ['## Candidates at L0 (3-seed ensemble = primary file)', '',
            W.table(['model', 'round', 'L0 TM [95 % CI]', 'paired vs distill', 'g1 TM > distill', 'g2 CI lo > -0.02', 'g3 stress mean vs chain >= 0', 'paired vs chain_v1 (people better)', 'beats chain', 'nulls (>= 3 distinct draws, mean <= 0.55)', 'missing levels'], t), '']
    st = []
    for c in rows:
        if c['r0'] is None:
            continue
        st.append([c['model']] + [('n/a' if W.lname(l) not in c['stress'] else f'{c["stress"][W.lname(l)]["diff"]:+.3f} [{c["stress"][W.lname(l)]["lo"]:+.3f}, {c["stress"][W.lname(l)]["hi"]:+.3f}] {c["stress"][W.lname(l)]["nb"]}/{c["stress"][W.lname(l)]["nc"]}') for l in STRESS_LEVELS]
                  + [W.f3(c['g3_mean'])])
    out += ['## Stress: paired difference to chain_v1 [95 % CI] people better', '', W.table(['model', 'L3', 'L4', 'L5', 'X1', 'mean (g3)'], st) if st else 'No model has an L0 file.', '']
    out += ['## Seeds, timeline, rule 3', '']
    rr = [[c['model'], len(c['seeds']), ', '.join(c['seeds']) or 'none', ', '.join(c['late']) or 'none', 'confirmed' if c['shippable'] else 'unconfirmed (manual)'] for c in rows]
    out += [W.table(['model', 'seed files with L0', 'seed files', 'late files (after 20:15 L0/nulls, 20:30 stress)', 'rule 3'], rr), '']
    if notes:
        out += ['## Skipped files', ''] + [f'* {n}' for n in notes] + ['']
    out += ['The 3-seed ensemble composition of a primary file cannot be verified from the score file; the seed files found are listed above. Nothing here is a medical-device claim.']
    return '\n'.join(out) + '\n'


def run(scores_dir=E.SCORES, out_dir=E.RES, B=2000, candidates=None, shippable=(), write=True, log=print, check_deadline=True):
    names = list(dict.fromkeys(ROUND2 + list(candidates or ROUND3)))
    models = sorted(W.discover(scores_dir))
    ev = W.Evaluator(scores_dir, B)
    today = dt.datetime.now()
    rows = evaluate(ev, models, names, set(shippable), today, check_deadline)
    inc = next((c for c in rows if c['model'] == INCUMBENT), None)
    winner, lines = verdict(rows, inc, bool(shippable))
    stamp = subprocess.run(['date', '+%a %b %d %H:%M:%S %Z %Y'], capture_output=True, text=True).stdout.strip()
    text = md(ev, rows, lines, winner, stamp, ev.notes)
    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'WINNER3.md').write_text(text)
    for n in ev.notes:
        log(n)
    for l in lines:
        log('VERDICT: ' + l)
    for l in beats_chain_lines(rows):
        log('CHAIN: ' + l)
    return winner, rows, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--scores', default=str(E.SCORES))
    ap.add_argument('--out', default=str(E.RES))
    ap.add_argument('--candidates', default=','.join(ROUND3), help='comma-separated round-3 primary model names')
    ap.add_argument('--shippable', default='', help='comma-separated models whose author confirms rule 3 (browser-shippable); manual input')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--ignore-deadline', action='store_true')
    a = ap.parse_args(argv)
    run(a.scores, a.out, B=a.boot, candidates=[x for x in a.candidates.split(',') if x], shippable=[x for x in a.shippable.split(',') if x],
        write=not a.no_write, check_deadline=not a.ignore_deadline)


if __name__ == '__main__':
    main()
