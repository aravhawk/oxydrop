"""Round-2 evaluation: results/photon/WINNER2.md (the mechanical winner rule of ROUND2.md / Amendment 7) and results/photon/SIDE_BY_SIDE.md.

    cd python && python -m oxydrop_ml.photon.winner2 [--boot 2000] [--models photonnet,distill,oxyssl,oxyseq] [--scores DIR] [--out DIR] [--no-write]

Runs at any time on whatever score files exist: a missing, partial (fewer than the 5,644 evaluated windows) or half-written file is skipped with a note, a
missing round-2 model is listed as "no score files yet", and with no round-2 file at all the verdict says so. It does NOT read TABLE.csv (other agents rewrite
it concurrently); it reads results/photon/scores/*.csv itself and calls the evalkit functions (Boot, load_scores, closed-form time-matched statistic, same
person-bootstrap seed), so every number equals the TABLE.csv value of `python -m oxydrop_ml.photon.evalkit`. No number is typed by hand.

Model groups (file names are <model>__L<0..5>.csv and <model>__X1.csv):
  eligible   primary files of the round-2 models: photonnet, distill, oxyssl, oxyseq (the 3-seed ensemble, ROUND2.md rule 6).
  seeds      <model>_s<k> / <model>_seed<k>: reported (seed mean), never eligible.
  ablations  every other <model>_<tag> file that is not a null control: shown, never eligible, never promotable.
  nulls      <model>[_s<k>|_seed<k>|_gperm|_perm...]_null[...] and ..._blank[...]: grouped to their model, mean TM over all draws.
  baselines  chain_v1, hybrid_final, cnn_full, ctl_dc_lr, twin_inverse (+ the shipped round-1 twin_gate for reference); compared in the side-by-side only.

The winner rule (ROUND2.md, "Winner rule for round 2", stated 18:25 CT before any round-2 score; operational reading in Amendment 7, section D):
  beats chain = (a) paired person-bootstrap 95 % CI of the L0 TM difference to chain_v1 excludes 0 above, (b) >= 5 of the 6 people improved,
                (c) worst person's TM >= chain_v1's worst person, (d) null mean <= 0.55 (and the blank mean <= 0.55; blank is required for temporal models: oxyseq).
  If some model beats chain: the highest L0 point estimate among them.
  Otherwise: among eligible models whose L0 paired CI includes 0 (a tie) and whose nulls passed, the highest L0 point estimate; models within 0.01 of it are
  near-ties, ordered by the mean paired difference to chain_v1 over L3, L4, L5, X1, then by the mean answered accuracy at their own coverage over the same levels.
  A model whose CI lies below 0 is a loss and cannot win; a model without a passed null control cannot win.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import evalkit as E

ROOT = E.REPO
RES = E.RES
SCORES = E.SCORES
REF = E.BASE                                                   # chain_v1
ROUND2 = ['photonnet', 'distill', 'oxyssl', 'oxyseq']
TEMPORAL = {'oxyseq'}                                          # blank-observation control required (ROUND2.md rule 5: temporal models)
BASELINES = ['chain_v1', 'hybrid_final', 'cnn_full', 'ctl_dc_lr', 'twin_inverse', 'twin_gate']
SHOW_LEVELS = [0, 3, 4, 5, 'X1']                               # clean next to the stress levels of the side-by-side
STRESS_LEVELS = [3, 4, 5, 'X1']                                # tie-break levels (ROUND2.md: L3..L5 and X1)
ALL_LEVELS = [0, 1, 2, 3, 4, 5, 'X1']
NULL_MAX = E.NULL_MAX                                          # 0.55
NEAR_TIE = 0.01
MIN_PEOPLE_BETTER = 5
SEED_RE = re.compile(r'_(?:s|seed)\d+$')
NULL_RE = re.compile(r'^(?P<base>.+?)_(?P<kind>null|blank)(?P<rest>(?:_.*)?)$')
BASE_TOKEN_RE = re.compile(r'_(?:s\d+|seed\d+|gperm|perm\d*|shift\d*|d\d+|draw\d+)$')


def lname(l):
    return f'L{l}' if isinstance(l, (int, np.integer)) else str(l)


def f3(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.3f}'


def f2(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.2f}'


# ------------------------------------------------------------------------------------------------ discovery
def discover(scores_dir):
    """{model: set(levels)} from the file names (no file is opened)."""
    out = {}
    for p in sorted(Path(scores_dir).glob('*__*.csv')):
        m, l = E.parse_name(p)
        if m is None or l not in E.ALL_LEVELS:
            continue
        out.setdefault(m, set()).add(l)
    return out


def classify(name, primaries):
    """-> (role, owner). role in primary | seed | null | blank | ablation | other; owner = the round-2 primary a file belongs to (or None)."""
    m = NULL_RE.match(name)
    if m:
        base = m.group('base')
        while BASE_TOKEN_RE.search(base):
            base = BASE_TOKEN_RE.sub('', base)
        return m.group('kind'), (base if base in primaries else None)
    if name in primaries:
        return 'primary', name
    for p in primaries:
        if name.startswith(p + '_'):
            return ('seed' if SEED_RE.search(name) and SEED_RE.sub('', name) == p else 'ablation'), p
    return 'other', None


# ------------------------------------------------------------------------------------------------ evaluation with a shared bootstrap
class Evaluator:
    def __init__(self, scores_dir, B):
        self.dir = Path(scores_dir)
        self.ref = E.eval_windows()
        self.boot = E.Boot(self.ref, B=B)
        self.notes = []
        self._cache = {}
        self.pcols = [f'tm_{s}' for s in self.boot.u]

    def get(self, model, level):
        """Metrics dict of (model, level) or None when the file is missing / unreadable / partial (a note is recorded)."""
        k = (model, level)
        if k in self._cache:
            return self._cache[k]
        p = self.dir / f'{model}__{level if isinstance(level, str) else "L" + str(level)}.csv'
        res = None
        if p.exists():
            try:
                d = E.load_scores(p, self.ref)
                tm, tm_b, per = self.boot.tm(d.pred.values)
                y, pred, ans = d.y.values, d.pred.values, d.answer.values
                res = dict(model=model, level=level, tm=tm, tm_b=tm_b, per=per, worst=float(np.nanmin(per)) if np.isfinite(per).any() else np.nan,
                           coverage=float(ans.mean()), acc_ans=float((pred[ans] == y[ans]).mean()) if ans.any() else np.nan,
                           acc_all=float((pred == y).mean()), n=len(d))
                res['lo'], res['hi'] = E.ci(tm_b)
            except Exception as e:                            # partial / in-progress / malformed: skip, say so
                self.notes.append(f'SKIPPED {p.name}: {type(e).__name__}: {str(e)[:120]}')
        self._cache[k] = res
        return res

    def paired(self, a, level):
        """Paired person-bootstrap difference of result `a` to chain_v1 at the same level (same resampled people)."""
        b = self.get(REF, level)
        if a is None or b is None or a['model'] == REF:
            return dict(diff=np.nan, lo=np.nan, hi=np.nan, nb=np.nan, nc=np.nan, p=np.nan)
        lo, hi = E.ci(a['tm_b'] - b['tm_b'])
        x, y = a['per'], b['per']
        ok = np.isfinite(x) & np.isfinite(y) & (x != y)
        nb = int((x[ok] > y[ok]).sum())
        return dict(diff=a['tm'] - b['tm'], lo=lo, hi=hi, nb=nb, nc=int(ok.sum()), p=E.sign_test(nb, int(ok.sum())))


# ------------------------------------------------------------------------------------------------ nulls
def null_summary(ev, files, primary):
    """{kind: dict(n, mean, max, tms)} of the null / blank files owned by `primary` (level 0 only)."""
    out = {}
    for kind in ('null', 'blank'):
        tms = []
        for name in files:
            role, owner = classify(name, [primary])
            if role == kind and owner == primary:
                r = ev.get(name, 0)
                if r is not None:
                    tms.append((name, r['tm']))
        if tms:
            v = np.array([t for _, t in tms])
            out[kind] = dict(n=len(v), mean=float(v.mean()), max=float(v.max()), tms=tms)
    return out


def nulls_status(primary, ns):
    """(passed: bool, text)."""
    need = ['null'] + (['blank'] if primary in TEMPORAL else [])
    miss = [k for k in need if k not in ns]
    bad = [k for k in ns if ns[k]['mean'] > NULL_MAX]
    if miss:
        return False, f'NOT PASSED: missing {"/".join(miss)} control'
    if bad:
        return False, 'NOT PASSED: ' + ', '.join(f'{k} mean {ns[k]["mean"]:.3f} > {NULL_MAX}' for k in bad)
    return True, 'passed: ' + ', '.join(f'{k} mean {ns[k]["mean"]:.3f} over {ns[k]["n"]} draw(s)' for k in ns)


# ------------------------------------------------------------------------------------------------ the rule
def evaluate_round2(ev, models, primaries):
    """One summary per eligible model (primary file with an L0 score)."""
    cands = []
    chain0 = ev.get(REF, 0)
    for m in primaries:
        r0 = ev.get(m, 0)
        c = dict(model=m, have=sorted([l for l in ALL_LEVELS if ev.get(m, l) is not None], key=E.level_key), r0=r0)
        c['missing'] = [lname(l) for l in ALL_LEVELS if l not in c['have']]
        files = [n for n in models if classify(n, [m])[1] == m]
        c['nulls'] = null_summary(ev, files, m)
        c['nulls_ok'], c['nulls_text'] = nulls_status(m, c['nulls'])
        if r0 is None:
            c['status'] = 'no L0 score file yet' if not c['have'] else 'no usable L0 file'
            cands.append(c); continue
        pr = ev.paired(r0, 0)
        c.update(tm=r0['tm'], lo=r0['lo'], hi=r0['hi'], diff=pr['diff'], dlo=pr['lo'], dhi=pr['hi'], nb=pr['nb'], nc=pr['nc'], sp=pr['p'],
                 worst=r0['worst'], chain_worst=chain0['worst'] if chain0 else np.nan)
        c['a'] = bool(pr['lo'] > 0)
        c['b'] = bool(pr['nb'] >= MIN_PEOPLE_BETTER)
        c['c'] = bool(np.isfinite(r0['worst']) and chain0 is not None and r0['worst'] >= chain0['worst'])
        c['d'] = c['nulls_ok']
        c['beats'] = c['a'] and c['b'] and c['c'] and c['d']
        c['tie'] = bool(pr['lo'] <= 0 <= pr['hi'])
        c['loss'] = bool(pr['hi'] < 0)
        diffs, accs = [], []
        for l in STRESS_LEVELS:
            r = ev.get(m, l)
            if r is None:
                continue
            q = ev.paired(r, l)
            if np.isfinite(q['diff']):
                diffs.append(q['diff'])
            accs.append(r['acc_ans'] if np.isfinite(r['acc_ans']) else -1.0)   # a model that answers nothing counts as -1 (never preferred)
        c['stress_levels_used'] = [lname(l) for l in STRESS_LEVELS if ev.get(m, l) is not None]
        c['stress_diff'] = float(np.mean(diffs)) if diffs else np.nan
        c['stress_acc'] = float(np.mean(accs)) if accs else np.nan
        c['status'] = 'beats chain' if c['beats'] else ('tie' if c['tie'] else ('loss' if c['loss'] else 'win on TM but fails the other conditions' if c['a'] else 'n/a'))
        cands.append(c)
    return cands


def decide(cands):
    """Mechanical verdict. Returns (winner model or None, lines of reasoning)."""
    lines = []
    have = [c for c in cands if c['r0'] is not None]
    if not have:
        return None, ['No round-2 model has an L0 score file yet; no verdict is possible.']
    beat = [c for c in have if c['beats']]
    if beat:
        beat.sort(key=lambda c: -c['tm'])
        w = beat[0]
        lines.append(f'Rule step 1: {len(beat)} model(s) beat chain_v1 on all of (a) paired CI above 0, (b) >= {MIN_PEOPLE_BETTER}/6 people better, (c) worst person not below chain_v1, '
                     f'(d) nulls passed: {", ".join(c["model"] for c in beat)}. The highest L0 point estimate wins.')
        lines.append(f'**WINNER: `{w["model"]}`, beats chain_v1** (L0 TM {w["tm"]:.3f}, paired {w["diff"]:+.3f} [{w["dlo"]:+.3f}, {w["dhi"]:+.3f}], {w["nb"]}/{w["nc"]} people better).')
        return w['model'], lines
    lines.append('Rule step 1: no round-2 model beats chain_v1 (all of: paired CI above 0, >= 5/6 people better, worst person not below chain_v1, nulls passed).')
    ties = [c for c in have if c['tie'] and c['nulls_ok']]
    blocked = [c for c in have if c['tie'] and not c['nulls_ok']]
    for c in blocked:
        lines.append(f'`{c["model"]}` ties chain_v1 at L0 ({c["tm"]:.3f}) but its null controls are {c["nulls_text"]}; it cannot be the tie winner.')
    for c in [c for c in have if c['loss']]:
        lines.append(f'`{c["model"]}` loses to chain_v1 at L0 (paired {c["diff"]:+.3f} [{c["dlo"]:+.3f}, {c["dhi"]:+.3f}]); not eligible for the tie rule.')
    for c in [c for c in have if c['a'] and not c['beats']]:
        why = [n for n, k in (('(b) people better', 'b'), ('(c) worst person', 'c'), ('(d) nulls', 'd')) if not c[k]]
        lines.append(f'`{c["model"]}` has a paired CI above 0 but fails {", ".join(why)}; it is neither a win nor a tie winner.')
    if not ties:
        lines.append('Rule step 2: no round-2 model is a tie with passed nulls. **NO WINNER**; the shipped chain stays the headline.')
        return None, lines
    top = max(c['tm'] for c in ties)
    near = [c for c in ties if c['tm'] >= top - NEAR_TIE]
    lines.append(f'Rule step 2: tie candidates (paired CI includes 0, nulls passed): {", ".join(f"{c["model"]} {c["tm"]:.3f}" for c in sorted(ties, key=lambda c: -c["tm"]))}. '
                 f'Highest L0 point estimate {top:.3f}; near-ties are within {NEAR_TIE}: {", ".join(c["model"] for c in near)}.')
    if len(near) > 1:
        near.sort(key=lambda c: (-(c['stress_diff'] if np.isfinite(c['stress_diff']) else -9), -(c['stress_acc'] if np.isfinite(c['stress_acc']) else -9)))
        lines.append('Near-tie break 1: mean paired TM difference to chain_v1 over L3, L4, L5, X1 (levels with files only), then break 2: mean answered accuracy at the own coverage over the same levels: '
                     + '; '.join(f'`{c["model"]}` diff {f3(c["stress_diff"])} ({"+".join(c["stress_levels_used"]) or "no stress file"}), answered acc {f3(c["stress_acc"])}' for c in near) + '.')
    else:
        near.sort(key=lambda c: -c['tm'])
    w = near[0]
    lines.append(f'**WINNER (tie): `{w["model"]}`**, L0 TM {w["tm"]:.3f} [{w["lo"]:.3f}, {w["hi"]:.3f}], paired {w["diff"]:+.3f} [{w["dlo"]:+.3f}, {w["dhi"]:+.3f}] ({w["nb"]}/{w["nc"]} people better): '
                 'a tie with chain_v1, not a win. The word "better" is not licensed.')
    if w['missing']:
        lines.append(f'Caveat: `{w["model"]}` has no score file for {", ".join(w["missing"])}; the verdict is based on the levels that exist.')
    return w['model'], lines


# ------------------------------------------------------------------------------------------------ markdown
def cell(r):
    return 'n/a' if r is None else f'{r["tm"]:.3f} [{r["lo"]:.3f}, {r["hi"]:.3f}]'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def row_groups(models, primaries):
    """Ordered [(title, [model names])] for the side-by-side."""
    disc = [m for m in models]
    groups = [('Baselines (round 1, for reference)', [b for b in BASELINES if b in disc])]
    for p in primaries:
        mem = [p] if p in disc else []
        mem += sorted(n for n in disc if classify(n, [p]) == ('ablation', p))
        if mem or any(classify(n, [p])[1] == p for n in disc):
            groups.append((f'Round 2: {p}', mem))
    return groups


def side_by_side(ev, models, primaries, levels=SHOW_LEVELS):
    L = [lname(l) for l in levels]
    out = ['# Side by side: round-2 trained models against the round-1 baselines', '',
           'Written by `python -m oxydrop_ml.photon.winner2` from `results/photon/scores/*.csv` (the same bootstrap as `TABLE.csv`: TM = Hoffman time-matched balanced accuracy at the forced '
           'decision `pred`, 95 % person-bootstrap CI, paired differences to `chain_v1` at the same level, same resampled people). Clean light (L0) next to the simulated distant-flash '
           'levels L3 to L5 and the out-of-family stress X1. All stress levels are simulations, not measurements of any phone. A difference is a **win** only when its paired CI excludes 0; '
           'otherwise it is a tie. Missing files show `n/a`. Models are not refit per level. Round-2 models were designed after the round-1 outer results were seen (Amendment 7), so their L0 numbers are optimistic.', '']
    groups = row_groups(models, primaries)
    # A: TM with CI
    out += ['## A. TM balanced accuracy [95 % CI]', '']
    rows = []
    for title, mem in groups:
        if not mem:
            continue
        rows.append([f'**{title}**'] + [''] * len(L))
        for m in mem:
            rows.append([m] + [cell(ev.get(m, l)) for l in levels])
        if title.startswith('Round 2: ') and title[9:] in primaries:
            seeds = sorted(n for n in models if classify(n, [title[9:]]) == ('seed', title[9:]))
            if seeds:
                mean = []
                for l in levels:
                    v = [ev.get(s, l)['tm'] for s in seeds if ev.get(s, l) is not None]
                    mean.append(f'{np.mean(v):.3f} (n={len(v)})' if v else 'n/a')
                rows.append([f'{title[9:]} seed mean over {len(seeds)} seed file(s)'] + mean)
    out += [table(['model'] + L, rows), '']
    # B: paired diff vs chain_v1
    out += ['## B. Paired difference to chain_v1 at the same level [95 % CI], people better of compared', '']
    rows = []
    for title, mem in groups:
        for m in mem:
            if m == REF:
                continue
            c = []
            for l in levels:
                r = ev.get(m, l); q = ev.paired(r, l)
                c.append('n/a' if r is None or not np.isfinite(q['diff']) else
                         f'{q["diff"]:+.3f} [{q["lo"]:+.3f}, {q["hi"]:+.3f}] {q["nb"]}/{q["nc"]} ' + ('win' if q['lo'] > 0 else 'loss' if q['hi'] < 0 else 'tie'))
            rows.append([m] + c)
    out += [table(['model'] + L, rows), '']
    # C: worst person
    out += ['## C. Worst person TM (the weakest of the six held-out people)', '']
    rows = [[m] + [f3(ev.get(m, l)['worst']) if ev.get(m, l) else 'n/a' for l in levels] for _, mem in groups for m in mem]
    out += [table(['model'] + L, rows), '']
    # D: coverage / answered accuracy
    out += ['## D. Own abstention: coverage, accuracy on answered windows, accuracy of `pred` on all windows (the random-abstention reference)', '',
            'Random abstention at the same coverage keeps the all-window accuracy in expectation, so answered accuracy only means something when it exceeds the all-window column.', '']
    rows = []
    for _, mem in groups:
        for m in mem:
            c = []
            for l in levels:
                r = ev.get(m, l)
                c.append('n/a' if r is None else f'cov {r["coverage"]:.2f} / ans {f3(r["acc_ans"])} / all {r["acc_all"]:.3f}')
            rows.append([m] + c)
    out += [table(['model'] + L, rows), '']
    # E: per person L0
    pc = [s for s in ev.boot.u]
    out += ['## E. Per-person TM at L0', '']
    rows = []
    for _, mem in groups:
        for m in mem:
            r = ev.get(m, 0)
            rows.append([m] + (['n/a'] * len(pc) if r is None else [f3(v) for v in r['per']]))
    out += [table(['model'] + [f'person {s}' for s in pc], rows), '']
    # F: nulls
    out += ['## F. Null and blank controls at L0 (mean over draws; must be <= 0.55)', '']
    rows = []
    for p in primaries:
        files = [n for n in models if classify(n, [p])[1] == p]
        ns = null_summary(ev, files, p)
        for kind in ('null', 'blank'):
            if kind in ns:
                s = ns[kind]
                rows.append([p, kind, s['n'], f'{s["mean"]:.3f}', f'{s["max"]:.3f}', 'ok' if s['mean'] <= NULL_MAX else '**FLAG**', ', '.join(f'{n} {t:.3f}' for n, t in s['tms'])])
        for kind in ('null', 'blank'):
            if kind not in ns and (kind == 'null' or p in TEMPORAL) and any(ev.get(p, l) is not None for l in ALL_LEVELS):
                rows.append([p, kind, 0, 'n/a', 'n/a', 'MISSING', ''])
    for b in BASELINES:                       # round-1 reference nulls
        files = [n for n in models if n.startswith(b + '_') and NULL_RE.match(n)]
        for n in files:
            r = ev.get(n, 0)
            if r is not None and classify(n, [])[0] in ('null', 'blank') and BASE_TOKEN_RE.sub('', NULL_RE.match(n).group('base')) == b:
                rows.append([b + ' (round 1)', classify(n, [])[0], 1, f'{r["tm"]:.3f}', f'{r["tm"]:.3f}', 'ok' if r['tm'] <= NULL_MAX else 'flag (single draw)', n])
    out += [table(['model', 'kind', 'draws', 'mean TM', 'max TM', 'check', 'files'], rows) if rows else 'No null controls found.', '']
    if ev.notes:
        out += ['## Skipped files', ''] + [f'* {n}' for n in ev.notes] + ['']
    return '\n'.join(out) + '\n'


def winner_md(ev, cands, verdict_lines, winner, models, primaries, stamp):
    out = ['# Round 2 winner (mechanical application of ROUND2.md, Amendment 7)', '',
           f'Written by `python -m oxydrop_ml.photon.winner2` ({stamp}) from the score files that existed at that moment. Rerun it after every new file: it ignores missing and partial files and says so. '
           'The rule is quoted verbatim from `.agents/tasks/task-oxydrop-photon-model/ROUND2.md` (stated 18:25 CT, before any round-2 score):', '',
           '> Only round-2 trained models are eligible. Primary: L0 TM bal. acc. (3-seed ensemble) paired vs chain_v1. "Beats chain" = paired CI excluding 0, >= 5/6 people improved, worst person not below chain\'s, nulls passed. '
           'If none beats chain: the eligible model with the highest L0 point estimate whose paired CI includes 0 (a tie), breaking near-ties (within 0.01) by mean paired diff over L3..L5 and X1, then by answered accuracy at its own coverage under stress. '
           'All round-2 models are designed after round-1 outer results were seen; this is disclosed as Amendment 7.', '',
           '## Verdict', '']
    out += [f'* {l}' for l in verdict_lines] + ['']
    c0 = ev.get(REF, 0)
    if c0:
        out += [f'Reference: `{REF}` L0 TM {cell(c0)}, worst person {f3(c0["worst"])}.', '']
    out += ['## Eligible models at L0 (primary file = the 3-seed ensemble)', '']
    rows = []
    for c in cands:
        if c['r0'] is None:
            rows.append([c['model'], c['status'], '', '', '', '', '', '', '', c['nulls_text'] if c['nulls'] else 'no null files yet', ', '.join(c['missing']) or ''])
            continue
        rows.append([c['model'], c['status'], cell(c['r0']), f'{c["diff"]:+.3f} [{c["dlo"]:+.3f}, {c["dhi"]:+.3f}]', f'{c["nb"]}/{c["nc"]} (p = {c["sp"]:.3f})',
                     f'{f3(c["worst"])} vs {f3(c["chain_worst"])}',
                     'yes' if c['a'] else 'no', 'yes' if c['b'] else 'no', 'yes' if c['c'] else 'no', c['nulls_text'], ', '.join(c['missing']) or 'none'])
    out += [table(['model', 'status', 'L0 TM [95 % CI]', 'paired diff vs chain_v1', 'people better (sign p)', 'worst person vs chain_v1', '(a) CI > 0', '(b) >= 5/6', '(c) worst ok', '(d) nulls', 'missing levels'], rows), '']
    out += ['## Tie-break inputs (only used among near-ties, within 0.01 of the top tie)', '']
    rows = [[c['model'], f3(c['tm']), f3(c['stress_diff']), f3(c['stress_acc']), '+'.join(c['stress_levels_used']) or 'none'] for c in cands if c['r0'] is not None]
    out += [table(['model', 'L0 TM', 'mean paired diff over L3..L5, X1', 'mean answered acc over the same levels', 'levels used'], rows) if rows else 'No round-2 L0 file yet.', '']
    # seeds
    out += ['## Seeds and ablations found', '']
    rows = []
    for p in primaries:
        seeds = sorted(n for n in models if classify(n, [p]) == ('seed', p))
        abl = sorted(n for n in models if classify(n, [p]) == ('ablation', p))
        r = [ev.get(s, 0) for s in seeds]
        rows.append([p, len([x for x in r if x]) if seeds else 0, f3(np.mean([x['tm'] for x in r if x])) if any(r) else 'n/a', ', '.join(abl) or 'none'])
    out += [table(['model', 'seed files with L0', 'seed-mean L0 TM', 'ablations (not eligible, not promotable)'], rows), '',
            'ROUND2.md rule 6 asks for three seeds of the final configuration and reports the seed mean next to the ensemble; the rule above uses the ensemble (primary file).', '']
    out += ['## Missing models and files', '']
    miss = [f'* `{c["model"]}`: ' + ('no score file yet' if not c['have'] else f'has {", ".join(lname(l) for l in c["have"])}; missing {", ".join(c["missing"]) or "none"}') for c in cands]
    out += miss + ['']
    if ev.notes:
        out += ['## Skipped files (partial, unreadable or in progress)', ''] + [f'* {n}' for n in ev.notes] + ['']
    out += ['See `SIDE_BY_SIDE.md` for the clean and stress tables of every round-2 model, ablation and baseline. Nothing here is a medical-device claim.']
    return '\n'.join(out) + '\n'


def run(scores_dir=SCORES, out_dir=RES, B=2000, primaries=None, write=True, log=print):
    primaries = list(primaries or ROUND2)
    disc = discover(scores_dir)
    models = sorted(disc)
    ev = Evaluator(scores_dir, B)
    cands = evaluate_round2(ev, models, primaries)
    winner, lines = decide(cands)
    import subprocess
    try:
        stamp = subprocess.run(['date', '+%a %b %d %H:%M:%S %Z %Y'], capture_output=True, text=True).stdout.strip()
    except Exception:
        stamp = 'time unknown'
    side = side_by_side(ev, models, primaries)
    wmd = winner_md(ev, cands, lines, winner, models, primaries, stamp)
    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'WINNER2.md').write_text(wmd)
        (Path(out_dir) / 'SIDE_BY_SIDE.md').write_text(side)
    for n in ev.notes:
        log(n)
    log('VERDICT: ' + ' '.join(lines))
    return winner, cands, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--scores', default=str(SCORES))
    ap.add_argument('--out', default=str(RES))
    ap.add_argument('--models', default=','.join(ROUND2), help='comma-separated primary round-2 model names')
    ap.add_argument('--no-write', action='store_true')
    a = ap.parse_args(argv)
    run(a.scores, a.out, B=a.boot, primaries=a.models.split(','), write=not a.no_write)


if __name__ == '__main__':
    main()
