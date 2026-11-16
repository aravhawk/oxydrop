#!/usr/bin/env python3
"""Generate paper/numbers.tex and paper/tables/*.tex from the harness outputs only.

Inputs (all under results/photon/, nothing is typed by hand):
  TABLE.csv    per (model, level) TM balanced accuracy, CIs, paired differences vs chain_v1, coverage, ...
  WINNER2.md   the "SHIPPED" line names the shipped model (override with --winner after the round-3 lock)
  NULLS.csv    50 global-permutation refits of the chain (null reference)

Usage:
  python paper/make_numbers.py                        # provisional numbers, DRAFT banner on
  python paper/make_numbers.py --winner distill2 --locked    # after the planner's lock

Macro names cannot contain digits, so levels are spelled out: Lzero, Lone, ..., Lfive, Xone.
Example: \\WinLzero (TM), \\WinLzeroCI, \\WinLzeroDiff, \\WinLzeroDiffCI, \\WinLzeroVerdict, \\WinLzeroBetter.
A missing model or level is written as "n/a", never as a guess.
"""
import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RES = REPO / 'results' / 'photon'

LEVELS = ['0', '1', '2', '3', '4', '5', 'X1']
LNAME = {'0': 'Lzero', '1': 'Lone', '2': 'Ltwo', '3': 'Lthree', '4': 'Lfour', '5': 'Lfive', 'X1': 'Xone'}
PEOPLE = ['100001', '100002', '100003', '100004', '100005', '100006']

# key in TABLE.csv -> (macro prefix, display name)
MODELS = {
    'chain_v1': ('Chain', 'Logistic chain v1'),
    'chain_v1_nestedC': ('ChainNested', 'Logistic chain v1, C by nested CV'),
    'hybrid_final': ('Hybrid', 'Hybrid (chain + CNN), nested'),
    'cnn_full': ('Cnn', 'CNN alone (OxyNet, 3 seeds)'),
    'ctl_dc_lr': ('DcLr', 'Control: DC-colour LR (3 values)'),
    'ctl_dc_ror_lr': ('DcRorLr', 'Control: DC colour + AC optics LR'),
    'ctl_dc_hr_lr': ('DcHrLr', 'Control: DC colour + dHR LR'),
    'twin_inverse': ('TwinInv', 'PhotonTwin-Inverse'),
    'twin_inverse_noresid': ('TwinInvNoResid', 'PhotonTwin-Inverse, physics path only'),
    'twin_gate': ('TwinGate', 'PhotonTwin gate (hand-coded, post hoc)'),
    'flashgap': ('Flashgap', 'FlashGap'),
    'flashgap_gate': ('FlashgapGate', 'FlashGap gate (post hoc)'),
    'oxytrack': ('Oxytrack', 'OxyTrack'),
    'distill': ('Distill', 'PhotonDistill'),
    'distill_nokd': ('DistillNoKd', 'PhotonDistill, no teachers'),
    'distill_v1': ('DistillVone', 'PhotonDistill v1 (before post hoc grid)'),
    'oxyseq': ('Oxyseq', 'OxySeq'),
    'oxyseq_notta': ('OxyseqNoTta', 'OxySeq, no test-time adaptation'),
    'photonnet': ('Photonnet', 'PhotonNet'),
    'photonnet_nophys': ('PhotonnetNoPhys', 'PhotonNet, no physics layer'),
    'oxyssl': ('Oxyssl', 'OxySSL'),
    'oxyssl_init': ('OxysslInit', 'OxySSL, init only'),
    'oxyssl_noaug': ('OxysslNoAug', 'OxySSL, no augmentation'),
    'distill2': ('DistillTwo', 'PhotonDistill2'),
    'multida': ('Multida', 'MultiDA'),
    'ssl3': ('SslThree', 'SSL-3'),
    'photonnet_xl': ('PhotonnetXl', 'PhotonNet-XL'),
    'ens3': ('EnsThree', 'Ens-3'),
    'multida_hofonly': ('MultidaHofonly', 'MultiDA, Hoffman rows only'),
}
ROUND_OF = {  # which table group a model belongs to
    'baseline': ['chain_v1', 'chain_v1_nestedC', 'hybrid_final', 'cnn_full'],
    'control': ['ctl_dc_lr', 'ctl_dc_ror_lr', 'ctl_dc_hr_lr'],
    'round1': ['twin_inverse', 'twin_inverse_noresid', 'flashgap', 'oxytrack', 'twin_gate', 'flashgap_gate'],
    'round2': ['distill', 'oxyseq', 'photonnet', 'oxyssl'],
    'round3': ['distill2', 'multida', 'ssl3', 'photonnet_xl', 'ens3'],
}
GROUP_TITLE = {'baseline': 'Baselines', 'control': 'Controls (no network)', 'round1': 'Round 1 (designed before any score)',
               'round2': 'Round 2 (trained end to end)', 'round3': 'Round 3 (scaled up)'}
ABLATION_OF = {
    'distill': ['distill_nokd', 'distill_v1'],
    'oxyseq': ['oxyseq_notta'],
    'photonnet': ['photonnet_nophys'],
    'oxyssl': ['oxyssl_init', 'oxyssl_noaug'],
    'multida': ['multida_hofonly'],
}


def fnum(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def tm(v, nd=3):
    return 'n/a' if v is None else f'{v:.{nd}f}'


def sg(v, nd=3):
    if v is None:
        return 'n/a'
    s = f'{abs(v):.{nd}f}'
    if v < 0 and float(s) != 0:
        return f'$-${s}'
    return f'$+${s}' if float(s) != 0 else '0.000'


def ci(lo, hi):
    return 'n/a' if lo is None or hi is None else f'[{lo:.3f}, {hi:.3f}]'


def sci(lo, hi):
    return 'n/a' if lo is None or hi is None else f'[{sg(lo)}, {sg(hi)}]'


def esc(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


class Table:
    def __init__(self, path):
        self.rows = {}
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                self.rows[(r['model'], str(r['level']))] = r
        self.models = sorted({m for m, _ in self.rows})

    def get(self, m, lv):
        return self.rows.get((m, lv))

    def val(self, m, lv, col):
        r = self.get(m, lv)
        return None if r is None else fnum(r.get(col))

    def has(self, m, lv='0'):
        return (m, lv) in self.rows


def parse_winner(path):
    if not path.exists():
        return None
    txt = path.read_text()
    m = re.search(r'SHIPPED[^`]*`([A-Za-z0-9_]+)`', txt)
    return m.group(1) if m else None


def null_rows(T, key):
    """Null / blank rows of model `key`, de-duplicated (byte-identical copies count once)."""
    pats = [rf'^{re.escape(key)}_null$', rf'^{re.escape(key)}_d\d+_null$', rf'^{re.escape(key)}_null_d\d+$',
            rf'^{re.escape(key)}_g?perm_null$', rf'^{re.escape(key)}_blank$']
    out, seen = [], set()
    for m in T.models:
        if any(re.match(p, m) for p in pats) and T.has(m, '0'):
            r = T.get(m, '0')
            sig = (r['tm_bacc'], r['tm_lo'], r['tm_hi'], r['auroc'])
            if sig in seen:
                continue
            seen.add(sig)
            out.append((m, fnum(r['tm_bacc']), 'blank' if m.endswith('_blank') else 'null'))
    return out


def seed_rows(T, key):
    return [m for m in T.models if re.match(rf'^{re.escape(key)}_s\d+$', m) and T.has(m, '0')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--winner', help='table key of the shipped model (default: parsed from WINNER2.md)')
    ap.add_argument('--locked', action='store_true', help='planner lock recorded: drop the DRAFT banner')
    ap.add_argument('--table', default=str(RES / 'TABLE.csv'))
    ap.add_argument('--out', default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    T = Table(a.table)
    parsed = parse_winner(RES / 'WINNER2.md')
    win = a.winner or parsed or 'distill'
    src = 'command line' if a.winner else ('WINNER2.md' if parsed else 'default')
    if win not in T.models:
        sys.exit(f'winner {win!r} has no rows in {a.table}')
    if win not in MODELS:
        MODELS[win] = (re.sub(r'[^A-Za-z]', '', win.title()) or 'Wx', win)
    print(f'shipped model: {win} (from {src}); locked={a.locked}')

    L = []
    w = L.append
    w('% GENERATED by paper/make_numbers.py from results/photon/TABLE.csv, WINNER2.md and NULLS.csv. Do not edit.')
    w(f'% shipped model key: {win} (source: {src})')
    w('\\newif\\ifDraft')
    w('\\Drafttrue' if not a.locked else '\\Draftfalse')
    w(f'\\newcommand{{\\WinKey}}{{\\texttt{{{esc(win)}}}}}')
    w(f'\\newcommand{{\\WinName}}{{{MODELS[win][1]}}}')
    w(f'\\newcommand{{\\WinSource}}{{{esc(src)}}}')

    def emit(prefix, key):
        for lv in LEVELS:
            n = LNAME[lv]
            r = T.get(key, lv)
            g = (lambda c: fnum(r.get(c))) if r else (lambda c: None)
            ppl = 'n/a'
            if r and r.get('n_people_compared'):
                ppl = f"{int(float(r['n_people_improved']))}/{int(float(r['n_people_compared']))}"
            w(f'\\newcommand{{\\{prefix}{n}}}{{{tm(g("tm_bacc"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}CI}}{{{ci(g("tm_lo"), g("tm_hi"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}Lo}}{{{tm(g("tm_lo"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}Hi}}{{{tm(g("tm_hi"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}Diff}}{{{sg(g("diff_vs_chain_v1"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}DiffCI}}{{{sci(g("diff_lo"), g("diff_hi"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}Verdict}}{{{(r.get("verdict") or "n/a") if r else "n/a"}}}')
            w(f'\\newcommand{{\\{prefix}{n}Better}}{{{ppl}}}')
            w(f'\\newcommand{{\\{prefix}{n}Worst}}{{{tm(g("worst_person_tm"))}}}')
            w(f'\\newcommand{{\\{prefix}{n}Cov}}{{{tm(g("coverage"), 2)}}}')
            w(f'\\newcommand{{\\{prefix}{n}AnsAcc}}{{{tm(g("acc_ans"), 3)}}}')
            w(f'\\newcommand{{\\{prefix}{n}Auroc}}{{{tm(g("auroc"), 3)}}}')
            w(f'\\newcommand{{\\{prefix}{n}Mae}}{{{tm(g("spo2_mae"), 1)}}}')

    for key, (pre, _) in MODELS.items():
        emit(pre, key)
    emit('Win', win)

    # verdict switches for the shipped model
    for lv in LEVELS:
        v = (T.get(win, lv) or {}).get('verdict', '')
        for kind in ('loss', 'tie', 'win'):
            nm = f'Win{kind.title()}{LNAME[lv]}'
            w(f'\\newif\\if{nm}')
            w(f'\\{nm}{"true" if v == kind else "false"}')
    # mean paired difference over L3, L4, L5, X1 for the shipped model
    ds = [T.val(win, lv, 'diff_vs_chain_v1') for lv in ('3', '4', '5', 'X1')]
    ds = [d for d in ds if d is not None]
    w(f'\\newcommand{{\\WinStressMeanDiff}}{{{sg(sum(ds) / len(ds)) if ds else "n/a"}}}')
    nwins = sum(1 for lv in ('3', '4', '5', 'X1') if (T.get(win, lv) or {}).get('verdict') == 'win')
    w(f'\\newcommand{{\\WinStressWins}}{{{nwins}}}')

    # seeds and nulls for the shipped model, null reference of the chain
    sr = seed_rows(T, win)
    sm = [T.val(m, '0', 'tm_bacc') for m in sr]
    sm = [x for x in sm if x is not None]
    w(f'\\newcommand{{\\WinSeedN}}{{{len(sm)}}}')
    w(f'\\newcommand{{\\WinSeedMeanLzero}}{{{tm(sum(sm) / len(sm)) if sm else "n/a"}}}')
    nr = null_rows(T, win)
    nv = [v for _, v, k in nr if k == 'null' and v is not None]
    w(f'\\newcommand{{\\WinNullDraws}}{{{len(nv)}}}')
    w(f'\\newcommand{{\\WinNullMean}}{{{tm(sum(nv) / len(nv)) if nv else "n/a"}}}')
    w(f'\\newcommand{{\\WinNullMax}}{{{tm(max(nv)) if nv else "n/a"}}}')
    w(f'\\newcommand{{\\WinNullList}}{{{", ".join(f"{v:.3f}" for v in nv) if nv else "n/a"}}}')
    w('\\newif\\ifWinNullPass')
    w('\\WinNullPass' + ('true' if nv and sum(nv) / len(nv) <= 0.55 else 'false'))

    w('\\newif\\ifWinIsDistillTwo')
    w('\\WinIsDistillTwo' + ('true' if win == 'distill2' else 'false'))
    for k_, nm_ in (('ens3', 'EnsThree'), ('ssl3', 'SslThree'), ('multida', 'Multida'), ('photonnet_xl', 'PhotonnetXl')):
        w(f'\\newif\\ifHas{nm_}')
        w(f'\\Has{nm_}' + ('true' if T.has(k_, '0') else 'false'))
    # paired difference of the shipped model to the round-2 engine (distill), from <win>_compare.csv written by its author
    cmp_ = RES / f'{win}_compare.csv'
    cmpd = {}
    if cmp_.exists():
        for r_ in csv.DictReader(open(cmp_)):
            if r_['model'] == win:
                cmpd[r_['level']] = r_
    for lv_, n_ in (('L0', 'Lzero'), ('L3', 'Lthree'), ('L4', 'Lfour'), ('L5', 'Lfive'), ('X1', 'Xone')):
        r_ = cmpd.get(lv_)
        if r_ and r_.get('d_distill'):
            w(f'\\newcommand{{\\WinVsDistill{n_}}}{{{sg(float(r_["d_distill"]))}}}')
            w(f'\\newcommand{{\\WinVsDistill{n_}CI}}{{{sci(float(r_["d_distill_lo"]), float(r_["d_distill_hi"]))}}}')
        else:
            w(f'\\newcommand{{\\WinVsDistill{n_}}}{{n/a}}')
            w(f'\\newcommand{{\\WinVsDistill{n_}CI}}{{n/a}}')
    w('\\newif\\ifWinIsDistill')
    w('\\WinIsDistill' + ('true' if win == 'distill' else 'false'))
    w('\\newif\\ifWinNinety')
    w('\\WinNinety' + ('true' if (T.val(win, '0', 'tm_bacc') or 0) >= 0.90 else 'false'))
    import subprocess
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ''
    head = git('rev-parse', 'HEAD') or 'unknown'
    dirty = bool(git('status', '--porcelain', '--', 'results/photon/scores', 'results/photon/TABLE.csv', 'python/oxydrop_ml/photon'))
    w(f'\\newcommand{{\\RepoCommit}}{{{head[:7]}}}')
    w(f'\\newcommand{{\\RepoBranch}}{{{esc(git("branch", "--show-current") or "unknown")}}}')
    w('\\newif\\ifRepoDirty')
    w('\\RepoDirty' + ('true' if dirty else 'false'))
    h = (RES / 'HARNESS.md').read_text() if (RES / 'HARNESS.md').exists() else ''
    m = re.search(r'pulsatile-band \(0\.7-3\.5 Hz\) SNR of the red channel over the 12 recordings: ([^`]*?) dB at levels 1-5', h)
    w(f'\\newcommand{{\\StressSNR}}{{{m.group(1).replace(" and ", ", and ").replace("-", "$-$") if m else "n/a"}}}')
    lvl_rows = []
    grab = False
    for ln in h.splitlines():
        if ln.startswith('| level |'):
            grab = True
            continue
        if grab and ln.startswith('|---'):
            continue
        if grab and ln.startswith('|'):
            lvl_rows.append([c.strip() for c in ln.strip('|').split('|')])
        elif grab:
            break
    body_ = []
    for r in lvl_rows:
        r = [re.sub(r'(\d+(?:\.\d+)?)e-0*(\d+)', r'$\1\\times10^{-\2}$', c).replace('%', '\\%') for c in r]
        body_.append(' & '.join(r) + ' \\\\')
    (out / 'tables' / 'tab_stress_def.tex').write_text(
        '% GENERATED from results/photon/HARNESS.md\n{\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{rrrrrrrrr}\n\\toprule\n'
        'Level & $d_\\rho$ (mm) & Red kept & Green kept & Blue kept & $m_c$ R / G / B & Gain & Leak & Read noise (DN) \\\\\n\\midrule\n'
        + '\n'.join(body_) + '\n\\bottomrule\n\\end{tabular}}\n')
    nulls = RES / 'NULLS.csv'
    if nulls.exists():
        x = [float(r['tm_bacc']) for r in csv.DictReader(open(nulls))]
        w(f'\\newcommand{{\\ChainNullN}}{{{len(x)}}}')
        w(f'\\newcommand{{\\ChainNullMean}}{{{statistics.mean(x):.3f}}}')
        w(f'\\newcommand{{\\ChainNullSD}}{{{statistics.stdev(x):.3f}}}')
        w(f'\\newcommand{{\\ChainNullMin}}{{{min(x):.3f}}}')
        w(f'\\newcommand{{\\ChainNullMax}}{{{max(x):.3f}}}')
        w(f'\\newcommand{{\\ChainNullAbove}}{{{100 * sum(v > 0.55 for v in x) / len(x):.0f}}}')
    nmod = len([m for m in T.models if not re.search(r'(_null|_blank|_perm_null|_s\d+$|_seed\d+$)', m)])
    w(f'\\newcommand{{\\NumModels}}{{{nmod}}}')
    (out / 'numbers.tex').write_text('\n'.join(L) + '\n')

    # ------------------------------------------------------------------ tables
    def nm(key):
        if key == win:
            return f'\\textbf{{{MODELS.get(key, ("", key))[1]}}} (shipped)'
        return MODELS.get(key, ('', esc(key)))[1] if key in MODELS else f'\\texttt{{{esc(key)}}}'

    def groups(include_round3=True):
        g = []
        for gk, keys in ROUND_OF.items():
            ks = [k for k in keys if T.has(k, '0')]
            if win not in keys and gk == 'round3' and False:
                pass
            if ks:
                g.append((GROUP_TITLE[gk], ks))
        if not any(win in ks for _, ks in g):
            g.append(('Shipped', [win]))
        return g

    def nullmean_str(key):
        nr_ = null_rows(T, key)
        v = [x for _, x, k in nr_ if k == 'null' and x is not None]
        if not v:
            return '--'
        flag = '' if sum(v) / len(v) <= 0.55 else '$^{*}$'
        return f'{sum(v) / len(v):.3f} ({len(v)}){flag}'

    def write_tab(name, header, body, colspec, note=None, font='\\footnotesize', sep='4pt'):
        s = [f'% GENERATED by make_numbers.py ({name})', '{' + font, f'\\setlength{{\\tabcolsep}}{{{sep}}}',
             '\\begin{adjustbox}{max width=\\linewidth}', f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', header, '\\midrule']
        s += body
        s += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '}']
        (out / 'tables' / f'{name}.tex').write_text('\n'.join(s) + '\n')

    ncol = 8

    # --- clean light
    body = []
    for title, ks in groups():
        body.append(f'\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{title}}}}} \\\\')
        for k in ks:
            r = T.get(k, '0')
            g = lambda c: fnum(r.get(c))
            if k == 'chain_v1':
                dcell, dv, bt, nl = 'reference', '', '', '--'
            else:
                dcell = f"{sg(g('diff_vs_chain_v1'))} {sci(g('diff_lo'), g('diff_hi'))}"
                dv = r.get('verdict', '')
                bt = f"{int(float(r['n_people_improved']))}/{int(float(r['n_people_compared']))}" if r.get('n_people_compared') else '--'
                nl = nullmean_str(k)
            body.append(f"{nm(k)} & {tm(g('tm_bacc'))} {ci(g('tm_lo'), g('tm_hi'))} & {dcell} & {dv} & {bt} & {tm(g('worst_person_tm'))} & {nl} & {tm(g('auroc'))} \\\\")
    write_tab('tab_clean', 'Model & TM bal. acc. [95\\% CI] & Paired diff. vs chain [95\\% CI] & Verdict & People better & Worst person & Null mean (draws) & AUROC \\\\',
              body, 'p{3.6cm}lllcccc')

    # --- stress, TM with CI
    cols = ['0', '3', '4', '5', 'X1']
    body = []
    for title, ks in groups():
        body.append(f'\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\')
        for k in ks:
            cells = []
            for lv in cols:
                r = T.get(k, lv)
                cells.append(f"{tm(fnum(r['tm_bacc']))} {ci(fnum(r['tm_lo']), fnum(r['tm_hi']))}" if r else 'n/a')
            body.append(f'{nm(k)} & ' + ' & '.join(cells) + ' \\\\')
    write_tab('tab_stress', 'Model & L0 clean & L3 & L4 & L5 & X1 (out of family) \\\\', body, 'p{3.4cm}lllll',
              font='\\scriptsize', sep='2.5pt')

    # --- stress, paired differences
    body = []
    for title, ks in groups():
        ks = [k for k in ks if k != 'chain_v1']
        if not ks:
            continue
        body.append(f'\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\')
        for k in ks:
            cells = []
            for lv in cols:
                r = T.get(k, lv)
                if not r or not r.get('diff_lo'):
                    cells.append('n/a')
                    continue
                bt = f"{int(float(r['n_people_improved']))}/{int(float(r['n_people_compared']))}" if r.get('n_people_compared') else ''
                cells.append(f"{sg(fnum(r['diff_vs_chain_v1']))} {sci(fnum(r['diff_lo']), fnum(r['diff_hi']))} {r['verdict']} {bt}")
            body.append(f'{nm(k)} & ' + ' & '.join(cells) + ' \\\\')
    write_tab('tab_paired', 'Model & L0 & L3 & L4 & L5 & X1 \\\\', body, 'p{3.0cm}lllll', font='\\tiny', sep='2pt')

    # --- all levels, few models
    show = [k for k in ['chain_v1', win, 'ctl_dc_lr', 'distill', 'twin_inverse'] if T.has(k, '0')]
    show = list(dict.fromkeys(show))
    body = []
    for k in show:
        body.append(f'{nm(k)} & ' + ' & '.join(tm(T.val(k, lv, 'tm_bacc')) for lv in LEVELS) + ' \\\\')
    write_tab('tab_levels', 'Model & L0 & L1 & L2 & L3 & L4 & L5 & X1 \\\\', body, 'p{4cm}ccccccc')

    # --- abstention
    body = []
    for k in [k for k in ['chain_v1', win, 'ctl_dc_lr', 'distill', 'oxyseq'] if T.has(k, '0')]:
        cells = []
        for lv in ['0', '3', '4', '5', 'X1']:
            r = T.get(k, lv)
            cells.append(f"{tm(fnum(r['coverage']), 2)} / {tm(fnum(r['acc_ans']))}" if r else 'n/a')
        body.append(f'{nm(k)} & ' + ' & '.join(cells) + ' \\\\')
    seen_ = set()
    body = [b for b in body if not (b in seen_ or seen_.add(b))]
    write_tab('tab_abstain', 'Model & L0 & L3 & L4 & L5 & X1 \\\\', body, 'p{3.8cm}lllll')

    # --- per person
    body = []
    pk = [k for k in ['chain_v1', win, 'ctl_dc_lr', 'distill'] if T.has(k, '0')]
    for k in dict.fromkeys(pk):
        body.append(f'{nm(k)} & ' + ' & '.join(tm(T.val(k, '0', f'tm_{p}')) for p in PEOPLE) + ' \\\\')
    write_tab('tab_person', 'Model & P1 & P2 & P3 & P4 & P5 & P6 \\\\', body, 'p{4cm}cccccc')

    # --- ablations
    body = []
    for parent, kids in ABLATION_OF.items():
        if not T.has(parent):
            continue
        for k in [parent] + kids:
            if not T.has(k):
                continue
            r = T.get(k, '0')
            body.append(f"{nm(k)} & {tm(fnum(r['tm_bacc']))} {ci(fnum(r['tm_lo']), fnum(r['tm_hi']))} & {sg(fnum(r['diff_vs_chain_v1'])) if k != 'chain_v1' else ''} {sci(fnum(r['diff_lo']), fnum(r['diff_hi']))} & {tm(T.val(k, 'X1', 'tm_bacc'))} & {tm(T.val(k, '4', 'tm_bacc'))} \\\\")
        body.append('\\addlinespace')
    # unknown variants of any shipped / round-3 model (e.g. win_noaug) appear in the full table of the appendix
    write_tab('tab_ablate', 'Variant & L0 TM [95\\% CI] & Paired diff. vs chain & X1 TM & L4 TM \\\\', body, 'p{4.2cm}lllc')

    # --- nulls
    body = []
    done = set()
    for gk, keys in ROUND_OF.items():
        for k in keys + ([win] if win not in sum(ROUND_OF.values(), []) else []):
            if k in done:
                continue
            done.add(k)
            nr_ = null_rows(T, k)
            if not nr_:
                continue
            for kind in ('null', 'blank'):
                v = [x for _, x, kk in nr_ if kk == kind and x is not None]
                if v:
                    flag = 'ok' if sum(v) / len(v) <= 0.55 else 'NOT PASSED'
                    body.append(f"{nm(k)} & {kind} & {len(v)} & {sum(v) / len(v):.3f} & {max(v):.3f} & {flag} \\\\")
    write_tab('tab_nulls', 'Model & Control & Draws & Mean TM & Max TM & Rule (mean $\\le$ 0.55) \\\\', body, 'p{4.4cm}lcccl')

    # --- full table for the appendix
    skip = re.compile(r'(_null|_blank|_perm_null|_null_d\d+|_s\d+$|_seed\d+$)')
    allm = [m for m in T.models if not skip.search(m) and T.has(m, '0')]
    order = {k: i for i, k in enumerate(sum(ROUND_OF.values(), []))}
    allm.sort(key=lambda m: (order.get(m, 999), m))
    body = []
    for k in allm:
        r0 = T.get(k, '0')
        body.append(f"\\texttt{{{esc(k)}}} & {tm(fnum(r0['tm_bacc']))} {ci(fnum(r0['tm_lo']), fnum(r0['tm_hi']))} & {sg(fnum(r0['diff_vs_chain_v1'])) if r0.get('diff_lo') else ''} {r0.get('verdict', '')} & {tm(T.val(k, '3', 'tm_bacc'))} & {tm(T.val(k, '4', 'tm_bacc'))} & {tm(T.val(k, '5', 'tm_bacc'))} & {tm(T.val(k, 'X1', 'tm_bacc'))} \\\\")
    hdr = 'Model key & L0 TM [95\\% CI] & vs chain & L3 & L4 & L5 & X1 \\\\'
    lt = ['% GENERATED by make_numbers.py (tab_full)', '{\\scriptsize\\setlength{\\tabcolsep}{3pt}',
          '\\begin{longtable}{@{}p{3.4cm}p{3.3cm}p{2.7cm}cccc@{}}', '\\caption{All models and variants on the harness (key as in \\texttt{TABLE.csv}; paired verdict to the chain at L0; TM at the other conditions).}\\label{tab:full}\\\\',
          '\\toprule', hdr, '\\midrule', '\\endfirsthead', '\\toprule', hdr, '\\midrule', '\\endhead', '\\bottomrule', '\\endfoot']
    lt += body + ['\\end{longtable}', '}']
    (out / 'tables' / 'tab_full.tex').write_text('\n'.join(lt) + '\n')

    print(f'wrote {out / "numbers.tex"} and {len(list((out / "tables").glob("*.tex")))} tables')


if __name__ == '__main__':
    main()
