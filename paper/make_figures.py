#!/usr/bin/env python3
"""Draw paper/figures/fig_stress.pdf and fig_forest.pdf from results/photon/TABLE.csv (harness numbers only).

Needs matplotlib (not in the repo's requirements; `python3 -m venv ~/.venvs/paper && ~/.venvs/paper/bin/pip install matplotlib`).
  ~/.venvs/paper/bin/python paper/make_figures.py [--winner KEY]
The winner is resolved exactly as in make_numbers.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_numbers as mn  # noqa: E402

LV = ['0', '1', '2', '3', '4', '5']
COL = {'distill2': '#c0392b', 'chain_v1': '#222222', 'ctl_dc_lr': '#8c564b', 'distill': '#e08e84', 'twin_inverse': '#0072b2', 'oxyseq': '#40a898',
       'photonnet': '#0f7a3a', 'oxyssl': '#8a2060'}


def arch(fig_dir):
    """Schematic of PhotonDistill2 (the student MLP and its training-only teachers). No result numbers in the figure."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec, bold=False, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.35)

    def arrow(x0, y0, x1, y1, color='#444444', style='-|>', ls='-'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.4, ls=ls))

    ax.text(0.5, 54, 'Inference, in the browser (per 10 s window, 2 s hop)', fontsize=8, style='italic', color='#666666')
    box(0.5, 36, 17, 15, 'Phone camera\nper-frame mean\nR, G, B\n(30 Hz, flash on)', '#fff6e5', '#e69f00')
    box(21, 36, 22, 15, '39 causal features,\nrelative to the person\'s\nown label-free first minute\n(19 chain deltas, 15 quality,\n5 respiratory)', '#fff6e5', '#e69f00')
    box(47, 36, 12, 15, 'impute,\nclip,\nstandardise\n(training\nstatistics)', '#f2f2f2', '#808080')
    box(63, 36, 16, 15, 'Student MLP\n39 \u2192 H ReLU\n\u2192 H ReLU \u2192 3\nH nested in\n{16, 32, 64}', '#eef3fa', '#0072b2', bold=True)
    box(83, 36, 16.5, 15, 'Heads\nP(SpO2 < 90)\nSpO2 estimate (hidden)\nabstain logit;\nanswer if p \u2265 \u03c4', '#eef3fa', '#0072b2')
    for x0, x1 in ((17.9, 20.6), (43.4, 46.6), (59.4, 62.6), (79.4, 82.6)):
        arrow(x0, 43.5, x1, 43.5)
    ax.text(0.5, 29, 'Training only (inside each outer fold: the five training people; the held-out person is never seen)', fontsize=8, style='italic', color='#666666')
    box(0.5, 8, 17, 17, 'Clean training\nrecordings', '#fff6e5', '#e69f00')
    box(21, 8, 22, 17, 'Physics-sampled\ndistant-flash copies\n(extra separation, leak,\nread noise, channel loss;\nown RNG, disjoint from\nthe stress levels and X1)', '#f3ecf7', '#cc79a7')
    box(47, 8, 21, 17, 'Teachers (refit per fold)\nchain v1 \u2192 soft targets,\nclean rows\nDC-colour LR \u2192 soft targets,\ndegraded rows\n(tree teacher: never chosen)', '#fbe9e7', '#d55e00')
    box(72, 8, 27.5, 17, 'Loss = BCE(label)\n+ \u03bb T\u00b2 BCE(student, teacher)\n+ 0.3 Huber(SpO2) + 0.5 BCE(abstain)\n\u03bb in {8, 16, 32, pure KD}, T in {2, 4},\nwidth, weight decay, epochs, \u03c4\nby leave-one-inner-person-out CV', '#fbe9e7', '#d55e00')
    for x0, x1 in ((17.9, 20.6), (43.4, 46.6), (68.4, 71.6)):
        arrow(x0, 16.5, x1, 16.5)
    arrow(85.7, 25.7, 71.5, 35.3, color='#d55e00', ls='--')
    ax.text(79.5, 29.7, 'gradients (training only)', fontsize=6.5, color='#d55e00', style='italic')
    ax.text(0.5, 2, 'Only the student runs at inference: no teacher, no gate, no chain. Three seeds are averaged. Schematic of the model code in python/oxydrop_ml/photon/.', fontsize=6.5, color='#555555')
    fig.savefig(fig_dir / 'fig_arch.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--winner')
    ap.add_argument('--locked', action='store_true')
    ap.add_argument('--table', default=str(mn.RES / 'TABLE.csv'))
    a = ap.parse_args()
    T = mn.Table(a.table)
    win = a.winner or mn.parse_winner(mn.RES / 'WINNER2.md') or 'distill'
    fig_dir = HERE / 'figures'
    fig_dir.mkdir(exist_ok=True)

    lines = ['chain_v1', win, 'ctl_dc_lr']
    for extra in ('distill',):
        if extra not in lines and T.has(extra):
            lines.append(extra)
    lines = [k for k in lines if T.has(k)]
    name = {k: mn.MODELS.get(k, ('', k))[1] for k in lines}
    name[win] = name[win] + ' (shipped)'

    # ---------------------------------------------------------------- stress curves
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    xs = list(range(6))
    xx = 7.2
    for k in lines:
        c = COL.get(k, '#555555')
        lw = 2.4 if k == win else 1.5
        ls = '--' if k in ('ctl_dc_lr', 'distill') and k != win else '-'
        y = [T.val(k, lv, 'tm_bacc') for lv in LV]
        ax[0].plot(xs, y, color=c, lw=lw, ls=ls, marker='o', ms=3.5, label=name[k])
        if k in ('chain_v1', win):
            lo = [T.val(k, lv, 'tm_lo') for lv in LV]
            hi = [T.val(k, lv, 'tm_hi') for lv in LV]
            ax[0].fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
        yx, lox, hix = T.val(k, 'X1', 'tm_bacc'), T.val(k, 'X1', 'tm_lo'), T.val(k, 'X1', 'tm_hi')
        if yx is not None:
            off = {'chain_v1': -0.25, 'ctl_dc_lr': 0.25}.get(k, 0.0)
            ax[0].errorbar([xx + off], [yx], yerr=[[yx - lox], [hix - yx]], fmt='D', color=c, ms=5, capsize=2, lw=1.2)
        ya = [T.val(k, lv, 'acc_ans') for lv in LV]
        ax[1].plot(xs, ya, color=c, lw=lw, ls=ls, marker='o', ms=3.5)
        v = T.val(k, 'X1', 'acc_ans')
        if v is not None:
            ax[1].plot([xx], [v], 'D', color=c, ms=5)
        yc = [T.val(k, lv, 'coverage') for lv in LV]
        ax[2].plot(xs, yc, color=c, lw=lw, ls=ls, marker='o', ms=3.5)
        v = T.val(k, 'X1', 'coverage')
        if v is not None:
            ax[2].plot([xx], [v], 'D', color=c, ms=5)
    ax[0].axhline(0.5, color='gray', ls=':', lw=1)
    ax[0].text(0.05, 0.505, 'chance', color='gray', fontsize=7)
    ax[1].axhline(0.9, color='gray', ls='--', lw=0.8)
    titles = ['(a) TM balanced accuracy, forced decision', '(b) Accuracy on answered windows', '(c) Coverage (share answered)']
    ylabels = ['TM balanced accuracy (95% person CI)', 'Accuracy of answered windows', 'Share of windows answered']
    for i, axx in enumerate(ax):
        axx.set_xticks(xs + [xx])
        axx.set_xticklabels([f'L{i_}\n+{2 * i_} mm' for i_ in xs] + ['X1'], fontsize=7)
        axx.axvline(6.35, color='gray', lw=0.8)
        axx.set_title(titles[i], fontsize=9, loc='left')
        axx.set_ylabel(ylabels[i], fontsize=8)
        axx.set_xlabel('Simulated flash-to-lens offset (level)', fontsize=8)
        axx.grid(alpha=0.25)
        axx.spines[['top', 'right']].set_visible(False)
        axx.tick_params(axis='y', labelsize=8)
    ax[0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig_stress.pdf')
    plt.close(fig)

    # ---------------------------------------------------------------- forest plot at L0
    rows = []
    for gk, keys in mn.ROUND_OF.items():
        ks = [k for k in keys if T.has(k)]
        if win not in sum(mn.ROUND_OF.values(), []) and gk == 'round3':
            ks = ks + [win]
        if ks:
            rows.append(('hdr', mn.GROUP_TITLE[gk]))
            rows += [('m', k) for k in ks if k not in ('chain_v1_nestedC', 'twin_inverse_noresid', 'flashgap_gate')]
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(rows) + 1.0))
    y = len(rows)
    ref = T.val('chain_v1', '0', 'tm_bacc')
    ax.axvspan(T.val('chain_v1', '0', 'tm_lo'), T.val('chain_v1', '0', 'tm_hi'), color='#eeeeee', zorder=0)
    ax.axvline(ref, color='k', ls='--', lw=1)
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    labels, ticks = [], []
    for kind, k in rows:
        y -= 1
        if kind == 'hdr':
            ax.text(0.205, y, k, fontsize=8, fontweight='bold', va='center')
            continue
        v, lo, hi = (T.val(k, '0', c) for c in ('tm_bacc', 'tm_lo', 'tm_hi'))
        if v is None:
            continue
        shipped = (k == win)
        ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt='*' if shipped else 'o', ms=10 if shipped else 5,
                    color='#c0392b' if shipped else '#333333', capsize=2, lw=1.8 if shipped else 1.1)
        labels.append(mn.MODELS.get(k, ('', k))[1] + (' [shipped]' if shipped else ''))
        ticks.append(y)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel('Clean-light (L0) time-matched balanced accuracy, 95% person-bootstrap CI', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig_forest.pdf')
    plt.close(fig)
    arch(fig_dir)
    print('shipped model:', win, '; wrote fig_stress.pdf, fig_forest.pdf, fig_arch.pdf')


if __name__ == '__main__':
    main()
