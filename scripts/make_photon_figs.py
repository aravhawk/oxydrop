"""Figures for the PhotonTwin / FlashGap / OxyTrack comparison, drawn only from results/photon/TABLE.csv and STRESS.csv.

    python scripts/make_photon_figs.py                          # all three figures, architecture of --winner (default twin_inverse)
    python scripts/make_photon_figs.py --winner oxytrack --mark-winner   # architecture layout of the winner, highlighted in the other two
    python scripts/make_photon_figs.py results stress           # a subset: results | stress | architecture
    python scripts/make_photon_figs.py --all-variants           # stress curves also for ablations and controls

Outputs (docs/figures/):
    photon_results_ci.png     forest plot of L0 time-matched balanced accuracy with person-bootstrap 95 % CIs, baselines greyed
    photon_stress_curve.png   (a) TM bal. acc. vs simulated distant-flash level with CI bands, (b) answered accuracy and coverage
    photon_architecture.png   block diagram of the model chosen with --winner (twin_inverse | flashgap | oxytrack)

Column names are read from the CSV header (tm_bacc, tm_lo, tm_hi, acc_ans, coverage, ...), nothing is recomputed. The stress levels are
a simulation (python/oxydrop_ml/photon/stress.py), not measurements of any phone. The architecture drawing is a schematic, not data.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'photon')
FIG = os.path.join(ROOT, 'docs', 'figures')
BLUE, ORANGE, GREEN, VERM, GREY, PURPLE, SKY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#7F7F7F', '#CC79A7', '#56B4E9'
DARK, LIGHT = '#444444', '#BBBBBB'
plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'savefig.dpi': 200, 'font.family': 'DejaVu Sans', 'legend.frameon': False})
NOTE = 'Research prototype, not a medical device. Stress levels are a simulation of a distant flash, not measurements of any phone.'

BASELINES = {'chain_v1': 'Logistic chain v1 (reference)', 'chain_v1_nestedC': 'Chain v1, C by nested CV', 'hybrid_final': 'Hybrid, per-person selected (nested)',
             'cnn_full': 'CNN alone (3-seed mean)'}
BASE_STYLE = {'chain_v1': '#222222', 'chain_v1_nestedC': '#6B6B6B', 'hybrid_final': '#8F8F8F', 'cnn_full': '#B0B0B0'}
FAMILIES = {'twin_inverse': ('PhotonTwin-Inverse', BLUE), 'flashgap': ('FlashGap', ORANGE), 'oxytrack': ('OxyTrack', GREEN)}
REFERENCE = 'chain_v1'


# ----------------------------------------------------------------------------------------------------------------------------- data
def load():
    t = pd.read_csv(os.path.join(RES, 'TABLE.csv'), dtype={'level': str})
    s = pd.read_csv(os.path.join(RES, 'STRESS.csv'), dtype={'level': str})
    return t, s


def family_of(model):
    for k in FAMILIES:
        if model == k or model.startswith(k + '_'):
            return k
    return None


def is_control(model):
    return model.endswith('_null') or model.endswith('_blank') or '_null' in model or '_blank' in model


def pretty(model):
    if model in BASELINES:
        return BASELINES[model]
    fam = family_of(model)
    if fam is None:
        return model
    base = FAMILIES[fam][0]
    rest = model[len(fam):].lstrip('_')
    return base if not rest else f'{base} [{rest}]'


def kind(model):
    if model in BASELINES:
        return 'baseline'
    if is_control(model):
        return 'control'
    if family_of(model) and model in FAMILIES:
        return 'candidate'
    return 'variant'


def style(model, winner=None):
    """colour, marker face, alpha, linewidth"""
    k = kind(model)
    if k == 'baseline':
        return BASE_STYLE[model], BASE_STYLE[model], 0.9, 1.4
    fam = family_of(model)
    col = FAMILIES[fam][1] if fam else PURPLE
    if k == 'candidate':
        return col, col, 1.0, 2.4 if model == winner else 1.8
    return col, 'white', 0.75, 1.1


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', os.path.relpath(p, ROOT))


# ------------------------------------------------------------------------------------------------------------------------- 1. forest
def fig_results(winner=None):
    t, _ = load()
    d = t[t['level'] == '0'].copy()
    ref = d[d['model'] == REFERENCE]
    order_key = {'baseline': 0, 'candidate': 1, 'variant': 2, 'control': 3}
    d['k'] = d['model'].map(kind)
    d['fam'] = d['model'].map(lambda m: list(FAMILIES).index(family_of(m)) if family_of(m) else -1)
    d['bo'] = d['model'].map(lambda m: list(BASELINES).index(m) if m in BASELINES else 0)
    d['ko'] = d['k'].map(order_key)
    d = d.sort_values(['ko', 'fam', 'bo', 'model']).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(10.5, 0.46 * n + 1.9))
    ys = np.arange(n)[::-1]
    if len(ref):
        r = ref.iloc[0]
        ax.axvspan(r['tm_lo'], r['tm_hi'], color='#DDDDDD', alpha=0.45, lw=0, zorder=0, label=f'chain_v1 95 % CI [{r["tm_lo"]:.3f}, {r["tm_hi"]:.3f}]')
        ax.axvline(r['tm_bacc'], color='#222222', lw=1.3, ls='--', zorder=1, label=f'chain_v1 = {r["tm_bacc"]:.3f}')
    ax.axvline(0.5, color=VERM, lw=1.0, ls=':', zorder=1, label='clock only / chance = 0.50')
    for y, (_, row) in zip(ys, d.iterrows()):
        col, face, alpha, lw = style(row['model'], winner)
        ax.errorbar(row['tm_bacc'], y, xerr=[[row['tm_bacc'] - row['tm_lo']], [row['tm_hi'] - row['tm_bacc']]], fmt='o' if kind(row['model']) != 'control' else 's',
                    color=col, mfc=face, mec=col, ms=7.5, elinewidth=lw, capsize=3.5, alpha=alpha, zorder=3)
        txt = f'{row["tm_bacc"]:.3f} [{row["tm_lo"]:.3f}, {row["tm_hi"]:.3f}]'
        v = row.get('verdict')
        if isinstance(v, str) and v != 'reference':
            txt += f'  {v}'
        ax.text(1.005, y, txt, transform=ax.get_yaxis_transform(), va='center', fontsize=8.6, color='#222' if kind(row['model']) != 'baseline' else '#555')
    ax.set_yticks(ys)
    labels = [pretty(m) + ('  (winner)' if m == winner else '') for m in d['model']]
    ax.set_yticklabels(labels)
    for tl, m in zip(ax.get_yticklabels(), d['model']):
        tl.set_color(style(m, winner)[0] if kind(m) != 'baseline' else '#555555')
        if m == winner:
            tl.set_fontweight('bold')
    ax.set_xlim(0.40, 0.95)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel('Time-matched balanced accuracy, clean L0 (Hoffman, leave-one-person-out), 95 % person-bootstrap CI')
    ax.set_title('Photon models against the baselines on clean data: tie unless the paired interval to chain_v1 excludes 0', loc='left', fontsize=11)
    ax.grid(axis='y', alpha=0)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, lab, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=8.6)
    ax.text(0.0, -0.13 - 0.18 / max(n / 10, 1), 'Baselines in grey. Open markers = ablations and variants, squares = null / blank controls (should sit near 0.50).\n'
            'hybrid_final and cnn_full exist at L0 only (no leave-one-person-out checkpoints under stress). Six people, one dark-skinned (n = 1): no skin-tone claim.\n' + NOTE,
            transform=ax.transAxes, fontsize=8, color='#555', va='top')
    save(fig, 'photon_results_ci.png')


# ------------------------------------------------------------------------------------------------------------------------- 2. stress
def fig_stress(winner=None, all_variants=False):
    _, s = load()
    levels = [str(i) for i in range(6)]
    xpos = {**{lv: int(lv) for lv in levels}, 'X1': 6.6}
    models = []
    for m in s['model'].unique():
        k = kind(m)
        if k in ('baseline', 'candidate') or all_variants:
            models.append(m)
    models.sort(key=lambda m: ({'baseline': 0, 'candidate': 1, 'variant': 2, 'control': 3}[kind(m)], list(BASELINES).index(m) if m in BASELINES else 0, m))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={'width_ratios': [1.15, 1]})
    axa, axb = axes
    for m in models:
        sub = s[s['model'] == m]
        col, face, alpha, lw = style(m, winner)
        ls = '-' if kind(m) in ('baseline', 'candidate') else '--'
        lab = pretty(m) + ('  (winner)' if m == winner else '')
        main = sub[sub['level'].isin(levels)].copy()
        main['x'] = main['level'].map(xpos)
        main = main.sort_values('x')
        if len(main) >= 2:
            axa.fill_between(main['x'], main['tm_lo'], main['tm_hi'], color=col, alpha=0.10 if kind(m) == 'baseline' else 0.14, lw=0)
        axa.plot(main['x'], main['tm_bacc'], ls=ls, color=col, lw=lw, marker='o', ms=5, mfc=face, alpha=alpha, label=lab)
        x1 = sub[sub['level'] == 'X1']
        if len(x1):
            r = x1.iloc[0]
            axa.errorbar([xpos['X1']], [r['tm_bacc']], yerr=[[r['tm_bacc'] - r['tm_lo']], [r['tm_hi'] - r['tm_bacc']]], fmt='D', color=col, mfc=face, ms=7, capsize=3,
                         elinewidth=lw, alpha=alpha)
        for c, ls2, mk in (('acc_ans', '-', 'o'), ('coverage', ':', 's')):
            if c not in sub.columns:
                continue
            mm = sub[sub['level'].isin(levels)].copy()
            mm['x'] = mm['level'].map(xpos)
            mm = mm.sort_values('x').dropna(subset=[c])
            axb.plot(mm['x'], mm[c], ls=ls2 if kind(m) in ('baseline', 'candidate') else '-.', color=col, lw=lw if c == 'acc_ans' else max(lw - 0.4, 0.9), marker=mk, ms=4.2,
                     mfc=face, alpha=alpha)
            xx = sub[(sub['level'] == 'X1')].dropna(subset=[c])
            if len(xx):
                axb.plot([xpos['X1']], [xx.iloc[0][c]], marker='D' if c == 'acc_ans' else 's', color=col, mfc=face, ms=6, alpha=alpha)
    ticks = [0, 1, 2, 3, 4, 5, xpos['X1']]
    d_rho = s.drop_duplicates('level').set_index('level')['d_rho_mm'].to_dict() if 'd_rho_mm' in s.columns else {}
    tl = [f'{lv}\n+{int(float(d_rho.get(lv, 2 * int(lv))))} mm' for lv in levels] + ['X1\nout-of-\nfamily']
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl)
        ax.axvline(5.8, color='#999', lw=0.8, ls='-')
        ax.set_xlim(-0.3, 7.1)
        ax.set_xlabel('Simulated flash-to-lens offset d_rho (level)')
    axa.axhline(0.5, color=VERM, lw=1, ls=':')
    axa.text(-0.25, 0.502, 'chance', color=VERM, fontsize=8, ha='left', va='bottom')
    axa.set_ylabel('Time-matched balanced accuracy (95 % person CI)')
    axa.set_ylim(0.38, 0.95)
    axa.set_title('(a) Accuracy against a simulated distant flash', loc='left')
    axa.legend(fontsize=8.2, loc='upper right')
    axb.axhline(0.9, color='#888', lw=0.8, ls='--')
    axb.set_ylim(0.0, 1.03)
    axb.set_ylabel('Answered accuracy (solid) and coverage (dotted)')
    axb.set_title('(b) Own abstention: answered accuracy and coverage', loc='left')
    # style key for panel b
    axb.plot([], [], color='#333', ls='-', marker='o', ms=4, label='answered accuracy')
    axb.plot([], [], color='#333', ls=':', marker='s', ms=4, label='coverage')
    axb.plot([], [], color='#333', ls='', marker='D', ms=5, label='X1 point')
    axb.plot([], [], color='#888', ls='--', lw=0.8, label='target answered accuracy 0.90')
    axb.legend(fontsize=8.2, loc='lower left')
    fig.suptitle('Distant-flash stress test (simulation, only the held-out person is degraded, models not refit per level)', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, -0.02, NOTE, fontsize=8, color='#555')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, 'photon_stress_curve.png')


# ------------------------------------------------------------------------------------------------------------------------ 3. architecture
class Diagram:
    def __init__(self, title, subtitle, figsize=(13, 6.6)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        ax = self.ax
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.grid(False)
        ax.text(0.0, 0.995, title, fontsize=13, fontweight='bold', va='top')
        ax.text(0.0, 0.945, subtitle, fontsize=9.5, va='top', color='#444')
        self.b = {}

    def box(self, key, x, y, w, h, text, fc='#EEF4FA', ec=BLUE, fs=9, bold=False, ls='-'):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.012', fc=fc, ec=ec, lw=1.5, ls=ls))
        self.ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.35)
        self.b[key] = (x, y, w, h)

    def side(self, key, where):
        x, y, w, h = self.b[key]
        return {'l': (x, y + h / 2), 'r': (x + w, y + h / 2), 't': (x + w / 2, y + h), 'b': (x + w / 2, y)}[where]

    def arrow(self, a, sa, b, sb, color='#444', lw=1.5, ls='-', label=None, rad=0.0, lx=0.0, ly=0.015):
        p0, p1 = self.side(a, sa), self.side(b, sb)
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13, lw=lw, color=color, ls=ls, shrinkA=1, shrinkB=1,
                                          connectionstyle=f'arc3,rad={rad}'))
        if label:
            self.ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label, fontsize=8, color=color, ha='center', va='bottom', style='italic')

    def elbow(self, pts, color='#444', lw=1.5, ls='-', label=None, lpos=None, fs=8):
        """polyline through pts with an arrow head on the last segment"""
        xs, ys = zip(*pts[:-1])
        self.ax.plot(xs + (pts[-1][0],), ys + (pts[-1][1],), color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=1)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=13, lw=lw, color=color, shrinkA=0, shrinkB=1))
        if label:
            self.ax.text(*lpos, label, fontsize=fs, color=color, ha='center', va='bottom', style='italic')

    def tag(self, x, y, text, color=GREY, fs=8.5):
        self.ax.text(x, y, text, fontsize=fs, color=color, style='italic', va='bottom')

    def done(self, name):
        self.ax.text(0.0, 0.0, NOTE.split(' Stress')[0] + ' Schematic of the model code in python/oxydrop_ml/photon/, not data.', fontsize=8, color='#555', va='bottom')
        save(self.fig, name)


CREAM, MINT, LAV, ROSE = '#FFF6E5', '#E8F5EE', '#F4ECF7', '#FBEAE6'


def arch_twin_inverse():
    d = Diagram('PhotonTwin-Inverse: infer the physical state of the finger, decode it with a differentiable optics model',
                'The flash-to-lens distance rho is a latent inferred from the signal, not assumed. Leave-one-person-out, every prior and scaler fitted inside the outer fold.')
    w, h, g = 0.15, 0.25, 0.02
    xs = [i * (w + g) for i in range(6)]
    y1 = 0.52
    d.box('rgb', xs[0], y1, w, h, 'Phone camera\nper-frame ROI mean\nR, G, B\n(30 Hz, flash on)', CREAM, ORANGE, fs=8.5)
    d.box('obs', xs[1], y1, w, h, 'Observables\nper 10 s window,\nbaseline-relative\nln DC, ln |AC| (R,G,B)\nln RoR RG, RB, GB\nperfusion index, SNR', CREAM, ORANGE, fs=8.2)
    d.box('enc', xs[2], y1, w, h, 'Encoder\nsmall neural net:\nobservables to the\nstate of the finger\n(window + first-\nminute baseline)', '#EEF4FA', BLUE, fs=8.5)
    d.box('lat', xs[3], y1, w, h, 'Physical latents\nSpO2 (saturation)\nblood volume\nmelanin, pulse swing\nflash-lens\ndistance rho', LAV, PURPLE, fs=8.5, bold=True)
    d.box('ren', xs[4], y1, w, h, 'PhotonTwin\nrenderer\ndifferentiable\nband-integrated\ndiffusion optics,\nphone spectral bands', LAV, PURPLE, fs=8.5, bold=True)
    d.box('rec', xs[5], y1, w, h, 'Reconstructed\nobservables\n(compared with\nthe measured ones)', CREAM, ORANGE, fs=8.5)
    for a, b in zip(['rgb', 'obs', 'enc', 'lat', 'ren'], ['obs', 'enc', 'lat', 'ren', 'rec']):
        d.arrow(a, 'r', b, 'l')
    top = y1 + h
    px0, px1 = d.side('rec', 't')[0], d.side('obs', 't')[0]
    d.elbow([(px0, top), (px0, top + 0.06), (px1, top + 0.06), (px1, top + 0.005)], VERM, ls='--', label='reconstruction likelihood (heteroscedastic)', lpos=((px0 + px1) / 2, top + 0.065))
    y2, h2 = 0.14, 0.22
    d.box('res', xs[1], y2, w, h2, 'Residual head\nheart-rate rise, chain\ndeltas (what optics\ncannot see); L2-\npenalised, share\nreported', '#EEF4FA', BLUE, fs=8.2)
    d.box('logit', xs[2], y2, w, h2, 'P(SpO2 < 90)\nlogit = physics path\n(saturation latent)\n+ residual', '#EEF4FA', BLUE, fs=8.5, bold=True)
    d.box('wid', xs[3], y2, w, h2, 'Posterior width\nof saturation:\nwidens when light\nis weak (distant\nflash)', LAV, PURPLE, fs=8.5)
    d.box('abs', xs[4], y2, w, h2, 'Abstention\nanswer only if width\nis below a threshold\nchosen by inner CV\non training people', '#EEF4FA', BLUE, fs=8.2)
    d.box('out', xs[5], y2, w, h2, 'Output\nNormal vs your\nbaseline / Possible\nlow oxygen / Unsure,\nre-measure', MINT, GREEN, fs=8.2, bold=True)
    d.arrow('lat', 'b', 'logit', 't', label='saturation', lx=-0.045)
    d.arrow('lat', 'b', 'wid', 't', label='width', lx=0.0)
    d.arrow('obs', 'b', 'res', 't')
    d.arrow('res', 'r', 'logit', 'l')
    d.arrow('wid', 'r', 'abs', 'l')
    lx, ax_ = d.side('logit', 'b')[0], d.side('abs', 'b')[0]
    d.elbow([(lx, y2), (lx, 0.095), (ax_, 0.095), (ax_, y2 - 0.003)])
    d.arrow('abs', 'r', 'out', 'l')
    d.tag(xs[4], 0.455, 'rho rises with the simulated flash offset:\nthe geometry-awareness check', PURPLE, fs=8)
    d.tag(0.0, 0.035, 'Controls: noresid (physics path only), null (other-person observables aligned to t), shifted-label run (>= 200 s).', GREY)
    d.done('photon_architecture.png')


def arch_flashgap():
    d = Diagram('FlashGap: learn to read a distant-flash signal, and say "unsure" when it cannot be read',
                'Physics-sampled degradation at training time only. At inference the classifier sees real features plus how much signal there is. Leave-one-person-out.')
    w, h, g = 0.172, 0.25, 0.035
    xs = [i * (w + g) for i in range(5)]
    y1, y2 = 0.53, 0.14
    d.tag(0.0, y1 + h + 0.015, 'Training (five clean people per fold)', GREY, fs=10)
    d.box('clean', xs[0], y1, w, h, 'Clean training\nrecordings\nper-frame RGB', CREAM, ORANGE, fs=8.8)
    d.box('aug', xs[1], y1, w + 0.02, h, 'Physics-sampled\ndistant-flash\naugmentation: d_rho\nU(0,12) mm, leak, noise,\nAE gain, channel kill\n(own RNG stream)', LAV, PURPLE, fs=8.2, bold=False)
    d.box('feat_t', xs[2] + 0.02, y1, w + 0.02, h, 'Features + SNR\n19 chain deltas + dHR\nper-channel SNR, PI,\nDC ratio, clipping,\nbeat fraction\nweights: clean 1,\naugmented 0.25', '#EEF4FA', BLUE, fs=8.2, ls='--')
    d.box('note', xs[3] + 0.04, y1, 2 * w + g - 0.04, h, 'Not used in training:\nthe stress levels L1-L5 and X1\n(augmentation seeds are\ndisjoint from the evaluation\n(level, seed) pairs)', '#F2F2F2', GREY, fs=8.2, ls='--')
    d.arrow('clean', 'r', 'aug', 'l')
    d.arrow('aug', 'r', 'feat_t', 'l')
    d.tag(0.0, y2 + h + 0.015, 'Inference (held-out person, real or degraded)', GREY, fs=10)
    d.box('rgb', xs[0], y2, w, h, 'Phone camera\nper-frame ROI mean\nR, G, B', CREAM, ORANGE, fs=8.8)
    d.box('feat', xs[1], y2, w, h, 'Features + SNR\n19 chain deltas + dHR\nper-channel SNR, PI,\nDC ratio, clipping,\nbeat fraction', '#EEF4FA', BLUE, fs=8.2)
    d.box('clf', xs[2], y2, w, h, 'Classifier\nlogistic regression\n(feature x SNR terms)\nor small LightGBM,\nnested inner CV', '#EEF4FA', BLUE, fs=8.2, bold=True)
    d.box('cp', xs[3], y2, w, h, 'Conformal risk\ncontrol: threshold\nfrom inner CV,\nanswered acc >= 0.90,\nconf from prob + SNR', LAV, PURPLE, fs=8.2, bold=True)
    d.box('out', xs[4], y2, w, h, 'Output\nNormal vs your\nbaseline / Possible\nlow oxygen / Unsure,\nre-measure', MINT, GREEN, fs=8.2, bold=True)
    for a, b in zip(['rgb', 'feat', 'clf', 'cp'], ['feat', 'clf', 'cp', 'out']):
        d.arrow(a, 'r', b, 'l')
    fx = d.side('clf', 't')[0]
    d.elbow([(d.side('feat_t', 'b')[0], y1), (d.side('feat_t', 'b')[0], (y1 + y2 + h) / 2), (fx, (y1 + y2 + h) / 2), (fx, y2 + h + 0.003)], label='fit', lpos=(fx - 0.07, (y1 + y2 + h) / 2 + 0.005))
    d.tag(0.0, 0.05, 'Controls: noaug (same features, no augmentation), nosnr (no SNR features), null. Stress X1 is never sampled by the augmentation.', GREY)
    d.done('photon_architecture.png')


def arch_oxytrack():
    d = Diagram('OxyTrack: a causal personal state-space tracker that leans on its prior when the light is weak',
                'Saturation is a slowly varying latent. Each window is a noisy observation weighted by its measured SNR. Window k uses windows <= k only.')
    w, h, g = 0.172, 0.25, 0.035
    xs = [i * (w + g) for i in range(5)]
    y1, y2, h2 = 0.50, 0.14, 0.22
    d.box('rgb', xs[0], y1, w, h, 'Phone camera\nper-frame ROI mean\nR, G, B', CREAM, ORANGE, fs=8.8)
    d.box('ev', xs[1], y1, w, h, 'Per-window evidence\n10 s window, 2 s hop\ndHR, chain deltas\n(projection),\nSNR of the window', CREAM, ORANGE, fs=8.2)
    d.box('em', xs[2], y1, w, h, 'Emission model\nweights pooled from\ntraining people +\nper-person offset and\nscale (label-free)', '#EEF4FA', BLUE, fs=8.2)
    d.box('kf', xs[3], y1, w, h, 'SNR-weighted\ncausal state-space\nfilter: observation\nnoise = f(SNR),\ndynamics prior from\ntraining trajectories', LAV, PURPLE, fs=8.2, bold=True)
    d.box('band', xs[4], y1, w, h, 'Posterior\nSpO2 mean and\n80 % band\nevery 2 s', '#EEF4FA', BLUE, fs=8.5)
    for a, b in zip(['rgb', 'ev', 'em', 'kf'], ['ev', 'em', 'kf', 'band']):
        d.arrow(a, 'r', b, 'l')
    x, y, ww, hh = d.b['kf']
    d.elbow([(x + ww * 0.75, y + hh), (x + ww * 0.75, y + hh + 0.05), (x + ww * 0.25, y + hh + 0.05), (x + ww * 0.25, y + hh + 0.003)], PURPLE,
            label='state s(k-1) to s(k)', lpos=(x + ww / 2, y + hh + 0.055))
    d.box('adapt', xs[2], y2, w, h2, 'Test-time adaptation\nfirst 60 s, then only on\nwindows with\nP(normoxic) > 0.95\n(no labels, causal EM)', '#F2F2F2', GREY, fs=8.2)
    d.box('p', xs[3], y2, w, h2, 'P(SpO2 < 90)\nfrom the posterior;\na wide band means\nlow confidence', '#EEF4FA', BLUE, fs=8.2, bold=True)
    d.box('abs', xs[4], y2, w, h2, 'Abstain if P is\nwithin 0.5 +- c\n(c from inner CV\non training people)', LAV, PURPLE, fs=8.2)
    d.box('out', xs[0], y2, w * 2 + g, h2, 'Output (three states)\nNormal vs your baseline /\nPossible low oxygen /\nUnsure, re-measure', MINT, GREEN, fs=9, bold=True)
    d.arrow('adapt', 't', 'em', 'b', ls='--', color=GREY, label='update', lx=0.03, ly=0.0)
    d.arrow('kf', 'b', 'p', 't')
    d.arrow('p', 'r', 'abs', 'l')
    ax_, ox = d.side('abs', 'b')[0], d.side('out', 'b')[0]
    d.elbow([(ax_, y2), (ax_, 0.095), (ox, 0.095), (ox, y2 - 0.003)])
    d.tag(0.0, 0.035, 'Controls: ema (chain v1 probability with a causal moving average), noadapt, nosnr, null. A gain over ema would be more than smoothing.', GREY)
    d.done('photon_architecture.png')


ARCH = {'twin_inverse': arch_twin_inverse, 'flashgap': arch_flashgap, 'oxytrack': arch_oxytrack}


# --------------------------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('figs', nargs='*', help='subset of: results stress architecture (default: all three)')
    ap.add_argument('--mark-winner', action='store_true', help='bold and label the --winner model in the results and stress figures (use only after the winner rule was applied)')
    ap.add_argument('--winner', choices=sorted(ARCH), default='twin_inverse', help='model whose architecture is drawn (default twin_inverse)')
    ap.add_argument('--all-variants', action='store_true', help='stress curves also for ablations and controls')
    a = ap.parse_args(argv)
    todo = a.figs or ['results', 'stress', 'architecture']
    bad = [f for f in todo if f not in ('results', 'stress', 'architecture')]
    if bad:
        ap.error(f'unknown figure(s): {bad}')
    mark = a.winner if a.mark_winner else None
    if 'results' in todo:
        fig_results(mark)
    if 'stress' in todo:
        fig_stress(mark, a.all_variants)
    if 'architecture' in todo:
        ARCH[a.winner]()


if __name__ == '__main__':
    main()
