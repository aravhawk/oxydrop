"""Figures for MODEL.md / WHITEPAPER.md / README.md, drawn only from results/ml/*.csv|json (and the model code for the parameter count).

    python scripts/make_ml_figs.py            # all figures -> docs/figures/ml_*.png
    python scripts/make_ml_figs.py ablation   # a subset by name

The conformal and OpenOximetry skin-tone figures are drawn from results/ml/conformal_*.csv and openox_spo2_*.csv. Schematic panels (failure-mode signal shapes) are labelled
"schematic, synthetic" in the figure itself: they show a mechanism, they are not data.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_numbers as N  # noqa: E402

ROOT = N.ROOT
RES = N.RES
FIG = os.path.join(ROOT, 'docs', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREEN, VERM, GREY, PURPLE, SKY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#7F7F7F', '#CC79A7', '#56B4E9'
plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.25, 'savefig.dpi': 200, 'font.family': 'DejaVu Sans', 'legend.frameon': False})
NOTE = 'Research prototype, not a medical device.'


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', os.path.relpath(p, ROOT))


def box(ax, x, y, w, h, text, fc='#EEF4FA', ec=BLUE, fs=9, bold=False, tc='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc, wrap=True)


def arrow(ax, x0, y0, x1, y1, color='#444', lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, lw=lw, color=color, shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.grid(False)


# ----------------------------------------------------------------------------------------------------------------------- 1. architecture
def fig_architecture():
    params = N.CUSTOM['params']()
    fig, ax = plt.subplots(figsize=(12, 6.2))
    blank(ax)
    ax.text(0.0, 0.995, f'OxyNet student: {params} parameters, 10 s window in, four heads out (shipped as dependency-free JavaScript)', fontsize=11.5, fontweight='bold', va='top')
    w, g, y0, h = 0.108, 0.0165, 0.62, 0.24
    items = [('Input x\n10 x 300\n(10 s, 30 Hz)\nac R,G,B, dt,\nratio-of-ratios x2,\nbeat mask,\nbaseline-relative\nlog DC x3', '#FFF6E5', ORANGE),
             ('Shared stem\nConv1d(1,8,k5)\non R, G, B\n+ 7 aux channels\nConv1d(31,24,k5)\n+ domain emb.', '#EEF4FA', BLUE)]
    for i, (wd, tt) in enumerate(zip([24, 40, 56, 80], [300, 150, 75, 38])):
        items.append((f'Stage {i + 1}\n2 residual blocks\nk3, dilations 1, 2\nwidth {wd}\nT = {tt}', '#EEF4FA', BLUE))
    items.append(('Global\navg + max\npool (160)', '#EEF4FA', BLUE))
    items.append(('Window\nembedding\n(64)', '#E8F5EE', GREEN))
    xs = []
    for i, (t, fc, ec) in enumerate(items):
        x = 0.0 + i * (w + g)
        xs.append(x)
        box(ax, x, y0, w, h, t, fc=fc, ec=ec, fs=7.2)
        if i:
            arrow(ax, x - g, y0 + h / 2, x, y0 + h / 2)
    cx = xs[-1] + w / 2
    ax.plot([cx, cx], [y0, 0.555], color='#888', lw=1.2)
    heads = ['HR head\n141 bins, 40..180 bpm', 'SpO2 + log-variance\n(experimental, never shown)', 'Low-oxygen logit\n(emb minus person baseline emb)\n+ 9 side features', 'Signal-quality logit\n(is HR within 5 bpm?)']
    hw = 0.22
    hx = [0.0 + i * 0.26 for i in range(4)]
    ax.plot([hx[0] + hw / 2, max(cx, hx[3] + hw / 2)], [0.555, 0.555], color='#888', lw=1.2)
    for t, x in zip(heads, hx):
        box(ax, x, 0.39, hw, 0.12, t, fs=7.4, fc='#F3E9F0', ec=PURPLE)
        arrow(ax, x + hw / 2, 0.555, x + hw / 2, 0.51, color='#888', lw=1.1)
    ax.text(0.0, 0.33, 'Person-relative by design: the 60 s label-free first minute supplies the baseline log DC, ratio-of-ratios and window embedding that are subtracted.', fontsize=8.3, va='top', color='#333')
    box(ax, 0.0, 0.03, 0.30, 0.20, 'Training (phone and contact domains)\nSSL SimCLR on 299,754 windows\nStage B: OpenOximetry warm-up\nFine-tune on Hoffman / MTHS / BUT PPG\nLOSO, nested selection', fs=7.8, fc='#F7F7F7', ec=GREY)
    box(ax, 0.34, 0.03, 0.30, 0.20, 'Hybrid (what is deployed)\nlogit = v1 chain logit (19 features)\n+ CNN residual, 3 seeds\nper-fold meta-learner, threshold 0', fs=7.8, fc='#E8F5EE', ec=GREEN)
    arrow(ax, 0.30, 0.13, 0.34, 0.13)
    arrow(ax, 0.64, 0.13, 0.68, 0.13)
    ax.text(0.685, 0.13, 'Normal vs your baseline /\nPossible low oxygen / Unsure.\nHeart rate: CNN averaged with the v2 spectral\nestimator when they agree within 5 bpm.\n' + NOTE, fontsize=7.8, va='center')
    save(fig, 'ml_architecture.png')


# ----------------------------------------------------------------------------------------------------------------------- 2. flow
def fig_flow():
    fig, ax = plt.subplots(figsize=(11, 3.7))
    blank(ax)
    ax.text(0, 0.98, 'Input to output: three numbers per frame leave the camera stage, and nothing leaves the phone', fontsize=11, fontweight='bold', va='top')
    stages = [('Fingertip over\nrear camera\n+ flash\n(30-60 s, at rest)', '#FFF6E5', ORANGE), ('Mean R, G, B\nper frame +\nframe timestamps', '#EEF4FA', BLUE), ('30 Hz resampler,\ngap accounting,\nplacement and\nsaturation gates', '#EEF4FA', BLUE),
              ('First 60 s:\nlabel-free personal\nbaseline', '#EEF4FA', BLUE), ('Every 2 s, 10 s window:\nchain logit\n(19 features)\n+ CNN residual\n(3 seeds)', '#E8F5EE', GREEN), ('Three states:\nNormal vs\nyour baseline /\nPossible low oxygen /\nUnsure, re-measure', '#F3E9F0', PURPLE)]
    n = len(stages)
    w = 0.145
    gap = (1 - n * w) / (n - 1)
    for i, (t, fc, ec) in enumerate(stages):
        x = i * (w + gap)
        box(ax, x, 0.38, w, 0.40, t, fc=fc, ec=ec, fs=7.4)
        if i:
            arrow(ax, x - gap, 0.58, x, 0.58)
    ax.text(0.0, 0.28, 'Also shown: heart rate and a quality score. The app never shows an SpO2 number: the absolute-SpO2 head is experimental,\nand its only supported test (OpenOximetry, a contact oximeter) is a different sensor.', fontsize=8.2, va='top')
    ax.text(0.0, 0.08, 'Unsure is the answer to bad placement, motion, no stable pulse, a poor calibration minute, a stall, or a score near the boundary.\n' + NOTE, fontsize=8.2, va='top')
    save(fig, 'ml_flow.png')


# ----------------------------------------------------------------------------------------------------------------------- 3. SSL
def fig_ssl():
    a, b = N.rd('ssl_loss_curve_full.csv'), N.rd('ssl_loss_curve_noox.csv')
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax[0].plot(a.step, a.loss, color=BLUE, label='all corpora')
    ax[0].plot(b.step, b.loss, color=ORANGE, label='no OpenOximetry')
    ax[0].set_xlabel('SSL step')
    ax[0].set_ylabel('NT-Xent loss')
    ax[0].set_title('SimCLR pretraining loss')
    ax[0].legend()
    ax[1].plot(a.step, a.top1, color=BLUE)
    ax[1].plot(b.step, b.top1, color=ORANGE)
    ax[1].set_xlabel('SSL step')
    ax[1].set_ylabel('top-1 positive retrieval')
    ax[1].set_title('Does the encoder find the other view?')
    fig.suptitle('Self-supervised pretraining on people disjoint from every test person', fontsize=11, y=1.02)
    save(fig, 'ml_ssl_curves.png')


# ----------------------------------------------------------------------------------------------------------------------- 4. ablation
def fig_ablation():
    a = N.rd('ablation_core.csv')
    full = N.ab('full', 'p23')
    full_k = N.ab('full', 'p23k')
    order = [('full CNN', 'full', 'p23'), ('no SSL', 'no_ssl', 'p23k'), ('no physics channels', 'no_physics', 'p23k'), ('no OpenOximetry', 'no_openox', 'p23k'), ('no augmentation', 'no_aug', 'p23k'),
             ('no baseline conditioning', 'no_basecond', 'p23k'), ('no pretraining (expl.)', 'no_pretrain', 'p23k'), ('20 s window (1 seed)', 'win20', 'p23k'), ('domain-adversarial (1 seed)', 'adv', 'p23k')]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
    y = np.arange(len(order))[::-1]
    for yi, (lab, var, kind) in zip(y, order):
        r = N.ab(var, kind)
        col = BLUE if var == 'full' else (VERM if var in ('no_openox', 'no_pretrain') else GREY)
        ax[0].errorbar(r.p1_tm_bacc, yi, xerr=[[r.p1_tm_bacc - r.p1_tm_bacc_lo], [r.p1_tm_bacc_hi - r.p1_tm_bacc]], fmt='o', color=col, capsize=3)
    ax[0].axvline(float(N.b1('chain_v1', 'tm_bacc').pooled), color=GREEN, ls='--', lw=1.3, label='v1 chain (best baseline)')
    ax[0].axvline(0.5, color=GREY, ls=':', lw=1)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([o[0] for o in order])
    ax[0].set_xlabel('P1 TM bal. acc., 95% person CI (green dashed: v1 chain)')
    ax[0].set_title('P1: Hoffman low-oxygen detection')
    for yi, (lab, var, kind) in zip(y, order):
        r = N.ab(var, kind)
        if N._ok(r.p2_within5):
            ref = full.p2_within5 if var == 'full' else None
            col = BLUE if var == 'full' else (VERM if var in ('no_openox', 'no_pretrain') else GREY)
            ax[1].errorbar(r.p2_within5, yi, xerr=[[r.p2_within5 - r.p2_within5_lo], [r.p2_within5_hi - r.p2_within5]], fmt='o', color=col, capsize=3)
    ax[1].axvline(float(N.b2('all', 'v2_spectral_acf', 'forced', 'within5_full').pooled), color=GREEN, ls='--', lw=1.3, label='v2 spectral (best baseline)')
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('P2 within 5 bpm (green dashed: v2 spectral)')
    ax[1].set_title('P2: heart rate')
    fig.suptitle('Ablations: removing OpenOximetry (or all pretraining) is the only change far outside the seed spread', fontsize=11.5, y=1.0)
    fig.tight_layout()
    save(fig, 'ml_ablation.png')


# ----------------------------------------------------------------------------------------------------------------------- 5. accuracy vs coverage
def fig_coverage():
    m, y = N.p1_selective()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    curves = [('Logistic chain v1', m['chain_v1__score'].to_numpy(), 0.5, GREEN), ('XGBoost, 19 chain features', m['xgb_fusion19_v2__score'].to_numpy(), 0.5, ORANGE),
              ('CNN alone (3 seeds)', m['p_low'].to_numpy(), 0.5, BLUE), ('Hybrid (selected)', m['logit'].to_numpy(), 0.0, VERM)]
    for lab, sc, c, col in curves:
        ok = ~np.isnan(sc)
        cov, acc, alert = N.selective_curve(sc[ok], y[ok], c)
        ax[0].plot(cov, acc, color=col, label=lab, lw=1.8)
        ax[1].plot(cov, alert, color=col, lw=1.8)
    cv = float(N.b1('chain_v1', 'coverage', 'band_shipped').pooled)
    av = float(N.b1('chain_v1', 'acc', 'band_shipped').pooled)
    ax[0].plot([cv], [av], marker='*', ms=14, color='k', label='v1 shipped band (reference point)')
    ax[0].axvline(cv, color=GREY, ls=':', lw=1)
    ax[1].axvline(cv, color=GREY, ls=':', lw=1)
    ax[0].set_xlabel('coverage (share of windows answered, most confident first)')
    ax[0].set_ylabel('accuracy on answered windows')
    ax[0].set_title('Accuracy versus coverage (descriptive)')
    ax[0].legend(fontsize=8.5, loc='lower left')
    ax[1].set_xlabel('coverage')
    ax[1].set_ylabel('share of ALL hypoxic windows alerted')
    ax[1].set_title('Abstention is not detection')
    fig.suptitle('Hoffman, 5,644 windows, leave-one-person-out scores; confidence = distance from the decision boundary, no cutoff tuned', fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, 'ml_coverage_accuracy.png')


# ----------------------------------------------------------------------------------------------------------------------- 6/9. FEAT-007 figures
def _pending_fig(name, title, lines):
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    blank(ax)
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.85, boxstyle='round,pad=0.01', fc='#FFF6E5', ec=ORANGE, lw=2, ls='--'))
    ax.text(0.5, 0.74, 'PENDING', ha='center', fontsize=26, fontweight='bold', color=ORANGE)
    ax.text(0.5, 0.58, title, ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.40, '\n'.join(lines), ha='center', va='center', fontsize=9.5)
    save(fig, name)


def _find_cols(df, *subs):
    for c in df.columns:
        lc = c.lower()
        if all(s in lc for s in subs):
            return c
    return None


def fig_conformal():
    """Left: accuracy versus coverage along alpha (nested Mondrian conformal, held-out people) for the hybrid, the deployed pipeline R and the v1 chain, with the v1 and v2
    shipped bands as reference points. Right: share of hypoxic windows called Normal per held-out person at alpha = 0.10, against the 10% target."""
    cur = N.rd('conformal_curve.csv')
    ref = N.rd('conformal_reference_bands.csv').set_index('system')
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for system, col, lab in (('final', VERM, 'Hybrid (nested pipeline)'), ('deployed_R', BLUE, 'Deployed pipeline R (fixed)'), ('chain_v1', GREEN, 'v1 chain alone (control)')):
        d = cur[(cur.system == system) & (cur.method == 'mondrian')].sort_values('param')
        d = d[d.param <= 0.15]                                   # past this alpha the class thresholds cross (see CONFORMAL.md)
        ax[0].plot(d.coverage, d.acc, color=col, lw=1.8, label=lab)
        ax[0].fill_between(d.coverage, d.acc_lo, d.acc_hi, color=col, alpha=0.10, lw=0)
        r = N._cf_row(system, 0.1)
        ax[0].plot([r.coverage], [r.acc], 'o', color=col, ms=7, mec='k')
    for key, mk, lab in (('v1_shipped_band', '*', 'v1 shipped band'), ('v2_shipped_band', 'P', 'v2 gated band')):
        r = ref.loc[key]
        ax[0].errorbar([r.coverage], [r.acc], yerr=[[r.acc - r.acc_lo], [r.acc_hi - r.acc]], fmt=mk, color='k', ms=12 if mk == '*' else 8, capsize=2, label=lab)
    ax[0].set_xlabel('coverage (share of windows answered Normal or Possible low)')
    ax[0].set_ylabel('accuracy on answered windows')
    ax[0].set_title('Held-out accuracy versus coverage; dots = alpha 0.10', fontsize=10)
    ax[0].legend(fontsize=8, loc='lower left')
    pp = N.rd('conformal_per_person.csv')
    pp = pp[(pp.system == 'final') & (pp.method == 'mondrian') & np.isclose(pp.alpha.astype(float), 0.1)]
    x = np.arange(len(pp))
    ax[1].bar(x, 100 * pp.confident_miss, color=[VERM if v > 0.2 else SKY for v in pp.confident_miss])
    ax[1].axhline(10, color='k', ls='--', lw=1)
    ax[1].text(len(pp) - 0.5, 11, 'target 10%', ha='right', fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([p.replace('hof_10000', 'P') for p in pp.person])
    ax[1].set_ylabel('hypoxic windows called Normal (%)')
    ax[1].set_title('Per held-out person, hybrid at alpha 0.10', fontsize=10)
    fig.suptitle('Conformal three-state output, Hoffman, thresholds from the other five people only', fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, 'ml_conformal.png')


def fig_openox_skin():
    """Left: SaO2 MAE by skin-tone group, CNN versus textbook ratio-of-ratios (OpenOximetry contact sensor, 24 held-out people, lag-corrected).
    Right: contact-oximeter display minus blood-gas SaO2 by group (reference bias)."""
    sk = N.rd('openox_spo2_skin.csv')
    sk = sk[sk.label == 'lag_star']
    cb = N.rd('openox_spo2_contact_bias.csv')
    cb = cb.iloc[[0, 1, 2, 5, 4, 3]].reset_index(drop=True) if list(cb.scheme) == ['monk'] * 3 + ['ita'] * 3 else cb
    groups = [('monk', '1-3', 'Monk 1-3'), ('monk', '4-6', 'Monk 4-6'), ('monk', '7+', 'Monk 7+'), ('ita', 'lightest tertile', 'ITA light'), ('ita', 'mid tertile', 'ITA mid'), ('ita', 'darkest tertile', 'ITA dark')]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(groups))
    for k, (m, lab, col) in enumerate((('R', 'CNN (hybrid R members)', BLUE), ('ror_textbook', 'Textbook ratio-of-ratios', ORANGE))):
        rr = [N._row(sk, scheme=s, group=g, model=m) for s, g, _ in groups]
        ax[0].bar(x + (k - 0.5) * 0.38, [r.mae for r in rr], 0.38, color=col, label=lab, yerr=np.array([[r.mae - r.mae_lo, r.mae_hi - r.mae] for r in rr]).T, capsize=2)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f'{lab}\n(n={int(N._row(sk, scheme=s, group=g, model="R").n_people)})' for s, g, lab in groups], fontsize=8)
    ax[0].set_ylabel('SaO2 MAE, SpO2 points')
    ax[0].set_title('Error by skin tone, 24 held-out people (n = people)\nMonk 7+ has 2 people: suppressed (minimum group size 5)', fontsize=9.5)
    ax[0].legend(fontsize=8, loc='upper left')
    ax[0].set_ylim(0, 6)
    xb = np.arange(len(cb))
    ax[1].bar(xb, cb.bias, color=[SKY, SKY, SKY, GREY, GREY, GREY][:len(cb)], yerr=np.array([cb.bias - cb.lo, cb.hi - cb.bias]), capsize=2)
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels([f'{"Monk" if s == "monk" else "ITA"} {g.replace(" tertile", "")}\n(n={int(n)})' for s, g, n in zip(cb.scheme, cb.group, cb.n_people)], fontsize=8)
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_ylabel('device display minus blood-gas SaO2')
    ax[1].set_title('The reference over-reads more in darker skin\n(all people with draws)', fontsize=9.5)
    fig.suptitle('OpenOximetry contact red/infrared sensor with arterial blood gas: not a phone test', fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, 'ml_openox_skin.png')


# ----------------------------------------------------------------------------------------------------------------------- 7. HR
def fig_hr():
    d = pd.read_csv(os.path.join(RES, 'scores', 'hybrid_final_butppg.csv'))
    d = d.dropna(subset=['hr_ref', 'hr_final', 'hr_spec'])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for a, col, ttl, c in ((ax[0], 'hr_spec', 'v2 spectral + autocorrelation (best baseline)', ORANGE), (ax[1], 'hr_final', 'Hybrid rule `agree` (CNN + spectral)', BLUE)):
        mean = (d[col] + d.hr_ref) / 2
        diff = d[col] - d.hr_ref
        a.scatter(mean, diff, s=4, alpha=0.18, color=c, rasterized=True)
        a.axhline(0, color='k', lw=1)
        a.axhline(5, color=GREY, ls='--', lw=1)
        a.axhline(-5, color=GREY, ls='--', lw=1)
        md, sd = float(diff.mean()), float(diff.std())
        a.axhline(md, color=VERM, lw=1.2)
        a.axhline(md + 1.96 * sd, color=VERM, ls=':', lw=1)
        a.axhline(md - 1.96 * sd, color=VERM, ls=':', lw=1)
        a.set_ylim(-80, 80)
        a.set_xlabel('mean of estimate and ECG reference (bpm)')
        a.set_ylabel('estimate minus reference (bpm)')
        a.set_title(ttl + f'\nBUT PPG, bias {md:+.1f} bpm, 95% limits {md - 1.96 * sd:+.0f} to {md + 1.96 * sd:+.0f}', fontsize=9.5)
    h2 = N.rd('hybrid_p2.csv')
    xg = np.arange(3)
    for k, (model, ml, col) in enumerate((('v2_spectral_acf__forced', 'v2 spectral (best baseline)', ORANGE), ('cnn_alone', 'CNN alone', SKY), ('HYBRID_agree', 'hybrid `agree`', BLUE))):
        rr = [N._row(h2, scope=sc, model=model) for sc in ('all', 'butppg', 'mths')]
        ax[2].bar(xg + (k - 1) * 0.27, [r.within5_full for r in rr], 0.27, yerr=np.array([[r.within5_full - r.lo, r.hi - r.within5_full] for r in rr]).T, color=col, capsize=2, label=ml)
    ax[2].set_xticks(xg)
    ax[2].set_xticklabels(['all 100 people', 'BUT PPG (38)', 'MTHS (62)'])
    ax[2].set_ylabel('share within 5 bpm (full coverage)')
    ax[2].set_title('By dataset (person-bootstrap 95% CI)', fontsize=9.5)
    ax[2].legend(fontsize=7.5, loc='upper left')
    fig.tight_layout()
    save(fig, 'ml_hr_blandaltman.png')


# ----------------------------------------------------------------------------------------------------------------------- 8. per person P1
def fig_per_person():
    h = N.rd('hybrid_p1_final.csv')
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(6)
    spec = [('CHAIN_SHIPPED', 'Logistic chain v1', GREEN, 'o'), ('default_P', 'CNN alone (3-seed ens.)', BLUE, 's'), ('HYBRID_FINAL', 'Hybrid (selected)', VERM, 'D'), ('CHAIN_SAME_LEVERS', 'Chain, same nested levers', GREY, 'x')]
    for k, (mod, lab, col, mk) in enumerate(spec):
        r = N._row(h, model=mod)
        ax.scatter(x + (k - 1.5) * 0.12, [r[f'tm_bacc_{p}'] for p in N.HOF], color=col, marker=mk, s=44, label=lab, zorder=3)
    ax.axhline(0.5, color=GREY, ls=':', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'person {i + 1}' for i in range(6)])
    ax.set_ylabel('time-matched balanced accuracy of the held-out person')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8.5, ncol=2, loc='upper right')
    ax.set_title('P1 per person: complementary errors, no combiner exploits them (6 people)')
    save(fig, 'ml_p1_per_person.png')


# ----------------------------------------------------------------------------------------------------------------------- 10. gap
def fig_gap():
    rows = []
    for m, lab in (('chain_v1', 'chain v1'), ('chain_v2', 'chain v2'), ('xgb_fusion19_v2', 'XGBoost 19'), ('xgb_hand26_cal', 'XGBoost 26'), ('lgbm_hand26_cal', 'LightGBM 26')):
        t, h, g = N.p1_gap_pooled(m)
        rows.append((lab, t, h))
    c = N.ab('full', 'p23')
    rows.append(('CNN alone', float(c.p1_tm_bacc_train), float(c.p1_tm_bacc)))
    for m, lab in (('HYBRID_FINAL', 'hybrid'), ('CHAIN_SAME_LEVERS', 'chain, same levers')):
        g = N._row(N.rd('hybrid_p1_final_gap.csv'), model=m)
        rows.append((lab, float(g.train_pooled), float(g.heldout_pooled)))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=SKY, label='fold models on their own training people')
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=BLUE, label='held-out people (pooled)')
    for i, r in enumerate(rows):
        ax.text(i, max(r[1], r[2]) + 0.012, f'{r[1] - r[2]:+.3f}', ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5, rotation=15)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel('P1 time-matched balanced accuracy')
    ax.set_title('Train-versus-test gap (like with like: pooled training people against pooled held-out)')
    ax.legend(fontsize=8.5, loc='upper right')
    save(fig, 'ml_gap.png')


# ----------------------------------------------------------------------------------------------------------------------- 11. failure modes
def _ppg(t, hr=72, notch=0.25):
    f0 = hr / 60.0
    ph = (t * f0) % 1.0
    return np.exp(-((ph - 0.18) / 0.07) ** 2) + notch * np.exp(-((ph - 0.48) / 0.07) ** 2)


def fig_failure():
    rng = np.random.default_rng(3)
    t = np.arange(0, 10, 1 / 30)
    fig = plt.figure(figsize=(12.5, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.28)
    panels = []
    clean = 1.0 * _ppg(t) + 0.03 * rng.standard_normal(len(t))
    motion = clean.copy()
    motion[100:150] += 2.2 * np.sin(np.linspace(0, 6, 50)) + 0.8 * rng.standard_normal(50)
    cold = 0.18 * _ppg(t, 64) + 0.05 * rng.standard_normal(len(t))
    far = 0.25 * _ppg(t, 80) + 0.12 * rng.standard_normal(len(t))
    sat = np.clip(1.4 * _ppg(t) + 0.03 * rng.standard_normal(len(t)), None, 0.7)
    schem = [('Motion burst', motion, 'The band-pass trace is dominated by a burst:\nthe pulse lock and motion gate fail,\nthe app answers Unsure.'),
             ('Cold finger, low perfusion', cold, 'Pulse amplitude is a fraction of the noise:\nthe quality score falls,\nthe app answers Unsure.'),
             ('Far-flash phone (weak diffuse signal)', far, 'Low AC/DC and more noise: the lens probe and\nexposure adaptation try to compensate;\nuntested on any such phone.'),
             ('Saturated red channel', sat, 'Clipping removes the pulse peaks:\nthe saturation gate fires\nand the answer is Unsure.')]
    for k, (ttl, sig, txt) in enumerate(schem[:3]):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(t, sig, color=BLUE, lw=1)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('s (schematic, synthetic)')
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va='top', fontsize=7.6, bbox=dict(fc='white', ec='#ccc', alpha=0.9))
        ax.set_yticks([])
    h = pd.read_csv(os.path.join(RES, 'scores', 'hybrid_final_hoffman.csv'))
    h = h[h['eval'].astype(bool)]
    c = pd.read_csv(os.path.join(RES, 'scores', 'cnn_full_hoffman.csv'))
    c = c[c['eval'].astype(bool)]
    for k, (sub, ttl) in enumerate((('hof_100001', 'Real: person 1, the hybrid lags the chain at onset'), ('hof_100006', 'Real: person 6, all scores lag the onset'))):
        ax = fig.add_subplot(gs[1, k])
        d = h[(h.subject == sub)]
        rec = d.rec.iloc[0]
        d = d[d.rec == rec]
        ax.plot(d.t_start, d.chain_logit, color=GREEN, lw=1, label='chain logit')
        ax.plot(d.t_start, d.logit, color=VERM, lw=1.2, label='hybrid logit')
        cc = c[(c.subject == sub) & (c.rec == rec)]
        cs = cc.groupby('t_start').p_low.mean()
        lg = np.log(np.clip(cs, 1e-4, 1 - 1e-4) / (1 - np.clip(cs, 1e-4, 1 - 1e-4)))
        ax.plot(cs.index, lg, color=BLUE, lw=1, label='CNN alone logit (3-seed mean prob.)')
        ax.axhline(0, color='k', lw=0.8)
        lo = d.low.to_numpy().astype(float)
        ax.fill_between(d.t_start, -8, 8, where=lo > 0, color=ORANGE, alpha=0.15, label='reference SpO2 < 90')
        ax.set_ylim(-8, 8)
        ax.set_xlabel('s since start (' + rec + ')')
        ax.set_title(ttl, fontsize=10)
        if k == 0:
            ax.legend(fontsize=6.8, loc='lower right')
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title('Saturated red channel (schematic)', fontsize=10)
    ax.plot(t, schem[3][1], color=BLUE, lw=1)
    ax.text(0.02, 0.97, schem[3][2], transform=ax.transAxes, va='top', fontsize=7.6, bbox=dict(fc='white', ec='#ccc', alpha=0.9))
    ax.set_yticks([])
    ax.set_xlabel('s (schematic, synthetic)')
    fig.suptitle('Failure modes: top row and bottom right are synthetic schematics of the mechanism; bottom left and middle are real out-of-fold scores', fontsize=10.5, y=0.97)
    save(fig, 'ml_failure_modes.png')


FIGS = {'architecture': fig_architecture, 'flow': fig_flow, 'ssl': fig_ssl, 'ablation': fig_ablation, 'coverage': fig_coverage, 'conformal': fig_conformal,
        'hr': fig_hr, 'per_person': fig_per_person, 'openox_skin': fig_openox_skin, 'gap': fig_gap, 'failure': fig_failure}

if __name__ == '__main__':
    todo = sys.argv[1:] or list(FIGS)
    for k in todo:
        FIGS[k]()
