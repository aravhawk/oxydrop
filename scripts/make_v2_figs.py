"""Figures for the v2 rebuild, from saved result CSVs only (numpy, pandas, matplotlib; no dataset needed).
Writes docs/figures/fig_hr_validation.png and docs/figures/fig_v1_vs_v2.png.
Inputs: results/hr_validation/but_clips.csv, results/hr_validation/HR_SUMMARY.csv, results/eval_v1_vs_v2.csv,
        results/loPO_v1_windows.csv, results/loPO_v2_windows.csv"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RES = ROOT + '/results/'; FIG = ROOT + '/docs/figures/'; os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREEN, VERM, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#7F7F7F'
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'savefig.dpi': 200, 'font.family': 'DejaVu Sans'})

# ---------------------------------------------------------------- HR validation on BUT PPG
C = pd.read_csv(RES + 'hr_validation/but_clips.csv'); S = pd.read_csv(RES + 'hr_validation/HR_SUMMARY.csv')
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
lim = (30, 200)
ok = C.A_hr.notna()
ax[0].scatter(C.ref[ok], C.A_hr[ok], s=5, alpha=0.25, color=VERM, rasterized=True)
acc = C.C_valid.astype(bool)
ax[1].scatter(C.ref[acc], C.C_hr[acc], s=7, alpha=0.4, color=BLUE, rasterized=True)
for a, ttl in ((ax[0], 'Old peak-interval estimator\nall clips (n=%d)' % int(ok.sum())), (ax[1], 'v2 spectral + autocorrelation, accepted by gate\n(n=%d of %d clips)' % (int(acc.sum()), len(C)))):
    a.plot(lim, lim, color='k', lw=1); a.plot(lim, [2 * x for x in lim], color=GREY, lw=1, ls='--'); a.plot(lim, [x / 2 for x in lim], color=GREY, lw=1, ls=':')
    a.set_xlim(*lim); a.set_ylim(30, 250); a.set_xlabel('ECG reference rate (bpm)'); a.set_ylabel('Estimated rate (bpm)'); a.set_title(ttl)
ax[0].text(120, 230, '2x line (dashed)', color=GREY, fontsize=10)
S2 = S.copy(); S2['subset'] = S2['subset'].str.strip()
names = S2['subset'].unique().tolist()
sel = []
for sub in names:
    if sub in ('all clips', 'expert-good'):
        d = S2[S2['subset'] == sub]
        for est, lab in (('old peak-interval', 'old peak'), ('v2 accepted (gated)', 'v2 accepted')):
            r = d[d.estimator == est]
            if len(r): sel.append((sub, lab, float(r.MAE.iloc[0]), float(r.within5.iloc[0]), float(r.doubling.iloc[0])))
labels = [f'{s.replace("expert-good", "expert\ngood")}\n{l}' for s, l, *_ in sel]; x = np.arange(len(sel))
ax[2].bar(x, [m for _, _, m, _, _ in sel], color=[VERM if 'old' in l else BLUE for _, l, *_ in sel])
for i, (_, _, m, w, d) in enumerate(sel):
    ax[2].text(i, (m + 0.3) if m > 5.5 else 5.5, f'{m:.1f} bpm\n{100 * w:.0f}% within 5\n2x: {100 * d:.2f}%', ha='center', fontsize=9)
ax[2].axhline(5, color='k', lw=1, ls='--', label='target: MAE below 5 bpm'); ax[2].legend(loc='upper right', fontsize=10)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9); ax[2].set_ylabel('Mean absolute error (bpm)'); ax[2].set_ylim(0, max(m for _, _, m, _, _ in sel) + 6)
ax[2].set_title('BUT PPG: error by subset\n(coverage: v2 accepts 35% of all, 71% of expert-good)')
fig.suptitle('Heart-rate validation on BUT PPG (3,840 smartphone clips, 38 subjects, ECG reference)', fontsize=14, y=1.02)
fig.tight_layout(); fig.savefig(FIG + 'fig_hr_validation.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- v1 vs v2 on Hoffman
E = pd.read_csv(RES + 'eval_v1_vs_v2.csv')
D1 = pd.read_csv(RES + 'loPO_v1_windows.csv'); D2 = pd.read_csv(RES + 'loPO_v2_windows.csv')
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
short = ['v1\nevery window', 'v2 forced\nto answer', 'v2 lock-gated', 'v1 + abstain\nband 0.10', 'v2 lock-gated\n+ abstain 0.10']
x = np.arange(len(E)); w = 0.27
ax[0].bar(x - w, E.coverage, w, label='Coverage', color=GREY)
ax[0].bar(x, E.acc, w, label='Accuracy on answered', color=BLUE)
ax[0].bar(x + w, E.worst_person, w, label='Worst person', color=ORANGE)
for i in range(len(E)):
    for off, col in ((-w, 'coverage'), (0, 'acc'), (w, 'worst_person')):
        ax[0].text(i + off, E[col].iloc[i] + 0.01, f'{E[col].iloc[i]:.2f}', ha='center', fontsize=8.5, rotation=90, va='bottom')
ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=9); ax[0].set_ylim(0, 1.2); ax[0].legend(loc='upper left', ncol=3, fontsize=9)
ax[0].set_title('Hoffman, leave-one-person-out, fixed-C protocol (eval_v2.py)')

def per(D, gate):
    out = {}
    for sid, d in D.groupby('sid'):
        a = d[d.valid] if gate else d
        out[sid] = (len(a) / len(d), ((a.p > .5).astype(int) == a.y).mean() if len(a) else np.nan)
    return out
p1 = per(D1, False); p2 = per(D2, True)
people = sorted(p1); xp = np.arange(len(people)); w2 = 0.38
ax[1].bar(xp - w2 / 2, [p1[p][1] for p in people], w2, color=VERM, label='v1 (every window)')
ax[1].bar(xp + w2 / 2, [np.nan_to_num(p2[p][1]) for p in people], w2, color=BLUE, label='v2 (answered = locked windows)')
for i, p in enumerate(people):
    ax[1].text(i + w2 / 2, np.nan_to_num(p2[p][1]) + 0.01, 'locked\n%d%%' % round(100 * p2[p][0]) if p2[p][0] > 0 else 'never\nlocks', ha='center', fontsize=8.5)
ax[1].set_xticks(xp); ax[1].set_xticklabels([f'P{int(p) % 100}' for p in people]); ax[1].set_ylim(0, 1.2); ax[1].set_ylabel('Accuracy on answered windows')
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9); ax[1].set_title('Per person (P4 rests at 43 bpm: v2 cannot lock)')
fig.tight_layout(); fig.savefig(FIG + 'fig_v1_vs_v2.png', bbox_inches='tight'); plt.close(fig)
print('wrote fig_hr_validation.png, fig_v1_vs_v2.png')
