"""Regenerate all evaluation figures/tables from the saved out-of-fold score arrays.
Dependencies: numpy, scipy, pandas, matplotlib only. Reads results/scores/*, writes docs/figures/*.png, results/*.csv, TABLES.md, SUBGROUPS.md.
Protocol: Hoffman 2022, 6 people / 12 hands, leave-one-PERSON-out (nested); eval windows = t >= 60 s; label = reference SpO2 < 90."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
SC = ROOT + '/results/scores/'; FIG = ROOT + '/docs/figures/'; RES = ROOT + '/results/'
os.makedirs(FIG, exist_ok=True)
BAND = 0.10                                                                  # default abstain band
# Reported (NOT recomputed) nested-cutoff numbers, copied from pivot/RESULTS.md (opus/t10_nested_abstain.py, target 0.90)
NESTED = dict(coverage=0.867, acc=0.903, bacc=0.902, sens=0.908, spec=0.896, auc=0.956, tm_acc=0.816, worst=0.80)

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YEL, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#CC79A7', '#F0E442', '#7F7F7F'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13, 'legend.fontsize': 11.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.dpi': 200, 'figure.dpi': 100,
                     'font.family': 'DejaVu Sans'})

# ---------------------------------------------------------------- data
W = pd.read_csv(SC + 'windows.csv')
sid = W.sid.values; t = W.t.values; yref = W.y.values; rec = W.rec.values
y_all = (yref < 90).astype(int); ev = t >= 60
S = dict(tiny=np.load(SC + 'tiny.npy'), dhr=np.load(SC + 'dhr_only.npy'), clock=np.load(SC + 'time_only.npy'))
PEOPLE = np.unique(sid); PNUM = {p: int(p) % 100 for p in PEOPLE}
assert all(len(v) == len(W) for v in S.values())

def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def counts(y, p):
    return dict(TP=int(((y == 1) & (p == 1)).sum()), FP=int(((y == 0) & (p == 1)).sum()), TN=int(((y == 0) & (p == 0)).sum()), FN=int(((y == 1) & (p == 0)).sum()))

def basic(y, p, s=None):
    c = counts(y, p); sens = c['TP'] / max(c['TP'] + c['FN'], 1); spec = c['TN'] / max(c['TN'] + c['FP'], 1)
    r = dict(n=len(y), acc=(p == y).mean(), bacc=(sens + spec) / 2, sens=sens, FNR=1 - sens, spec=spec, FPR=1 - spec, **c)
    r['auc'] = auc(y, s) if s is not None and 0 < y.sum() < len(y) else np.nan
    return r

def metrics(score, keep=None):
    m = ev if keep is None else ev & keep
    r = basic(y_all[m], (score[m] > 0.5).astype(int), score[m]); r['coverage'] = m.sum() / ev.sum()
    pp = {PNUM[p]: ((score[m & (sid == p)] > .5) == y_all[m & (sid == p)]).mean() for p in PEOPLE if (m & (sid == p)).sum()}
    r['worst_person_acc'] = min(pp.values()); r['worst_person'] = min(pp, key=pp.get); r['per_person'] = pp
    return r

# time-matched accuracy (same construction as kit.tm: 300 subsets, each 100 s elapsed-time bin class-balanced)
_te = np.where(ev)[0]; _yv = y_all[_te]; _bv = (t[_te] // 100).astype(int)
USABLE = [k for k in np.unique(_bv) if min(_yv[_bv == k].sum(), (_bv == k).sum() - _yv[_bv == k].sum()) >= max(5, 0.08 * (_bv == k).sum())]

def tm_acc(score, nrep=300, seed=0):
    rng = np.random.default_rng(seed); p = (score[_te] > .5).astype(int); out = []
    for _ in range(nrep):
        sel = []
        for k in USABLE:
            i1 = np.where((_bv == k) & (_yv == 1))[0]; i0 = np.where((_bv == k) & (_yv == 0))[0]; m = min(len(i1), len(i0))
            sel += list(rng.choice(i1, m, replace=False)) + list(rng.choice(i0, m, replace=False))
        sel = np.array(sel); out.append((p[sel] == _yv[sel]).mean())
    return float(np.mean(out))

def tm_answered(score, keep, nrep=200, seed=0):
    """time-matched accuracy over answered windows (same as opus/t10: bins with >=5 of each class), and the clock on the same subsets"""
    rng = np.random.default_rng(seed); idx = np.where(ev & keep)[0]; b = t[idx] // 100; a = []; c = []
    for _ in range(nrep):
        sel = []
        for k in np.unique(b):
            i1 = idx[(b == k) & (y_all[idx] == 1)]; i0 = idx[(b == k) & (y_all[idx] == 0)]; mm = min(len(i1), len(i0))
            if mm < 5: continue
            sel += list(rng.choice(i1, mm, replace=False)) + list(rng.choice(i0, mm, replace=False))
        sel = np.array(sel); a.append(((S['tiny'][sel] > .5) == y_all[sel]).mean()); c.append(((S['clock'][sel] > .5) == y_all[sel]).mean())
    return float(np.mean(a)), float(np.mean(c))

def boot_ci(score, keep=None, nboot=500, seed=1):
    rng = np.random.default_rng(seed); m = ev if keep is None else ev & keep; ids = np.unique(sid); a = []
    for _ in range(nboot):
        pick = rng.choice(ids, len(ids)); idx = np.concatenate([np.where(m & (sid == q))[0] for q in pick])
        a.append(((score[idx] > .5) == y_all[idx]).mean())
    return np.percentile(a, [2.5, 97.5])

def answered(score, c): return np.abs(score - 0.5) >= c

# ---------------------------------------------------------------- headline check vs RESULTS.md
m_tiny = metrics(S['tiny']); m_clock = metrics(S['clock']); m_dhr = metrics(S['dhr'])
tm_tiny, tm_clock, tm_dhr = tm_acc(S['tiny']), tm_acc(S['clock']), tm_acc(S['dhr'])
EXPECT = dict(acc=0.870, bacc=0.869, auc=0.942, tm=0.795, worst=0.78)
GOT = dict(acc=m_tiny['acc'], bacc=m_tiny['bacc'], auc=m_tiny['auc'], tm=tm_tiny, worst=m_tiny['worst_person_acc'])
check = pd.DataFrame({'RESULTS.md': EXPECT, 'recomputed': {k: round(float(v), 3) for k, v in GOT.items()}})
check['match(at 2-3 dp)'] = [abs(round(r, 2 if k == 'worst' else 3) - e) < 1e-9 for k, e, r in zip(check.index, check['RESULTS.md'], [GOT[k] for k in check.index])]
check['recomputed_unrounded'] = [float(GOT[k]) for k in check.index]
check.to_csv(RES + 'check_vs_RESULTS.csv'); print(check)

# ---------------------------------------------------------------- tables
def row(name, m, keep_note, tm=np.nan, ci=(np.nan, np.nan), source='recomputed'):
    return dict(model=name, source=source, coverage=m['coverage'], acc=m['acc'], acc_ci_lo=ci[0], acc_ci_hi=ci[1], bacc=m['bacc'],
                sens=m['sens'], FNR_missed_hypoxemia=m['FNR'], spec=m['spec'], FPR_false_alarm=m['FPR'], auroc=m['auc'], tm_acc=tm,
                worst_person_acc=m['worst_person_acc'], TP=m['TP'], FP=m['FP'], TN=m['TN'], FN=m['FN'], n_windows=m['n'])

keep10 = answered(S['tiny'], BAND); m10 = metrics(S['tiny'], keep10); tm10, clk10 = tm_answered(S['tiny'], keep10)
rows = [row('Time-only clock (elapsed s; not deployable)', m_clock, '', tm_clock, boot_ci(S['clock'])),
        row('DeltaHR only (1 feature, nested LOPO)', m_dhr, '', tm_dhr, boot_ci(S['dhr'])),
        row('TINY chain, all windows (default threshold 0.5)', m_tiny, '', tm_tiny, boot_ci(S['tiny'])),
        row('TINY chain, answered windows at abstain band |p-0.5|<0.10', m10, '', tm10, boot_ci(S['tiny'], keep10))]
nested = dict(coverage=NESTED['coverage'], acc=NESTED['acc'], bacc=NESTED['bacc'], sens=NESTED['sens'], FNR=1 - NESTED['sens'], spec=NESTED['spec'],
              FPR=1 - NESTED['spec'], auc=NESTED['auc'], worst_person_acc=NESTED['worst'], TP=np.nan, FP=np.nan, TN=np.nan, FN=np.nan, n=np.nan)
rows.append(row('TINY chain, nested cutoff (target 0.90) - REPORTED in RESULTS.md, not recomputed', nested, '', NESTED['tm_acc'], source='RESULTS.md (opus/t10_nested_abstain.py)'))
TA = pd.DataFrame(rows); TA.to_csv(RES + 'table_a_main.csv', index=False)

pp_rows = []
for p in PEOPLE:
    m = ev & (sid == p); mt = basic(y_all[m], (S['tiny'][m] > .5).astype(int)); mc = basic(y_all[m], (S['clock'][m] > .5).astype(int))
    ab = (~keep10[m]).mean()
    mk = m & keep10; acc_ans = ((S['tiny'][mk] > .5) == y_all[mk]).mean()
    pp_rows.append(dict(person=PNUM[p], n_windows=int(m.sum()), base_rate_hypoxemic=y_all[m].mean(), acc=mt['acc'], sens=mt['sens'], spec=mt['spec'],
                        FN=mt['FN'], FP=mt['FP'], abstain_rate_band010=ab, acc_on_answered_band010=acc_ans, clock_acc=mc['acc'], clock_sens=mc['sens'], clock_spec=mc['spec']))
pp_rows.append(dict(person='pooled', n_windows=int(ev.sum()), base_rate_hypoxemic=y_all[ev].mean(), acc=m_tiny['acc'], sens=m_tiny['sens'], spec=m_tiny['spec'],
                    FN=m_tiny['FN'], FP=m_tiny['FP'], abstain_rate_band010=(~keep10[ev]).mean(), acc_on_answered_band010=m10['acc'], clock_acc=m_clock['acc'],
                    clock_sens=m_clock['sens'], clock_spec=m_clock['spec']))
TB = pd.DataFrame(pp_rows); TB.to_csv(RES + 'table_b_per_person.csv', index=False)

bands = [0.0, 0.05, 0.10, 0.15, 0.20]; crow = []
for c in bands:
    k = answered(S['tiny'], c); m = metrics(S['tiny'], k); ta, ca = tm_answered(S['tiny'], k)
    crow.append(dict(band=c, coverage=m['coverage'], acc=m['acc'], bacc=m['bacc'], sens=m['sens'], spec=m['spec'], FN=m['FN'], FP=m['FP'], TP=m['TP'], TN=m['TN'],
                     FNR=m['FNR'], FPR=m['FPR'], n_answered=m['n'], worst_person_acc=m['worst_person_acc'], tm_acc_answered=ta, clock_acc_same_windows=((S['clock'][ev & k] > .5) == y_all[ev & k]).mean(),
                     clock_tm_same_windows=ca))
TC = pd.DataFrame(crow); TC.to_csv(RES + 'table_c_coverage.csv', index=False)

def md(df, floats=3):
    INT = {'n', 'n_windows', 'n_answered', 'TP', 'FP', 'TN', 'FN', 'person'}
    def f(c, v):
        if pd.isna(v): return ''
        if c in INT or isinstance(v, (int, np.integer, str)): return str(int(v)) if c in INT and not isinstance(v, str) else str(v)
        return f'{v:.{floats}f}'
    h = '| ' + ' | '.join(df.columns) + ' |\n|' + '---|' * len(df.columns) + '\n'
    return h + ''.join('| ' + ' | '.join(f(c, v) for c, v in zip(df.columns, r)) + ' |\n' for r in df.itertuples(index=False))

with open(RES + 'TABLES.md', 'w') as f:
    f.write('# OxyDrop evaluation tables\n\nHoffman 2022 (UW), 6 people / 12 hands, leave-one-PERSON-out (nested), eval windows t >= 60 s (5,644), label = reference SpO2 < 90 (base rate %.3f). '
            'Generated by `scripts/make_eval.py` from `results/scores/`. Sensitivity (FNR = 1 - sens = missed hypoxemia) and specificity (FPR = 1 - spec = false alarms) are reported separately. '
            'The 6-person sample cannot resolve differences of a few points; the person-bootstrap CI for TINY vs clock accuracy overlaps (+3.5 pts on the saved scores, paired person-bootstrap CI [-6.8, +16.7]; see `scripts/review_checks.py`).\n\n' % y_all[ev].mean())
    f.write('## (a) Main results\n\n' + md(TA) + '\nWindows counts (TP/FP/TN/FN) are 2-second-stride windows, not independent samples. The nested-cutoff row is copied from RESULTS.md (cutoff chosen in inner LOPO), not recomputed here; '
            'the band-0.10 row is a fixed a-priori band. Time-matched (TM) accuracy: 300 subsets in which every 100 s elapsed-time bin is class-balanced (clock scores ~0.52). '
            'Answered-window TM uses 200 subsets over answered windows only.\n\n')
    f.write('## (b) Per person (TINY, default threshold 0.5; abstain-rate at band 0.10)\n\n' + md(TB) + '\n')
    f.write('## (c) Coverage vs accuracy at several abstain bands (TINY, answered windows only)\n\n' + md(TC) + '\nBand 0 = no abstention. Abstained windows are excluded from every count; coverage = answered / 5,644.\n\n')
    f.write('## Check against RESULTS.md\n\n' + md(check.reset_index().rename(columns={'index': 'metric'})) + '\n')

# ---------------------------------------------------------------- Fig 1 per person
fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]; labs = [f'P{PNUM[p]}' for p in PEOPLE] + ['Pooled']; x = np.arange(len(labs)); w = 0.38
ta = TB.acc.values; ca = TB.clock_acc.values
ax.bar(x - w / 2, ta, w, color=BLUE, label='TINY chain (deployable)'); ax.bar(x + w / 2, ca, w, color=ORANGE, label='Time-only clock (control, not deployable)')
for i in range(len(x)):
    ax.text(x[i] - w / 2, ta[i] + 0.012, f'{ta[i]:.2f}', ha='center', fontsize=10.5, color=BLUE); ax.text(x[i] + w / 2, ca[i] + 0.012, f'{ca[i]:.2f}', ha='center', fontsize=10.5, color='#8a5d00')
wi = int(np.argmin(ta[:6])); ax.bar(x[wi] - w / 2, ta[wi], w, color='none', edgecolor=VERM, linewidth=3)
ax.annotate(f'Worst person (P{wi + 1}: {ta[wi]:.2f})', xy=(x[wi] - w / 2, ta[wi] + 0.06), xytext=(x[wi], 1.07), ha='center', fontsize=12, color=VERM, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=VERM, lw=2))
ax.axhline(0.5, color=GREY, ls=':', lw=1); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.3, 1.2); ax.set_ylabel('Accuracy (windows t >= 60 s)')
ax.set_title('Accuracy per held-out person'); ax.legend(loc='lower center', frameon=True, framealpha=0.95, fontsize=10.5)
ax = axs[1]
ax.scatter(x[:6] - 0.1, TB.sens.values[:6], s=110, marker='o', color=GREEN, label='TINY sensitivity (catches hypoxemia)', zorder=3, edgecolor='k', linewidth=0.6)
ax.scatter(x[:6] + 0.1, TB.spec.values[:6], s=110, marker='s', color=PURPLE, label='TINY specificity (no false alarm)', zorder=3, edgecolor='k', linewidth=0.6)
ax.scatter([6 - 0.1], [TB.sens.values[6]], s=150, marker='o', color=GREEN, edgecolor='k', linewidth=1.5, zorder=3); ax.scatter([6 + 0.1], [TB.spec.values[6]], s=150, marker='s', color=PURPLE, edgecolor='k', linewidth=1.5, zorder=3)
for i in range(7):
    ax.plot([i - 0.1, i + 0.1], [TB.sens.values[i], TB.spec.values[i]], color='#bbbbbb', lw=1.5, zorder=2)
    ax.text(i - 0.3, TB.sens.values[i], f'{TB.sens.values[i]:.2f}', ha='center', va='center', fontsize=10, color='#00674a'); ax.text(i + 0.3, TB.spec.values[i], f'{TB.spec.values[i]:.2f}', ha='center', va='center', fontsize=10, color='#8a4a70')
ax.axvspan(wi - 0.4, wi + 0.4, color=VERM, alpha=0.12); ax.text(wi, 0.70, 'worst', ha='center', color=VERM, fontsize=11, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.45, 1.08); ax.set_ylabel('Rate'); ax.set_title('TINY sensitivity and specificity per person'); ax.legend(loc='lower left', frameon=True, framealpha=0.95)
fig.suptitle('Hoffman 2022, leave-one-person-out: per-person performance (default threshold, no abstention)', fontsize=16, y=1.0)
fig.text(0.5, -0.02, 'The clock baseline scores only from elapsed time since recording start; it reaches 0.54 to 0.96 per person. TM accuracy (clock-free): TINY %.2f vs clock %.2f.' % (tm_tiny, tm_clock), ha='center', fontsize=11.5)
fig.tight_layout(); fig.savefig(FIG + 'fig_per_person.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2 ROC
def roc(y, s):
    o = np.argsort(-s, kind='stable'); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
fig, ax = plt.subplots(figsize=(7.8, 7))
for key, lab, col, ls in [('tiny', 'TINY chain', BLUE, '-'), ('dhr', 'DeltaHR only', GREEN, '--'), ('clock', 'Time-only clock (control)', ORANGE, ':')]:
    s = S[key][ev]; yy = y_all[ev]; fpr, tpr = roc(yy, s); a = auc(yy, s)
    ax.plot(fpr, tpr, color=col, lw=3, ls=ls, label=f'{lab} (AUROC {a:.3f})')
    p = (s > .5).astype(int); b = basic(yy, p)
    ax.scatter([b['FPR']], [b['sens']], s=140, color=col, edgecolor='k', linewidth=1.5, zorder=5, marker='o' if key == 'tiny' else 's' if key == 'dhr' else '^')
    if key == 'tiny': ax.annotate(f"default threshold 0.5\nsens {b['sens']:.2f}, FPR {b['FPR']:.2f}", xy=(b['FPR'], b['sens']), xytext=(0.28, 0.72), fontsize=11.5, arrowprops=dict(arrowstyle='->', lw=1.8))
ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls='-.'); ax.set_xlabel('False-positive rate (1 - specificity)'); ax.set_ylabel('True-positive rate (sensitivity)')
ax.set_title('ROC, hypoxemia (SpO2 < 90), held-out people'); ax.legend(loc='lower right', frameon=True, framealpha=0.95, title='Markers = default threshold (p = 0.5)', title_fontsize=11); ax.set_aspect('equal'); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.tight_layout(); fig.savefig(FIG + 'fig_roc.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 coverage-accuracy
cs = np.arange(0, 0.3001, 0.005); cov = []; acc = []; bac = []
for c in cs:
    m = metrics(S['tiny'], answered(S['tiny'], c)); cov.append(m['coverage']); acc.append(m['acc']); bac.append(m['bacc'])
cov, acc, bac = map(np.array, (cov, acc, bac))
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.plot(cov, acc, color=BLUE, lw=3, label='TINY accuracy (answered windows)'); ax.plot(cov, bac, color=GREEN, lw=3, ls='--', label='TINY balanced accuracy')
ax.axhline(m_clock['acc'], color=ORANGE, lw=2.5, ls=':', label=f"Time-only clock accuracy, no abstention ({m_clock['acc']:.3f})")
ax.axhline(0.90, color=GREY, lw=1, ls='-.'); ax.text(0.505, 0.9015, '90 %', color=GREY, fontsize=10, va='bottom')
ax.scatter([m10['coverage']], [m10['acc']], s=190, color=VERM, edgecolor='k', linewidth=1.5, zorder=6, marker='o')
ax.annotate(f"Default band c = 0.10 (fixed a priori)\ncoverage {m10['coverage']:.3f}, acc {m10['acc']:.3f}, bacc {m10['bacc']:.3f}", xy=(m10['coverage'], m10['acc']), xytext=(0.965, 0.978), fontsize=11.5,
            arrowprops=dict(arrowstyle='->', lw=1.8, color=VERM), color=VERM)
ax.scatter([NESTED['coverage']], [NESTED['acc']], s=210, color='white', edgecolor=PURPLE, linewidth=3, zorder=6, marker='D')
ax.annotate(f"Nested cutoff, target 0.90 (reported in RESULTS.md, not recomputed here)\ncoverage {NESTED['coverage']:.3f}, acc {NESTED['acc']:.3f}, bacc {NESTED['bacc']:.3f}", xy=(NESTED['coverage'], NESTED['acc']),
            xytext=(0.965, 0.858), fontsize=11.5, color='#8a4a70', arrowprops=dict(arrowstyle='->', lw=1.8, color=PURPLE))
ax.scatter([1.0], [m_tiny['acc']], s=110, color=BLUE, edgecolor='k', zorder=6); ax.annotate(f"no abstention\nacc {m_tiny['acc']:.3f}", xy=(1.0, m_tiny['acc']), xytext=(0.99, 0.80), ha='left', fontsize=11, arrowprops=dict(arrowstyle='->', lw=1.2))
ax.set_xlim(1.02, 0.50); ax.set_ylim(0.78, 1.0); ax.set_xlabel('Coverage (fraction of readings answered)  <-  abstain more'); ax.set_ylabel('Accuracy on answered windows')
ax.set_title('Selective prediction: abstain when |p - 0.5| < c  (TINY, c swept 0 to 0.30)'); ax.legend(loc='lower right', frameon=True, framealpha=0.95)
fig.tight_layout(); fig.savefig(FIG + 'fig_coverage_accuracy.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 4 timelines
worst_p = int(TB.person[TB.acc.values[:6].argmin()]); good_p = 5
shown = [(worst_p, 'L', 'worst person'), (worst_p, 'R', 'worst person'), (good_p, 'L', 'a good person')]
fig, axs = plt.subplots(2, 3, figsize=(20, 9), sharex=True, sharey='row', gridspec_kw=dict(height_ratios=[1, 1.2]))
notes = {}
for j, (pn, hand, tag) in enumerate(shown):
    r = f'{100000 + pn}{hand}'; m = rec == r; tt = t[m]; yy = yref[m]; pp = S['tiny'][m]; lab = y_all[m]; e = ev[m]
    dec = (pp > .5).astype(int); ab = np.abs(pp - .5) < BAND; wrong_ans = (dec != lab) & e & ~ab; ab_e = ab & e
    acc_hand = (dec[e] == lab[e]).mean(); acc_ans = (dec[e & ~ab] == lab[e & ~ab]).mean()
    fa = (dec == 1) & (lab == 0) & e; fn = (dec == 0) & (lab == 1) & e
    t_hyp = int(tt[lab == 1].min()); fa_pre = fa & (tt < t_hyp)
    d = dict(rec=r, acc=round(float(acc_hand), 3), acc_answered=round(float(acc_ans), 3), abstain=round(float(ab_e.sum() / e.sum()), 3), FA_windows=int(fa.sum()), FN_windows=int(fn.sum()),
             first_ref_below_90_s=t_hyp, FA_before_first_hypoxemia=int(fa_pre.sum()),
             FA_first_s=int(tt[fa].min()) if fa.any() else None, FA_ref_spo2_range=[round(float(yy[fa].min()), 1), round(float(yy[fa].max()), 1)] if fa.any() else None,
             FN_first_s=int(tt[fn].min()) if fn.any() else None, FN_last_s=int(tt[fn].max()) if fn.any() else None)
    notes[r] = d
    for ax_ in axs[:, j]:
        ax_.axvspan(0, 60, color=SKY, alpha=0.18, lw=0)
        ax_.fill_between(tt, 0, 1, where=wrong_ans, step='post', color=VERM, alpha=0.30, lw=0, transform=ax_.get_xaxis_transform())
        ax_.fill_between(tt, 0, 1, where=ab_e, step='post', color=GREY, alpha=0.30, lw=0, transform=ax_.get_xaxis_transform())
    a0 = axs[0, j]; a0.plot(tt, yy, color='k', lw=2.4); a0.axhline(90, color=VERM, ls='--', lw=1.5); a0.set_ylim(64, 102)
    a0.text(tt.max(), 90.6, 'hypoxemia threshold (90 %)', ha='right', color=VERM, fontsize=10.5)
    a0.set_title(f"P{pn} {'left' if hand == 'L' else 'right'} hand ({tag}): accuracy {acc_hand:.2f}", fontsize=14)
    a0.text(30, 66, 'calibration\n(not scored)', ha='center', fontsize=10, color='#1a6a96', va='bottom')
    a1 = axs[1, j]; a1.plot(tt, pp, color=BLUE, lw=2.2); a1.axhline(0.5, color='k', lw=1)
    a1.step(tt, np.where(e, dec * 0.07 - 0.10, np.nan), where='post', color=ORANGE, lw=3); a1.step(tt, lab * 0.07 - 0.20, where='post', color='k', lw=2)
    a1.set_ylim(-0.24, 1.45); a1.set_xlabel('Time since start of recording (s)'); a1.set_xlim(0, 1115)
    a1.text(1112, -0.065, 'TINY decision (hypoxic = high)', color='#8a5d00', fontsize=9.5, va='center', ha='right'); a1.text(1112, -0.165, 'reference < 90 (high)', color='k', fontsize=9.5, va='center', ha='right')
    # factual annotation box
    lines = [f"false alarms: {d['FA_windows']} windows" + (f" (ref SpO2 {d['FA_ref_spo2_range'][0]:.0f}-{d['FA_ref_spo2_range'][1]:.0f}, first at {d['FA_first_s']} s)" if d['FA_windows'] else ''),
             f"missed hypoxemia: {d['FN_windows']} windows" + (f" ({d['FN_first_s']}-{d['FN_last_s']} s)" if d['FN_windows'] else ''),
             f"abstained: {100 * d['abstain']:.0f} % of windows"]
    a1.text(0.985, 0.985, '\n'.join(lines), transform=a1.transAxes, ha='right', va='top', fontsize=10.5, bbox=dict(boxstyle='round', fc='white', ec='#999', alpha=0.95))
axs[0, 0].set_ylabel('Reference SpO2 (%)'); axs[1, 0].set_ylabel('TINY p(hypoxic)')
axs[1, 0].legend(handles=[Line2D([], [], color=BLUE, lw=2.2, label='p(hypoxic)'), Patch(color=VERM, alpha=0.3, label='answered but wrong'), Patch(color=GREY, alpha=0.3, label='abstained (|p-0.5|<0.10)'),
                          Patch(color=SKY, alpha=0.3, label='calibration, not scored')], loc='center left', bbox_to_anchor=(0.0, 0.62), fontsize=10, frameon=True, framealpha=0.95)
fig.suptitle('Failure and success cases (leave-one-person-out, default threshold; hands of one person shown separately)', fontsize=16, y=0.995)
fig.tight_layout(); fig.savefig(FIG + 'fig_timeline_failure.png', bbox_inches='tight'); plt.close(fig)
json.dump(notes, open(RES + 'timeline_notes.json', 'w'), indent=1); print(json.dumps(notes, indent=0))

# ---------------------------------------------------------------- Fig 5 pipeline diagram
fig, ax = plt.subplots(figsize=(17, 11)); ax.set_xlim(0, 17); ax.set_ylim(0, 11); ax.axis('off')
ABUN = '#DCEEF9'; SCAR = '#FBE6C8'; NEUT = '#F2F2F2'
def box(x, y, w, h, title, sub, fc, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.15', fc=fc, ec='#333', lw=2))
    ax.text(x + w / 2, y + h * 0.70, title, ha='center', va='center', fontsize=fs, fontweight='bold'); ax.text(x + w / 2, y + h * 0.30, sub, ha='center', va='center', fontsize=10, color='#222', linespacing=1.3)
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=22, lw=2.2, color='#222', shrinkA=0, shrinkB=0))
def call(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=2.2)); ax.text(x + 0.2, y + h - 0.18, title, fontsize=11.5, fontweight='bold', color=ec, va='top')
    ax.text(x + 0.2, y + h - 0.62, body, fontsize=10.3, va='top', linespacing=1.4)
bw, bh = 3.1, 1.5; y1 = 6.9; xs = [0.2, 3.5, 6.8, 10.1, 13.4]
box(xs[0], y1, bw, bh, '1. Camera frames', 'fingertip on lens, flash on,\n30 fps video', NEUT)
box(xs[1], y1, bw, bh, '2. Mean RGB per frame', 'one (R, G, B) triple\nper frame', NEUT)
box(xs[2], y1, bw, bh, '3. Causal band-pass', '3rd-order IIR, 0.7-3.5 Hz\n(no look-ahead)', ABUN)
box(xs[3], y1, bw, bh, '4. Beat detection', 'beat times and\ninter-beat intervals', ABUN)
box(xs[4], y1, bw, bh, '5. 19 window statistics', 'HR, HRV, resp. rate, perfusion,\nDC absorbance, ratio-of-ratios', ABUN, fs=11)
for i in range(4): arrow((xs[i] + bw, y1 + bh / 2), (xs[i + 1], y1 + bh / 2))
# row 2: 6 | 7a over 7b | 8 | 9
box(13.4, 4.2, bw, bh, '6. Own-baseline delta', "subtract the person's own\nfirst-60-s median", NEUT)
box(9.9, 5.0, 3.1, 1.25, '7a. Logistic regression', 'DeltaHR only (1 feature)', ABUN, fs=11)
box(9.9, 3.3, 3.1, 1.25, '7b. Logistic regression', 'all 19 deltas (fusion)', SCAR, fs=11)
box(6.4, 4.2, 3.0, bh, '8. Average the two', 'mean of the two\nprobabilities (62 floats)', SCAR)
box(2.9, 4.2, 3.3, bh, '9. Decision', 'p >= 0.6: hypoxic\np <= 0.4: normal\notherwise: "unsure, re-measure"', NEUT, fs=11)
arrow((xs[4] + bw / 2, y1), (13.4 + bw / 2, 4.2 + bh))
arrow((13.4, 4.2 + bh * 0.75), (13.0, 5.62)); arrow((13.4, 4.2 + bh * 0.25), (13.0, 3.92))
arrow((9.9, 5.62), (9.4, 4.2 + bh * 0.75)); arrow((9.9, 3.92), (9.4, 4.2 + bh * 0.25)); arrow((6.4, 4.95), (6.2, 4.95))
call(0.2, 8.95, 9.6, 1.85, 'Validated on abundant data (HR extraction: stages 3-5)',
     'BUT PPG (18 people, ECG reference): finger-at-rest HR error 11.1 bpm MAE overall,\n4.5 bpm on expert-good clips. Hoffman (locked camera settings, at rest): 1.2 bpm MAE\nvs the reference oximeter pulse, 99 % within 5 bpm.', ABUN, '#005a8d')
call(10.0, 8.95, 6.8, 1.85, 'DeltaHR to hypoxia link (stage 7a)', 'Reproduced on a 20-person altitude-chamber dataset:\nzero-shot AUROC 0.88 on Hoffman windows,\nwith no Hoffman training.', ABUN, '#005a8d')
call(0.2, 0.9, 8.2, 2.1, 'Scarce hypoxemia data (stages 7b and 8, decision band)', 'Only 6 people with phone-camera video and SpO2 < 90 exist in open data\n(UW Hoffman 2022). Fusion weights are fit and evaluated by leave-one-PERSON-out;\naccuracy CI is wide (0.83 to 0.91). Evaluated at rest only.', SCAR, '#9a5b00')
call(8.8, 0.9, 8.0, 2.1, 'Why the chain is tiny', 'Two logistic regressions, 62 floats (1.9 KB). No neural network.\nEvery stage is causal, so it can run live on the phone.\nKnown limit: DeltaHR is not oxygen-specific; exercise raises it.', NEUT, '#444')
ax.legend(handles=[Patch(fc=ABUN, ec='#333', label='informed or validated by abundant data'), Patch(fc=SCAR, ec='#333', label='fit on scarce hypoxemia data (6 people)'), Patch(fc=NEUT, ec='#333', label='fixed / deterministic step')],
          loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=11.5)
ax.set_title('OxyDrop TINY chain: from fingertip video to a hypoxemia decision', fontsize=17, pad=8)
fig.savefig(FIG + 'fig_chain.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 6 exercise false alarms
EX = pd.read_csv(SC + 'exercise_fp.csv'); EXu = EX[EX.dhr_p50.notna() & EX.fp_rate_normoxic.notna()]       # Subject4: no HR could be extracted
ex_pooled = float((EXu.fp_rate_normoxic * EXu.n_norm).sum() / EXu.n_norm.sum()); ex_median = float(EXu.fp_rate_normoxic.median())
ex_pooled14 = float((EX.fp_rate_normoxic * EX.n_norm).sum() / EX.n_norm.sum())   # all 14 csv rows: this is the 63 % quoted in RESULTS.md
rest_fpr_tiny = m_tiny['FPR']; rest_fpr_dhr = m_dhr['FPR']
rest_pp = [1 - TB.spec.values[i] for i in range(6)]
fig, ax = plt.subplots(figsize=(8.6, 6.4)); lab_ = ['Normoxic exercise\n(DeltaHR rule; moshekam,\n%d usable people)' % len(EXu), 'At rest, TINY\n(Hoffman, 6 people)', 'At rest, DeltaHR only\n(Hoffman, 6 people)']
vals = [ex_pooled, rest_fpr_tiny, rest_fpr_dhr]; ax.bar(range(3), vals, color=[VERM, BLUE, GREEN], width=0.55)
for i, v in enumerate(vals): ax.text(i, v / 2, f'{100 * v:.0f} %', ha='center', va='center', fontsize=17, fontweight='bold', color='white')
rng_ = np.random.default_rng(0); ax.scatter(rng_.uniform(-0.12, 0.12, len(EXu)), EXu.fp_rate_normoxic, color='k', s=28, zorder=4, label='per person')
ax.scatter(rng_.uniform(0.88, 1.12, 6), rest_pp, color='k', s=28, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels(lab_, fontsize=11.5); ax.set_ylim(0, 1.12); ax.set_ylabel('False-positive rate\n(flagged hypoxic while SpO2 normal)')
ax.set_title('Exercise raises heart rate: false alarms when not at rest', fontsize=14.5); ax.legend(loc='upper right')
fig.text(0.5, -0.06, 'Exercise bar: windows with Nonin SpO2 >= 92 after the first 60 s, window-weighted over the %d people with an extracted heart rate (median person %.0f %%).\nPooling all 14 csv rows, incl. Subject4 with no extracted HR (never flagged), gives %.0f %% (the 63 %% quoted in RESULTS.md). Rest bars: 1 - specificity.\nThe app therefore requires 60 s seated rest and a motion gate; the claim is limited to "at rest".' % (len(EXu), 100 * ex_median, 100 * ex_pooled14), ha='center', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG + 'fig_exercise_false_alarm.png', bbox_inches='tight'); plt.close(fig)
pd.DataFrame(dict(setting=['exercise normoxic (moshekam, window-weighted, %d people)' % len(EXu), 'rest TINY (1-spec)', 'rest DeltaHR-only (1-spec)'], false_positive_rate=vals)).to_csv(RES + 'exercise_vs_rest_fpr.csv', index=False)
print('exercise pooled FP %.3f over %d people (median %.3f); all 14 rows %.3f' % (ex_pooled, len(EXu), ex_median, ex_pooled14))

# ---------------------------------------------------------------- SUBGROUPS.md
D = pd.read_csv(SC + 'demographics.csv'); Dl = D[D.hand == 'Left'].copy(); Dr = D[D.hand == 'Right']
consistent = all((Dl.set_index('sid')[c] == Dr.set_index('sid')[c]).all() for c in ['ethnicity', 'age', 'sex', 'skintone_hex'])
SG = Dl.drop(columns='hand').assign(person=lambda d: d.sid % 100).merge(TB[TB.person != 'pooled'].assign(person=lambda d: d.person.astype(int)), on='person')
SG = SG[['person', 'sid', 'age', 'sex', 'ethnicity', 'skintone_hex', 'n_windows', 'base_rate_hypoxemic', 'acc', 'sens', 'spec', 'abstain_rate_band010', 'acc_on_answered_band010', 'clock_acc']]
SG.to_csv(RES + 'subgroups_per_person.csv', index=False)
dark = SG[SG.ethnicity.str.contains('Black')]
with open(RES + 'SUBGROUPS.md', 'w') as f:
    f.write('# Subgroup note\n\nSource of demographics: the Hoffman 2022 repository (`ubicomplab/oximetry-phone-cam-data`), `data/info/{Left,Right}/*.info`, field `subject_info` (ethnicity, age, sex, skintone hex). '
            'These are the only per-person demographics the dataset documents; the README gives none, and the repository has no Fitzpatrick or Monk score (only a skin-tone hex colour). '
            'Left and right-hand files agree for every person: %s. The table below copies them verbatim and puts the TINY per-person metrics (default threshold 0.5, windows t >= 60 s, leave-one-person-out) next to them.\n\n' % consistent)
    f.write(md(SG.rename(columns={'acc': 'TINY_acc', 'sens': 'TINY_sens', 'spec': 'TINY_spec', 'clock_acc': 'clock_acc'})) + '\n')
    if len(dark):
        q = dark.iloc[0]
        f.write(f"**Dark-skinned person.** Person {int(q.person)} (sid {int(q.sid)}) is the only participant documented as African American/Black, with the darkest skin-tone hex ({q.skintone_hex}); the other five are documented as White "
                f"(hex {', '.join(sorted(set(SG[SG.person != q.person].skintone_hex)))}). TINY on this person: accuracy {q.acc:.3f}, sensitivity {q.sens:.3f}, specificity {q.spec:.3f}, abstain-rate {q.abstain_rate_band010:.3f} at band 0.10. "
                f"For comparison the other five people: accuracy {SG[SG.person != q.person].acc.min():.2f} to {SG[SG.person != q.person].acc.max():.2f}.\n\n")
    f.write('**What this does and does not show.** The groups contain 1 Black participant, 3 women and 3 men aged 20 to 34 (all adults); person 4 (the worst-performing person, accuracy %.3f) is a 20-year-old White man with a skin-tone hex of %s. '
            'With only one Black participant and five White participants, per-person differences cannot be attributed to skin tone, sex or age, and no subgroup or fairness claim is made. '
            'Not documented anywhere in the repository: BMI, hand size, nail polish, perfusion at rest, medications, or any clinical history. Dataset-wide caveats: one phone (Nexus 6P), fingertip only, controlled FiO2 protocol.\n' % (
                SG[SG.person == worst_p].acc.iloc[0], SG[SG.person == worst_p].skintone_hex.iloc[0]))
print('done')
