"""OPTICS v2 experiment: does a physically motivated per-beat optics path beat the TINY chain on Hoffman (LOPO) ?
Run: /home/aravhawk/.t3/scratch/veritas/oxy_push/.venv/bin/python -s scripts/optics_experiment.py
Scratch/cache: /home/aravhawk/.t3/scratch/optics_tmp. Outputs: results/optics_experiment/*.csv (+ JSON weights only if a variant clearly wins).
Protocol: person-level leave-one-person-out (both hands together); hyperparameters chosen by inner LOPO on the 5 training people;
per-recording calibration = median of the first 60 s (gated windows); no elapsed-time input; windows 10 s / 2 s step; t >= 60 s evaluated.
"""
import os, sys, json, pickle, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/home/aravhawk/.t3/scratch/veritas'
SUB = ROOT + '/submission/oxydrop'
sys.path.insert(0, HERE); sys.path.insert(0, SUB + '/python'); sys.path.insert(0, ROOT + '/pivot')
import optics_feats as O
import kit
from oxydrop.hoffman import load
OUT = SUB + '/results/optics_experiment'; TMP = '/home/aravhawk/.t3/scratch/optics_tmp'
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
SC = SUB + '/results/scores/'
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- features
y_all, sid, t, ev, PEOPLE = kit.y_all, kit.sid, kit.t, kit.ev, kit.PEOPLE
recs_of = kit.df.rec.values
yref = kit.df.y.values
cache = TMP + '/hof_optics.pkl'
if os.path.exists(cache):
    feats = pickle.load(open(cache, 'rb'))
else:
    feats = {}
    for r in load(ROOT + '/_feas_photon/repo'):
        ts, rows = O.rec_optics(r['rgb']); feats[r['rec']] = pd.DataFrame(rows).assign(t=ts)
    pickle.dump(feats, open(cache, 'wb'))
F = kit.df[['rec', 't']].merge(pd.concat([feats[r].assign(rec=r) for r in feats]), on=['rec', 't'], how='left')
assert (F.rec.values == recs_of).all() and (F.t.values == t).all()


def gate_of(F):
    return (F.valid_hr.astype(bool) & (F.sat <= O.SAT_MAX) & (F.amp_cv <= O.CV_MAX) & (F.nb >= O.NB_MIN)
            & F.hr.between(30, 200)).values


def calibrate(F, gate, tcol, cal=60):
    """Per-recording delta features = raw - median over the first `cal` s (gated windows; all finite windows if <3 gated)."""
    D = pd.DataFrame(index=F.index, columns=O.RAW, dtype=float)
    for rec in F.rec.unique():
        m = (F.rec == rec).values
        base_m = m & (tcol + O.WIN <= cal)
        b = base_m & gate
        if b.sum() < 3:
            b = base_m
        base = F.loc[b, O.RAW].median()
        D.loc[m, O.RAW] = F.loc[m, O.RAW].values - base.values
    return D


gate = gate_of(F)
D = calibrate(F, gate, t)
log(f'gate coverage on evaluated windows: {gate[ev].mean():.4f}; valid_hr {F.valid_hr[ev].mean():.4f}')

ROR = ['ror_RG', 'ror_RB', 'ror_GB']; DCF = ['dc_R', 'dc_G', 'dc_B']; PI = ['pi_R', 'pi_G', 'pi_B']; HR = ['hr']
SETS = {
    'OPT (RoR+PI+DC+CV)': ROR + PI + DCF + ['amp_cv'],
    'OPT+dHR': ROR + PI + DCF + ['amp_cv'] + HR,
    'RoR only': ROR, 'DC only': DCF, 'PI only': PI, 'dHR only (new HR, gated)': HR,
    'RoR+dHR': ROR + HR, 'RoR_GB only (post hoc)': ['ror_GB'], 'RoR_GB+dHR (post hoc)': ['ror_GB'] + HR, 'DC+dHR': DCF + HR, 'PI+dHR': PI + HR, 'RoR_RG only': ['ror_RG'],
}
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
tiny = np.load(SC + 'tiny.npy'); dhr_old = np.load(SC + 'dhr_only.npy'); clock = kit.TIME_ONLY
Xall = {k: D[v].values for k, v in SETS.items()}
EVG = ev & gate                      # evaluated, answered by the gate
TRG = ev & gate                      # training windows: same rule (t >= 60, gated-in)

# ---------------------------------------------------------------- models
import lightgbm as lgb


def fit_pred(kind, X, ytr, Xte, hp):
    if kind == 'lr':
        sc = StandardScaler().fit(X); m = LogisticRegression(C=hp, max_iter=2000).fit(sc.transform(X), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    n_est, leaves = hp
    m = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=0.05, min_child_samples=150, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=20.0, max_depth=3, n_jobs=2, verbose=-1, random_state=0)
    m.fit(X, ytr); return m.predict_proba(Xte)[:, 1]


GRID = {'lr': CS, 'gbm': [(40, 3), (120, 3), (40, 6)]}


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def inner_oof(kind, X, tr_mask, hp, people):
    """inner LOPO out-of-fold scores on training people."""
    s = np.full(len(X), np.nan)
    for q in people:
        a = tr_mask & (sid != q); b = tr_mask & (sid == q)
        if a.sum() < 50 or b.sum() == 0: continue
        s[b] = fit_pred(kind, X[a], y_all[a], X[b], hp)
    return s


def lopo(kind, X, mask_tr=TRG, grid=None, return_hp=False):
    """Nested LOPO: for each held-out person choose hp by inner LOPO log loss on the other 5, refit on all 5, score the held-out person."""
    grid = grid or GRID[kind]
    score = np.full(len(X), np.nan); chosen = {}
    allfin = np.isfinite(X).all(1)
    trm = mask_tr & allfin
    for p in PEOPLE:
        tr = trm & (sid != p); others = [q for q in PEOPLE if q != p]
        best, bl = None, 9
        for hp in grid:
            s = inner_oof(kind, X, tr, hp, others); m = tr & np.isfinite(s)
            l = ll(y_all[m], s[m])
            if l < bl: bl, best = l, hp
        te = (sid == p) & allfin
        score[te] = fit_pred(kind, X[tr], y_all[tr], X[te], best); chosen[int(p) % 100] = best
    return (score, chosen) if return_hp else score


# ---------------------------------------------------------------- metrics (same definitions as kit, any keep mask)
def basic(y, p, s):
    tp = ((y == 1) & (p == 1)).sum(); tn = ((y == 0) & (p == 0)).sum()
    sens = tp / max(y.sum(), 1); spec = tn / max((1 - y).sum(), 1)
    return dict(acc=(p == y).mean(), bacc=(sens + spec) / 2, sens=sens, spec=spec, auc=kit.auc(y, s))


def tm_exact(y, p, tt):
    """Expected time-matched accuracy = kit.tm in expectation (bins of 100 s; per bin equal numbers of hypoxic/normal windows)."""
    b = (tt // 100).astype(int); num = den = 0.0
    for k in np.unique(b):
        mk = b == k; n1 = y[mk].sum(); n0 = mk.sum() - n1
        if min(n1, n0) < max(5, 0.08 * mk.sum()): continue
        m = min(n1, n0); sens_k = (p[mk & (y == 1)] == 1).mean(); spec_k = (p[mk & (y == 0)] == 0).mean()
        num += m * (sens_k + spec_k); den += 2 * m
    return num / den if den else np.nan


def metrics(score, keep, thr=0.5, pred=None):
    m = ev & keep & np.isfinite(score)
    s = score[m]; p = (s > thr).astype(int) if pred is None else pred[m]; y = y_all[m]
    r = basic(y, p, s); r['tm_acc'] = tm_exact(y, p, t[m]); r['coverage'] = m.sum() / ev.sum(); r['n'] = int(m.sum())
    pp = {int(q) % 100: float((p[sid[m] == q] == y[sid[m] == q]).mean()) for q in PEOPLE if (sid[m] == q).sum() > 0}
    r['worst_person'] = min(pp.values()); r['per_person'] = pp
    return r


def boot_ci(score, keep, nb=400, seed=1, thr=0.5, base=None):
    """person bootstrap CIs of acc,bacc,auc,tm_acc (and paired difference to `base` score on same windows)."""
    m = ev & keep & np.isfinite(score) & (np.isfinite(base) if base is not None else True)
    idx = {q: np.where(m & (sid == q))[0] for q in PEOPLE}
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        pick = rng.choice(PEOPLE, len(PEOPLE)); ii = np.concatenate([idx[q] for q in pick])
        y = y_all[ii]; s = score[ii]; p = (s > thr).astype(int); r = basic(y, p, s); r['tm_acc'] = tm_exact(y, p, t[ii])
        if base is not None:
            pb = (base[ii] > thr).astype(int); rb = basic(y, pb, base[ii]); rb['tm_acc'] = tm_exact(y, pb, t[ii])
            r = {k: r[k] - rb[k] for k in ('acc', 'bacc', 'auc', 'tm_acc')}
        out.append([r['acc'], r['bacc'], r['auc'], r['tm_acc']])
    out = np.array(out); lo, hi = np.nanpercentile(out, [2.5, 97.5], 0)
    return dict(zip(['acc', 'bacc', 'auc', 'tm_acc'], zip(lo, hi)))


ROWS = []


def add_row(name, score, keep, group, thr=0.5, ci=True, base=None):
    r = metrics(score, keep, thr)
    row = dict(group=group, model=name, coverage=r['coverage'], n=r['n'])
    for k in ('acc', 'bacc', 'sens', 'spec', 'auc', 'tm_acc', 'worst_person'):
        row[k] = r[k]
    if ci:
        c = boot_ci(score, keep, thr=thr)
        for k, (lo, hi) in c.items(): row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
    if base is not None:
        c = boot_ci(score, keep, thr=thr, base=base)
        for k, (lo, hi) in c.items(): row[f'd_{k}_vsTINY_lo'], row[f'd_{k}_vsTINY_hi'] = lo, hi
        r2 = metrics(base, keep); row['d_acc_vsTINY'] = r['acc'] - r2['acc']; row['d_tm_vsTINY'] = r['tm_acc'] - r2['tm_acc']
        row['d_worst_vsTINY'] = r['worst_person'] - r2['worst_person']
    for q, v in r['per_person'].items(): row[f'p{q}'] = v
    ROWS.append(row); return row


def fmt(row):
    return (f"{row['model'][:46]:46s} acc {row['acc']:.3f} bacc {row['bacc']:.3f} sens {row['sens']:.3f} spec {row['spec']:.3f} auc {row['auc']:.3f} "
            f"TM {row['tm_acc']:.3f} worst {row['worst_person']:.3f} cov {row['coverage']:.3f}")


# ---------------------------------------------------------------- run: baselines on the gated windows
t0 = time.time()
SCORES = {}
for nm, sc in [('time-only clock', clock), ('dHR-only (TINY stage 1, shipped)', dhr_old), ('TINY chain', tiny)]:
    row = add_row(nm, sc, gate, 'baseline-on-gated-windows'); log(fmt(row))
row = add_row('TINY chain (ALL windows, reference)', tiny, np.ones(len(t), bool), 'baseline-all-windows'); log(fmt(row))
row = add_row('time-only clock (ALL windows)', clock, np.ones(len(t), bool), 'baseline-all-windows'); log(fmt(row))
row = add_row('dHR-only (ALL windows)', dhr_old, np.ones(len(t), bool), 'baseline-all-windows'); log(fmt(row))

VARIANTS = []
for nm, X in Xall.items():
    sc, hp = lopo('lr', X, return_hp=True); SCORES[f'LR {nm}'] = sc
    VARIANTS.append(f'LR {nm}')
    row = add_row(f'LR {nm}', sc, gate, 'optics-v2 (LR)', base=tiny); log(fmt(row), 'C chosen', sorted(set(hp.values())), f'{time.time()-t0:.0f}s')
for nm in ['OPT+dHR', 'OPT (RoR+PI+DC+CV)']:
    sc, hp = lopo('gbm', Xall[nm], return_hp=True); SCORES[f'GBM {nm}'] = sc; VARIANTS.append(f'GBM {nm}')
    row = add_row(f'GBM {nm}', sc, gate, 'optics-v2 (tiny GBM)', base=tiny); log(fmt(row), 'hp chosen', list(hp.values()), f'{time.time()-t0:.0f}s')

# ensemble (post hoc, listed as a variant): mean of TINY and the OPT+dHR LR probability
ens = 0.5 * (tiny + SCORES['LR OPT+dHR']); SCORES['TINY+OPT+dHR average'] = ens; VARIANTS.append('TINY+OPT+dHR average')
row = add_row('TINY + (LR OPT+dHR) average', ens, gate, 'optics-v2 + TINY', base=tiny); log(fmt(row))

# ungated version (windows without beats imputed with delta 0 = calibration baseline) -> shows the effect of gating
Xun = np.nan_to_num(Xall['OPT+dHR'], nan=0.0)
EVN = np.ones(len(t), bool)
sc_un = np.full(len(t), np.nan)
allw = ev & np.isfinite(Xun).all(1)
for p in PEOPLE:
    tr = allw & (sid != p); others = [q for q in PEOPLE if q != p]
    bl, best = 9, None
    for hp in CS:
        s = inner_oof('lr', Xun, tr, hp, others); m = tr & np.isfinite(s); l = ll(y_all[m], s[m])
        if l < bl: bl, best = l, hp
    te = sid == p; sc_un[te] = fit_pred('lr', Xun[tr], y_all[tr], Xun[te], best)
SCORES['LR OPT+dHR ungated'] = sc_un; VARIANTS.append('LR OPT+dHR ungated')
row = add_row('LR OPT+dHR, no gating (all windows)', sc_un, EVN, 'optics-v2 (no gate)'); log(fmt(row))
row = add_row('TINY on same all windows', tiny, EVN, 'baseline-all-windows'); log(fmt(row))

# ---------------------------------------------------------------- per-person Spearman of calibrated optical features vs reference SpO2
rows = []
okm = ev & gate
for fn in O.RAW:
    r = dict(feature=fn)
    for q in PEOPLE:
        m = okm & (sid == q) & np.isfinite(D[fn].values)
        r[f'p{int(q)%100}'] = spearmanr(D[fn].values[m], yref[m])[0]
        # partial Spearman controlling for elapsed time (rank-based partial correlation; time is used ONLY for this diagnostic)
        rf, ry, rt = (rankdata(D[fn].values[m]), rankdata(yref[m]), rankdata(t[m]))
        def rr(a, b): return np.corrcoef(a, b)[0, 1]
        num = rr(rf, ry) - rr(rf, rt) * rr(ry, rt); den = np.sqrt((1 - rr(rf, rt) ** 2) * (1 - rr(ry, rt) ** 2))
        r[f'pt{int(q)%100}'] = num / den
    vals = np.array([r[f'p{int(q)%100}'] for q in PEOPLE]); valsp = np.array([r[f'pt{int(q)%100}'] for q in PEOPLE])
    r['n_pos'] = int((vals > 0).sum()); r['n_neg'] = int((vals < 0).sum()); r['mean_rho'] = vals.mean()
    r['n_pos_timepartial'] = int((valsp > 0).sum()); r['n_neg_timepartial'] = int((valsp < 0).sum()); r['mean_rho_timepartial'] = valsp.mean()
    rows.append(r)
SP = pd.DataFrame(rows); SP.to_csv(OUT + '/spearman_per_person.csv', index=False)
log('\nPer-person Spearman (feature delta vs reference SpO2; plain / time-partial):')
log(SP.round(2).to_string(index=False))

# ---------------------------------------------------------------- selective prediction (abstain on |p-0.5|<c)
def nested_cutoffs(kind, X, hp_by_person, target):
    CUTS = np.round(np.arange(0, 0.481, 0.02), 2); allfin = np.isfinite(X).all(1); trm = TRG & allfin
    keepc = np.zeros(len(t), bool); cuts = {}
    for p in PEOPLE:
        tr = trm & (sid != p); others = [q for q in PEOPLE if q != p]
        so = inner_oof(kind, X, tr, hp_by_person[int(p) % 100], others); mi = tr & np.isfinite(so)
        c = CUTS[-1]
        for cc in CUTS:
            a = mi & (np.abs(so - .5) >= cc)
            if a.sum() >= 100:
                b = basic(y_all[a], (so[a] > .5).astype(int), so[a])
                if min(b['acc'], b['bacc']) >= target: c = cc; break
        cuts[int(p) % 100] = float(c)
    return cuts


SEL = []


def sel_rows(name, score, keepbase, cuts=None, tag=''):
    bands = [0.0, 0.05, 0.10, 0.15, 0.20]
    for bnd in bands:
        k = keepbase & np.isfinite(score) & (np.abs(np.nan_to_num(score, nan=.5) - .5) >= bnd)
        r = metrics(score, k); SEL.append(dict(model=name, rule=f'fixed band {bnd:.2f}', coverage_of_all_eval=r['coverage'], acc=r['acc'], bacc=r['bacc'],
                                              sens=r['sens'], spec=r['spec'], auc=r['auc'], tm_acc=r['tm_acc'], worst_person=r['worst_person']))
    if cuts is not None:
        k = keepbase & np.isfinite(score)
        for p, c in cuts.items(): k &= ~((sid % 100 == p) & (np.abs(np.nan_to_num(score, nan=.5) - .5) < c))
        r = metrics(score, k); SEL.append(dict(model=name, rule=f'nested cutoff {tag}', coverage_of_all_eval=r['coverage'], acc=r['acc'], bacc=r['bacc'],
                                              sens=r['sens'], spec=r['spec'], auc=r['auc'], tm_acc=r['tm_acc'], worst_person=r['worst_person']))


def best_name():
    cand = [r for r in ROWS if r['group'] == 'optics-v2 (LR)' and 'dHR only' not in r['model'] and 'post hoc' not in r['model']]
    return max(cand, key=lambda r: r['bacc'])['model']      # post hoc: best optics-bearing LR by balanced accuracy


sel_rows('TINY chain', tiny, gate)
best_lr = best_name(); log('\nbest optics-bearing LR variant by balanced accuracy (post hoc):', best_lr)
Xbest = Xall[best_lr[3:]]
sc_b, hp_b = lopo('lr', Xbest, return_hp=True)
for tgt in (0.90,):
    cuts = nested_cutoffs('lr', Xbest, hp_b, tgt)
    sel_rows(best_lr, SCORES[best_lr], gate, cuts, f'(target acc&bacc>={tgt})'); log('nested cuts', cuts)
if 'LR OPT+dHR' != best_lr:
    hp_o = lopo('lr', Xall['OPT+dHR'], return_hp=True)[1]
    cuts = nested_cutoffs('lr', Xall['OPT+dHR'], hp_o, 0.90); sel_rows('LR OPT+dHR', SCORES['LR OPT+dHR'], gate, cuts, '(target 0.90)')
sel_rows('TINY+OPT+dHR average', ens, gate)
SELD = pd.DataFrame(SEL); SELD.to_csv(OUT + '/selective_prediction.csv', index=False)
log('\nSelective prediction:\n' + SELD.round(3).to_string(index=False))

# ---------------------------------------------------------------- Person 2 (only Black participant) and per-person table
R = pd.DataFrame(ROWS); R.to_csv(OUT + '/main_table.csv', index=False)
pp = R[['group', 'model'] + [c for c in R.columns if c.startswith('p') and c[1:].isdigit()]]
pp.to_csv(OUT + '/per_person_accuracy.csv', index=False)
pickle.dump(dict(SCORES=SCORES, gate=gate, D=D), open(TMP + '/scores.pkl', 'wb'))
log('\nPer-person accuracy (person 2 = only Black participant):\n' + pp.round(3).to_string(index=False))
log('\nMAIN TABLE written;', len(R), 'rows; elapsed', f'{time.time()-t0:.0f}s')
open(OUT + '/run_log_hoffman.txt', 'w').write('\n'.join(LOG))

# ================================================================ MTHS (out-of-distribution phone, normoxic people; evaluation only, CC BY-NC-ND)
LOG.clear()
import glob
import oxydrop.chain as chain
MTHS = ROOT + '/oxy_push/data/mths/repo/MTHS/Data'
W = json.load(open(SUB + '/python/tiny_chain_weights.json'))
MCAL = 20         # MTHS clips are only 26-134 s (most ~60 s): calibration = first 20 s instead of 60 s; evaluated windows t >= 20 s


def fit_final(kind, X, mask):
    others = list(PEOPLE); bl, best = 9, None
    for hp in GRID[kind]:
        s_ = inner_oof(kind, X, mask, hp, others); m = mask & np.isfinite(s_); l = ll(y_all[m], s_[m])
        if l < bl: bl, best = l, hp
    sc = StandardScaler().fit(X[mask]); m_ = LogisticRegression(C=best, max_iter=2000).fit(sc.transform(X[mask]), y_all[mask])
    return sc, m_, best


FINAL = {}
for nm in ['OPT+dHR', 'DC+dHR', 'RoR+dHR']:
    X = Xall[nm]; mk = TRG & np.isfinite(X).all(1)
    FINAL[nm] = fit_final('lr', X, mk); log(f'final fit on all 6 Hoffman people: {nm} C={FINAL[nm][2]} n={mk.sum()}')

mrecs = []
for f in sorted(glob.glob(MTHS + '/signal_*.npy'), key=lambda s_: int(s_.split('_')[-1][:-4])):
    i = int(f.split('_')[-1][:-4]); rgb = np.load(f).astype(float); lab = np.load(f'{MTHS}/label_{i}.npy').astype(float)
    mrecs.append(dict(i=i, rgb=rgb, hr_ref=np.where(lab[:, 0] > 0, lab[:, 0], np.nan), gt=np.where(lab[:, 1] > 0, lab[:, 1], np.nan)))
log(f'MTHS: {len(mrecs)} clips, 30 fps assumed (frames/label-rows = {np.median([len(r["rgb"])/len(r["gt"]) for r in mrecs]):.1f}); duration {min(len(r["rgb"])/30 for r in mrecs):.0f}-{max(len(r["rgb"])/30 for r in mrecs):.0f} s')
chain.CAL = MCAL
mrows = []; mfeat = []
for r in mrecs:
    n = len(r['rgb']) // 30
    if n < MCAL + 10: continue
    ts, rows = O.rec_optics(r['rgb']); Fm = pd.DataFrame(rows).assign(t=ts, i=r['i'])
    yw = np.array([np.nanmean(r['gt'][s_:s_ + 10]) if np.isfinite(r['gt'][s_:s_ + 10]).sum() >= 5 else np.nan for s_ in ts])
    hw = np.array([np.nanmean(r['hr_ref'][s_:s_ + 10]) if np.isfinite(r['hr_ref'][s_:s_ + 10]).sum() >= 5 else np.nan for s_ in ts])
    Fm['y'] = yw; Fm['hr_ref'] = hw
    g_full = gate_of(Fm); g_nosat = (Fm.valid_hr.astype(bool) & (Fm.amp_cv <= O.CV_MAX) & (Fm.nb >= O.NB_MIN) & Fm.hr.between(30, 200)).values
    Dm = calibrate(Fm, g_full, Fm.t.values, cal=MCAL); Dn = calibrate(Fm, g_nosat, Fm.t.values, cal=MCAL)
    # TINY (shipped weights) with 20 s calibration
    res = {s_: (p, lab) for s_, p, lab in chain.infer(r['rgb'], W)}
    Fm['p_tiny'] = [res[s_][0] if s_ in res else np.nan for s_ in ts]
    for nm, (sc, m_, C) in FINAL.items():
        X = Dm[SETS[nm]].values; ok = np.isfinite(X).all(1)
        p = np.full(len(ts), np.nan); p[ok] = m_.predict_proba(sc.transform(X[ok]))[:, 1]; Fm[f'p_{nm}_gated'] = p
        Xu = np.nan_to_num(Dn[SETS[nm]].values, nan=0.0)      # ungated: all windows, sat check ignored, missing deltas = 0
        Fm[f'p_{nm}_ungated'] = m_.predict_proba(sc.transform(np.nan_to_num(calibrate(Fm, np.ones(len(Fm), bool), Fm.t.values, cal=MCAL)[SETS[nm]].values, nan=0.0)))[:, 1]
    for c in SETS['OPT+dHR']: Fm['d_' + c] = Dm[c].values; Fm['dn_' + c] = Dn[c].values
    Fm['g_full'] = g_full; Fm['g_nosat'] = g_nosat; Fm['ev'] = Fm.t.values >= MCAL
    mfeat.append(Fm)
M = pd.concat(mfeat, ignore_index=True); M.to_pickle(TMP + '/mths_windows.pkl')
log(f'MTHS clips used (>= {MCAL+10} s): {M.i.nunique()}; evaluated windows (t>={MCAL}): {int(M.ev.sum())}; with reference SpO2: {int((M.ev & M.y.notna()).sum())}')
log(f'sat check: fraction of MTHS windows with >5% of frames having a channel >=250: {(M[M.ev].sat>O.SAT_MAX).mean():.3f}  (Hoffman: 0.000). Green mean channel typically ~0-70 of 255.')

# (0) phone heart-rate sanity vs the dataset's HR reference
mh = M[M.ev & M.valid_hr.astype(bool) & M.hr_ref.notna()]
log(f'MTHS HR check: valid-HR windows {len(mh)}/{int((M.ev & M.hr_ref.notna()).sum())}; MAE vs reference HR {np.abs(mh.hr - mh.hr_ref).mean():.1f} bpm, within 5 bpm {(np.abs(mh.hr - mh.hr_ref) <= 5).mean():.2f}')

# (1) false-alarm rate (person level): fraction of windows flagged hypoxic among windows whose reference SpO2 >= 90
def fa_table(col, gate_col=None):
    out = []
    for i, g in M[M.ev & M.y.notna()].groupby('i'):
        norm = g[g.y >= 90]
        if gate_col is not None: norm = norm[norm[gate_col]]
        pv = norm[col].dropna()
        out.append(dict(person=i, n_norm_windows=len(pv), flagged=(pv > 0.5).mean() if len(pv) else np.nan,
                        unsure_band010=(np.abs(pv - .5) < .10).mean() if len(pv) else np.nan, mean_p=pv.mean() if len(pv) else np.nan))
    T = pd.DataFrame(out); return T


FA = []; rng = np.random.default_rng(3)
for lab, col, gc in [('TINY (shipped weights, 20 s calibration)', 'p_tiny', None),
                     ('OPTICS v2 LR OPT+dHR, gated (full gate incl. saturation)', 'p_OPT+dHR_gated', None),
                     ('OPTICS v2 LR OPT+dHR, ungated (sat ignored)', 'p_OPT+dHR_ungated', None),
                     ('OPTICS v2 LR DC+dHR, ungated', 'p_DC+dHR_ungated', None),
                     ('OPTICS v2 LR RoR+dHR, ungated', 'p_RoR+dHR_ungated', None)]:
    T_ = fa_table(col, gc); T_.to_csv(OUT + f"/mths_false_alarm_per_person_{col.replace('+','_')}.csv", index=False)
    V = T_.dropna(subset=['flagged']); b = [V.flagged.values[rng.integers(0, len(V), len(V))].mean() for _ in range(2000)]
    pooled_n = M[M.ev & M.y.notna() & (M.y >= 90) & M[col].notna()]
    FA.append(dict(model=lab, people=len(V), mean_person_FPR=V.flagged.mean(), ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5),
                   median_person_FPR=V.flagged.median(), pooled_window_FPR=(pooled_n[col] > .5).mean(), people_with_FPR_gt_50pct=int((V.flagged > .5).sum()),
                   mean_unsure_band010=V.unsure_band010.mean(), windows=len(pooled_n)))
FAD = pd.DataFrame(FA); FAD.to_csv(OUT + '/mths_false_alarm_summary.csv', index=False)
log('\nMTHS false-alarm (flagged hypoxic among windows with ref SpO2 >= 90; Hoffman LOPO reference for TINY = 13.3% FPR, answers every window):\n' + FAD.round(3).to_string(index=False))
# coverage of the full gate on MTHS
gcov = M[M.ev & M.y.notna() & (M.y >= 90)].groupby('i').g_full.mean()
log(f'OPTICS gate coverage on MTHS normoxic windows (full gate): mean per person {gcov.mean():.3f}, people with any gated-in window {(gcov>0).sum()}/{len(gcov)}')
log('TINY on MTHS: clips with any ref window < 90: ' + str(sorted(M[M.ev & (M.y < 90)].i.unique().tolist())))

# (2) within MTHS: per-person calibrated optical deltas vs calibrated reference SpO2 inside the normal range
rows = []
for pref, gcol, tag in [('d_', 'g_full', 'full gate'), ('dn_', 'g_nosat', 'saturation check ignored')]:
    for c in SETS['OPT+dHR']:
        rhos = []; n_used = 0
        for i, g in M[M.ev & M.y.notna()].groupby('i'):
            g = g[g[gcol] & (g.y >= 90) & g[pref + c].notna()]
            if len(g) >= 8 and g.y.nunique() >= 3:
                rhos.append(spearmanr(g[pref + c], g.y)[0]); n_used += 1
        rhos = np.array(rhos); allg = M[M.ev & M.y.notna() & M[gcol] & (M.y >= 90) & M[pref + c].notna()]
        # pooled: calibrated feature delta vs calibrated SpO2 delta (label minus the person's own first-20 s median reference)
        base_y = M[(M.t + 10 <= MCAL) & M.y.notna()].groupby('i').y.median()
        dy = allg.y - allg.i.map(base_y)
        rp = spearmanr(allg[pref + c], dy)[0] if len(allg) > 10 else np.nan
        rows.append(dict(gate=tag, feature=c, people_used=n_used, mean_rho=rhos.mean() if n_used else np.nan, n_pos=int((rhos > 0).sum()), n_neg=int((rhos < 0).sum()),
                         pooled_rho_vs_calibrated_spo2=rp, pooled_windows=len(allg)))
MC = pd.DataFrame(rows); MC.to_csv(OUT + '/mths_feature_corr.csv', index=False)
log('\nMTHS within-person Spearman of calibrated optical features vs reference SpO2 (windows with ref >= 90, >=3 distinct label values, >=8 windows):\n' + MC.round(3).to_string(index=False))
log('Hoffman sign reference (6 people): see spearman_per_person.csv (hr -, dc_* -, ror_GB +).')
open(OUT + '/run_log_mths.txt', 'w').write('\n'.join(LOG))
