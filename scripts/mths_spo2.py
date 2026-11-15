"""MTHS SpO2 optics experiment: does the per-window red/green/blue ratio-of-ratios (plus per-person calibration and a small learned correction)
predict pulse-oximeter SpO2 from phone fingertip video, validated person by person?

Run: /home/aravhawk/.t3/scratch/veritas/oxy_push/.venv/bin/python -s scripts/mths_spo2.py
Data: MTHS (Samavati et al. 2022, CC BY-NC-ND). 62 people, mean R,G,B per frame at 30 fps + 1 Hz reference HR/SpO2. Not redistributed here.
Env overrides: OXY_MTHS (the MTHS Data dir), OXY_TMP (feature cache dir, default /home/aravhawk/.t3/scratch/optics_tmp, NOT the RAM-backed /tmp).
Outputs: results/mths_spo2/{summary.csv,per_person.csv,label_range.csv,README.md}.

Protocol (fixed before looking at any model score)
  * windows: 10 s, 2 s hop, same causal band-pass and the same robust-HR lock (hr.py) as the app (optics_feats.rec_optics).
  * quality gate: pulse locked (hr.py valid), saturation <= 10 % of frames in the window (any channel), per-beat amplitude CV <= 0.60, >= 4 beats, HR 30-200.
  * per-person calibration (label free): median of the person's own gated windows that end within the first 30 s. Persons with < 3 such windows
    are not calibrated and are excluded from EVERY model and baseline (all rows share one evaluation set). Evaluation windows start at t >= 30 s.
  * person-level validation only: leave-one-person-out (LeaveOneGroupOut over person id); hyperparameters (ridge alpha, GBM size) chosen by an
    inner GroupKFold over the training people only. No window-level split exists anywhere in this file.
  * uncertainty: person bootstrap (resample people, 1000 draws) 95 % intervals, also for the paired difference to the train-mean baseline.
Two tasks (MTHS is almost entirely normoxic, see label_range.csv, so detection of true hypoxemia cannot be tested):
  A. absolute SpO2 from uncalibrated features (what a stand-alone ratio-of-ratios sensor would need) and AUROC for SpO2 < 95.
  B. change from the person's own first-30 s reference, from calibrated deltas (what the shipped app does), AUROC for a drop of >= 2 points.
"""
import os, sys, time, resource, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SUB, 'python'))
import optics_feats as O          # same window features and HR lock as the Hoffman optics experiment
MTHS = os.environ.get('OXY_MTHS', '/home/aravhawk/.t3/scratch/veritas/oxy_push/data/mths/repo/MTHS/Data')
TMP = os.environ.get('OXY_TMP', '/home/aravhawk/.t3/scratch/optics_tmp') + '/mths_feats'
OUT = os.path.join(SUB, 'results', 'mths_spo2')
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
CAL = 30; SAT_MAX = 0.10; BOOT = 1000; DROP = 2.0; POS_THR = 95.0
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ 1. per-person windows (one person at a time, cached)
def person_windows(i):
    cp = f'{TMP}/p{i}.pkl'
    if os.path.exists(cp):
        return pd.read_pickle(cp)
    rgb = np.load(f'{MTHS}/signal_{i}.npy').astype(float); lab = np.load(f'{MTHS}/label_{i}.npy').astype(float)
    assert len(rgb) // 30 == len(lab) or abs(len(rgb) / 30 - len(lab)) <= 1, 'unexpected frame rate'
    hr_ref = np.where(lab[:, 0] > 0, lab[:, 0], np.nan); sp = np.where(lab[:, 1] > 0, lab[:, 1], np.nan)
    ts, rows = O.rec_optics(rgb)
    F = pd.DataFrame(rows).assign(t=ts, i=i)
    mean = lambda v, s: np.nanmean(v[s:s + 10]) if np.isfinite(v[s:s + 10]).sum() >= 5 else np.nan
    F['y'] = [mean(sp, s) for s in ts]; F['hr_ref'] = [mean(hr_ref, s) for s in ts]
    F['satR'] = [float((rgb[s * 30:(s + 10) * 30, 0] >= 250).mean()) for s in ts]
    F['satG'] = [float((rgb[s * 30:(s + 10) * 30, 1] >= 250).mean()) for s in ts]
    F['y_base'] = np.nanmedian(sp[:CAL]) if np.isfinite(sp[:CAL]).sum() >= 5 else np.nan     # reference at the start, used ONLY to score task B
    F['dur'] = len(rgb) / 30.0
    F.to_pickle(cp)
    return F


ids = sorted(int(f.split('_')[-1][:-4]) for f in os.listdir(MTHS) if f.startswith('signal_'))
W = pd.concat([person_windows(i) for i in ids], ignore_index=True)
log(f'MTHS: {len(ids)} people, {len(W)} windows (10 s / 2 s hop), recordings {W.groupby("i").dur.first().min():.0f}-{W.groupby("i").dur.first().max():.0f} s '
    f'(median {W.groupby("i").dur.first().median():.0f} s)  [{time.time()-T0:.0f}s]')

# ------------------------------------------------------------------ 2. label range first (decides which metrics make sense)
lab_all = np.concatenate([np.load(f'{MTHS}/label_{i}.npy')[:, 1] for i in ids]); lab_all = lab_all[lab_all > 0]
pm = pd.Series({i: np.nanmean(np.where(np.load(f'{MTHS}/label_{i}.npy')[:, 1] > 0, np.load(f'{MTHS}/label_{i}.npy')[:, 1], np.nan)) for i in ids})
pmin = pd.Series({i: np.nanmin(np.where(np.load(f'{MTHS}/label_{i}.npy')[:, 1] > 0, np.load(f'{MTHS}/label_{i}.npy')[:, 1], np.nan)) for i in ids})
lr = dict(n_label_seconds=len(lab_all), min=lab_all.min(), p5=np.percentile(lab_all, 5), median=np.median(lab_all), max=lab_all.max(),
          frac_below_95=float((lab_all < 95).mean()), frac_below_90=float((lab_all < 90).mean()), n_below_90_seconds=int((lab_all < 90).sum()),
          people_mean_below_95=int((pm < 95).sum()), people_mean_below_90=int((pm < 90).sum()), people_any_below_95=int((pmin < 95).sum()),
          people_any_below_90=int((pmin < 90).sum()), person_mean_min=pm.min(), person_mean_max=pm.max(),
          sd_between_person_means=pm.std(), mean_within_person_sd=float(np.mean([np.nanstd(np.where(np.load(f'{MTHS}/label_{i}.npy')[:, 1] > 0, np.load(f'{MTHS}/label_{i}.npy')[:, 1], np.nan)) for i in ids])))
pd.DataFrame([lr]).T.rename(columns={0: 'value'}).to_csv(f'{OUT}/label_range.csv')
log('\nLABEL RANGE (reference oximeter, 1 Hz, valid seconds only): ' + ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in lr.items()))
log('=> a normoxic cohort. True hypoxemia (<90) is ~1 % of label seconds, so the questions answerable here are (A) absolute-SpO2 error and a '
    '"<95" screen, and (B) change from own start; NOT hypoxemia detection.')

# ------------------------------------------------------------------ 3. features, gate, per-person calibration
RAW = O.RAW; ROR = ['ror_RG', 'ror_RB', 'ror_GB']; PI = ['pi_R', 'pi_G', 'pi_B']; DCF = ['dc_R', 'dc_G', 'dc_B']
W['has_y'] = W.y.notna()
W['g_pulse'] = (W.valid_hr.astype(bool) & (W.amp_cv <= O.CV_MAX) & (W.nb >= O.NB_MIN) & W.hr.between(30, 200)).values
W['g_full'] = W.g_pulse & (W.sat <= SAT_MAX)
log(f'\nGATE (labelled windows with t >= {CAL} s): pulse lock/beat checks pass {W[W.has_y & (W.t >= CAL)].g_pulse.mean():.3f}; '
    f'+ saturation <= {SAT_MAX:.0%} (any channel) {W[W.has_y & (W.t >= CAL)].g_full.mean():.3f}; '
    f'red saturated >{SAT_MAX:.0%} in {(W[W.has_y & (W.t >= CAL)].satR > SAT_MAX).mean():.3f} of windows, green in {(W[W.has_y & (W.t >= CAL)].satG > SAT_MAX).mean():.3f}')


def prepare(gate_col):
    """Returns the frame of EVALUATION windows with calibrated deltas (shared by every model for this gate)."""
    D = W.copy(); D['gate'] = D[gate_col]
    deltas = []; ncal = {}
    for i, g in D.groupby('i'):
        b = g[g.gate & (g.t + 10 <= CAL)]
        ncal[i] = len(b)
        base = b[RAW].median() if len(b) >= 3 else pd.Series(np.nan, index=RAW)
        deltas.append(pd.DataFrame(g[RAW].values - base.values, index=g.index, columns=['d_' + c for c in RAW]))
    D = D.join(pd.concat(deltas)); D['calibrated'] = D.i.map(lambda i: ncal[i] >= 3)
    E = D[D.gate & D.has_y & (D.t >= CAL) & D.calibrated & np.isfinite(D[RAW + ['d_' + c for c in RAW]]).all(1) & D.y_base.notna()].copy()
    E['dy'] = E.y - E.y_base; E['pos'] = (E.y < POS_THR).astype(int); E['drop'] = (E.dy <= -DROP).astype(int)
    n_cal = sum(v >= 3 for v in ncal.values())
    return D, E, n_cal


# ------------------------------------------------------------------ 4. models (person-level LOPO, inner GroupKFold for hyperparameters)
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]; GBM_GRID = [(30, 0.05), (100, 0.05)]


def fit_pred(kind, hp, Xtr, ytr, Xte):
    if kind == 'ridge':
        sc = StandardScaler().fit(Xtr); m = Ridge(alpha=hp).fit(sc.transform(Xtr), ytr); return m.predict(sc.transform(Xte))
    import lightgbm as lgb
    n, lr_ = hp
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=lr_, num_leaves=4, max_depth=3, min_child_samples=30, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=10.0, n_jobs=1, verbose=-1, random_state=0)
    m.fit(Xtr, ytr); return m.predict(Xte)


def lopo(kind, X, y, groups):
    """Leave-one-person-out predictions; hyperparameter by inner GroupKFold(5) over the training people only."""
    grid = ALPHAS if kind == 'ridge' else GBM_GRID
    pred = np.full(len(y), np.nan); chosen = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        best, be = None, np.inf
        gk = GroupKFold(n_splits=5)
        for hp in grid:
            err = 0.0
            for a, b in gk.split(X[tr], y[tr], groups[tr]):
                err += ((fit_pred(kind, hp, X[tr][a], y[tr][a], X[tr][b]) - y[tr][b]) ** 2).sum()
            if err < be: be, best = err, hp
        pred[te] = fit_pred(kind, best, X[tr], y[tr], X[te]); chosen.append(best)
    return pred, chosen


def auroc(pos, score):
    pos = np.asarray(pos); n1 = pos.sum(); n0 = len(pos) - n1
    if n1 < 1 or n0 < 1: return np.nan
    r = rankdata(score); return (r[pos == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_person_rho(df, pred_col, y_col):
    rs = []
    for _, g in df.groupby('i'):
        if len(g) >= 8 and g[y_col].nunique() >= 3 and g[pred_col].nunique() >= 3:
            r = spearmanr(g[pred_col], g[y_col])[0]
            if np.isfinite(r): rs.append(r)
    return (float(np.mean(rs)) if rs else np.nan), len(rs)


def dev_of(df, pc):
    """Score used for ranking metrics: the model's deviation from its own fold's training mean (task A) so that the leave-one-person-out
    intercept shift (a held-out person with a HIGH label gets a LOWER training mean) cannot create a spurious or inverted ranking signal.
    Textbook (not fitted) predictions are used as they are."""
    if pc.startswith('A_textbook'): return df[pc].values
    if pc.startswith('A_'): return df[pc].values - df['A_train_mean'].values
    return df[pc].values


def point_metrics(df, pc, ycol, task):
    e = df[pc].values - df[ycol].values
    m = dict(n_windows=len(df), n_people=df.i.nunique(), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
             within2=(np.abs(e) <= 2).mean(), within3=(np.abs(e) <= 3).mean(), within5=(np.abs(e) <= 5).mean())
    dv = dev_of(df, pc); g = pd.DataFrame({'p': dv, 'y': df[ycol].values, 'i': df.i.values}).groupby('i').mean()
    if g.p.nunique() > 2 and g.y.nunique() > 2 and g.p.std() > 1e-9:
        m['pearson_person'] = pearsonr(g.p, g.y)[0]; m['spearman_person'] = spearmanr(g.p, g.y)[0]
    else:
        m['pearson_person'] = m['spearman_person'] = np.nan
    m['within_person_rho'], m['within_person_n'] = within_person_rho(df, pc, ycol)
    # detection score: lower predicted SpO2 / more negative predicted change = more suspicious
    lab = 'pos' if task == 'A' else 'drop'
    m['auroc'] = auroc(df[lab].values, -dev_of(df, pc)) if np.std(dev_of(df, pc)) > 1e-9 else 0.5
    return m


def person_bootstrap(df, pc, ycol, task, base_pc=None, seed=7):
    rng = np.random.default_rng(seed); groups = {i: g.index.values for i, g in df.groupby('i')}; keys = np.array(list(groups))
    pos = df['pos' if task == 'A' else 'drop']
    y = df[ycol].values; p = df[pc].values; s = -dev_of(df, pc); b = df[base_pc].values if base_pc else None; ix = {k: np.searchsorted(df.index.values, v) for k, v in groups.items()}
    mae, au, dmae = [], [], []
    for _ in range(BOOT):
        idx = np.concatenate([ix[k] for k in rng.choice(keys, len(keys))])
        e = np.abs(p[idx] - y[idx]); mae.append(e.mean()); au.append(auroc(pos.values[idx], s[idx]) if np.nanstd(s[idx]) > 1e-9 else 0.5)
        if b is not None: dmae.append(e.mean() - np.abs(b[idx] - y[idx]).mean())
    q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if len(v) and np.isfinite(v).any() else (np.nan, np.nan)
    return q(mae), q(au), (q(dmae) if dmae else (np.nan, np.nan))


def run(gate_col, tag):
    D, E, n_cal = prepare(gate_col)
    E = E.sort_index()
    log(f'\n=== GATE "{tag}" ===  people calibrated (>=3 gated windows in first {CAL} s): {n_cal}/{len(ids)};  evaluation windows (t>={CAL}, gated, labelled): '
        f'{len(E)} from {E.i.nunique()} people  (gated coverage of labelled t>={CAL} windows of calibrated people: '
        f'{len(E) / max(((D.has_y & (D.t >= CAL) & D.calibrated)).sum(), 1):.3f})  [{time.time()-T0:.0f}s]')
    log(f'Evaluation-set labels: min {E.y.min():.1f} median {E.y.median():.1f} max {E.y.max():.1f}; windows < 95: {int(E.pos.sum())} ({E.pos.mean():.3f}) from '
        f'{E[E.pos == 1].i.nunique()} people; windows < 90: {int((E.y < 90).sum())}; drops >= {DROP:.0f} points vs own start: {int(E["drop"].sum())} windows from {E[E["drop"] == 1].i.nunique()} people')
    grp = E.i.values; rows = []

    HRF = ['hr']; OPT = ROR + PI + DCF + ['amp_cv']; OPTHR = OPT + HRF
    DOPTHR = ['d_' + c for c in OPTHR]
    # ---------------- Task A: absolute SpO2 from UNcalibrated features
    yA = E.y.values; ctr = {}
    pred_mean = np.full(len(E), np.nan)
    for tr, te in LeaveOneGroupOut().split(E, yA, grp): pred_mean[te] = yA[tr].mean()
    E['A_train_mean'] = pred_mean
    E['A_textbook_RB'] = np.clip(110 - 25 * np.exp(E.ror_RB), 70, 100); E['A_textbook_RG'] = np.clip(110 - 25 * np.exp(E.ror_RG), 70, 100)
    specsA = {'HR only': HRF, 'DC only': DCF, 'time-in-recording only (t)': ['t'], 'RoR only (RG,RB,GB)': ROR, 'OPT (RoR+PI+DC+CV)': OPT, 'OPT+HR': OPTHR,
              'OPT+HR raw + calibrated deltas': OPTHR + DOPTHR, 'calibrated deltas only': DOPTHR}
    for nm, cols in specsA.items():
        E['A_ridge ' + nm], ctr[nm] = lopo('ridge', E[cols].values, yA, grp)
    E['A_GBM OPT+HR'], ctr['gbm'] = lopo('gbm', E[OPTHR].values, yA, grp)
    namesA = ['A_train_mean', 'A_textbook_RB', 'A_textbook_RG'] + ['A_ridge ' + k for k in specsA] + ['A_GBM OPT+HR']
    for pc in namesA:
        m = point_metrics(E, pc, 'y', 'A'); ci, cia, cid = person_bootstrap(E, pc, 'y', 'A', 'A_train_mean')
        rows.append(dict(task='A absolute SpO2 (uncalibrated features)', gate=tag, model=pc[2:], **m, mae_lo=ci[0], mae_hi=ci[1], auroc_lo=cia[0], auroc_hi=cia[1],
                         dmae_vs_train_mean=m['mae'] - point_metrics(E, 'A_train_mean', 'y', 'A')['mae'], dmae_lo=cid[0], dmae_hi=cid[1]))
    # ---------------- Task B: change vs own start from calibrated deltas (label-free calibration), scored against y - y_base
    yB = E.dy.values; E['B_no_change'] = 0.0
    specsB = {'HR delta only': ['d_hr'], 'DC delta only': ['d_dc_R', 'd_dc_G', 'd_dc_B'], 'time-in-recording only (t)': ['t'], 'RoR delta only': ['d_' + c for c in ROR],
              'OPT+HR deltas': DOPTHR}
    for nm, cols in specsB.items():
        E['B_ridge ' + nm], _ = lopo('ridge', E[cols].values, yB, grp)
    E['B_GBM OPT+HR deltas'], _ = lopo('gbm', E[DOPTHR].values, yB, grp)
    namesB = ['B_no_change'] + ['B_ridge ' + k for k in specsB] + ['B_GBM OPT+HR deltas']
    for pc in namesB:
        m = point_metrics(E, pc, 'dy', 'B'); ci, cia, cid = person_bootstrap(E, pc, 'dy', 'B', 'B_no_change')
        rows.append(dict(task=f'B change vs own start (calibrated deltas)', gate=tag, model=pc[2:], **m, mae_lo=ci[0], mae_hi=ci[1], auroc_lo=cia[0], auroc_hi=cia[1],
                         dmae_vs_train_mean=m['mae'] - point_metrics(E, 'B_no_change', 'dy', 'B')['mae'], dmae_lo=cid[0], dmae_hi=cid[1]))
    S = pd.DataFrame(rows)
    show = S[['task', 'model', 'mae', 'mae_lo', 'mae_hi', 'rmse', 'within2', 'within3', 'within5', 'pearson_person', 'within_person_rho', 'auroc', 'auroc_lo', 'auroc_hi', 'dmae_vs_train_mean', 'dmae_lo', 'dmae_hi']]
    log(show.round(3).to_string(index=False))
    log(f'ridge alpha / gbm hp chosen (outer folds, most common): ' + '; '.join(f'{k}: {pd.Series([str(x) for x in v]).mode()[0]}' for k, v in ctr.items()))
    return S, E, D


S1, E1, D1 = run('g_full', 'full gate (pulse lock, beats, CV, saturation<=10%)')
S2, E2, D2 = run('g_pulse', 'saturation ignored (sensitivity)')
S = pd.concat([S1, S2], ignore_index=True); S.to_csv(f'{OUT}/summary.csv', index=False)

# ------------------------------------------------------------------ 5. HR check on the same windows (reference HR from the oximeter)
h = D1[D1.g_full & D1.hr_ref.notna() & (D1.t >= 0)]
hr_mae = (h.hr - h.hr_ref).abs().mean(); hr_w5 = ((h.hr - h.hr_ref).abs() <= 5).mean()
hp_ = D1[D1.valid_hr.astype(bool) & D1.hr_ref.notna()]
log(f'\nHR vs reference (all windows with a lock and a reference HR): MAE {(hp_.hr - hp_.hr_ref).abs().mean():.1f} bpm, within 5 bpm {((hp_.hr - hp_.hr_ref).abs() <= 5).mean():.3f} '
    f'({len(hp_)}/{int(D1.hr_ref.notna().sum())} windows locked); full-gate windows: MAE {hr_mae:.1f}, within 5 {hr_w5:.3f}, n={len(h)}')

# ------------------------------------------------------------------ 6. per-person table (primary gate)
pp = []
for i, g in D1.groupby('i'):
    e = E1[E1.i == i]
    r = dict(person=i, duration_s=g.dur.iloc[0], windows=len(g), labelled=int(g.has_y.sum()), gated_frac=float(g[g.has_y].g_full.mean()) if g.has_y.any() else np.nan,
             pulse_gate_frac=float(g[g.has_y].g_pulse.mean()) if g.has_y.any() else np.nan, red_sat_frac=float(g.satR.mean()), calibrated=bool(g.calibrated.iloc[0]),
             eval_windows=len(e), mean_ref_spo2=float(g.y.mean()), min_ref_spo2=float(g.y.min()))
    for pc in ['A_train_mean', 'A_ridge OPT+HR', 'A_ridge OPT+HR raw + calibrated deltas', 'A_GBM OPT+HR', 'A_textbook_RB']:
        r['mae_' + pc[2:]] = float((e[pc] - e.y).abs().mean()) if len(e) else np.nan
        r['meanpred_' + pc[2:]] = float(e[pc].mean()) if len(e) else np.nan
    pp.append(r)
pd.DataFrame(pp).to_csv(f'{OUT}/per_person.csv', index=False)

# ------------------------------------------------------------------ 7. README (numbers from this run)
def md(df, cols):
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == 'f': d[c] = d[c].map(lambda v: '' if not np.isfinite(v) else f'{v:.3f}')
    return '| ' + ' | '.join(d.columns) + ' |\n|' + '---|' * len(d.columns) + '\n' + '\n'.join('| ' + ' | '.join(str(x) for x in r) + ' |' for r in d.values)


def ci(a, lo, hi): return f'{a:.2f} [{lo:.2f}, {hi:.2f}]'


prim = S1.copy(); sens = S2.copy()
for d in (prim, sens):
    d['MAE [95% CI]'] = [ci(a, b, c) for a, b, c in zip(d.mae, d.mae_lo, d.mae_hi)]
    d['AUROC [95% CI]'] = [('n/a' if not np.isfinite(a) else ci(a, b, c)) for a, b, c in zip(d.auroc, d.auroc_lo, d.auroc_hi)]
    d['dMAE vs baseline [95% CI]'] = [('' if x == 0 else ci(x, b, c)) for x, b, c in zip(d.dmae_vs_train_mean, d.dmae_lo, d.dmae_hi)]
cols = ['model', 'n_windows', 'n_people', 'MAE [95% CI]', 'rmse', 'within2', 'within3', 'within5', 'spearman_person', 'within_person_rho', 'AUROC [95% CI]', 'dMAE vs baseline [95% CI]']
rd = ['# MTHS SpO2 optics experiment (person-level leave-one-person-out)\n', 'Generated by `scripts/mths_spo2.py`. Numbers below are from that run (see `summary.csv`, `per_person.csv`, `label_range.csv`).\n',
      '## Label range (decides what can be tested)\n', '```', *[l for l in LOG if l.startswith('MTHS:') or l.startswith('\nLABEL') or l.startswith('=> a normoxic')], '```\n',
      '## Gate\n', '```', *[l for l in LOG if l.startswith('\nGATE')], '```\n']
for tag, d in (('Primary gate', prim), ('Sensitivity: saturation criterion ignored', sens)):
    for task in sorted(d.task.unique()):
        x = d[d.task == task]
        rd += [f'## {tag}: Task {task}\n', f'Baselines are the first rows ({"train-mean, textbook 110-25R" if task.startswith("A") else "no change"}); ridge/GBM rows are the learned models. '
               f'Same windows and people for every row.\n', md(x, cols), '\n']
rd += ['## HR check\n', '```', *[l for l in LOG if l.startswith('\nHR vs')], '```\n', f'Runtime {time.time()-T0:.0f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB.\n']
open(f'{OUT}/README.md', 'w').write('\n'.join(rd))
log(f'\nDONE in {time.time()-T0:.0f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB')
open(f'{OUT}/run_log.txt', 'w').write('\n'.join(LOG))
