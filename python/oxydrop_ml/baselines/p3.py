"""P3 baselines: absolute SpO2 on MTHS (62 people, iPhone fingertip video), leave-one-subject-out, 10 s windows with t >= 30 s.

Primary metric: SpO2 MAE (percentage points). MTHS IS NORMOXIC (about 1 % of reference seconds below 90, label range in baselines_P3_info.json), so a model that
always predicts the training mean is a very strong baseline and nothing here measures hypoxemia detection.
Models: train_mean, ror_textbook_{RB,RG} (110 - 25 R clipped to 70-100, no fitted parameter), ridge_ratio (ridge on the six log ratio features, alpha by inner CV),
lgbm_hand26 / xgb_hand26 (gradient boosting on the 26 absolute + 26 baseline-relative handcrafted features, hyperparameters by inner CV). All hyperparameters are selected
on training subjects only (inner subject-grouped CV, criterion = MAE). Train-vs-test gap = held-out MAE minus MAE on the refit model's own training windows.
Evaluation set shared by every model: eval windows (t >= 30 s) with a reference SpO2, a personal baseline and finite features.
Per-window MTHS predictions stay in the cache (CC BY-NC-ND); the committed per-person table has aggregates only.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .. import metrics as M
from .. import splits as S
from .. import windows as W
from . import common as C
from . import ror, xgb_hand as XH

METRICS = ['n', 'mae', 'rmse', 'bias', 'within2', 'within5', 'auroc_lt90', 'auroc_lt95']
REF = 'train_mean'
GRID_RIDGE = [dict(alpha=a) for a in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)]
GRID_XGB = [dict(depth=d, n=n) for d in (2, 3) for n in (50, 150)]
GRID_LGB = [dict(leaves=l, n=n) for l in (4, 8) for n in (50, 150)]


class Ctx:
    pass


def build_ctx():
    ws = W.load('mths'); m = ws.meta
    Hall = XH.hand26_for(ws); Hc = XH.calibrate(m, Hall)
    ok = m['eval'].values & m.has_spo2.values & m.has_base.values & np.isfinite(Hall).all(1) & np.isfinite(Hc).all(1)
    sel = np.nonzero(ok)[0]
    ctx = Ctx()
    ctx.n_eval_all = int((m['eval'] & m.has_spo2).sum()); ctx.people_all = int(m[m['eval'] & m.has_spo2].subject.nunique())
    ctx.meta = m.iloc[sel].reset_index(drop=True)
    ctx.y = ctx.meta.spo2_ref.values.astype(float); ctx.subject = ctx.meta.subject.values.astype(str); ctx.t = ctx.meta.t_start.values.astype(float)
    ctx.F = dict(hand26=Hall[sel], hand26_cal=Hc[sel])
    ctx.F['hand52'] = np.concatenate([ctx.F['hand26'], ctx.F['hand26_cal']], 1)
    ctx.F['ratio6'] = Hall[sel][:, [XH.FEATS26.index(c) for c in XH.RATIO6]]
    return ctx


class _P:
    def __init__(self, f):
        self.score = f


def specs(ctx):
    y = ctx.y

    def mean_fit(hp, rows):
        mu = float(y[rows].mean())
        return _P(lambda q: np.full(len(q), mu))

    def textbook(ch):
        col = XH.FEATS26.index('rr' + ch)
        return lambda hp, rows: _P(lambda q: ror.textbook_spo2(ctx.F['hand26'][q, col]))

    def ridge_fit(hp, rows):
        X = ctx.F['ratio6']; sc = StandardScaler().fit(X[rows]); r = Ridge(alpha=hp['alpha']).fit(sc.transform(X[rows]), y[rows])
        return _P(lambda q: r.predict(sc.transform(X[q])))

    def tree_fit(lib):
        def fit(hp, rows):
            m = XH.xgb_reg(hp) if lib == 'xgb' else XH.lgb_reg(hp)
            m.fit(ctx.F['hand52'][rows], y[rows])
            return _P(lambda q: m.predict(ctx.F['hand52'][q]))
        return fit
    return [
        dict(name='train_mean', fit=mean_fit, grid=[{}], fitted=True, note='mean reference SpO2 of the training people'),
        dict(name='ror_textbook_RB', fit=textbook('RB'), grid=[{}], fitted=False, note='110 - 25 R, R from red/blue AC/DC ratio; no fitted parameter'),
        dict(name='ror_textbook_RG', fit=textbook('RG'), grid=[{}], fitted=False, note='110 - 25 R, R from red/green AC/DC ratio; no fitted parameter'),
        dict(name='ridge_ratio', fit=ridge_fit, grid=GRID_RIDGE, fitted=True, note='ridge on six log ratio features (time and spectral domain), alpha by inner CV'),
        dict(name='lgbm_hand26', fit=tree_fit('lgbm'), grid=GRID_LGB, fitted=True, note='LightGBM on 26 absolute + 26 baseline-relative handcrafted features, hp by inner CV'),
        dict(name='xgb_hand26', fit=tree_fit('xgb'), grid=GRID_XGB, fitted=True, note='XGBoost on 26 absolute + 26 baseline-relative handcrafted features, hp by inner CV'),
    ]


def reg_metrics(pred, ref, dev=None):
    """MAE etc. plus AUROC for SpO2 < 90 / < 95 using -dev (deviation from the fold's training mean for fitted models, so the leave-one-person-out intercept
    shift cannot create a ranking signal; the textbook rule is used as it is)."""
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    e = pred - ref
    out = dict(n=len(ref), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), within2=float((np.abs(e) <= 2).mean()),
               within5=float((np.abs(e) <= 5).mean()))
    for thr in (90, 95):
        out[f'auroc_lt{thr}'] = M.auroc((ref < thr).astype(int), -(pred if dev is None else dev)) if dev is not None else np.nan
    return out


def run_reg(ctx, sp):
    folds = []
    for f in S.loso(ctx.subject):
        tr, te = f.train_idx, f.test_idx
        inner = list(S.inner_group_folds(ctx.subject[tr], k=C.INNER_K, seed=C.INNER_SEED))
        best = None
        if len(sp['grid']) == 1:
            best = (0.0, sp['grid'][0])
        else:
            for hp in sp['grid']:
                err = 0.0
                for fi in inner:
                    m = sp['fit'](hp, tr[fi.train_idx])
                    err += np.abs(m.score(tr[fi.test_idx]) - ctx.y[tr[fi.test_idx]]).sum()
                if best is None or err < best[0] - 1e-9:
                    best = (err, hp)
        m = sp['fit'](best[1], tr)
        folds.append(dict(subject=f.test_group, te=te, tr=tr, hp=best[1], p_te=m.score(te), p_tr=m.score(tr), mu=float(ctx.y[tr].mean())))
    return folds


def run(B=1000, only=None, log=print, out_dir=None):
    t0 = time.time()
    ctx = build_ctx()
    ref_all = ctx.y
    lab = ref_all
    log(f'P3: {len(ctx.y)} windows from {len(np.unique(ctx.subject))} people (of {ctx.n_eval_all} eval windows with a label from {ctx.people_all} people); reference SpO2 '
        f'min {lab.min():.0f} median {np.median(lab):.0f} max {lab.max():.0f}, windows < 95: {(lab < 95).mean():.3f}, < 90: {int((lab < 90).sum())}')
    rows, fold_rows, scores = [], [], {}
    cache = {}
    for sp in specs(ctx):
        if only and not any(o in sp['name'] for o in only):
            continue
        folds = run_reg(ctx, sp)
        order = np.concatenate([f['te'] for f in folds])
        pred = np.concatenate([f['p_te'] for f in folds])
        dev = pred if not sp['fitted'] else np.concatenate([f['p_te'] - f['mu'] for f in folds])
        if sp['name'] == 'train_mean':
            dev = None                                                   # constant per fold: no ranking information
        ft = [reg_metrics(f['p_te'], ctx.y[f['te']]) for f in folds]; ftr = [reg_metrics(f['p_tr'], ctx.y[f['tr']]) for f in folds]
        for f, a, b in zip(folds, ft, ftr):
            fold_rows.append(dict(task='P3', dataset='mths', model=sp['name'], variant='full', subject=f['subject'], hp=json.dumps(f['hp']), **{k: a[k] for k in ('n', 'mae', 'rmse', 'bias', 'within2', 'within5')},
                                  train_mae=b['mae']))
        y = ctx.y[order]; subj = ctx.subject[order]
        fn = lambda idx, dev=dev: reg_metrics(pred[idx], y[idx], None if dev is None else dev[idx])
        pooled = fn(np.arange(len(y)))
        ci = C.bootstrap_all(subj, fn, B=B)
        rows += C.summary_rows(dict(task='P3', dataset='mths', model=sp['name'], variant='full', note=sp['note']), ft, ftr, pooled, ci, METRICS)
        cache[sp['name']] = (order, pred)
        scores[sp['name']] = pd.Series(pred, index=order)
        log(f'  {sp["name"]}: MAE {pooled["mae"]:.3f} [{ci["mae"][0]:.3f}, {ci["mae"][1]:.3f}] ({time.time() - t0:.0f}s)')
    if REF in cache:
        o_r, p_r = cache[REF]
        for name, (o, p) in cache.items():
            if name == REF:
                continue
            assert (o == o_r).all()
            y = ctx.y[o]; subj = ctx.subject[o]
            d = M.paired_person_bootstrap(subj, lambda idx: np.abs(p[idx] - y[idx]).mean(), lambda idx: np.abs(p_r[idx] - y[idx]).mean(), B=B, higher_is_better=False)
            rows.append(dict(task='P3', dataset='mths', model=name, variant='full', metric='mae_improvement_vs_train_mean', pooled=d['diff'], pooled_lo=d['lo'], pooled_hi=d['hi'],
                             note=f'paired person bootstrap (B={B}) of train-mean MAE minus model MAE (positive = better than train-mean); interval excludes zero: {d["excludes_zero"]}'))
    summ = pd.DataFrame(rows).reindex(columns=C.SUMMARY_COLS)
    fold_df = pd.DataFrame(fold_rows)
    od = out_dir or C.RES
    C.write_csv(summ, od / 'baselines_P3.csv'); C.write_perperson(fold_df, od, 'baselines_P3')   # all P3 people are MTHS: aggregate spread only (licence)
    sc = pd.DataFrame(scores).sort_index()
    key = ctx.meta.loc[sc.index, ['subject', 'rec', 't_start']].reset_index(drop=True)
    C.write_csv(pd.concat([key, pd.Series(ctx.y[sc.index], name='spo2_ref'), sc.reset_index(drop=True)], axis=1), C.CACHE_B / 'scores' / 'baseline_P3_mths.csv')
    lr = dict(n_windows=int(len(ctx.y)), n_people=int(len(np.unique(ctx.subject))), n_eval_windows_with_label=ctx.n_eval_all, n_people_with_label=ctx.people_all,
              ref_min=float(lab.min()), ref_p5=float(np.percentile(lab, 5)), ref_median=float(np.median(lab)), ref_max=float(lab.max()), share_below_95=float((lab < 95).mean()),
              windows_below_90=int((lab < 90).sum()), people_with_window_below_90=int(len(np.unique(ctx.subject[lab < 90]))), seconds=round(time.time() - t0),
              caveat='MTHS is normoxic: the train-mean predictor is nearly optimal and nothing here measures hypoxemia detection')
    json.dump(lr, open(od / 'baselines_P3_info.json', 'w'), indent=1)
    log(f'P3 done in {time.time() - t0:.0f}s')
    return summ, fold_df, ctx
