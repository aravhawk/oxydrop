"""P1 baselines: low-oxygen (reference SpO2 < 90) detection on UW Hoffman, leave-one-person-out (both hands together), 5,644 windows with t >= 60 s.

Primary metric: time-matched balanced accuracy at full coverage. Models (all through common.run_cls, nested selection on training people only):
  majority, clock (time only), ror_textbook_{RB,RG}, ror_calibrated_{RB,RG}, chain_v1 / chain_v1_nestedC, chain_v2 / chain_v2_nestedC,
  {xgb, lgbm} x {fusion19_v1, fusion19_v2, hand26_cal}.
Variants per model: full (forced answer at the selected threshold), band_nested (abstain band chosen on inner OOF), band_shipped (0.10, chains only),
nolock_* (v2 chains: no answer when the pulse is not locked), spec90 (threshold from inner OOF negatives at 90 % specificity).
"""
import json
import time

import numpy as np
import pandas as pd

from .. import metrics as M
from .. import windows as W
from . import chain_v1, chain_v2, common as C, ror, xgb_hand as XH

METRICS = ['n', 'coverage', 'acc', 'bacc', 'sens', 'spec', 'auroc', 'tm_bacc', 'tm_auroc', 'alert_share_all', 'acc_all', 'ece', 'brier', 'worst_person_acc']
FOLD_KEYS = [k for k in METRICS if k != 'worst_person_acc']
REF = ('chain_v1', 'full')                    # reference row of the paired differences (the shipped v1 chain)
DEMOG = 'results/scores/demographics.csv'


class Ctx:
    pass


def build_ctx():
    ws = W.load('hoffman'); m = ws.meta
    sel = np.nonzero(m['eval'].values & m.has_low.values)[0]
    meta = m.iloc[sel].reset_index(drop=True)
    ctx = Ctx()
    ctx.meta = meta
    ctx.y = meta.low.values.astype(int); ctx.t = meta.t_start.values.astype(float); ctx.subject = meta.subject.values.astype(str)
    Hall = XH.hand26_for(ws)
    H = Hall[sel]; Hc = XH.calibrate(m, Hall)[sel]
    X1, v1, src = chain_v1.chain_features(ws, meta, 1)
    X2, v2, _ = chain_v1.chain_features(ws, meta, 2)
    ctx.F = dict(fusion19_v1=X1, fusion19_v2=X2, hand26=H, hand26_cal=Hc)
    ctx.valid = {1: v1, 2: v2}
    ctx.feature_source = src
    return ctx


class _P:
    def __init__(self, f):
        self.score = f


def basic_specs(ctx):
    y, t = ctx.y, ctx.t

    def maj(hp, rows):
        p = float(float(y[rows].mean()) > 0.5)             # constant 0/1 score: no ranking information, AUROC 0.5 in every fold and pooled
        return _P(lambda q: np.full(len(q), p))

    def clock(hp, rows):
        b = (t[rows] // hp['bin']).astype(int); nb = b.max() + 2
        n = np.bincount(b, minlength=nb); s = np.bincount(b, weights=y[rows], minlength=nb); g = float(y[rows].mean())
        rate = np.where(n >= 3, s / np.maximum(n, 1), g)
        return _P(lambda q: rate[np.clip((t[q] // hp['bin']).astype(int), 0, nb - 1)])
    return [C.Spec('majority', maj, thr_mode='fixed', fixed_thr=0.5, bands=False, note='always the training majority class (hypoxic, 57.5 % of the evaluated windows)'),
            C.Spec('clock', clock, [dict(bin=b) for b in (25, 50, 100, 200)], thr_mode='tune', select_on='bacc',
                   note='elapsed time only: training-people hypoxic rate in time bins (bin width and threshold by inner CV on plain balanced accuracy, because the time-matched metric is blind to a clock by construction)')]


def all_specs(ctx, only=None):
    sp = basic_specs(ctx) + ror.specs(ctx) + chain_v1.specs(ctx, 1) + chain_v1.specs(ctx, 2)
    for lib in ('xgb', 'lgbm'):
        for fk in ('fusion19_v1', 'fusion19_v2', 'hand26_cal'):
            sp.append(XH.tree_cls_spec(ctx, lib, fk))
    return [s for s in sp if only is None or any(o in s.name for o in only)]


def run(B=1000, only=None, log=print, out_dir=None):
    t0 = time.time()
    ctx = build_ctx()
    log(f'P1: {len(ctx.y)} windows, {len(np.unique(ctx.subject))} people, hypoxic share {ctx.y.mean():.4f} (majority class {max(ctx.y.mean(), 1 - ctx.y.mean()):.4f}); '
        f'chain features from {ctx.feature_source}')
    specs = all_specs(ctx, only)
    rows, fold_rows, scores = [], [], {}
    pooled_cache = {}
    cls_fold = {}
    for spec in specs:
        folds = C.run_cls(ctx, spec)
        cls_fold[spec.name] = folds
        order = np.concatenate([fo.te for fo in folds])
        has_valid = spec.name.startswith('chain_v2')
        scores[spec.name + '__score'] = pd.Series(np.concatenate([fo.s_te for fo in folds]), index=order)
        for var in C.variants_of(spec, has_valid):
            ft, ftr = [], []
            pr, an = [], []
            for fo in folds:
                p_te, a_te = C.variant_arrays(var, spec, fo, 'te'); p_tr, a_tr = C.variant_arrays(var, spec, fo, 'tr')
                ft.append(C.cls_metrics(ctx.y[fo.te], ctx.t[fo.te], fo.s_te, p_te, a_te)); ftr.append(C.cls_metrics(ctx.y[fo.tr], ctx.t[fo.tr], fo.s_tr, p_tr, a_tr))
                pr.append(p_te); an.append(a_te)
                fold_rows.append(dict(task='P1', dataset='hoffman', model=spec.name, variant=var, subject=fo.subject, hp=json.dumps(fo.hp), thr=fo.thr,
                                      band_nested=fo.band, band_nested_nolock=fo.band_valid, spec90_thr=fo.spec_thr, inner_value=fo.inner_value,
                                      **ft[-1], **{'train_' + k: ftr[-1][k] for k in ('acc', 'bacc', 'tm_bacc', 'auroc', 'coverage')}))
            pred = np.concatenate(pr); ans = np.concatenate(an); s = scores[spec.name + '__score'].values
            y, t, subj = ctx.y[order], ctx.t[order], ctx.subject[order]
            pooled = C.cls_metrics(y, t, s, pred, ans); pooled['worst_person_acc'] = C.worst_person_acc(subj, y, pred, ans)
            ci = C.bootstrap_all(subj, lambda idx: C.cls_metrics(y[idx], t[idx], s[idx], pred[idx], ans[idx]), B=B, skip=C.NO_CI)
            base = dict(task='P1', dataset='hoffman', model=spec.name, variant=var, note=spec.note)
            rows += C.summary_rows(base, ft, ftr, pooled, ci, METRICS)
            pooled_cache[(spec.name, var)] = (order, s, pred, ans)
            scores[f'{spec.name}__{var}'] = pd.Series(np.where(ans, pred, -1), index=order)
        log(f'  {spec.name}: done ({time.time() - t0:.0f}s)')
    # paired difference of the primary metric to the reference row (shipped v1 chain), same windows, same resampled people
    if REF in pooled_cache:
        o_r, s_r, p_r, a_r = pooled_cache[REF]
        subj = ctx.subject[o_r]; y = ctx.y[o_r]; t = ctx.t[o_r]
        for (name, var), (o, s, p, a) in pooled_cache.items():
            if var != 'full' or name == REF[0]:
                continue
            assert (o == o_r).all()
            sa = lambda idx, s=s, p=p, a=a: M.time_matched(t[idx], y[idx], p[idx], s[idx])['tm_bacc']
            sb = lambda idx: M.time_matched(t[idx], y[idx], p_r[idx], s_r[idx])['tm_bacc']
            d = M.paired_person_bootstrap(subj, sa, sb, B=B)
            r = dict(task='P1', dataset='hoffman', model=name, variant='full', metric=f'tm_bacc_diff_vs_{REF[0]}', pooled=d['diff'], pooled_lo=d['lo'], pooled_hi=d['hi'],
                     note=f'paired person bootstrap (B={B}) of TM balanced accuracy minus {REF[0]}; interval excludes zero: {d["excludes_zero"]}; descriptive (6 people)')
            rows.append(r)
    summ = pd.DataFrame(rows).reindex(columns=C.SUMMARY_COLS)
    folds_df = pd.DataFrame(fold_rows)
    # per-window scores (Hoffman is MIT licensed: derived scores may be committed)
    sc = pd.DataFrame(scores).sort_index()
    key = ctx.meta.loc[sc.index, ['subject', 'rec', 't_start']].reset_index(drop=True)
    sc = pd.concat([key, pd.Series(ctx.y[sc.index], name='y_low'), pd.Series(ctx.meta.spo2_ref.values[sc.index], name='spo2_ref'), sc.reset_index(drop=True)], axis=1)
    od = out_dir or C.RES
    C.write_csv(summ, od / 'baselines_P1.csv'); C.write_csv(folds_df, od / 'baselines_P1_perperson.csv')
    C.write_csv(sc, od / 'scores' / 'baseline_P1_hoffman.csv', '%.5g')
    skin_table(folds_df, od)
    info = dict(n_windows=int(len(ctx.y)), n_people=int(len(np.unique(ctx.subject))), hypoxic_share=float(ctx.y.mean()), majority_share=float(max(ctx.y.mean(), 1 - ctx.y.mean())),
                feature_source=ctx.feature_source, seconds=round(time.time() - t0))
    json.dump(info, open(od / 'baselines_P1_info.json', 'w'), indent=1)
    log(f'P1 done in {time.time() - t0:.0f}s')
    return summ, folds_df, ctx


def skin_table(folds_df, od):
    """Per-skin-tone gap on Hoffman: the one documented dark-skinned participant (hex #32210F, sid 100002) against the five lighter ones. n = 1 versus 5."""
    import os
    p = C.REPO / DEMOG
    if not p.exists():
        return
    d = pd.read_csv(p).drop_duplicates('sid')
    dark = {f'hof_{r.sid}': ('dark' if r.skintone_hex.upper() == '#32210F' else 'light') for r in d.itertuples()}
    f = folds_df[folds_df.variant.isin(['full', 'band_nested'])].copy()
    f['tone'] = f.subject.map(dark)
    out = []
    for (m, v), g in f.groupby(['model', 'variant']):
        dk = g[g.tone == 'dark']; lt = g[g.tone == 'light']
        r = dict(model=m, variant=v, n_dark=len(dk), n_light=len(lt))
        for k in ('acc', 'bacc', 'sens', 'spec', 'tm_bacc', 'coverage', 'alert_share_all'):
            r[f'dark_{k}'] = float(dk[k].mean()); r[f'light_mean_{k}'] = float(lt[k].mean()); r[f'gap_{k}'] = r[f'light_mean_{k}'] - r[f'dark_{k}']
        out.append(r)
    pd.DataFrame(out).to_csv(od / 'baselines_P1_skin.csv', index=False, float_format='%.5g')
