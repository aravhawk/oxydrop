"""P2 baselines: heart rate on BUT PPG (38 people with SpO2, 3,840 clips) and MTHS (62 people, 2,065 windows), leave-one-subject-out over the 100 people.

Primary metric: share of windows within 5 bpm of the reference at FULL coverage (a window with no answer counts as wrong), with the doubling rate.
Every window with a reference HR is scored (HR needs no personal baseline, so the baseline clips/windows are included; this differs from the P1/P3 eval masks).
Estimators (hr_est.py): fft_plain, v1_peak, v2_spectral_acf. The estimators have no fitted parameters; the only selection is the v2 SQI/ACF gate, chosen per outer fold
on the training subjects only (rule in select_gate: most permissive gate whose answered training windows are within 5 bpm >= 80 % of the time).
Variants: forced (always answers; for v2 the primary channel's estimate if no channel passes the shipped gate), gate_shipped (SQI >= 0.30 and ACF >= 0.30, no answer
otherwise), gate_nested (gate selected on training subjects). Train-vs-test gap: the same fold's gate and estimator on the training subjects' windows versus the held-out subject.
Scopes: all (100 people pooled), butppg, mths.
"""
import json
import time

import numpy as np
import pandas as pd

from .. import metrics as M
from .. import splits as S
from .. import windows as W
from . import common as C
from . import hr_est as H

METRICS = ['n', 'n_answered', 'coverage', 'mae', 'bias', 'within5', 'within5_full', 'doubling', 'halving']
REF = ('v2_spectral_acf', 'forced')


class Ctx:
    pass


def build_ctx():
    ws = {d: W.load(d) for d in ('butppg', 'mths')}
    parts, raws = [], []
    for d, w in ws.items():
        m = w.meta
        sel = np.nonzero(m.has_hr.values)[0]
        r = H.raw_estimates(w)
        parts.append(m.iloc[sel].assign(ds=d).reset_index(drop=True)); raws.append({k: v[sel] for k, v in r.items()})
    meta = pd.concat(parts, ignore_index=True)
    raw = {k: np.concatenate([r[k] for r in raws]) for k in raws[0]}
    ctx = Ctx()
    ctx.meta = meta; ctx.raw = raw
    ctx.ref = meta.hr_ref.values.astype(float); ctx.subject = meta.subject.values.astype(str); ctx.ds = meta.ds.values
    ctx.gates = [(a, b) for a in H.GATE_SQI for b in H.GATE_ACF]
    ctx.gate_est = {g: H.select(raw, *g) for g in ctx.gates}
    return ctx


HR_TARGET = 0.80       # nested gate rule: the most permissive gate whose answered training windows are within 5 bpm at least 80 % of the time
HR_MIN_COVERAGE = 0.20  # fallback when no gate reaches the target: the most accurate gate among those answering >= 20 % of the windows


def select_gate(ctx, tr):
    """Nested selection of the v2 SQI/ACF gate on the outer TRAINING subjects only. The estimators have no fitted parameters, so the inner subject-grouped
    validation folds partition the training subjects and their union is the training set: the criterion is evaluated on training windows of other people than
    the test person and never on the test person. Rule: highest coverage among the gates with answered within-5 >= HR_TARGET (ties: looser gate),
    else the gate with the best answered within-5 among those with coverage >= HR_MIN_COVERAGE."""
    best_t = (-1.0, None); best_f = (-1.0, None)
    for g in sorted(ctx.gates):
        hr, valid = ctx.gate_est[g]
        est = np.where(valid, hr, np.nan)[tr]; ref = ctx.ref[tr]
        ans = np.isfinite(est)
        if ans.sum() < 20:
            continue
        acc = float((np.abs(est[ans] - ref[ans]) <= 5).mean()); cov = float(ans.mean())
        if acc >= HR_TARGET and cov > best_t[0] + 1e-12:
            best_t = (cov, g)
        if cov >= HR_MIN_COVERAGE and acc > best_f[0] + 1e-12:
            best_f = (acc, g)
    return (best_t[1], best_t[0]) if best_t[1] is not None else (best_f[1], best_f[0])


def estimates(ctx, model, var, gate=None):
    """Window estimates (NaN = no answer) of a model variant for all rows."""
    raw = ctx.raw
    if model == 'fft_plain':
        return raw['fft'].copy()
    if model == 'v1_peak':
        return raw['v1'].copy()
    hr, valid = ctx.gate_est[H.SHIPPED_GATE if var != 'gate_nested' else gate]
    return hr.copy() if var == 'forced' else np.where(valid, hr, np.nan)


def run(B=1000, log=print, out_dir=None):
    t0 = time.time()
    ctx = build_ctx()
    n = len(ctx.ref)
    log(f'P2: {n} windows with a reference HR from {len(np.unique(ctx.subject))} people ({", ".join(f"{d}: {int((ctx.ds == d).sum())}" for d in np.unique(ctx.ds))})')
    models = [('fft_plain', ['forced']), ('v1_peak', ['forced']), ('v2_spectral_acf', ['forced', 'gate_shipped', 'gate_nested'])]
    folds = list(S.loso(ctx.subject))
    sel = {f.test_group: select_gate(ctx, f.train_idx) for f in folds}
    log(f'  gates selected: ' + ', '.join(f'{g}:{sum(1 for v in sel.values() if v[0] == g)}' for g in sorted({v[0] for v in sel.values()})) + f' ({time.time() - t0:.0f}s)')
    rows, fold_rows, scores = [], [], {}
    pooled_cache = {}
    order = np.concatenate([f.test_idx for f in folds])
    for model, variants in models:
        for var in variants:
            # per-fold estimates for test and train rows (only the nested gate changes between folds)
            est_fold = {}
            full = None if var == 'gate_nested' else estimates(ctx, model, var)
            pooled_est = np.full(n, np.nan)
            for f in folds:
                e = estimates(ctx, model, var, sel[f.test_group][0]) if var == 'gate_nested' else full
                est_fold[f.test_group] = e
                pooled_est[f.test_idx] = e[f.test_idx]
            scores[f'{model}__{var}'] = pooled_est
            for scope in ('all', 'butppg', 'mths'):
                inscope = np.ones(n, bool) if scope == 'all' else (ctx.ds == scope)
                ft, ftr, ids = [], [], []
                for f in folds:
                    if not inscope[f.test_idx].all():
                        continue
                    e = est_fold[f.test_group]
                    tr = f.train_idx[inscope[f.train_idx]]
                    ft.append(M.hr_metrics(e[f.test_idx], ctx.ref[f.test_idx])); ftr.append(M.hr_metrics(e[tr], ctx.ref[tr]))
                    ids.append(f.test_group)
                    fold_rows.append(dict(task='P2', dataset=scope, model=model, variant=var, subject=f.test_group, gate_sqi=sel[f.test_group][0][0] if var == 'gate_nested' else np.nan,
                                          gate_acf=sel[f.test_group][0][1] if var == 'gate_nested' else np.nan, **ft[-1], **{'train_' + k: ftr[-1][k] for k in ('mae', 'within5', 'within5_full', 'coverage', 'doubling')}))
                rows_s = np.nonzero(inscope[order])[0]
                o = order[rows_s]
                est = pooled_est[o]; ref = ctx.ref[o]; subj = ctx.subject[o]
                pooled = M.hr_metrics(est, ref)
                ci = C.bootstrap_all(subj, lambda idx: M.hr_metrics(est[idx], ref[idx]), B=B)
                base = dict(task='P2', dataset=scope, model=model, variant=var, note=NOTES[(model, var)])
                rows += C.summary_rows(base, ft, ftr, pooled, ci, METRICS)
                pooled_cache[(model, var, scope)] = (o, est)
        log(f'  {model} done ({time.time() - t0:.0f}s)')
    for (model, var, scope), (o, est) in pooled_cache.items():
        if (model, var) == REF:
            continue
        o_r, est_r = pooled_cache[REF + (scope,)]
        assert (o == o_r).all()
        ref = ctx.ref[o]; subj = ctx.subject[o]
        d = M.paired_person_bootstrap(subj, lambda idx: M.hr_metrics(est[idx], ref[idx])['within5_full'], lambda idx: M.hr_metrics(est_r[idx], ref[idx])['within5_full'], B=B)
        rows.append(dict(task='P2', dataset=scope, model=model, variant=var, metric=f'within5_full_diff_vs_{REF[0]}_{REF[1]}', pooled=d['diff'], pooled_lo=d['lo'], pooled_hi=d['hi'],
                         note=f'paired person bootstrap (B={B}) of within-5 at full coverage minus the shipped v2 estimator; interval excludes zero: {d["excludes_zero"]}'))
    summ = pd.DataFrame(rows).reindex(columns=C.SUMMARY_COLS)
    fold_df = pd.DataFrame(fold_rows)
    od = out_dir or C.RES
    C.write_csv(summ, od / 'baselines_P2.csv'); C.write_perperson(fold_df, od, 'baselines_P2')   # MTHS people: aggregate spread only (licence)
    # per-window estimates: BUT PPG (CC BY 4.0) is committed, MTHS (CC BY-NC-ND) stays in the cache
    sc = pd.DataFrame({k: v for k, v in scores.items()})
    base_cols = pd.DataFrame(dict(subject=ctx.subject, rec=ctx.meta.rec.values, t_start=ctx.meta.t_start.values, hr_ref=ctx.ref, ds=ctx.ds))
    full = pd.concat([base_cols, sc], axis=1)
    C.write_csv(full[full.ds == 'butppg'].drop(columns='ds'), od / 'scores' / 'baseline_P2_butppg.csv')
    C.write_csv(full[full.ds == 'mths'].drop(columns='ds'), C.CACHE_B / 'scores' / 'baseline_P2_mths.csv')
    json.dump(dict(n_windows=int(n), n_people=int(len(np.unique(ctx.subject))), gate_counts={str(g): int(sum(1 for v in sel.values() if v[0] == g)) for g in sorted({v[0] for v in sel.values()})}, seconds=round(time.time() - t0)),
              open(od / 'baselines_P2_info.json', 'w'), indent=1)
    log(f'P2 done in {time.time() - t0:.0f}s')
    return summ, fold_df, ctx


NOTES = {
    ('fft_plain', 'forced'): 'largest FFT peak 0.7-3.5 Hz of the primary channel, 10 s window, always answers',
    ('v1_peak', 'forced'): 'peak-interval counter (engine_v1 logic) on the 15 s buffer; no answer when < 6 peaks / < 5 valid intervals (counted wrong)',
    ('v2_spectral_acf', 'forced'): 'python/oxydrop/hr.py spectral + autocorrelation estimator, forced answer (hr of the first channel passing the shipped gate, else the primary channel)',
    ('v2_spectral_acf', 'gate_shipped'): 'v2 with the shipped gate (SQI >= 0.30, ACF >= 0.30, agreement 15 %); no answer when no channel passes',
    ('v2_spectral_acf', 'gate_nested'): 'v2 with SQI/ACF gate selected per outer fold on training subjects only (most permissive gate with answered within-5 >= 80 %)',
}
