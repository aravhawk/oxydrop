"""Shared pieces of the baseline runner: nested leave-one-subject-out classification, summary rows, abstention rule, bootstrap.

Protocol (python/oxydrop_ml/PREREGISTRATION.md, binding):
* every outer fold is one held-out subject (`splits.loso`); both hands / all clips of that person are held out together;
* EVERYTHING that is selected (hyperparameters, decision threshold, abstention half-width, the "sensitivity at fixed specificity" threshold, the SQI gate)
  is selected on an inner subject-grouped CV over the outer TRAINING subjects only (`splits.inner_group_folds`). The outer test rows are only
  scored, never looked at during selection: `run_cls` only passes `tr` rows to `choose_*` and `fit`, and tests/test_baselines.py checks that scrambling the
  test labels does not change a single score, threshold or hyperparameter;
* train-vs-test gap: the refit model's metric on its own training rows minus its metric on the held-out person (positive = over-fit).

Output rows (long format, one row per task / dataset / model / variant / metric), see `summary_rows`:
  pooled(+lo,hi)   metric on all out-of-fold windows pooled, with the 95 % person-bootstrap interval (people resampled, windows kept together)
  fold_mean, fold_sd, fold_lo, fold_hi, fold_min, fold_max, n_folds   per-held-out-person values (mean, SD ddof 1, bootstrap CI of the mean over people)
  train_mean, test_mean, gap, gap_sd, gap_lo, gap_hi   train-vs-test gap over folds
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import CACHE
from .. import metrics as M
from .. import splits as S

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'ml'
SCORES = RES / 'scores'
CACHE_B = Path(CACHE) / 'baselines'

# metrics for which a larger value is better (the sign of the train-vs-test gap)
LOWER_IS_BETTER = {'mae', 'rmse', 'doubling', 'halving', 'brier', 'ece', 'bias_abs'}
NO_GAP = {'n', 'n_answered', 'worst_person_acc', 'n_pos', 'bias', 'auroc_lt90', 'auroc_lt95'}
NO_CI = {'worst_person_acc'}

INNER_K = 5
INNER_SEED = 0
BAND_GRID = np.round(np.arange(0.0, 0.451, 0.01), 2)
BAND_TARGET = 0.90          # nested abstention rule: smallest half-width c whose answered inner-OOF windows reach acc >= 0.90 and bacc >= 0.90
BAND_MIN_COVERAGE = 0.20    # fallback when no c reaches the target: the c with the best answered bacc among those with coverage >= 0.20
SPEC_TARGET = 0.90          # 'sensitivity at fixed specificity': threshold from the inner OOF negatives at 90 % specificity


# ------------------------------------------------------------------------------------------------ bootstrap over people
def bootstrap_all(subject, fn, B=1000, seed=0, alpha=0.05, skip=()):
    """Person bootstrap of every key of the dict returned by fn(idx). Returns dict key -> (lo, hi)."""
    u, rows = M._person_index(subject)
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(B):
        pick = rng.integers(0, len(u), len(u))
        d = fn(np.concatenate([rows[i] for i in pick]))
        for k, v in d.items():
            if k in skip:
                continue
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        v = np.asarray(v, float)
        out[k] = (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))) if np.isfinite(v).any() else (np.nan, np.nan)
    return out


# ------------------------------------------------------------------------------------------------ classification metrics
def cls_metrics(y, t, score, pred, ans, subject=None):
    """Classification metrics on the ANSWERED windows (all windows when ans is all True). Unanswered windows count as wrong only in acc_all.
    alert_share_all = share of ALL hypoxic windows that receive a low-oxygen alert."""
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int); ans = np.asarray(ans, bool); score = np.asarray(score, float)
    n1 = int((y == 1).sum())
    out = dict(n=len(y), n_answered=int(ans.sum()), coverage=float(ans.mean()) if len(y) else np.nan,
               alert_share_all=float(((pred == 1) & (y == 1) & ans).sum() / n1) if n1 else np.nan,
               acc_all=float(((pred == y) & ans).sum() / max(len(y), 1)))
    keys = ['acc', 'bacc', 'sens', 'spec', 'auroc', 'ece', 'brier', 'tm_bacc', 'tm_auroc', 'tm_sens', 'tm_spec']
    if ans.sum() == 0:
        out.update({k: np.nan for k in keys}); return out
    c = M.classification_metrics(y[ans], score[ans], pred=pred[ans])
    out.update({k: c[k] for k in ('acc', 'bacc', 'sens', 'spec', 'auroc', 'ece', 'brier')})
    tm = M.time_matched(np.asarray(t)[ans], y[ans], pred[ans], score[ans])
    out.update(tm_bacc=tm['tm_bacc'], tm_auroc=tm['tm_auroc'], tm_sens=tm['tm_sens'], tm_spec=tm['tm_spec'])
    return out


def worst_person_acc(subject, y, pred, ans, min_n=20):
    subject = np.asarray(subject); y = np.asarray(y); pred = np.asarray(pred); ans = np.asarray(ans, bool)
    v = [float((pred[ans & (subject == s)] == y[ans & (subject == s)]).mean()) for s in np.unique(subject) if (ans & (subject == s)).sum() >= min_n]
    return min(v) if v else np.nan


# ------------------------------------------------------------------------------------------------ selection helpers (inner OOF only)
def _crit(kind, y, t, score, thr):
    pred = (score > thr).astype(int)
    if kind == 'tm_bacc':
        v = M.time_matched(t, y, pred, None)['tm_bacc']
    else:
        v = M.classification_metrics(y, score, pred=pred)['bacc']
    return -1.0 if not np.isfinite(v) else float(v)


def thr_candidates(oof, n=37):
    q = np.quantile(oof[np.isfinite(oof)], np.linspace(0.04, 0.96, n))
    return np.unique(np.concatenate([q, [0.5]])) if np.nanmin(oof) < 0.5 < np.nanmax(oof) else np.unique(q)


def choose_thr(oof, y, t, kind, mode, fixed_thr):
    """Decision threshold and the inner criterion value. mode 'fixed' keeps fixed_thr, 'tune' searches thr_candidates on the inner OOF scores."""
    ok = np.isfinite(oof)
    oof, y, t = oof[ok], y[ok], t[ok]
    if mode == 'fixed':
        return float(fixed_thr), _crit(kind, y, t, oof, fixed_thr)
    best = (-2.0, None)
    for c in thr_candidates(oof):
        v = _crit(kind, y, t, oof, c)
        if v > best[0] + 1e-12 or (abs(v - best[0]) <= 1e-12 and abs(c - 0.5) < abs(best[1] - 0.5)):
            best = (v, float(c))
    return best[1], best[0]


def choose_band(oof, y, thr, mask=None, target=BAND_TARGET):
    """Nested abstention rule (shared by every baseline and meant for the CNN): answer when |score - thr| >= c. c is the smallest value on BAND_GRID
    whose answered inner-OOF windows reach accuracy AND balanced accuracy >= target (with >= 20 answered windows and both classes);
    if none does, the c with the best answered balanced accuracy among those that keep >= BAND_MIN_COVERAGE of the windows."""
    ok = np.isfinite(oof) if mask is None else (np.isfinite(oof) & mask)
    oof, y = oof[ok], y[ok]
    if len(y) == 0:
        return 0.0
    best = (-1.0, 0.0)
    for c in BAND_GRID:
        a = np.abs(oof - thr) >= c
        if a.sum() < 20 or len(np.unique(y[a])) < 2:
            continue
        cm = M.classification_metrics(y[a], oof[a], pred=(oof[a] > thr).astype(int))
        if cm['acc'] >= target and cm['bacc'] >= target:
            return float(c)
        if a.mean() >= BAND_MIN_COVERAGE and cm['bacc'] > best[0]:
            best = (cm['bacc'], float(c))
    return best[1]


def choose_spec_thr(oof, y, spec=SPEC_TARGET):
    """Threshold giving `spec` specificity on the inner OOF negatives (alert when score > thr)."""
    ok = np.isfinite(oof)
    neg = oof[ok & (y == 0)]
    return float(np.quantile(neg, spec)) if len(neg) else 0.5


# ------------------------------------------------------------------------------------------------ nested LOSO classification
@dataclass
class Spec:
    """One classifier. fit(hp, rows) -> object with score(rows) -> array and optional valid(rows) -> bool array (False = the model has no answer,
    e.g. v2 'not locked'). `rows` are integer indices into the arrays of the context passed to run_cls."""
    name: str
    fit: Callable
    grid: list = field(default_factory=lambda: [{}])
    thr_mode: str = 'tune'             # 'tune' = threshold selected on inner OOF, 'fixed' = fixed_thr (chains: 0.5 as shipped)
    fixed_thr: float = 0.5
    select_on: str = 'tm_bacc'         # inner criterion for hp and threshold (pre-registered P1 metric); the clock uses 'bacc' (TM is blind to it by design)
    bands: bool = True                 # False for scores that are not probabilities (the 0.01-0.45 band grid is meaningless on them): only 'full' and 'spec90' are reported
    shipped_band: float = None         # fixed abstain half-width of the shipped model (0.10 for the logistic chains), reported as its own variant
    note: str = ''


@dataclass
class ClsFold:
    subject: object
    te: np.ndarray
    tr: np.ndarray
    hp: dict
    thr: float
    band: float
    band_valid: float
    spec_thr: float
    s_te: np.ndarray
    s_tr: np.ndarray
    v_te: np.ndarray
    v_tr: np.ndarray
    inner_value: float
    hp_trace: list


def run_cls(ctx, spec, log=print):
    """Outer LOSO over ctx.subject (rows = evaluated windows), nested selection, one ClsFold per held-out subject."""
    out = []
    for f in S.loso(ctx.subject):
        tr, te = f.train_idx, f.test_idx
        y_tr, t_tr = ctx.y[tr], ctx.t[tr]
        inner = list(S.inner_group_folds(ctx.subject[tr], k=INNER_K, seed=INNER_SEED, balance=y_tr))
        best = None; trace = []
        for hp in spec.grid:
            oof = np.full(len(tr), np.nan); voof = np.ones(len(tr), bool)
            for fi in inner:
                m = spec.fit(hp, tr[fi.train_idx])
                oof[fi.test_idx] = m.score(tr[fi.test_idx])
                if hasattr(m, 'valid'):
                    voof[fi.test_idx] = m.valid(tr[fi.test_idx])
            thr, val = choose_thr(oof, y_tr, t_tr, spec.select_on, spec.thr_mode, spec.fixed_thr)
            trace.append((hp, thr, val))
            if best is None or val > best[0] + 1e-12:
                best = (val, hp, thr, oof, voof)
        val, hp, thr, oof, voof = best
        m = spec.fit(hp, tr)
        v_te = m.valid(te) if hasattr(m, 'valid') else np.ones(len(te), bool)
        v_tr = m.valid(tr) if hasattr(m, 'valid') else np.ones(len(tr), bool)
        out.append(ClsFold(f.test_group, te, tr, hp, thr, choose_band(oof, y_tr, thr), choose_band(oof, y_tr, thr, voof), choose_spec_thr(oof, y_tr),
                           m.score(te), m.score(tr), v_te, v_tr, val, trace))
    return out


def variants_of(spec, has_valid):
    """Variant names reported for a spec."""
    if not spec.bands:
        return ['full', 'spec90']
    v = ['full', 'band_nested', 'spec90']
    if spec.shipped_band is not None:
        v.insert(1, 'band_shipped')
    if has_valid:
        v += ['nolock_abstain', 'nolock_band_nested'] + (['nolock_band_shipped'] if spec.shipped_band is not None else [])
    return v


def variant_arrays(var, spec, fo, which):
    """(pred, answered) of a variant for the 'te' or 'tr' rows of a fold."""
    s = fo.s_te if which == 'te' else fo.s_tr
    val = fo.v_te if which == 'te' else fo.v_tr
    n = len(s)
    thr = fo.thr
    if var == 'full':
        return (s > thr).astype(int), np.ones(n, bool)
    if var == 'spec90':
        return (s > fo.spec_thr).astype(int), np.ones(n, bool)
    if var == 'band_nested':
        return (s > thr).astype(int), np.abs(s - thr) >= fo.band
    if var == 'band_shipped':
        return (s > thr).astype(int), np.abs(s - thr) >= spec.shipped_band
    if var == 'nolock_abstain':
        return (s > thr).astype(int), val.copy()
    if var == 'nolock_band_nested':
        return (s > thr).astype(int), val & (np.abs(s - thr) >= fo.band_valid)
    if var == 'nolock_band_shipped':
        return (s > thr).astype(int), val & (np.abs(s - thr) >= spec.shipped_band)
    raise KeyError(var)


# ------------------------------------------------------------------------------------------------ summary rows
def summarize_metric(test_vals, train_vals, key):
    """fold_* and gap_* fields for one metric from per-fold values."""
    te = np.asarray(test_vals, float); tr = np.asarray(train_vals, float)
    s = M.summarize_folds(te)
    r = dict(fold_mean=s['mean'], fold_sd=s['sd'], fold_lo=s['lo'], fold_hi=s['hi'], fold_min=s['min'], fold_max=s['max'], n_folds=s['n'])
    if key in NO_GAP or len(tr) != len(te):
        r.update(train_mean=np.nan, test_mean=np.nan, gap=np.nan, gap_sd=np.nan, gap_lo=np.nan, gap_hi=np.nan)
    else:
        g = M.train_vs_test_gap(tr, te, higher_is_better=key not in LOWER_IS_BETTER)
        r.update(train_mean=g['train_mean'], test_mean=g['test_mean'], gap=g['gap'], gap_sd=g['gap_sd'], gap_lo=g['gap_lo'], gap_hi=g['gap_hi'])
    return r


def summary_rows(base, fold_test, fold_train, pooled, ci, metrics_order):
    """base: dict(task, dataset, model, variant, note). fold_test/fold_train: list (per fold) of metric dicts. pooled: dict of pooled metric values.
    ci: dict key -> (lo, hi) person-bootstrap. Returns list of row dicts."""
    rows = []
    for k in metrics_order:
        te = [d.get(k, np.nan) for d in fold_test]; tr = [d.get(k, np.nan) for d in fold_train]
        r = dict(base); r['metric'] = k
        r['pooled'] = pooled.get(k, np.nan)
        lo, hi = ci.get(k, (np.nan, np.nan)); r['pooled_lo'] = lo; r['pooled_hi'] = hi
        r.update(summarize_metric(te, tr, k))
        rows.append(r)
    return rows


SUMMARY_COLS = ['task', 'dataset', 'model', 'variant', 'metric', 'pooled', 'pooled_lo', 'pooled_hi', 'fold_mean', 'fold_sd', 'fold_lo', 'fold_hi', 'fold_min', 'fold_max',
                'n_folds', 'train_mean', 'test_mean', 'gap', 'gap_sd', 'gap_lo', 'gap_hi', 'note']


def write_csv(df, path, float_fmt='%.5g'):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_fmt)
    return path


# ------------------------------------------------------------------------------------------------------------------ licence-safe per-person tables
# MTHS is CC BY-NC-ND (DATA_LICENSES.md: aggregates only): per-person rows are committed only for the datasets whose licence allows them (Hoffman MIT, BUT PPG CC BY).
# For MTHS the per-person values are replaced by an aggregate spread (people, mean, SD, quantiles) over the held-out people.
AGG_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
KEY_COLS = ('task', 'dataset', 'model', 'variant')


def is_restricted_subject(subject) -> bool:
    return str(subject).startswith('mths_')


def split_restricted(fold_df: pd.DataFrame, subject_col='subject'):
    """(rows that may be committed per person, spread table of the restricted-licence people).
    Spread: one row per (task, dataset, model, variant) with n_people and, for every numeric column, mean, SD (ddof 1) and the 10/25/50/75/90th percentiles over people."""
    mask = fold_df[subject_col].map(is_restricted_subject)
    keep, hid = fold_df[~mask].copy(), fold_df[mask]
    keys = [k for k in KEY_COLS if k in fold_df.columns]
    num = [c for c in fold_df.columns if c not in keys + [subject_col] and pd.api.types.is_numeric_dtype(fold_df[c])]
    rows = []
    for key, g in hid.groupby(keys, dropna=False, sort=False):
        r = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        r['n_people'] = int(g[subject_col].nunique())
        for c in num:
            v = g[c].dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            r[c + '_mean'] = v.mean(); r[c + '_sd'] = v.std(ddof=1) if len(v) > 1 else np.nan
            for q in AGG_QUANTILES:
                r[f'{c}_q{int(round(100 * q)):02d}'] = float(np.quantile(v, q))
        rows.append(r)
    return keep, pd.DataFrame(rows)


def write_perperson(fold_df: pd.DataFrame, out_dir, name: str, subject_col='subject'):
    """Write `<name>_perperson.csv` (people of datasets whose licence allows it) and, if any restricted-licence people exist, `<name>_spread.csv` (aggregates over them).
    A stale per-person file that would be empty is removed."""
    out_dir = Path(out_dir)
    keep, spread = split_restricted(fold_df, subject_col)
    pp = out_dir / f'{name}_perperson.csv'
    if len(keep):
        write_csv(keep, pp)
    elif pp.exists():
        pp.unlink()
    if len(spread):
        write_csv(spread, out_dir / f'{name}_spread.csv')
    return keep, spread
