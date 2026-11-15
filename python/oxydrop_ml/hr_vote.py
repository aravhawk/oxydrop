"""Heart-rate voting between autocorrelation, spectral and CNN estimates with harmonic / subharmonic correction and a causal median of 5 (FEAT-007,
amendments/AMENDMENT_5_conformal_openox.md, Part C; every row is exploratory and the rule below was fixed before any result).

    python -m oxydrop_ml.hr_vote inputs --res ../results/ml     # needs the cache (hr_raw_*.npz, shards, MTHS scores); writes scores/hr_vote_inputs_{butppg,hoffman}.csv (committed) and the MTHS file (cache only)
    python -m oxydrop_ml.hr_vote run    --res ../results/ml     # hr_vote_*.csv from the inputs
    python -m oxydrop_ml.hr_vote report --res ../results/ml     # HR_VOTE.md

Estimators per 10 s window: s spectral harmonic-sum, a autocorrelation (`python/oxydrop/hr.py`, channel as in `baselines/hr_est.select`), c the CNN (true leave-one-subject-out seed median
for BUT PPG and MTHS; leave-one-person-out for Hoffman). s and a have no learned parameter.
Rule (fixed in Amendment 5):
  1. tol(v) = max(5 bpm, 6 % of v); two estimates agree when they differ by at most tol of their mean.
  2. If at least two of {s, a, c} agree, the answer is the mean of the largest agreeing group (a tie goes to the group that contains c; with no c, to the first).
  3. Else harmonic / subharmonic correction: every estimate is moved to the octave {x / 2, x, 2x} (kept inside 40..200 bpm) closest to c, the vote of step 2 is repeated; if still nothing agrees the answer is c.
  4. Octave check against the CNN: if the voted value is within tol of 2c or c / 2 and not within tol of c, the answer is c.
  5. Causal median of the last 5 answers of the same recording (fewer at the start).
Each step is reported alone (voting only, + harmonic correction, + octave check, + causal median) next to the three estimators and the pre-registered P2 final rule.
`clean` windows pass the shipped v2 quality gate (spectral and autocorrelation agree within 15 %, SQI >= 0.30, ACF >= 0.30).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import metrics as M

LO, HI = 40.0, 200.0
TARGET_MAE, TARGET_DOUBLING = 3.0, 0.01
VARIANTS = ['cnn', 'spec', 'acf', 'p2_final_rule', 'vote', 'vote+harmonic', 'vote+harmonic+octave', 'vote+harmonic+octave+median5']
STEPS = {'vote': (False, False, False), 'vote+harmonic': (True, False, False), 'vote+harmonic+octave': (True, True, False), 'vote+harmonic+octave+median5': (True, True, True)}
CACHE = os.environ.get('OXY_CACHE', '/home/aravhawk/.t3/scratch/veritas/ml_cache')


def tol(v):
    return max(5.0, 0.06 * float(v))


def agree(x, y):
    return abs(x - y) <= tol(0.5 * (x + y))


def _vote(vals, c):
    """Largest agreeing group among the finite estimates `vals` (list of (name, value)); ties go to the group that contains the CNN. Returns (value, size)."""
    best = (None, 0, False)
    for i, (ni, vi) in enumerate(vals):
        grp = [j for j, (nj, vj) in enumerate(vals) if j == i or agree(vi, vj)]
        has_c = any(vals[j][0] == 'c' for j in grp)
        if len(grp) > best[1] or (len(grp) == best[1] and has_c and not best[2]):
            best = (float(np.mean([vals[j][1] for j in grp])), len(grp), has_c)
    return best[0], best[1]


def vote_window(s, a, c, harmonic=True, octave=True):
    """The per-window rule (steps 1 to 4). NaN inputs are ignored; the answer is NaN only when every input is NaN."""
    ests = [(n, float(v)) for n, v in (('s', s), ('a', a), ('c', c)) if np.isfinite(v) and LO <= v <= HI]
    if not ests:
        return np.nan
    cc = float(c) if np.isfinite(c) else np.nan
    if len(ests) == 1:
        return ests[0][1]
    v, n = _vote(ests, cc)
    if n < 2:
        if harmonic and np.isfinite(cc):
            moved = []
            for nm, x in ests:
                cand = [y for y in (x / 2, x, 2 * x) if LO <= y <= HI]
                moved.append((nm, min(cand, key=lambda y: abs(y - cc)) if cand else x))
            v, n = _vote(moved, cc)
        if n < 2:
            return cc if np.isfinite(cc) else v
    if octave and np.isfinite(cc) and not agree(v, cc) and (agree(v, 2 * cc) or agree(v, cc / 2)):
        return cc
    return v


def causal_median(x, rec, t, k=5):
    """Median of the last k answers (NaN ignored) of the same recording, ordered by t."""
    x = np.asarray(x, float); rec = np.asarray(rec); t = np.asarray(t, float)
    out = np.full(len(x), np.nan)
    for r in pd.unique(rec):
        idx = np.nonzero(rec == r)[0]
        o = idx[np.argsort(t[idx], kind='stable')]
        v = x[o]
        for i in range(len(o)):
            w = v[max(0, i - k + 1):i + 1]
            w = w[np.isfinite(w)]
            out[o[i]] = np.median(w) if len(w) else np.nan
    return out


def apply_variant(df, name):
    """Estimates of one variant for every row of df (columns hr_spec, hr_acf, hr_cnn, hr_p2, rec, t_start)."""
    if name == 'cnn':
        return df.hr_cnn.values.astype(float)
    if name == 'spec':
        return df.hr_spec.values.astype(float)
    if name == 'acf':
        return df.hr_acf.values.astype(float)
    if name == 'p2_final_rule':
        return df.hr_p2.values.astype(float)
    harmonic, octave, med = STEPS[name]
    est = np.array([vote_window(s, a, c, harmonic, octave) for s, a, c in zip(df.hr_spec.values, df.hr_acf.values, df.hr_cnn.values)])
    return causal_median(est, df.rec.values, df.t_start.values) if med else est


# ------------------------------------------------------------------------------------------------ inputs (cache-dependent)
def _estimates(ds, n, meta):
    from .baselines import hr_est as HE
    z = np.load(os.path.join(CACHE, 'baselines', f'hr_raw_{ds}_{n}.npz'))
    raw = {k: z[k] for k in z.files}
    ok = HE.gate_ok_vec(raw['hr'], raw['hr_acf'], raw['sqi'], raw['acf'], *HE.SHIPPED_GATE)
    first = np.where(ok.any(1), ok.argmax(1), 0)
    r = np.arange(len(first))
    return dict(hr_spec=raw['hr'][r, first], hr_acf=raw['hr_acf'][r, first], sqi=raw['sqi'][r, first], acf=raw['acf'][r, first], gate_ok=ok.any(1))


def build_inputs(res):
    out = {}
    for ds, n, src in (('butppg', 3840, os.path.join(res, 'scores', 'hybrid_final_butppg.csv')), ('mths', 2065, os.path.join(CACHE, 'scores', 'hybrid_final_mths.csv'))):
        meta = pd.read_parquet(os.path.join(CACHE, 'data', ds, 'meta.parquet'), columns=['subject', 'rec', 't_start'])
        meta['row'] = np.arange(len(meta))
        e = _estimates(ds, n, meta)
        f = pd.read_csv(src)
        f['t_start'] = f.t_start.astype(float); meta['t_start'] = meta.t_start.astype(float)
        d = f.merge(meta[['rec', 't_start', 'row']], on=['rec', 't_start'], how='left')
        assert d.row.notna().all() and len(d) == len(f)
        r = d.row.values.astype(int)
        t = pd.DataFrame(dict(subject=d.subject.values, rec=d.rec.values, t_start=d.t_start.values, hr_ref=d.hr_ref.values, hr_spec=e['hr_spec'][r], hr_acf=e['hr_acf'][r], hr_cnn=d.hr_cnn.values,
                              hr_p2=d.hr_final.values, gate_ok=e['gate_ok'][r], sqi=e['sqi'][r], acf=e['acf'][r], cnn_sqi=d.cnn_sqi.values, spec_forced_p2=d.hr_spec.values))
        out[ds] = t
    meta = pd.read_parquet(os.path.join(CACHE, 'data', 'hoffman', 'meta.parquet'), columns=['subject', 'rec', 't_start', 'hr_ref'])
    e = _estimates('hoffman', 6004, meta)
    h = pd.read_csv(os.path.join(res, 'scores', 'hybrid_final_hoffman.csv'))
    r = h.local.values.astype(int)
    out['hoffman'] = pd.DataFrame(dict(subject=h.subject.values, rec=h.rec.values, t_start=h.t_start.values, hr_ref=meta.hr_ref.values[r], hr_spec=e['hr_spec'][r], hr_acf=e['hr_acf'][r], hr_cnn=h.cnn_hr.values,
                                       hr_p2=np.nan, gate_ok=e['gate_ok'][r], sqi=e['sqi'][r], acf=e['acf'][r], cnn_sqi=h.cnn_sqi.values, spec_forced_p2=np.nan))
    for ds in ('butppg', 'hoffman'):
        out[ds].to_csv(os.path.join(res, 'scores', f'hr_vote_inputs_{ds}.csv'), index=False)
    os.makedirs(os.path.join(CACHE, 'scores'), exist_ok=True)
    out['mths'].to_csv(os.path.join(CACHE, 'scores', 'hr_vote_inputs_mths.csv'), index=False)          # licence: MTHS rows stay in the cache
    print({k: len(v) for k, v in out.items()})
    return out


def slow_pulse_band(res, person='hof_100004', f_lo_bpm=40.0):
    """The slow-pulse Hoffman person (reference about 43 bpm, below the shipped spectral floor F_LO = 0.85 Hz = 51 bpm): spectral and autocorrelation estimates with the search band
    widened to `f_lo_bpm`, voted with the CNN by the same rule. Needs the Hoffman shard. Returns one row per estimator / rule."""
    from oxydrop import hr as HR
    from . import features as F
    from . import windows as W
    ws = W.load('hoffman')
    m = ws.meta
    inp = pd.read_csv(os.path.join(res, 'scores', 'hr_vote_inputs_hoffman.csv'))
    inp = inp[inp.subject == person]
    key = {(r, float(t)): i for i, (r, t) in enumerate(zip(m.rec.values, m.t_start.values))}
    rows = np.array([key[(r, float(t))] for r, t in zip(inp.rec.values, inp.t_start.values)])
    est = {k: np.full(len(rows), np.nan) for k in ('spec_band51', 'acf_band51', 'spec_band40', 'acf_band40')}
    for a in range(0, len(rows), 256):
        idx = rows[a:a + 256]
        ac = F.causal_ac(ws.lrgb(idx).astype(np.float64))
        for j in range(len(idx)):
            w = ac[j, -300:]
            sd = w.std(0)
            prim = 1 if sd[1] > 0.3 * sd[0] else 0
            for tag, flo in (('band51', None), ('band40', f_lo_bpm / 60.0)):
                e = HR.estimate_hr(w[:, prim], 30, f_lo=flo)
                est['spec_' + tag][a + j] = e['hr']; est['acf_' + tag][a + j] = e['hr_acf']
    ref = inp.hr_ref.values; c = inp.hr_cnn.values
    out = []

    def add(name, v):
        mm = M.hr_metrics(v, ref)
        out.append(dict(person=person, estimator=name, n_windows=len(ref), mae=mm['mae'], within5=mm['within5'], halving=mm['halving'], doubling=mm['doubling'], mean_estimate=float(np.nanmean(v)), mean_ref=float(np.mean(ref))))
    add('cnn', c)
    for k, v in est.items():
        add(k, v)
    for tag in ('band51', 'band40'):
        v = np.array([vote_window(s_, a_, c_, True, True) for s_, a_, c_ in zip(est['spec_' + tag], est['acf_' + tag], c)])
        add(f'vote+harmonic+octave ({tag})', v)
        add(f'vote+harmonic+octave+median5 ({tag})', causal_median(v, inp.rec.values, inp.t_start.values))
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(res, 'hr_vote_slow_band.csv'), index=False)
    return t


# ------------------------------------------------------------------------------------------------ evaluation
def metric_row(df, est, B, seed=0):
    ref = df.hr_ref.values.astype(float)
    ok = np.isfinite(ref)
    d = df[ok]; e = np.asarray(est, float)[ok]; r = ref[ok]
    sub = d.subject.values
    base = M.hr_metrics(e, r)
    row = dict(n_windows=int(ok.sum()), n_people=int(len(np.unique(sub))), coverage=base['coverage'], mae=base['mae'], within5=base['within5'], within5_full=base['within5_full'], doubling=base['doubling'],
               halving=base['halving'], bias=base['bias'])
    if B and len(np.unique(sub)) >= 2:
        for k in ('mae', 'within5', 'doubling', 'halving'):
            b = M.person_bootstrap(sub, lambda i, k=k: M.hr_metrics(e[i], r[i])[k], B=B, seed=seed)
            row[k + '_lo'], row[k + '_hi'] = b['lo'], b['hi']
    per = pd.DataFrame(dict(subject=sub, e=e, r=r)).groupby('subject').apply(lambda g: pd.Series(dict(mae=M.hr_metrics(g.e.values, g.r.values)['mae'], within5=M.hr_metrics(g.e.values, g.r.values)['within5'],
                                                                                                          doubling=M.hr_metrics(g.e.values, g.r.values)['doubling'], n=len(g))), include_groups=False)
    row.update(subj_mae_mean=float(per.mae.mean()), subj_mae_sd=float(per.mae.std(ddof=1)) if len(per) > 1 else np.nan, subj_mae_min=float(per.mae.min()), subj_mae_max=float(per.mae.max()),
               subj_within5_mean=float(per.within5.mean()), subj_within5_sd=float(per.within5.std(ddof=1)) if len(per) > 1 else np.nan, subj_within5_min=float(per.within5.min()))
    return row, per


def write_subject_tables(subj: pd.DataFrame, res: str):
    """Per-subject rows only for the datasets whose licence allows it (Hoffman MIT, BUT PPG CC BY): hr_vote_per_subject.csv.
    MTHS (CC BY-NC-ND, aggregates only) gets the spread over its subjects instead: hr_vote_subject_spread.csv (subjects, mean, SD, 10/25/50/75/90th percentiles, count with MAE < 3 bpm)."""
    qs = (0.10, 0.25, 0.50, 0.75, 0.90)
    subj[subj.dataset != 'mths'].to_csv(os.path.join(res, 'hr_vote_per_subject.csv'), index=False)
    rows = []
    for (ds, v, sc), g in subj[subj.dataset == 'mths'].groupby(['dataset', 'variant', 'scope'], sort=False):
        r = dict(dataset=ds, variant=v, scope=sc, n_subjects=int(g.subject.nunique()), n_mae_below_3=int((g.mae < TARGET_MAE).sum()))
        for c in ('mae', 'within5', 'doubling'):
            x = g[c].dropna().to_numpy(dtype=float)
            r[c + '_mean'] = float(x.mean()); r[c + '_sd'] = float(x.std(ddof=1)) if len(x) > 1 else float('nan')
            for q in qs:
                r[f'{c}_q{int(round(100 * q)):02d}'] = float(np.quantile(x, q))
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(res, 'hr_vote_subject_spread.csv'), index=False)


def run(res, B=2000):
    ins = {ds: pd.read_csv(os.path.join(res, 'scores', f'hr_vote_inputs_{ds}.csv')) for ds in ('butppg', 'hoffman')}
    pm = os.path.join(CACHE, 'scores', 'hr_vote_inputs_mths.csv')
    if os.path.exists(pm):
        ins['mths'] = pd.read_csv(pm)
    ins['butppg+mths'] = pd.concat([ins[k] for k in ('butppg', 'mths') if k in ins], ignore_index=True)
    rows, subj_rows, est_store = [], [], {}
    for ds, df in ins.items():
        for v in VARIANTS:
            if v == 'p2_final_rule' and ds == 'hoffman':
                continue
            est = apply_variant(df, v)
            est_store[(ds, v)] = est
            for scope in ('all', 'clean'):
                m = df.gate_ok.values.astype(bool) if scope == 'clean' else np.ones(len(df), bool)
                sub = df[m].reset_index(drop=True)
                r, per = metric_row(sub, est[m], B)
                r.update(dataset=ds, variant=v, scope=scope, share_of_windows=float(m.mean()), meets_target=bool((r['mae'] < TARGET_MAE) and (r['doubling'] < TARGET_DOUBLING)) if scope == 'clean' else np.nan)
                rows.append(r)
                if ds != 'butppg+mths' and v in ('cnn', 'spec', 'p2_final_rule', 'vote+harmonic+octave+median5'):
                    for s, p in per.iterrows():
                        subj_rows.append(dict(dataset=ds, variant=v, scope=scope, subject=s, n=int(p.n), mae=p.mae, within5=p.within5, doubling=p.doubling))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(res, 'hr_vote_summary.csv'), index=False)
    write_subject_tables(pd.DataFrame(subj_rows), res)
    # paired person bootstrap of the full rule against the CNN alone and the P2 final rule (all windows, then clean)
    prow = []
    for ds in ('butppg', 'mths', 'butppg+mths', 'hoffman'):
        if ds not in ins:
            continue
        df = ins[ds]
        full = est_store[(ds, 'vote+harmonic+octave+median5')]
        for base in ('cnn', 'spec', 'p2_final_rule'):
            if (ds, base) not in est_store:
                continue
            b = est_store[(ds, base)]
            for scope in ('all', 'clean'):
                m = df.gate_ok.values.astype(bool) if scope == 'clean' else np.ones(len(df), bool)
                ref = df.hr_ref.values
                ok = m & np.isfinite(ref)
                sub = df.subject.values[ok]
                a_, b_, r_ = full[ok], b[ok], ref[ok]
                for key, hib in (('mae', False), ('within5_full', True)):
                    pb = M.paired_person_bootstrap(sub, lambda i, key=key: M.hr_metrics(a_[i], r_[i])[key], lambda i, key=key: M.hr_metrics(b_[i], r_[i])[key], B=B, seed=0, higher_is_better=hib)
                    prow.append(dict(dataset=ds, scope=scope, rule='vote+harmonic+octave+median5', baseline=base, metric=key, rule_value=pb['a'], baseline_value=pb['b'], better_is_positive_diff=pb['diff'], lo=pb['lo'], hi=pb['hi'],
                                     verdict='better' if pb['lo'] > 0 else ('worse' if pb['hi'] < 0 else 'tie')))
    pd.DataFrame(prow).to_csv(os.path.join(res, 'hr_vote_paired.csv'), index=False)
    # slow pulse: windows with reference HR below 50 bpm
    srows = []
    for ds in ('hoffman', 'butppg', 'mths'):
        if ds not in ins:
            continue
        df = ins[ds]
        for v in VARIANTS:
            if (ds, v) not in est_store:
                continue
            m = (df.hr_ref.values < 50)
            # minimum group size 5 people for every dataset except Hoffman (MIT data, one named slow-pulse case study); a smaller MTHS or BUT PPG group is one person's result
            if m.sum() >= 5 and (ds == 'hoffman' or df.subject[m].nunique() >= 5):
                mm = M.hr_metrics(est_store[(ds, v)][m], df.hr_ref.values[m])
                srows.append(dict(dataset=ds, variant=v, n_windows=int(m.sum()), n_people=int(df.subject[m].nunique()), mae=mm['mae'], within5=mm['within5'], halving=mm['halving'], doubling=mm['doubling'], mean_estimate=float(np.nanmean(est_store[(ds, v)][m])), mean_ref=float(df.hr_ref.values[m].mean())))
    pd.DataFrame(srows).to_csv(os.path.join(res, 'hr_vote_slow_pulse.csv'), index=False)
    # votes written back (BUT PPG, Hoffman committed; MTHS in the cache)
    for ds in ('butppg', 'hoffman'):
        o = ins[ds].copy()
        o['hr_vote'] = est_store[(ds, 'vote+harmonic+octave+median5')]
        o.to_csv(os.path.join(res, 'scores', f'hr_vote_{ds}.csv'), index=False)
    if 'mths' in ins:
        o = ins['mths'].copy(); o['hr_vote'] = est_store[('mths', 'vote+harmonic+octave+median5')]
        o.to_csv(os.path.join(CACHE, 'scores', 'hr_vote_mths.csv'), index=False)
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('inputs', 'run', 'report', 'slow'):
        p = sp.add_parser(n); p.add_argument('--res', default='../results/ml'); p.add_argument('--B', type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == 'inputs':
        build_inputs(a.res)
    elif a.cmd == 'run':
        run(a.res, a.B)
    elif a.cmd == 'slow':
        slow_pulse_band(a.res)
    else:
        from .hr_vote_md import report
        report(a.res)


if __name__ == '__main__':
    main()
