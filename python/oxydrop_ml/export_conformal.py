"""Shipped conformal parameters for the app: `app/data/conformal.json` (parameters only) and `results/ml/conformal_shipped.csv`.

    python -m oxydrop_ml.export_conformal oof   --runs RUNS/hyb_A,RUNS/hyb_B --seeds 0 1 2 --res ../results/ml     # out-of-fold logits of the DEPLOYED pipeline on the six Hoffman people (needs the run tables)
    python -m oxydrop_ml.export_conformal build --res ../results/ml --out ../app/data/conformal.json                # thresholds from that CSV (no run tables needed)

Which pipeline. The one `export_hybrid.py` deploys (`app/data/hybrid_weights.json`: source, per-user calibration, k, meta-learner), chosen by the declared inner-fold rule on the six
out-of-fold tables. Its decision logit z is what the app computes (z > 0 = possible low oxygen).
Which scores. Out-of-fold logits of all six Hoffman people: every person's CNN logits come from the fits that exclude that person, and the meta-learner scoring a person is fit on the other
five (leave-one-person-out), so no calibration window was scored by a model that saw its own label. Thresholds are the Mondrian (class-conditional) split-conformal quantiles of these pooled
windows, in the logit domain (`conformal.thresholds`), for a few miscoverage levels alpha. Nested evaluation of the same procedure (thresholds fit without the evaluated person): `CONFORMAL.md`.
The app rule, per window with logit z at the chosen alpha: contains0 = z <= t0, contains1 = z >= t1;
  normal if contains0 and not contains1, possible low oxygen if contains1 and not contains0, otherwise unsure (re-measure).
Not tuned to any phone: six people recorded with the Hoffman devices; other phones are untested and the report says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import conformal as C

ALPHAS_APP = (0.02, 0.05, 0.10, 0.15, 0.20)
DEFAULT_ALPHA = C.HEADLINE
FORMAT = 'oxydrop-conformal-v1'
APP_STATE = {0: 'NORMAL', 1: 'LOW', 2: 'UNSURE'}                 # keys of STATE in app/js/decision.js
EXAMPLE_Z = (-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0)


def oof_logits(runs_dirs, seeds, res_dir, families=('P', 'R', 'D', 'L')):
    from . import export_hybrid as EH
    from . import hybrid as H
    runs, _ = H.load_runs(runs_dirs, families=families, seeds=seeds)
    meta, X19 = H.hoffman_meta()
    w = H.add_chain(H.wide(runs), meta, X19, None)
    ch, cols, f, oof = EH.deploy_pipeline(w)
    Fe = H.lever_features(oof, cols, ch['cal'], ch['k'])
    ev = oof['eval'].values.astype(bool)
    y = oof.low.values.astype(int)
    who = oof.subject.values
    z = np.full(len(oof), np.nan)
    for q in np.unique(who[ev]):
        te, tr = ev & (who == q), ev & (who != q)
        z[te] = H.meta_fit(Fe[tr], y[tr])(Fe[te])
    zf = f(Fe)                                                               # the deployed meta-learner (fit on all six people): for the consistency check only
    t = oof[ev][['subject', 'rec', 'local', 't_start', 'low']].copy()
    t['z'] = z[ev]; t['z_deployed_fit'] = zf[ev]
    t['source'], t['cal'], t['k'] = ch['source'], bool(ch['cal']), int(ch['k'])
    path = os.path.join(res_dir, 'scores', 'conformal_deployed_oof_hoffman.csv')
    t.to_csv(path, index=False)
    print('deployed pipeline', ch['source'], ch['cal'], ch['k'], 'rows', len(t), 'people', t.subject.nunique(), '->', path)
    return t


def nested_evaluation(res_dir):
    """Held-out (nested over people) result at the default alpha for the SHIPPED pipeline R, read from conformal_curve.csv, with the optimism caveat and the chain-only control next to it.
    The app text, README, MODEL.md and WHITEPAPER quote the same rows."""
    p = os.path.join(res_dir, 'conformal_curve.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    out = {}
    for key, system in (('deployed_pipeline_R', 'deployed_R'), ('chain_only_control', 'chain_v1'), ('per_person_selected_hybrid', 'final')):
        r = d[(d.system == system) & (d.method == 'mondrian') & ((d.param - DEFAULT_ALPHA).abs() < 1e-9)]
        if len(r) != 1:
            return None
        r = r.iloc[0]
        out[key] = dict(coverage=round(float(r.coverage), 4), accuracy_on_answered=round(float(r.acc), 4), share_of_hypoxic_windows_alerted=round(float(r.alert_share_all), 4),
                        hypoxic_called_normal=round(float(r.confident_miss), 4), coverage_ci95=[round(float(r.coverage_lo), 4), round(float(r.coverage_hi), 4)])
    out['note'] = ('Leave-one-person-out over the 6 Hoffman people, thresholds for each held-out person from the other five only, alpha = ' + str(DEFAULT_ALPHA) + '. deployed_pipeline_R is the pipeline in hybrid_weights.json (the model the app runs); '
                   'its row is exploratory and optimistic because R was chosen with all six people in view. per_person_selected_hybrid is the nested estimate of the selection procedure. '
                   'chain_only_control (the v1 logistic chain through the same conformal step, available in the app as ?engine=logistic) has the higher point estimates; all intervals overlap. Source: results/ml/conformal_curve.csv.')
    return out


def build(res_dir, out_json, hybrid_json=None):
    t = pd.read_csv(os.path.join(res_dir, 'scores', 'conformal_deployed_oof_hoffman.csv'))
    z, y = t.z.values.astype(float), t.low.values.astype(int)
    rows, alphas = [], {}
    for method in C.METHODS:
        for a in ALPHAS_APP:
            t0, t1 = C.thresholds(z, y, a, method)
            s, c0, c1 = C.states(z, t0, t1)
            m = C.metrics_from_counts(C.count_vector(z, y, t0, t1))
            rows.append(dict(method=method, alpha=a, t0=t0, t1=t1, past_crossing=bool(t0 < t1), pooled_in_sample_coverage=float(m['coverage']), pooled_in_sample_acc=float(m['acc']),
                             n_normal=int((y == 0).sum()), n_hypoxic=int((y == 1).sum()), n_people=int(t.subject.nunique())))
            if method == 'mondrian':
                alphas[f'{a:.2f}'] = dict(t0=float(t0), t1=float(t1), normal_below=float(min(t0, t1)), low_above=float(max(t0, t1)), past_crossing=bool(t0 < t1))
    csv = pd.DataFrame(rows)
    csv.to_csv(os.path.join(res_dir, 'conformal_shipped.csv'), index=False)
    hw = json.load(open(hybrid_json)) if hybrid_json and os.path.exists(hybrid_json) else None
    pipe = None
    if hw:
        p = hw['pipeline']
        pipe = dict(source=p['source'], features=p['features'], calibration=p['calibration'], aggregate_windows=p['aggregate_windows'], meta_intercept=p['meta']['intercept'], meta_coef=p['meta']['coef'])
    doc = dict(format=FORMAT, not_a_medical_device=True,
               description='Class-conditional (Mondrian) split-conformal thresholds on the decision logit z of the hybrid pipeline in hybrid_weights.json. Parameters only.',
               score='z = decision logit of the pipeline (z > 0 = possible low oxygen); app/js computes it from hybrid_weights.json',
               method='mondrian split conformal, nonconformity 1 - p_y, finite-sample quantile ceil((n + 1)(1 - alpha)) / n per class, in the logit domain',
               states=dict(NORMAL='Normal vs your baseline', LOW='Possible low oxygen: re-check with a pulse oximeter', UNSURE='Unsure, re-measure'),
               rule=dict(contains_normal='z <= t0', contains_low='z >= t1', normal='contains_normal and not contains_low', possible_low_oxygen='contains_low and not contains_normal',
                         unsure='otherwise (both or neither), re-measure',
                         shortcut='normal if z < normal_below, possible low oxygen if z > low_above, unsure in between (equality has probability zero); past_crossing = true means alpha is so large that the middle band holds empty sets, not used'),
               default_alpha=DEFAULT_ALPHA,
               alphas=alphas,
               calibration=dict(source='out-of-fold logits of the deployed pipeline on all six Hoffman people (scores/conformal_deployed_oof_hoffman.csv), leave-one-person-out meta-learner',
                                n_people=int(t.subject.nunique()), n_windows=int(len(t)), n_normal=int((y == 0).sum()), n_hypoxic=int((y == 1).sum()), min_t_start_s=float(t.t_start.min())),
               pipeline=pipe,
               nested_evaluation=nested_evaluation(res_dir),
               examples=[dict(z=zz, state={k: APP_STATE[int(C.states(np.array([zz]), v['t0'], v['t1'])[0][0])] for k, v in alphas.items()}) for zz in EXAMPLE_Z],
               limits='Fit on six people with the Hoffman phones; windows overlap and people differ, so the coverage guarantee is approximate (see results/ml/CONFORMAL.md for the nested evaluation). Other phones are untested.')
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(doc, f, indent=1)
    print('wrote', out_json)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    o = sp.add_parser('oof'); o.add_argument('--runs', required=True); o.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2]); o.add_argument('--res', default='../results/ml')
    b = sp.add_parser('build'); b.add_argument('--res', default='../results/ml'); b.add_argument('--out', default='../app/data/conformal.json'); b.add_argument('--hybrid', default='../app/data/hybrid_weights.json')
    a = ap.parse_args(argv)
    if a.cmd == 'oof':
        oof_logits(a.runs.split(','), a.seeds, a.res)
    else:
        build(a.res, a.out, a.hybrid)


if __name__ == '__main__':
    main()
