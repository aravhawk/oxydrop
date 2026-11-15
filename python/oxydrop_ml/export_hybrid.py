"""Export the deployable hybrid for the app (FEAT-008 -> FEAT-009): app/data/hybrid_weights.json (+ one CNN weight file per ensemble member) and a combiner fixture.

    python -m oxydrop_ml.export_hybrid --runs RUNS/hyb_A,RUNS/hyb_host --seeds 0 1 2 --final-dir RUNS/hyb_final --out-dir ../app/data --fixture ../tests/fixtures/hybrid_ref.json

What is exported is the pipeline that the declared selection rule (amendments/AMENDMENT_2_hybrid.md) picks when it is applied to the six leave-one-person-out
out-of-fold prediction tables together (the same rule as in the nested evaluation, one more person), with the meta-learner fit on those six people's out-of-fold
windows, and the ensemble members fit on all Hoffman people (`hybrid_run final`). The schema is documented in results/ml/HYBRID.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import hybrid as H
from . import model as M
from .distill import Chain
from .export_js import export_graph

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'oxydrop-hybrid-v1'
FAMILY_OF = {'p': 'P', 'd': 'D', 'corr': 'R', 'r': 'R', 'corrL': 'L', 'rL': 'L'}


def oof_table(w):
    """The six outer out-of-fold tables stacked as one 'inner'-style table (the person is the CV group), for the deployment-time selection."""
    t = w[w.target == 'outer'].copy()
    t['inner'] = t.subject
    return t.reset_index(drop=True)


def deploy_pipeline(w, order=None):
    order = order or [n for n in H.CNN_ORDER_EXT if H.has_cols(w, n)]
    oof = oof_table(w)
    ch = H.select(oof, order)
    cols = H.cand_cols(ch['source'])
    ev = oof['eval'].values.astype(bool)
    f = H.meta_fit(H.lever_features(oof, cols, ch['cal'], ch['k'])[ev], oof.low.values[ev].astype(int))
    return ch, cols, f, oof


def members_for(cols):
    fams = sorted({FAMILY_OF[c] for c in cols if c in FAMILY_OF})
    return fams


def export_member(ckpt, path, family, seed):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    net = M.OxyNet(M.OxyCfg(**ck['cfg'])); net.load_state_dict(ck['state']); net.eval()
    g = export_graph(net)
    g['random_init'] = False
    g['hybrid_family'] = family; g['seed'] = seed
    g['ckpt_sha256'] = hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()
    Path(path).write_text(json.dumps(g, separators=(',', ':')))
    return g['ckpt_sha256'], Path(path).stat().st_size


def engine_v1_weights(chain, shipped):
    """The chain in the weights format that app/js/engine_v1.js `Chain` takes: {hr:{cols,mu,sd,coef,b}, fusion:{...}} (column NAMES from app/data/weights.json when present)."""
    names = shipped['fusion']['cols'] if shipped else list(range(19))
    out = {}
    for key, m in zip(('hr', 'fusion'), chain.params()):
        out[key] = dict(cols=[names[i] for i in m['cols']], mu=m['mean'], sd=m['scale'], coef=m['coef'], b=m['intercept'])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--final-dir', default=None, help='checkpoints from `hybrid_run final` (family_s<seed>.pt)')
    ap.add_argument('--out-dir', default=str(ROOT / 'app/data')); ap.add_argument('--fixture', default=str(ROOT / 'tests/fixtures/hybrid_ref.json'))
    ap.add_argument('--chain-weights', default=str(ROOT / 'app/data/weights.json'))
    ap.add_argument('--strict', action='store_true', help='fail when a member checkpoint is missing')
    a = ap.parse_args(argv)
    runs, counts = H.load_runs(a.runs.split(','), seeds=a.seeds)
    meta, X19 = H.hoffman_meta()
    w = H.add_chain(H.wide(runs), meta, X19, None)
    ch, cols, f, oof = deploy_pipeline(w)
    fams = members_for(cols)
    ev = (meta['eval'].values & meta.has_low.values)
    chain = Chain(X19[ev], meta.low.values[ev].astype(int))
    shipped = json.load(open(a.chain_weights)) if os.path.exists(a.chain_weights) else None
    chain_diff = None
    if shipped:
        chain_diff = max(float(np.max(np.abs(np.array(m['coef']) - np.array(shipped[k]['coef'])))) for m, k in zip(chain.params(), ('hr', 'fusion')))
    members = []
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for fam in fams:
        for s in a.seeds:
            ck = Path(a.final_dir or '.') / f'{fam}_s{s}.pt'
            item = dict(family=fam, seed=s, weights=f'hybrid_{fam}_s{s}.json', role={'P': 'p', 'D': 'd', 'R': 'corr', 'L': 'corrL'}[fam])
            if ck.exists():
                sha, size = export_member(ck, out / item['weights'], fam, s)
                item.update(ckpt_sha256=sha, bytes=size)
            else:
                item.update(weights=None, missing=True)
                if a.strict:
                    raise SystemExit(f'missing member checkpoint {ck}')
            members.append(item)
    doc = dict(
        format=FORMAT, not_a_medical_device=True,
        description='Physics-informed hybrid: v1 chain logit + CNN member logits -> per-user calibration -> trailing mean -> logistic meta-learner -> low-oxygen logit (> 0 = possible low oxygen).',
        pipeline=dict(source=ch['source'], features=cols, calibration=bool(ch['cal']), aggregate_windows=int(ch['k']), threshold=0.0,
                      meta=f.params, inner_cv_tm_bacc=ch['inner_cv']),
        features_definition=dict(c='logit(p_chain) with p_chain the mean probability of the two v1 chain models, clipped to [1e-6, 1-1e-6]',
                                 p='mean over p-members of lowox_logit', d='mean over d-members of lowox_logit', corr='mean over R-members of lowox_logit (the residual correction)', corrL='mean over L-members (frozen backbone) of lowox_logit',
                                 r='c + corr', rL='c + corrL'),
        chain=dict(models=chain.params(), combine='mean_probability_then_logit', clip=1e-6, features='app/js/engine_v1.js FUSION deltas (19), model 0 uses column 0 only',
                   engine_v1_weights=engine_v1_weights(chain, shipped),
                   max_abs_coef_diff_vs_app_weights_json=chain_diff,
                   note='Refit under the LOSO harness on the 5,644 evaluated Hoffman windows (shipped C 0.01 / 0.1). Slightly different from app/data/weights.json (the pure v1 chain): the hybrid meta-learner was fit on THIS chain, so use these weights for c.'),
        members=members,
        ensemble='mean logit over seeds within a family',
        calibration_definition='feature minus its median over the windows of the same recording that end inside the first 60 s (label free)',
        aggregation_definition='causal mean of the last k windows of the same recording, window hop 2 s, baseline windows included, fewer rows at the start of the recording',
        selection=dict(rule='amendments/AMENDMENT_2_hybrid.md applied to the 6 out-of-fold tables', stage1_cv={k: float(v) for k, v in ch['stage1'].items()},
                       cal_on_cv=float(ch['cal_on']), stage3_cv={str(k): float(v) for k, v in ch['stage3'].items()}),
        seeds=a.seeds)
    (out / 'hybrid_weights.json').write_text(json.dumps(doc, indent=1))
    # combiner fixture: one recording, features -> expected logit (the chain and the CNN members are checked by their own parity tests)
    rec = oof[oof.rec == oof.rec.iloc[0]].sort_values('t_start')
    ev_mask = rec['eval'].values.astype(bool)
    sc, _ = H.apply_pipeline(oof, rec.reset_index(drop=True), ch['source'], ch['cal'], ch['k']) if ch['source'] != 'chain_shipped' else (rec.c.values, None)
    fx = dict(recording=rec.rec.iloc[0], source=ch['source'], calibration=bool(ch['cal']), k=int(ch['k']),
              rows=[dict(t_start=float(t), in_base=bool(b), eval=bool(e), **{c: float(rec[c].iloc[i]) for c in ('c', 'p', 'd', 'corr') if c in rec})
                    for i, (t, b, e) in enumerate(zip(rec.t_start, rec.in_base, ev_mask))],
              expected_logit=[float(x) for x in sc])
    Path(a.fixture).parent.mkdir(parents=True, exist_ok=True)
    Path(a.fixture).write_text(json.dumps(fx, separators=(',', ':')))
    print('pipeline', ch['source'], 'cal', ch['cal'], 'k', ch['k'], 'inner CV', round(ch['inner_cv'], 3), 'members', [(m['family'], m['seed'], not m.get('missing')) for m in members])


if __name__ == '__main__':
    main()
