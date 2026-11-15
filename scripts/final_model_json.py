"""Write results/ml/FINAL_MODEL.json: the identity of the model that the app ships (weights hashes, parameters, seeds, training subjects, pipeline, thresholds).

    source scripts/ml_env.sh && python scripts/final_model_json.py [--ckpt-dir RUNS/hyb_final]

Reads app/data/{hybrid_weights.json, hybrid_R_s*.json, conformal.json, oxynet.onnx, weights.json} and, for the training subjects, the window shards in ml_cache (the same
Pool that `hybrid_run final` fitted on; the held-out OpenOximetry people are not in it). Writes aggregates only: subject counts and the Hoffman ids, no row-level data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=os.path.join(os.environ.get('VERITAS', '/home/aravhawk/.t3/scratch/veritas'), 'ml_cache/runs/hyb_final'))
    ap.add_argument('--out', default=str(ROOT / 'results/ml/FINAL_MODEL.json'))
    a = ap.parse_args()
    import torch
    from oxydrop_ml.train import Pool
    A = ROOT / 'app/data'
    hw = json.load(open(A / 'hybrid_weights.json')); cf = json.load(open(A / 'conformal.json'))
    pool = Pool()
    subj = {d: sorted(set(pool.subjects_of(d))) for d in ('hoffman', 'mths', 'butppg', 'openox')}
    members = []
    for m in hw['members']:
        ck = torch.load(os.path.join(a.ckpt_dir, f"{m['family']}_s{m['seed']}.pt"), map_location='cpu', weights_only=False)
        assert sha(os.path.join(a.ckpt_dir, f"{m['family']}_s{m['seed']}.pt")) == m['ckpt_sha256'], 'checkpoint does not match hybrid_weights.json'
        members.append(dict(family=m['family'], seed=m['seed'], role=m['role'], file='app/data/' + m['weights'], file_sha256=sha(A / m['weights']), bytes=os.path.getsize(A / m['weights']),
                            ckpt_sha256=m['ckpt_sha256'], params_shipped=ck['params_shipped'], best_step=ck['best_step'], early_stopping_val_subjects=ck['val_subjects']))
    doc = dict(
        name='oxydrop-hybrid-v1', not_a_medical_device=True,
        description='v1 physics chain logit + residual correction of a 3-seed OxyNet CNN ensemble -> logistic meta-learner -> Mondrian conformal three-state rule; CNN heart rate with the rule agree',
        params_per_member=members[0]['params_shipped'], members=members, params_total_members=sum(m['params_shipped'] for m in members), seeds=hw['seeds'],
        chain=dict(models=len(hw['chain']['models']), note=hw['chain']['note'], max_abs_coef_diff_vs_app_weights_json=hw['chain']['max_abs_coef_diff_vs_app_weights_json']),
        pipeline=hw['pipeline'], conformal=dict(file='app/data/conformal.json', sha256=sha(A / 'conformal.json'), default_alpha=cf['default_alpha'], thresholds_default=cf['alphas']['0.10'], calibration=cf['calibration']),
        files=dict(hybrid_weights=dict(file='app/data/hybrid_weights.json', sha256=sha(A / 'hybrid_weights.json')), onnx_member0=dict(file='app/data/oxynet.onnx', sha256=sha(A / 'oxynet.onnx'), bytes=os.path.getsize(A / 'oxynet.onnx')),
                   v2_quality_chain=dict(file='app/data/weights.json', sha256=sha(A / 'weights.json'))),
        training_subjects=dict(hoffman=subj['hoffman'], n_hoffman=len(subj['hoffman']), n_mths=len(subj['mths']), n_butppg=len(subj['butppg']), n_openox_train=len(subj['openox']),
                               note='The final members are fitted on every Hoffman person plus the MTHS, BUT PPG and OpenOximetry training people (the 34 held-out OpenOximetry people are excluded). One early-stopping validation split per seed is drawn from these people (early_stopping_val_subjects). '
                                    'MTHS (CC BY-NC-ND) and OpenOximetry weights: see DATA_LICENSES.md.'),
        low_amplitude_guard=dict(lac_max_min=-6.02, ac_dc_ratio_percent=0.24, derivation='0.1th percentile of the largest-channel log AC amplitude over all Hoffman windows (scripts/ood_range.py, results/ml/ood_ac_range.csv); window label low_ac, state Unsure'),
        evidence=dict(p1_hybrid_tm_bacc='results/ml/hybrid_p1_final.csv', conformal='results/ml/CONFORMAL.md', hr='results/ml/hybrid_p2.csv', openox_spo2='results/ml/OPENOX_SPO2.md'),
        caveat='The members saw all six Hoffman people, so the replay demo (100001L, 100004R) is in-sample: demo only, not a test. Held-out numbers come from the nested leave-one-person-out runs in results/ml.')
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print('wrote', a.out, 'hoffman', len(subj['hoffman']), 'mths', len(subj['mths']), 'but', len(subj['butppg']), 'openox', len(subj['openox']))


if __name__ == '__main__':
    main()
