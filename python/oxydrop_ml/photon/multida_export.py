"""Export `multida` (multida.py) to JSON: results/photon/multida_weights.json (distill_export.py format, so distill_app_export.py --src can turn it
into the app file) and results/photon/multida_app.json (nn-runtime-v1 graphs + feature normaliser, the app schema of distill_app_export.app_json),
each with a Python reference output on 2 fixture windows.

    cd python && MULTIDA_PACK=... MULTIDA_AUX=... DISTILL_FIXTURE=.../distill_fixture.npz python -m oxydrop_ml.photon.multida_export

Recipe (no held-out number belongs to this refit; the quoted numbers are the leave-one-person-out ones in results/photon/multida_log.md):
  * lam_kd, w_aux and epochs = the modal nested choice over the 6 outer folds of each seed (ties toward the smaller value); tau = median of the fold taus;
  * teachers and the network are refit on all 6 Hoffman people (clean + the seed's 4 augmented copies) + the MTHS / BUT PPG auxiliary rows, seeds 0-2;
  * deployed output = 3-seed ensemble: mean sigmoid(hyp), mean spo2, mean sigmoid(abstain) against the mean tau. The 4th output (dHR / 10, auxiliary
    head) is in the weights but not used by the app. The domain head is training-only and not exported.
Ops: standardise (median-impute, clip, (x - mu) / sd), Linear(39, 32), ReLU, Linear(32, 32), ReLU, Linear(32, 4), sigmoid: nn-runtime-v1 plain ops.
The fixture windows are the 2 windows (t = 60 s, 300 s) of tests/fixtures/ppg-csv/Left/100001.csv used by distill_export.py (DISTILL_FIXTURE).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from . import distill_app_export as DAE
from . import distill_export as DE
from . import distill_student as DS
from . import flashgap as F
from . import multida as M

OUT = M.RES / 'multida_weights.json'
OUT_APP = M.RES / 'multida_app.json'
FORMAT = 'oxydrop-photon-multida-v1'


def recipe(seed):
    with open(M.OUT_DIR / 'models' / f'multida_s{seed}.pkl', 'rb') as fh:
        folds = pickle.load(fh)
    return (DE.modal([fo['info']['lam'] for fo in folds]), DE.modal([fo['info']['w_aux'] for fo in folds]), DE.modal([fo['info']['epochs'] for fo in folds]),
            float(np.median([fo['tau'] for fo in folds])))


def fit_all(seed, lam, w_aux, epochs):
    os.environ.setdefault(DS.PACK_ENV, os.environ.get('MULTIDA_PACK', ''))
    DS._pack()
    mt = F.meta()
    conds = DS.train_conds(seed)
    Xd, _ = DS.features(conds, log=lambda *a: None)
    rows = np.arange(len(mt))
    H = M.hof_design(Xd, conds, rows, mt.y.values.astype(int), mt.y_spo2.values.astype(float))
    A = M.aux_for(M.MODELS[f'multida_s{seed}'], 'all')
    return M.train_net(H, A, lam, w_aux, M.GRL, epochs, seed, [epochs])[epochs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--out-app', default=str(OUT_APP))
    ap.add_argument('--app-only', action='store_true', help='only convert an existing --out weights file to the nn-runtime-v1 app schema (no training)')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    if a.app_only:
        return app_export(a)
    nets, taus, rec = [], [], []
    for s in range(3):
        lam, wa, ep, tau = recipe(s)
        nets.append(fit_all(s, lam, wa, ep)); taus.append(tau); rec.append(dict(seed=s, lam_kd=lam, w_aux=wa, grl=M.GRL, epochs=ep, tau=tau))
    tw, Xw = DE.fixture_windows()
    sc, sp, cf, an = DE.predict_ens(nets, taus, Xw)
    js = dict(format=FORMAT, cols=F.COLS, hidden=M.HID, n_out=M.NOUT, heads=['hyp_logit', 'spo2_(x-90)/5', 'abstain_logit', 'dhr_/10'], recipe=rec,
              tau=float(np.mean(taus)), students=[DE.student_json(m) for m in nets],
              reference=dict(fixture=str(DE.FIXTURE.relative_to(F.REPO)), t=DE._l(tw), X=DE._l(Xw), score=DE._l(sc), spo2_hat=DE._l(sp), conf=DE._l(cf),
                             answer=[bool(v) for v in an]))
    Path(a.out).write_text(json.dumps(js))
    print(f'wrote {a.out}: recipe {rec}, reference scores {np.round(sc, 4).tolist()}, conf {np.round(cf, 4).tolist()}, answer {an.tolist()}')
    app_export(a)


def app_export(a):
    src = json.loads(Path(a.out).read_text())
    sc = np.asarray(src['reference']['score']); an = np.asarray(src['reference']['answer'])
    app = DAE.app_json(src, Path(a.out).resolve())
    app.update(format='oxydrop-photon-multida-app-v1', id='oxydrop-photon-multida-v1', model='multida',
               note='Trained on Hoffman 2022 (6 people) + MTHS (CC BY-NC-ND 4.0, non-commercial) + BUT PPG (CC BY 4.0); research prototype, not a medical device.',
               reference=src['reference'])
    Path(a.out_app).write_text(json.dumps(app))
    back = json.loads(Path(a.out_app).read_text())
    s2, p2, c2, a2 = DAE.forward(back, np.asarray(src['reference']['X'], np.float64))
    err = float(np.max(np.abs(s2 - sc)))
    print(f'wrote {a.out_app}: nn-runtime-v1 round trip vs the training-code reference, max |dp| {err:.2e}')
    assert err < 1e-6 and list(a2) == list(an)


if __name__ == '__main__':
    from oxydrop_ml.photon import multida_export as _m
    _m.main()
