"""Export the PhotonDistill student (distill_student.py) to a JSON weight file plus a Python reference output on 2 fixture windows.

    cd python && python -m oxydrop_ml.photon.distill_export [--out results/photon/distill_weights.json]

Recipe (no held-out number belongs to this refit; the quoted numbers are the leave-one-person-out ones in results/photon/TABLE.csv):
  * lam_kd, weight decay and epochs = the modal nested choice over the 6 outer folds of each seed (ties toward the smaller value); tau = the median of the 6 fold taus;
  * teachers (chain v1, DC-colour LR) and the student are refit on all 6 Hoffman people (clean + the seed's 4 augmented copies), seeds 0-2;
  * the deployed output is the 3-seed ensemble: mean of sigmoid(hyp), mean spo2, mean sigmoid(abstain) against the mean tau.
Ops: standardise (median-impute, clip, (x - mu) / sd), Linear, ReLU, Linear, ReLU, Linear(3), sigmoid. The reference block holds the 39 input features
of 2 windows of tests/fixtures/ppg-csv/Left/100001.csv (computed by flashgap.rec_features) and the expected outputs, so a JS port can be checked.
The 39-feature extractor (chain deltas + FlashGap quality + respiratory group) is not yet ported to JS; only the network is plain-op exportable here.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import distill_student as D
from . import flashgap as F

OUT = D.RES / 'distill_weights.json'
FIXTURE = F.REPO / 'tests' / 'fixtures' / 'ppg-csv' / 'Left' / '100001.csv'
FORMAT = 'oxydrop-photon-distill-v1'


def _l(x):
    return np.asarray(x, np.float64).round(8).tolist()


def modal(vals):
    c = Counter(vals); top = max(c.values())
    return min(v for v in c if c[v] == top)


def recipe(seed):
    with open(D.CACHE_DIR / 'models' / f'distill_s{seed}.pkl', 'rb') as fh:
        folds = pickle.load(fh)
    return (modal([fo['info']['lam'] for fo in folds]), modal([fo['info'].get('wd', D.WD) for fo in folds]), modal([fo['info']['epochs'] for fo in folds]),
            float(np.median([fo['tau'] for fo in folds])))


def fit_all(seed, lam, wd, epochs):
    D._pack()
    mt = F.meta()
    conds = D.train_conds(seed)
    Xd, _ = D.features(conds, log=lambda *a: None)
    rows = np.arange(len(mt))
    Xs, ys, w, q = D.soft_targets(Xd, conds, rows, mt.y.values.astype(int))
    return D.train_student(Xs, ys, w, q, np.tile(mt.y_spo2.values.astype(float), len(conds)), lam, epochs, seed, wd=wd)[epochs]


def student_json(m):
    p = m.prep
    return dict(prep=dict(med=_l(p.med), lo=_l(p.lo), hi=_l(p.hi), mu=_l(p.mu), sd=_l(p.sd)), **{k: _l(v) for k, v in m.W.items()})


def predict_ens(students, taus, X):
    outs = [s.predict(X) for s in students]
    sc = np.mean([o[0] for o in outs], 0); sp = np.mean([o[1] for o in outs], 0); cf = np.mean([o[2] for o in outs], 0)
    return sc, sp, cf, cf >= float(np.mean(taus))


def fixture_windows():
    """The 2 reference windows (t = 60 s and 300 s). For remote runs, DISTILL_FIXTURE points to an .npz (t, X) written locally by `--fixture-npz`."""
    import os
    if os.environ.get('DISTILL_FIXTURE'):
        z = np.load(os.environ['DISTILL_FIXTURE'])
        return z['t'], z['X']
    rgb = pd.read_csv(FIXTURE)[['R', 'G', 'B']].values.astype(float)
    t, X = F.rec_features(rgb)
    pick = [int(np.nonzero(t >= 60)[0][0]), int(np.nonzero(t >= 300)[0][0])]
    return t[pick], X[pick]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--fixture-npz', default=None, help='only write the 2 fixture windows (t, X) to this .npz and exit')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    if a.fixture_npz:
        tw, Xw = fixture_windows(); np.savez(a.fixture_npz, t=tw, X=Xw); print('wrote', a.fixture_npz); return
    students, taus, rec = [], [], []
    for s in range(3):
        lam, wd, ep, tau = recipe(s)
        students.append(fit_all(s, lam, wd, ep)); taus.append(tau); rec.append(dict(seed=s, lam_kd=lam, weight_decay=wd, epochs=ep, tau=tau))
    tw, Xw = fixture_windows()
    sc, sp, cf, an = predict_ens(students, taus, Xw)
    js = dict(format=FORMAT, cols=F.COLS, hidden=D.HID, recipe=rec, tau=float(np.mean(taus)), students=[student_json(m) for m in students],
              reference=dict(fixture=str(FIXTURE.relative_to(F.REPO)), t=_l(tw), X=_l(Xw), score=_l(sc), spo2_hat=_l(sp), conf=_l(cf), answer=[bool(v) for v in an]))
    Path(a.out).write_text(json.dumps(js))
    # round trip: rebuild from JSON with plain numpy and compare
    back = json.loads(Path(a.out).read_text())
    Z = np.asarray(back['reference']['X'])
    outs = []
    for st in back['students']:
        p = {k: np.asarray(v) for k, v in st['prep'].items()}
        z = (np.clip(np.where(np.isfinite(Z), Z, p['med']), p['lo'], p['hi']) - p['mu']) / p['sd']
        h = np.maximum(z @ np.asarray(st['w1']).T + st['b1'], 0); h = np.maximum(h @ np.asarray(st['w2']).T + st['b2'], 0)
        o = h @ np.asarray(st['wo']).T + st['bo']
        outs.append(1 / (1 + np.exp(-o[:, 0])))
    err = float(np.max(np.abs(np.mean(outs, 0) - sc)))
    print(f'wrote {a.out}: recipe {rec}, reference scores {np.round(sc, 4).tolist()}, round-trip max abs diff {err:.2e}')
    assert err < 1e-5


if __name__ == '__main__':
    from oxydrop_ml.photon import distill_export as _m
    _m.main()
