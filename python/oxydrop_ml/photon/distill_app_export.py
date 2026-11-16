"""App export of the PhotonDistill student (FEAT-006): app/data/distill.json (nn-runtime-v1 graphs + feature normaliser) and the parity fixture
tests/fixtures/ref_distill.json (39 features and model outputs of every window of the fixture recordings, computed by the Python reference).

    cd python && python -m oxydrop_ml.photon.distill_app_export

Source of the weights: results/photon/distill_weights.json, written by distill_export.py: the 3 students refit on all 6 Hoffman people with the modal
nested recipe (teachers refit on all 6 as well). The numbers quoted for this model are the leave-one-person-out ones (results/photon/distill_log.md),
never the refit's. Nothing is retrained here; the weights are copied (8 decimals, as stored) into the nn_runtime JSON schema, and the Python reference
below uses exactly those copied weights, so the JS port is checked against the shipped file.

Deployed output (3-seed ensemble, as distill_student.ensemble): p = mean sigmoid(hyp), spo2 = mean (90 + 5 spo2_head), conf = mean sigmoid(abstain),
answer = conf >= tau (mean of the seed taus). App states: not answer -> Unsure, re-measure; p > 0.5 -> Possible low oxygen; else Normal vs your baseline.
Features: flashgap.rec_features (19 chain v1 deltas, 15 quality, 5 baseline-relative respiratory), computed on the 30 Hz RGB of the recording.
Fixtures: tests/fixtures/ppg-csv/Left/100001.csv, Right/100004.csv (clean) and tests/fixtures/photon_L4_100001L.csv: 100001L degraded by
stress.degrade(rgb, level=4, seed=0) (a SIMULATION of a distant flash / low light, see stress.py and results/photon/HARNESS.md), whole recording
degraded, then the first 480 s kept; values are multiples of 1/32 DN written exactly (repr). stress_csv() rewrites it if missing and otherwise
asserts that the file equals a fresh stress.degrade output bit for bit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import flashgap as F

REPO = F.REPO
SRC = REPO / 'results' / 'photon' / 'distill_weights.json'
OUT_APP = REPO / 'app' / 'data' / 'distill.json'
OUT_REF = REPO / 'tests' / 'fixtures' / 'ref_distill.json'
FIX = REPO / 'tests' / 'fixtures'
RECS = [('ppg-csv/Left/100001.csv', '100001L'), ('ppg-csv/Right/100004.csv', '100004R'), ('photon_L4_100001L.csv', '100001L_L4')]
FORMAT = 'oxydrop-photon-distill-app-v1'
ENGINE_ID = 'oxydrop-photon-distill-v1'
CAL = 60
STRESS_SRC, STRESS_LEVEL, STRESS_SEED, STRESS_SECONDS, STRESS_CSV = 'ppg-csv/Left/100001.csv', 4, 0, 480, 'photon_L4_100001L.csv'


def load_csv(p):
    return pd.read_csv(p)[['R', 'G', 'B']].values.astype(np.float64)


def stress_csv():
    from . import stress as ST
    d = ST.degrade(load_csv(FIX / STRESS_SRC), STRESS_LEVEL, STRESS_SEED)[:STRESS_SECONDS * F.FS]
    assert np.array_equal(d * 32.0, np.round(d * 32.0)), 'stress.degrade output is quantised to 1/32 DN'
    path = FIX / STRESS_CSV
    if not path.exists():
        path.write_text('\n'.join(['R,G,B'] + [f'{r!r},{g!r},{b!r}' for r, g, b in d.tolist()]) + '\n')
    assert np.array_equal(load_csv(path), d), f'{path} differs from stress.degrade(level={STRESS_LEVEL}, seed={STRESS_SEED})'
    return d


def graph(st, n_in):
    t = lambda a: dict(shape=list(np.shape(a)), data=np.asarray(a, np.float64).ravel().tolist())
    return {
        'format': 'nn-runtime-v1',
        'inputs': {'z': {'shape': [n_in]}},
        'tensors': {k: t(st[k]) for k in ('w1', 'b1', 'w2', 'b2', 'wo', 'bo')},
        'ops': [dict(op='linear', **{'in': 'z'}, out='h1', w='w1', b='b1'), dict(op='act', **{'in': 'h1'}, out='a1', fn='relu'),
                dict(op='linear', **{'in': 'a1'}, out='h2', w='w2', b='b2'), dict(op='act', **{'in': 'h2'}, out='a2', fn='relu'),
                dict(op='linear', **{'in': 'a2'}, out='o', w='wo', b='bo')],
        'outputs': ['o'],
    }


def app_json(src):
    n = len(src['cols'])
    return dict(format=FORMAT, id=ENGINE_ID, model='distill', cols=src['cols'], tau=src['tau'], recipe=src['recipe'],
                heads=dict(hyp='logit of P(SpO2 < 90)', spo2='(SpO2 - 90) / 5', abstain='logit of P(hyp head correct); answer = mean sigmoid >= tau'),
                source='results/photon/distill_weights.json (python/oxydrop_ml/photon/distill_export.py); app export python/oxydrop_ml/photon/distill_app_export.py',
                note='Trained on the Hoffman 2022 dataset (6 people, all used in this refit); research prototype, not a medical device.',
                students=[dict(prep=st['prep'], graph=graph(st, n)) for st in src['students']])


def forward(app, X):
    """Python reference of the app engine on a [n, 39] feature matrix (numpy float64, same weights as the app file)."""
    outs = []
    for st in app['students']:
        p = {k: np.asarray(v, np.float64) for k, v in st['prep'].items()}
        T = {k: np.asarray(v['data'], np.float64).reshape(v['shape']) for k, v in st['graph']['tensors'].items()}
        z = (np.clip(np.where(np.isfinite(X), X, p['med']), p['lo'], p['hi']) - p['mu']) / p['sd']
        h = np.maximum(z @ T['w1'].T + T['b1'], 0); h = np.maximum(h @ T['w2'].T + T['b2'], 0)
        o = h @ T['wo'].T + T['bo']
        sig = lambda v: 1 / (1 + np.exp(-np.clip(v, -30, 30)))
        outs.append((sig(o[:, 0]), 90 + 5 * o[:, 1], sig(o[:, 2])))
    sc = np.mean([o[0] for o in outs], 0); sp = np.mean([o[1] for o in outs], 0); cf = np.mean([o[2] for o in outs], 0)
    return sc, sp, cf, cf >= float(app['tau'])


def main():
    import warnings
    warnings.filterwarnings('ignore')
    src = json.loads(SRC.read_text())
    assert src['cols'] == F.COLS
    app = app_json(src)
    OUT_APP.write_text(json.dumps(app))
    app = json.loads(OUT_APP.read_text())
    # the trained export's own 2-window reference must be reproduced from the app file
    Xr = np.asarray(src['reference']['X'], np.float64)
    sc, sp, cf, an = forward(app, Xr)
    d = float(np.max(np.abs(sc - np.asarray(src['reference']['score']))))
    print(f'app file vs distill_weights.json reference: max |dp| = {d:.2e}')
    assert d < 1e-6 and list(an) == src['reference']['answer']
    stress_csv()
    recs = {}
    for rel, name in RECS:
        rgb = load_csv(FIX / rel)
        t, X = F.rec_features(rgb)
        sc, sp, cf, an = forward(app, X)
        ev = t >= CAL
        recs[name] = dict(csv=rel, n_samples=int(len(rgb)), t=t.astype(int).tolist(), X=[[None if not np.isfinite(v) else float(v) for v in row] for row in X],
                          p=sc.tolist(), spo2=sp.tolist(), conf=cf.tolist(), answer=[bool(v) for v in an], eval=[bool(v) for v in ev])
        print(f'{name}: {len(t)} windows, eval {int(ev.sum())}, answered {int(an[ev].sum())}, p>0.5 among answered {int((sc[ev & an] > 0.5).sum())}')
    OUT_REF.write_text(json.dumps(dict(engine=ENGINE_ID, model_file='app/data/distill.json', cols=F.COLS, tau=app['tau'], cal=CAL,
                                       generator='python/oxydrop_ml/photon/distill_app_export.py', recordings=recs)))
    print('wrote', OUT_APP, OUT_REF)


if __name__ == '__main__':
    main()
