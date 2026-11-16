"""Export of photonnet_xl: 3 seeds refit on all 6 Hoffman people with the modal nested recipe, as nn-runtime-v1 graphs (the PhotonTwin decoder is
training-only and is NOT exported), plus a Python reference on 2 fixture windows (t = 60 s and 200 s of tests/fixtures/ppg-csv/Left/100001.csv; their
39 features are results/photon/photonnet_ref_features.json, flashgap.rec_features = the features app/js/distill_features.js computes).

    PHOTONNET_BUNDLE=... python -m oxydrop_ml.photon.photonnet_xl_export     (remote; needs results/photon/photonnet_xl_raw/choices_main_s*.csv)

Per seed: x_std = (clip(nan_to_num(f, 0), lo, hi) - mu) / sd ; graph outputs zt = tanh(z_pre) (d_sat channel), r, s, y_pre, a_pre; then
  mu = 100 sat_b + beta * 100 * 2 P_SD[0] * zt + r ; sigma = softplus(s) + 0.5 ; score = sigmoid(y_pre + 1.6 (90 - mu) / sigma) ; abst = sigmoid(a_pre).
Ensemble: mean score, mean mu, mean abst; answer = mean abst >= c* (median of the 18 nested c*). The LOSO numbers are the ones to quote, never this refit's.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import photonnet as PN
from . import photonnet_xl as XL
from . import twin_inverse as TI

OUT_W = XL.RES / 'photonnet_xl_weights.json'
OUT_REF = XL.RES / 'photonnet_xl_ref.json'


def _t(a):
    a = np.asarray(a, np.float64)
    return dict(shape=list(a.shape), data=a.ravel().tolist())


def graph(net):
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    T = dict(w1=sd['trunk.0.weight'], b1=sd['trunk.0.bias'], w2=sd['trunk.2.weight'], b2=sd['trunk.2.bias'],
             wz=sd['z_head.weight'][:1], bz=sd['z_head.bias'][:1], we=sd['e_head.weight'], be=sd['e_head.bias'],
             wr=sd['r_head.weight'], br=sd['r_head.bias'], ws=sd['s_head.weight'], bs=sd['s_head.bias'],
             wy=sd['y_head.weight'], by=sd['y_head.bias'], wa=sd['a_head.weight'], ba=sd['a_head.bias'])
    L = lambda i, o, w, b: {'op': 'linear', 'in': i, 'out': o, 'w': w, 'b': b}
    A = lambda i, o, fn: {'op': 'act', 'in': i, 'out': o, 'fn': fn}
    ops = [L('x', 'h1', 'w1', 'b1'), A('h1', 'a1', 'tanh'), L('a1', 'h2', 'w2', 'b2'), A('h2', 'h', 'tanh'),
           L('h', 'zp', 'wz', 'bz'), A('zp', 'zt', 'tanh'), L('h', 'ep', 'we', 'be'), A('ep', 'e', 'tanh'), L('e', 'r', 'wr', 'br'),
           L('h', 's', 'ws', 'bs'), {'op': 'concat', 'in': ['h', 'e'], 'out': 'he'}, L('he', 'y', 'wy', 'by'), L('he', 'a', 'wa', 'ba')]
    return {'format': 'nn-runtime-v1', 'inputs': {'x': {'shape': [int(T['w1'].shape[1])]}}, 'tensors': {k: _t(v) for k, v in T.items()},
            'ops': ops, 'outputs': ['zt', 'r', 's', 'y', 'a']}


def np_forward(m, X):
    """Plain-numpy reference of the exported graph + head arithmetic for one seed."""
    p = {k: np.asarray(v) for k, v in m['prep'].items()}
    T = {k: np.asarray(v['data']).reshape(v['shape']) for k, v in m['graph']['tensors'].items()}
    x = (np.clip(np.nan_to_num(X, nan=0.0), p['lo'], p['hi']) - p['mu']) / p['sd']
    h = np.tanh(np.tanh(x @ T['w1'].T + T['b1']) @ T['w2'].T + T['b2'])
    zt = np.tanh(h @ T['wz'].T + T['bz'])[:, 0]
    e = np.tanh(h @ T['we'].T + T['be'])
    r = (e @ T['wr'].T + T['br'])[:, 0]
    s = (h @ T['ws'].T + T['bs'])[:, 0]
    he = np.concatenate([h, e], 1)
    y = (he @ T['wy'].T + T['by'])[:, 0]; a = (he @ T['wa'].T + T['ba'])[:, 0]
    mu = 100 * m['sat_b'] + m['beta'] * 100 * 2 * m['psd0'] * zt + r
    sg = np.logaddexp(0, s) + 0.5
    sig = lambda v: 1 / (1 + np.exp(-v))
    return mu, sg, sig(y + 1.6 * (90 - mu) / sg), sig(a)


def main():
    torch.set_num_threads(2)
    ch = pd.concat([pd.read_csv(f) for f in sorted(XL.RAW.glob('choices_main_s*.csv'))], ignore_index=True)
    D = PN.load_data()
    mt = D['mt']
    sids = sorted(mt.sid.unique())
    ylab = mt.y_spo2.values.astype(float)
    satb = mt.sat_b.values.astype(float)
    satb_all = float(np.nanmean(mt.groupby('rec').sat_b.first().values))
    ref = json.loads((XL.RES / 'photonnet_ref_features.json').read_text())
    Xr = np.asarray([[np.nan if v is None else v for v in row] for row in ref['features']], np.float64)
    models = []
    for sd in XL.SEEDS:
        c = ch[ch.seed == sd]
        key = Counter(zip(c.w_rec, c.w_phys, c.steps)).most_common(1)[0][0]
        ci = [i for i, g in enumerate(XL.GRID) if (g['w_rec'], g['w_phys']) == key[:2]][0]
        prep, net, st = XL.fit(D, sids, [], sd, XL.GRID[ci], ylab, satb, None, steps=int(key[2]), snaps=(int(key[2]),))
        net.load_state_dict(st[-1][1])
        m = dict(seed=sd, recipe=dict(w_rec=key[0], w_phys=key[1], steps=int(key[2])), prep=dict(lo=prep.lo.tolist(), hi=prep.hi.tolist(),
                 mu=prep.mu.tolist(), sd=prep.sd.tolist()), graph=graph(net), sat_b=satb_all, beta=float(net.beta), psd0=float(TI.P_SD[0]),
                 n_infer_params=XL.n_infer_params(net))
        with torch.no_grad():                                 # torch vs numpy parity of the exported pieces
            o = net(prep.x(Xr), PN._f64(np.full(len(Xr), satb_all)))
        mu, sg, sc, ab = np_forward(m, Xr)
        assert np.allclose(mu, o['mu'].numpy(), atol=1e-8) and np.allclose(sc, torch.sigmoid(o['logit']).numpy(), atol=1e-10)
        models.append(m)
        print(f'seed {sd}: recipe {m["recipe"]}, beta {m["beta"]:.3f}, {m["n_infer_params"]} inference parameters', flush=True)
    c_star = float(np.median(ch.c_star.values))
    outs = [np_forward(m, Xr) for m in models]
    mu = np.mean([o[0] for o in outs], 0); sc = np.mean([o[2] for o in outs], 0); ab = np.mean([o[3] for o in outs], 0)
    OUT_W.write_text(json.dumps(dict(format='oxydrop-photonnet-xl-v1', cols=PN.FG.COLS, c_star=c_star, models=models,
                                     note='nn-runtime-v1 graph per seed; PhotonTwin decoder was training-only; head arithmetic in photonnet_xl_export.py')))
    OUT_REF.write_text(json.dumps(dict(fixture=ref['fixture'], t=ref['t'], X=ref['features'], mu=mu.tolist(), score=sc.tolist(), conf=ab.tolist(),
                                       answer=[bool(v) for v in ab >= c_star], c_star=c_star)))
    print('wrote', OUT_W, OUT_REF, 'mu', np.round(mu, 2), 'score', np.round(sc, 3), flush=True)


if __name__ == '__main__':
    main()
