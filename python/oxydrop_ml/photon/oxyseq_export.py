"""Export OxySeq to a browser-ready JSON weight file plus a Python reference output on the 2 clean fixture recordings.

Deployment recipe (no new selection, nothing tuned on test data): the network is retrained on ALL 6 Hoffman people (12 recordings) with the
configuration fixed in oxyseq.py, the number of training steps = the most frequent nested choice of the 6 outer folds of the oxyseq run, and the
3 seeds of the final configuration (the app averages the 3 probabilities). Decision threshold, abstain tau and 80 % band half-width = the medians
of the 6 nested per-fold values (results/photon/oxyseq_log.md). Plain ops only: Linear, ReLU, sigmoid, tanh, one GRU layer (PyTorch gate order
r, z, n), a gated EMA.

Reference: tests/fixtures/ppg-csv/Left/100001.csv (100001L) and Right/100004.csv (100004R) -> causal per-window features (oxytrack.rec_features)
-> outputs for every window; two windows per recording are singled out (`ref_windows`). A NumPy re-implementation of the forward pass from the JSON is
checked against PyTorch (max abs diff written to the file).
Run: cd python && python -m oxydrop_ml.photon.oxyseq_export
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from . import oxyseq as O
from . import oxytrack as OT

OUT = OT.RES / 'oxyseq_model.json'
FIX = OT.REPO / 'tests' / 'fixtures'
CLEAN = [('ppg-csv/Left/100001.csv', '100001L'), ('ppg-csv/Right/100004.csv', '100004R')]


def fold_choices():
    txt = (OT.RES / 'oxyseq_log.md').read_text().split('### oxyseq\n')[1].split('###')[0]
    rows = [l.split('|')[1:-1] for l in txt.splitlines() if re.match(r'^\| 1000', l)]
    return [dict(steps=int(r[1]), thr=float(r[3]), tau=float(r[4]), half=float(r[6])) for r in rows]


def _np(t):
    return np.asarray(t.detach().cpu().numpy(), float).round(8).tolist()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_np(W, x, base):
    """Reference forward of one seed from the JSON dict W. x (n, 22) scaled inputs, base (n,) first-minute indicator."""
    A = {k: np.asarray(v, float) for k, v in W.items()}
    lin = x @ A['lin_w'][0] + A['lin_b'][0]
    xd, xs = x[:, :O.ND].copy(), x[:, O.ND:]
    b = np.zeros(O.ND)
    hh = np.zeros(A['inp_b'].shape[0])
    Hn = len(hh)
    out = np.zeros((len(x), 3))
    for k in range(len(x)):
        g = 1.0 if (base[k] > 0 or sigmoid(lin[k]) < O.GATE_P) else 0.0
        xt_d = xd[k] - b
        a = A['rate'] * g
        b = (1 - a) * b + a * xd[k]
        xt = np.concatenate([xt_d, xs[k]])
        z = np.maximum(A['inp_w'] @ xt + A['inp_b'], 0)
        gi = A['w_ih'] @ z + A['b_ih']; gh = A['w_hh'] @ hh + A['b_hh']
        r = sigmoid(gi[:Hn] + gh[:Hn]); u = sigmoid(gi[Hn:2 * Hn] + gh[Hn:2 * Hn]); n = np.tanh(gi[2 * Hn:] + r * gh[2 * Hn:])
        hh = (1 - u) * n + u * hh
        out[k] = A['head_w'] @ np.concatenate([hh, xt]) + A['head_b']
    return dict(score=sigmoid(out[:, 0]), dhat=out[:, 1] * 5.0, q=sigmoid(out[:, 2]))


def main():
    ch = fold_choices()
    steps = Counter(c['steps'] for c in ch).most_common(1)[0][0]
    thr, tau, half = (float(np.median([c[k] for c in ch])) for k in ('thr', 'tau', 'half'))
    recs = OT.rec_table(O.feats(0), OT.window_table())
    tr = list(recs.values())
    sc = O.scaler(tr)
    data = [O.tensors(sc, r) for r in tr]
    seeds = []
    states = []
    for sd in O.SEEDS:
        st = O.train(data, seed=sd, ckpts=(steps,))[steps]
        states.append(st)
        seeds.append(dict(lin_w=_np(st['lin.weight']), lin_b=_np(st['lin.bias']), rate=_np(0.5 * torch.sigmoid(st['rho'])),
                          inp_w=_np(st['inp.weight']), inp_b=_np(st['inp.bias']), w_ih=_np(st['gru.weight_ih_l0']), w_hh=_np(st['gru.weight_hh_l0']),
                          b_ih=_np(st['gru.bias_ih_l0']), b_hh=_np(st['gru.bias_hh_l0']), head_w=_np(st['head.weight']), head_b=_np(st['head.bias'])))
    ref, maxdiff = {}, 0.0
    for path, name in CLEAN:
        rgb = pd.read_csv(FIX / path)[['R', 'G', 'B']].to_numpy(float)
        f = OT.rec_features(rgb)
        r = dict(t=f['t'], D=f['D'], snr=f['snr'], y=np.full(len(f['t']), np.nan), sbase=np.nan)
        td = O.tensors(sc, r)
        pt = [O.predict(st, td) for st in states]
        pn = [forward_np(w, td['x'], td['base']) for w in seeds]
        for a, b in zip(pt, pn):
            maxdiff = max(maxdiff, *(float(np.max(np.abs(a[k] - b[k]))) for k in ('score', 'q', 'dhat')))
        s = np.mean([p['score'] for p in pn], 0); q = np.mean([p['q'] for p in pn], 0); dh = np.mean([p['dhat'] for p in pn], 0)
        ev = np.nonzero(f['t'] >= OT.CAL)[0]
        pick = [int(ev[len(ev) // 2]), int(ev[-1])]
        ref[name] = dict(source=path, t=f['t'].tolist(), score=s.round(6).tolist(), q=q.round(6).tolist(), spo2_hat=(sc['mu_b'] + dh).round(4).tolist(),
                         ref_windows=[dict(index=i, t=int(f['t'][i]), x=td['x'][i].round(6).tolist(), score=round(float(s[i]), 6), q=round(float(q[i]), 6),
                                           pred=int(s[i] > thr), answer=bool(q[i] >= tau), spo2_hat=round(float(sc['mu_b'] + dh[i]), 4)) for i in pick])
    doc = dict(model='oxyseq', version=1, arch=dict(nd=O.ND, ns=O.NS, hidden=O.H, gate_p=O.GATE_P, win_s=OT.WIN, step_s=OT.STEP, cal_s=OT.CAL,
                                                     gru_gate_order='r,z,n (PyTorch)', inputs='19 chain v1 FUSION deltas / sd (clipped +-8), (log10 SNR R,G,B - lmu) / lsd'),
               scaler=dict(sd=np.round(sc['sd'], 8).tolist(), lmu=np.round(sc['lmu'], 8).tolist(), lsd=np.round(sc['lsd'], 8).tolist(), mu_b=sc['mu_b']),
               decision=dict(thr=thr, tau=tau, band_half=half, ensemble='mean of the seed probabilities'), train=dict(steps=steps, seeds=list(O.SEEDS),
               people='all 6 Hoffman people (deployment fit)', fold_choices=ch), seeds=seeds, reference=ref, numpy_vs_torch_maxdiff=maxdiff)
    OUT.write_text(json.dumps(doc))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB); steps {steps}, thr {thr:.3f}, tau {tau:.3f}; numpy vs torch max diff {maxdiff:.2e}')


if __name__ == '__main__':
    main()
