"""Export PhotonNet (round-2 candidate R2A) to a plain-ops JSON weight file plus a Python reference output on 2 fixture windows.

    cd python && python -m oxydrop_ml.photon.photonnet_export     # final fit on all 6 Hoffman people (3 seeds), writes the JSON + reference

RECIPE (nested-modal, as the shipped chain's C): the training length of each seed is the MODAL nested choice of that seed over the 6 outer folds
(results/photon/photonnet_choices.csv, model `photonnet`; ties broken toward the median), c* is the median of the 18 nested fold/seed choices. Everything
else (winsorisation bounds, scaler, sat_b of an unseen recording, network and physics parameters) is fitted here on all 6 people (clean + the declared
augmentation). The numbers quoted for PhotonNet are the leave-one-person-out ones (results/photon/TABLE.csv); this refit has no held-out number.

INFERENCE uses only plain ops; the PhotonTwin layer is a TRAINING-time reconstruction head and is not needed to predict:
  x = (clip(f, lo, hi) - mu) / sd                         f = the 39 window features (flashgap.COLS, flashgap.rec_features)
  h = tanh(W2 tanh(W1 x + b1) + b2)
  z = 2 psd tanh(Wz h + bz);  r = Wr tanh(We h + be) + br;  mu = 100 (sat_b + z_0) + r;  sigma = softplus(Ws h + bs) + 0.5
  per seed: score_s = Phi((90 - mu_s) / sigma_s); ensemble: mu = mean mu_s, sigma = sqrt(mean sigma_s^2 + var mu_s), score = mean score_s
  conf = 1 - min(1, sigma / |mu - 90|), answer = conf >= c_star, pred = score > 0.5
JS STATUS: the network above is ~40 lines of JS (Linear, tanh, softplus, erf). The 39-feature extractor (flashgap.rec_features: chain deltas, 15 quality and
5 respiratory features) has no JS port yet; the reference file stores the feature vectors of the 2 fixture windows so the network part can be checked
on its own and the feature port can be checked separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import flashgap as FG
from . import photonnet as PN

OUT = PN.RES / 'photonnet_weights.json'
REF = PN.RES / 'photonnet_ref.json'
FIXTURE = PN.REPO / 'tests' / 'fixtures' / 'ppg-csv' / 'Left' / '100001.csv'
REF_T = (60, 200)
FORMAT = 'oxydrop-photonnet-v1'


def _modal(v):
    v = np.asarray(v)
    vals, cnt = np.unique(v, return_counts=True)
    top = vals[cnt == cnt.max()]
    med = np.median(v)
    return top[np.argmin(np.abs(top - med))].item()


def _np(t):
    return t.detach().cpu().numpy().tolist()


def net_json(net):
    tr = net.trunk
    return dict(W1=_np(tr[0].weight), b1=_np(tr[0].bias), W2=_np(tr[2].weight), b2=_np(tr[2].bias), Wz=_np(net.z_head.weight), bz=_np(net.z_head.bias),
                We=_np(net.e_head.weight), be=_np(net.e_head.bias), Wr=_np(net.r_head.weight), br=_np(net.r_head.bias), Ws=_np(net.s_head.weight),
                bs=_np(net.s_head.bias))


def _erf_phi(x):
    from scipy.special import ndtr
    return ndtr(x)


def forward_json(M, F):
    """Plain-numpy forward of the exported model on raw feature rows F [n, 39] (each net has its own input scaler `prep`).
    Returns dict mu, sigma, score, conf, answer, pred (3-seed ensemble)."""
    F = np.nan_to_num(np.asarray(F, float), nan=0.0)
    psd = np.asarray(M['psd'])
    mus, sgs, scs = [], [], []
    for n in M['nets']:
        A = {k: np.asarray(v) for k, v in n.items() if k not in ('steps', 'prep')}
        P = {k: np.asarray(v) for k, v in n['prep'].items()}
        x = (np.clip(F, P['lo'], P['hi']) - P['mu']) / P['sd']
        h = np.tanh(np.tanh(x @ A['W1'].T + A['b1']) @ A['W2'].T + A['b2'])
        z = 2 * psd * np.tanh(h @ A['Wz'].T + A['bz'])
        r = (np.tanh(h @ A['We'].T + A['be']) @ A['Wr'].T + A['br'])[:, 0]
        mu = 100 * (M['sat_b'] + z[:, 0]) + r
        sg = np.logaddexp(0.0, (h @ A['Ws'].T + A['bs'])[:, 0]) + 0.5
        mus.append(mu); sgs.append(sg); scs.append(_erf_phi((90 - mu) / sg))
    mus, sgs = np.stack(mus), np.stack(sgs)
    mu = mus.mean(0); sg = np.sqrt((sgs ** 2).mean(0) + mus.var(0)); score = np.mean(scs, 0)
    conf = PN.conf_of(mu, sg)
    return dict(mu=mu, sigma=sg, score=score, conf=conf, answer=conf >= M['c_star'], pred=(score > 0.5).astype(int))


def build(log=print):
    torch.set_num_threads(2)
    D = PN.load_data(log)
    ch = pd.read_csv(PN.CHOICES)
    ch = ch[ch.model == 'photonnet']
    assert len(ch) == 18, f'expected 18 nested choices (6 folds x 3 seeds), found {len(ch)}'
    mt = D['mt']
    sids = sorted(mt.sid.unique())
    ylab = mt.y_spo2.values.astype(float)
    satb = mt.sat_b.values.astype(float)
    sat_b_new = float(np.nanmean(mt.groupby('rec').sat_b.first().values))
    nets = []
    for sd in PN.SEEDS:
        steps = int(_modal(ch[ch.seed == sd].steps.values))
        prep, net, st = PN.fit(D, sids, [], sd, True, ylab, satb, steps=steps, snaps=(steps,))
        net.load_state_dict(st[-1][1])
        nets.append((sd, steps, prep, net))
        log(f'seed {sd}: {steps} steps, {PN.n_params(net)} trainable parameters')
    M = dict(format=FORMAT, cols=FG.COLS, psd=PN.TI.P_SD.tolist(), sat_b=sat_b_new, c_star=float(np.median(ch.c_star.values)), nets=[],
             recipe=dict(steps={str(sd): s for sd, s, _, _ in nets}, c_star_from='median of the 18 nested fold/seed choices'))
    for sd, steps, prep, net in nets:
        j = net_json(net)
        j['steps'] = steps
        j['prep'] = dict(lo=prep.lo.tolist(), hi=prep.hi.tolist(), mu=prep.mu.tolist(), sd=prep.sd.tolist())
        M['nets'].append(j)
    return M, nets


def reference(M, nets):
    pre = PN.RES / 'photonnet_ref_features.json'
    if pre.exists():            # features of the 2 fixture windows computed by flashgap.rec_features (the remote run has no raw fixture / legacy package)
        r = json.loads(pre.read_text())
        assert tuple(r['t']) == REF_T
        F = np.asarray(r['features'], float)
    else:
        rgb = pd.read_csv(FIXTURE)[['R', 'G', 'B']].values.astype(float)
        t, X = FG.rec_features(rgb)
        F = X[[int(np.where(t == tt)[0][0]) for tt in REF_T]]
    o = forward_json(M, F)
    # torch path of the same nets
    mus, sgs = [], []
    for sd, steps, prep, net in nets:
        p = PN.predict(prep, net, net.state_dict(), F, np.full(len(F), M['sat_b']))
        mus.append(p['mu']); sgs.append(p['sigma'])
    mu_t = np.mean(mus, 0)
    assert np.allclose(mu_t, o['mu'], atol=1e-8), (mu_t, o['mu'])
    return dict(fixture=str(FIXTURE.relative_to(PN.REPO)), t=list(REF_T), features=F.tolist(),
                expected={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in o.items()})


def main():
    M, nets = build()
    ref = reference(M, nets)
    OUT.write_text(json.dumps(M))
    REF.write_text(json.dumps(ref, indent=1))
    print(f'wrote {OUT} ({OUT.stat().st_size // 1024} KB) and {REF}: mu {np.round(ref["expected"]["mu"], 2)}, score {np.round(ref["expected"]["score"], 3)}')


if __name__ == '__main__':
    main()
