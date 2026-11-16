"""Export OxySSL as ONE OxyNet in the app/js/cnn.js JSON format (export_js.export_graph) plus a Python reference on 2 fixture windows.

    cd python && python -m oxydrop_ml.photon.oxyssl_export [--seed 0]

Deployment fit (not evaluated; the evaluation is the LOSO score files of oxyssl.py): the trainable top of OxyNet is fit on all 6 Hoffman people
(clean + the K_AUG distant-flash copies) with the modal learning rate / epoch count of the 6 nested outer-fold choices of that seed, and the abstain
threshold is the median of the 6 nested thresholds. The frozen encoder is the pretrained stage-B encoder. The per-feature side scale is folded
into the SpO2 and hypoxia heads (exact), so the JSON consumes the raw features.windows_to_tensor output.

Outputs: results/photon/oxyssl_oxynet.json (cnn.js graph + 'oxyssl' block: decision threshold 0.5 on sigmoid(lowox_logit), abstain when
sigmoid(sqi_logit) < threshold), results/photon/oxyssl_ref.json (2 windows of tests/fixtures/ppg-csv/Left/100001.csv: x, side, emb_base, outputs).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .. import export_js as EJ
from .. import features as FE
from . import oxyssl as O

FIXTURE = O.REPO / 'tests' / 'fixtures' / 'ppg-csv' / 'Left' / '100001.csv'
OUT = O.RES / 'oxyssl_oxynet.json'
REF = O.RES / 'oxyssl_ref.json'
REF_IDX = (60, 200)           # window indices (after the calibration minute) of the reference


def deploy_fit(seed=0):
    sel = json.load(open(O.RES / 'oxyssl_raw' / f'{O.NAME}_s{seed}_selection.json'))
    (lr, ep), _ = Counter((s['lr'], s['epochs']) for s in sel).most_common(1)[0]
    thr = float(np.median([s['thr'] for s in sel]))
    net, _ = O.load_init()
    bank = O.load_bank()
    meta = O.shard()[0]
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    rows = np.nonzero(lab)[0]
    s_scale = O.side_scale(bank, rows)
    D = O.Data(bank, meta, ['clean'] + [f'aug{k}' for k in range(O.K_AUG)], rows, O.labels_of(meta), s_scale)
    top = O.train(net, D, lr, (ep,), seed)
    return net, top, s_scale, dict(lr=lr, epochs=ep, thr=thr, seed=seed, choices=[(s['lr'], s['epochs']) for s in sel])


def assemble(net, top, s_scale):
    """Full OxyNet with the fine-tuned top and the side scale folded in."""
    import copy
    full = copy.deepcopy(net).eval()
    E = full.cfg.emb
    with torch.no_grad():
        for name in ('fc_emb', 'hr1', 'hr2', 'spo2', 'lowox', 'sqi'):
            getattr(full, name).load_state_dict(getattr(top, name).state_dict())
        for lin in (full.spo2, full.lowox):
            lin.weight[:, E:] /= torch.tensor(s_scale, dtype=lin.weight.dtype)[None]
    return full


def reference(full, top, s_scale):
    rgb = pd.read_csv(FIXTURE)[['R', 'G', 'B']].values.astype(float)
    rw = EJ.recorded_windows(rgb)
    x, side, cal = rw['x'], rw['side'], rw['calibrating']
    with torch.no_grad():
        eb = full(torch.from_numpy(x[np.nonzero(cal)[0][-O.BASE_ROWS:]]))['emb'].mean(0)
        post = np.nonzero(~cal)[0][list(REF_IDX)]
        xs = torch.from_numpy(x[post]); ss = torch.from_numpy(side[post])
        o = full(xs, None, ss, eb[None].expand(len(post), -1))
        top.eval()
        p = full.encode(xs)
        o2 = top(p, ss / torch.from_numpy(s_scale), eb[None].expand(len(post), -1))
    for k in ('lowox_logit', 'sqi_logit', 'spo2'):
        assert torch.allclose(o[k], o2[k], atol=1e-4), k
    return dict(fixture=str(FIXTURE.relative_to(O.REPO)), window_te=[int(rw['te'][i]) for i in post], emb_base_from='mean emb of the last 8 calibration windows',
                x=[EJ.b64(x[i]) for i in post], side=side[post].tolist(), emb_base=eb.tolist(),
                out={k: o[k].numpy().tolist() for k in ('lowox_logit', 'sqi_logit', 'spo2', 'spo2_logvar')})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    O.set_threads(2)
    net, top, s_scale, info = deploy_fit(a.seed)
    full = assemble(net, top, s_scale)
    g = EJ.export_graph(full)
    g['oxyssl'] = dict(info, decision='hypoxic if sigmoid(lowox_logit) > 0.5', abstain='answer only if sigmoid(sqi_logit) >= thr', domain=0,
                       emb_base='mean emb of the baseline (first-minute) windows')
    json.dump(g, open(OUT, 'w'))
    json.dump(reference(full, top, s_scale), open(REF, 'w'), indent=1)
    print('wrote', OUT, OUT.stat().st_size, 'bytes;', REF, info)


if __name__ == '__main__':
    main()
