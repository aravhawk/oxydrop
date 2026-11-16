"""Export SSL3 as ONE OxyNet in the app/js/cnn.js JSON format (export_js.export_graph) plus a Python reference on 2 fixture windows.

    python -m oxydrop_ml.photon.ssl3_export [--seed 0]

Deployment fit: all 6 Hoffman people (clean + the K_AUG distant-flash copies), the modal nested epoch count of seed 0's outer folds, abstain threshold =
median of the nested fold thresholds. The side-feature scale is folded into the spo2 / lowox weights, so the app feeds raw side features.
Writes results/photon/ssl3_oxynet.json (oxynet-v1 graph + an `ssl3` block with the decision and abstain rule) and results/photon/ssl3_ref.json
(2 windows of tests/fixtures/ppg-csv/Left/100001.csv: x, side, emb_base, torch outputs of the exported graph's network).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .. import export_js as EJ
from . import ssl3 as S3

FIXTURE = S3.REPO / 'tests' / 'fixtures' / 'ppg-csv' / 'Left' / '100001.csv'
OUT = S3.RES / 'ssl3_oxynet.json'
REF = S3.RES / 'ssl3_ref.json'
REF_IDX = (0, 5)


def deploy_fit(seed=0):
    sel = json.load(open(S3.RAW / f'{S3.NAME}_s{seed}_selection.json'))
    ep, _ = Counter(s['epochs'] for s in sel).most_common(1)[0]
    thr = float(np.median([max(s['thr'], 0.0) for s in sel]))
    pre = S3.pre_state()
    bank = S3.Bank()
    meta = S3.shard()[0]
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    rows = np.nonzero(lab)[0]
    sc = S3.side_scale(bank, rows)
    D = S3.Data(bank, meta, ['clean'] + [f'aug{k}' for k in range(S3.K_AUG)], rows, S3.labels_of(meta), sc)
    net = S3.train(pre, D, (ep,), seed).cpu().eval()
    return net, sc, dict(epochs=int(ep), thr=thr, seed=seed, choices=[s['epochs'] for s in sel], lr_head=S3.LR_HEAD, lr_enc=S3.LR_ENC)


def fold_side(net, sc):
    import copy
    full = copy.deepcopy(net).eval()
    E = full.cfg.emb
    with torch.no_grad():
        for lin in (full.spo2, full.lowox):
            lin.weight[:, E:] /= torch.tensor(sc, dtype=lin.weight.dtype)[None]
    return full


def reference(full):
    rgb = pd.read_csv(FIXTURE)[['R', 'G', 'B']].values.astype(float)
    rw = EJ.recorded_windows(rgb)
    x, side, cal = rw['x'].astype(np.float32), rw['side'].astype(np.float32), rw['calibrating']
    with torch.no_grad():
        eb = full(torch.from_numpy(x[np.nonzero(cal)[0][-S3.BASE_ROWS:]]))['emb'].mean(0)
        post = np.nonzero(~cal)[0][list(REF_IDX)]
        o = full(torch.from_numpy(x[post]), None, torch.from_numpy(side[post]), eb[None].expand(len(post), -1))
    return dict(fixture=str(FIXTURE.relative_to(S3.REPO)), window_te=[int(rw['te'][i]) for i in post],
                emb_base_from='mean emb of the last 8 calibration windows', x=[EJ.b64(x[i]) for i in post], side=side[post].tolist(),
                emb_base=eb.tolist(), out={k: o[k].numpy().tolist() for k in ('lowox_logit', 'sqi_logit', 'spo2', 'spo2_logvar')})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    net, sc, info = deploy_fit(a.seed)
    full = fold_side(net, sc)
    g = EJ.export_graph(full)
    g['ssl3'] = dict(info, decision='hypoxic if sigmoid(lowox_logit) > 0.5', abstain='answer only if sigmoid(sqi_logit) >= thr', domain=0,
                     emb_base='mean emb of the baseline (first-minute) windows', pretrain=torch.load(S3.PRE_CK, weights_only=False)['ssl3'])
    json.dump(g, open(OUT, 'w'))
    json.dump(reference(full), open(REF, 'w'), indent=1)
    torch.save(dict(cfg=full.cfg.to_dict(), state=full.state_dict(), info=info), S3.WORK / 'ssl3_deploy.pt')
    print('wrote', OUT, OUT.stat().st_size, 'bytes;', REF, info, flush=True)


if __name__ == '__main__':
    main()
