"""Reference for the in-app streaming hybrid engine (app/js/hybrid_engine.js): tests/fixtures/hybrid_engine_ref.json.

    python python/oxydrop_ml/make_engine_ref.py --ckpt-dir RUNS/hyb_final --out tests/fixtures/hybrid_engine_ref.json

For the two Hoffman fixture recordings (tests/fixtures/ppg-csv) it computes, window by window, with PyTorch and numpy only (no JS):
  c      v1 chain logit with the chain weights embedded in app/data/hybrid_weights.json (python/oxydrop/chain.py, the reference of engine_v1.js)
  corr   lowox_logit of each R member (checkpoints R_s{0,1,2}.pt, the ones exported to app/data/hybrid_R_s*.json), baseline embedding = mean embedding
         of <= 8 evenly spaced baseline windows built WITH the recording's baseline (train.group_emb convention)
  hr     decoded CNN heart rate of each member (expectation within +-7 bins of the argmax)
  z      decision logit of the exported pipeline (meta-learner on r = c + mean corr)
Windows: te = 300 + 60 k samples (hop 2 s), first 60 s = calibration; rows are written for windows that start at s >= 60 s (the first decision).
The samples are fed with a constant frame interval (dt channel 0), like the app does when no frame is late.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

from oxydrop import chain as C            # noqa: E402
from oxydrop_ml import features as F      # noqa: E402
from oxydrop_ml import model as M         # noqa: E402

FS, WIN, HOP, L = 30, 300, 60, 450
RECS = [('Left/100001.csv', '100001L'), ('Right/100004.csv', '100004R')]


def buffers(lr, tes):
    out = np.empty((len(tes), L, 3)); dt = np.full((len(tes), L), 1.0 / FS)
    for i, te in enumerate(tes):
        ii = te - L + np.arange(L)
        below = ii < 0
        src = np.where(below, -ii, ii)
        v = lr[src]
        out[i] = np.where(below[:, None], 2 * lr[0][None] - v, v)
    return out, dt


def load_members(ckpt_dir, seeds=(0, 1, 2)):
    nets = []
    for s in seeds:
        ck = torch.load(Path(ckpt_dir) / f'R_s{s}.pt', map_location='cpu', weights_only=False)
        net = M.OxyNet(M.OxyCfg(**ck['cfg'])); net.load_state_dict(ck['state']); net.eval(); nets.append(net)
    return nets


def run_record(csv, nets, W, meta, conf):
    rgb = np.loadtxt(csv, delimiter=',', skiprows=1)[:, :3]
    n = len(rgb)
    lr = np.log(np.maximum(rgb, 1e-3) + 1.0)
    tes = np.arange(WIN, n + 1, HOP)
    # ---- chain logit c (v1 chain, embedded weights)
    t, X = C.deltas(rgb)
    ev = {int(s): i for i, s in enumerate(t)}
    p = 0.5 * (C._lr(W['hr'], X[:, :1]) + C._lr(W['fusion'], X))
    pc = np.clip(p, 1e-6, 1 - 1e-6); cl = np.log(pc / (1 - pc))
    # ---- CNN windows: baseline from the windows fully inside the first minute, then every window rebuilt with it
    lrb, dt = buffers(lr, tes)
    mask = np.ones((len(tes), 3), bool)
    _, _, ab = F.windows_to_tensor(lrb, dt, None, mask)
    in_base = tes <= 60 * FS
    base = F.baseline_from_abs(ab[in_base])
    x, s, _ = F.windows_to_tensor(lrb, dt, np.tile(base, (len(tes), 1)), mask)
    rows = np.nonzero(in_base)[0]
    sel = rows[np.unique(np.linspace(0, len(rows) - 1, 8).round().astype(int))]
    dom = torch.zeros(len(tes), dtype=torch.long)
    xt, st = torch.from_numpy(x), torch.from_numpy(s)
    corr, hrs, ebs = [], [], []
    with torch.no_grad():
        for net in nets:
            eb = net(xt[sel], dom[sel], st[sel], None)['emb'].mean(0)
            o = net(xt, dom, st, eb[None].expand(len(tes), -1))
            corr.append(o['lowox_logit'].numpy().astype(float))
            hrs.append(M.hr_from_logits(o['hr_logits'], 'expect', 7).numpy().astype(float))
            ebs.append(eb.numpy().astype(float))
    corr, hrs = np.array(corr), np.array(hrs)
    pm = meta
    out = []
    for i, te in enumerate(tes):
        s0 = te // FS - 10
        if s0 < 60 or s0 not in ev:
            continue
        c = float(cl[ev[s0]]); cr = float(corr[:, i].mean())
        z = pm['intercept'] + pm['coef'][0] * ((c + cr) - pm['mean'][0]) / pm['scale'][0]
        out.append(dict(te=int(te), s=int(s0), c=c, corr=[float(v) for v in corr[:, i]], hr=[float(v) for v in hrs[:, i]], z=float(z)))
    return dict(n_samples=int(n), base=[float(b) for b in base], emb_base=[[float(v) for v in e] for e in ebs], windows=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--out', default=str(ROOT / 'tests/fixtures/hybrid_engine_ref.json'))
    ap.add_argument('--max-windows', type=int, default=0, help='keep only the first N decision windows per recording (0 = all)')
    a = ap.parse_args(argv)
    hw = json.load(open(ROOT / 'app/data/hybrid_weights.json'))
    W = hw['chain']['engine_v1_weights']
    nets = load_members(a.ckpt_dir)
    res = {}
    for rel, name in RECS:
        r = run_record(ROOT / 'tests/fixtures/ppg-csv' / rel, nets, W, hw['pipeline']['meta'], None)
        if a.max_windows:
            r['windows'] = r['windows'][:a.max_windows]
        res[name] = r
        print(name, len(r['windows']), 'decision windows')
    doc = dict(format='oxydrop-hybrid-engine-ref-v1', note='generated by python/oxydrop_ml/make_engine_ref.py; see its docstring', pipeline_meta=hw['pipeline']['meta'], records=res)
    Path(a.out).write_text(json.dumps(doc, separators=(',', ':')))
    print('wrote', a.out, os.path.getsize(a.out), 'bytes')


if __name__ == '__main__':
    main()
