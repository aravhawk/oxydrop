"""Can app/js/cnn.js run an OxyNet of ANOTHER architecture? Exports random-init OxyNets with non-default configs through export_js.export_graph and
compares the JS forward pass with PyTorch on random inputs (every head). Writes nothing in the repository (temporary files under $TMPDIR).

    cd python && python -m oxydrop_ml.photon.cnn_arch_check

Result and the op coverage are recorded in docs/photon_wiring_notes.md (section 10).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .. import model as M
from ..export_js import export_graph, randomize_bn

REPO = Path(__file__).resolve().parents[3]
CONFIGS = {
    'joint_stem_2stages_k5': dict(stem='joint', widths=(16, 32), depths=(1, 2), k=5, dil_cycle=3, emb=32, hr_hidden=16),
    'shared_stem_causal_3stages': dict(slot_width=4, stem_width=12, widths=(12, 20, 28), depths=(1, 1, 1), k=3, causal=True, emb=24, hr_hidden=24),
    'teacher_size': dict(slot_width=16, stem_width=64, widths=(64, 128, 192, 256), depths=(2, 3, 3, 2), emb=128, hr_hidden=128),
}
HEADS = ['hr_logits', 'spo2', 'spo2_logvar', 'lowox_logit', 'sqi_logit', 'emb']
NODE = r"""
import fs from 'node:fs';
import { loadModel, forward } from '%s';
const [graph, cases] = [JSON.parse(fs.readFileSync(process.argv[2])), JSON.parse(fs.readFileSync(process.argv[3]))];
const m = loadModel(graph), out = [];
for (const c of cases) { const o = forward(m, c.x, { domain: c.domain, side: c.side, embBase: c.emb_base });
  out.push({ hr_logits: Array.from(o.hr_logits), spo2: o.spo2, spo2_logvar: o.spo2_logvar, lowox_logit: o.lowox_logit, sqi_logit: o.sqi_logit, emb: Array.from(o.emb) }); }
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def check(name, kw, n_cases=4, seed=0):
    torch.manual_seed(seed)
    net = M.OxyNet(M.OxyCfg(**kw))
    randomize_bn(net, seed)
    net.eval()
    g = export_graph(net)
    gen = torch.Generator().manual_seed(seed + 7)
    cfg = net.cfg
    x = torch.randn(n_cases, cfg.in_ch, M.T_WIN, generator=gen)
    dom = torch.randint(0, cfg.n_domains, (n_cases,), generator=gen)
    side, eb = torch.randn(n_cases, cfg.side_dim, generator=gen), torch.randn(n_cases, cfg.emb, generator=gen)
    with torch.no_grad():
        o = net(x, dom, side, eb)
    cases = [dict(x=x[i].reshape(-1).tolist(), domain=int(dom[i]), side=side[i].tolist(), emb_base=eb[i].tolist()) for i in range(n_cases)]
    tmp = Path(tempfile.mkdtemp(prefix='cnnarch_', dir=os.environ.get('TMPDIR')))
    (tmp / 'g.json').write_text(json.dumps(g)); (tmp / 'c.json').write_text(json.dumps(cases))
    (tmp / 'run.mjs').write_text(NODE % (REPO / 'app' / 'js' / 'cnn.js').as_uri())
    subprocess.run(['node', str(tmp / 'run.mjs'), str(tmp / 'g.json'), str(tmp / 'c.json'), str(tmp / 'o.json')], check=True, timeout=300)
    js = json.loads((tmp / 'o.json').read_text())
    worst = {}
    for h in HEADS:
        ref = o[h].numpy().astype(np.float64)
        got = np.array([j[h] for j in js], np.float64).reshape(ref.shape)
        worst[h] = float(np.max(np.abs(got - ref)))
    ops = sorted({op['op'] for op in g['ops']})
    return dict(name=name, params=g['params'], n_ops=len(g['ops']), ops=ops, worst=worst)


def main():
    torch.set_num_threads(3)
    for name, kw in CONFIGS.items():
        r = check(name, kw)
        print(f"{r['name']}: {r['params']:,} parameters, {r['n_ops']} ops ({', '.join(r['ops'])}); max abs diff JS vs torch: "
              + ', '.join(f'{h} {v:.2e}' for h, v in r['worst'].items()))


if __name__ == '__main__':
    main()
