"""Export OxyNet for the browser and for interoperability.

  * app/data/cnn_weights.json : architecture as an op list + BatchNorm-folded float32 tensors (base64). Read by app/js/cnn.js.
  * app/data/oxynet.onnx      : the same eval-mode graph as ONNX (verified against torch with onnxruntime, 1e-5).
  * tests/fixtures/cnn_ref.json : random and real-looking inputs with torch outputs for every head (tests/cnn_parity.mjs).

Usage (repo root):  python python/oxydrop_ml/export_js.py [--ckpt path.pt] [--seed 0]
Without --ckpt the weights are RANDOM (seeded) with randomised BatchNorm statistics so that BN folding is really exercised.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oxydrop_ml import model as M  # noqa: E402
try:   # FEAT-003 reference window builder; absent only in a bare ml-model-cnn checkout
    from oxydrop_ml import features as FE  # noqa: E402
except ImportError:  # pragma: no cover
    FE = None

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'oxynet-v1'


# ---------------------------------------------------------------- helpers
def randomize_bn(net: torch.nn.Module, seed: int = 0):
    """Make a random-init net a meaningful parity target: He-normal conv weights, BatchNorm running statistics measured on random
    inputs (so activations keep O(1) variance and every head responds to the input, instead of collapsing to its bias), then
    non-trivial gamma/beta and perturbed statistics so that BN folding is really exercised."""
    g = torch.Generator().manual_seed(seed + 1000)
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)) and m.weight.dim() >= 2:
            fan_in = m.weight[0].numel()
            m.weight.data = torch.randn(m.weight.shape, generator=g) * (2.0 / fan_in) ** 0.5
    for lin in (net.hr2, net.spo2, net.lowox, net.sqi, net.aux_spo2):   # keep head outputs O(1) so float32 round-off stays well below the parity limits
        lin.weight.data *= 0.5
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None   # cumulative average over the calibration batches
    cfg = net.cfg
    net.train()
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(32, cfg.in_ch, M.T_WIN, generator=g)
            net(x, torch.randint(0, cfg.n_domains, (32,), generator=g))
    net.eval()
    for m in bns:
        n = m.num_features
        m.momentum = 0.1
        m.weight.data = 0.7 + 0.6 * torch.rand(n, generator=g)
        m.bias.data = 0.2 * torch.randn(n, generator=g)
        m.running_mean = m.running_mean + 0.1 * torch.randn(n, generator=g) * m.running_var.sqrt()
        m.running_var = m.running_var * (0.8 + 0.4 * torch.rand(n, generator=g))
    net.fc_emb.weight.data *= 0.1   # pooled features are O(10); this puts the embedding at O(1) so cross-runtime float32 round-off stays ~1e-6
    with torch.no_grad():   # centre the pre-activations of the two ReLU layers after the pooled features so that about half of the units are active
        x = torch.randn(64, cfg.in_ch, M.T_WIN, generator=g)
        dom = torch.randint(0, cfg.n_domains, (64,), generator=g)
        p = net.encode(x, dom)
        for lin, inp in ((net.fc_emb, p), (net.hr1, None)):
            if inp is None:
                inp = F.relu(net.fc_emb(p))
            pre = lin(inp)
            lin.bias.data = lin.bias.data - pre.mean(0)


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode('ascii')


def fold_conv(c: M.Conv):
    """-> (w [co,ci,k], b [co]) float64 with BatchNorm folded in."""
    w = c.conv.weight.detach().double().numpy()
    b = np.zeros(w.shape[0]) if c.conv.bias is None else c.conv.bias.detach().double().numpy()
    if c.bn is not None:
        bn = c.bn
        s = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy() + bn.eps)
        w = w * s[:, None, None]
        b = (b - bn.running_mean.detach().double().numpy()) * s + bn.bias.detach().double().numpy()
    return w, b


class Graph:
    def __init__(self):
        self.tensors, self.ops = {}, []

    def t(self, name, a):
        a = np.asarray(a, dtype=np.float64)
        self.tensors[name] = {'shape': list(a.shape), 'b64': b64(a)}
        return name

    def conv(self, name, src, dst, c: M.Conv, relu):
        w, b = fold_conv(c)
        self.ops.append({'op': 'conv1d', 'in': src, 'out': dst, 'w': self.t(name + '.w', w), 'b': self.t(name + '.b', b),
                         'stride': c.stride, 'dilation': c.dilation, 'pad': list(c.pad), 'relu': bool(relu)})

    def dense(self, name, ins, dst, lin: torch.nn.Linear, relu=False):
        self.ops.append({'op': 'dense', 'in': ins, 'out': dst, 'w': self.t(name + '.w', lin.weight.detach().double().numpy()),
                         'b': self.t(name + '.b', lin.bias.detach().double().numpy()), 'relu': relu})


def export_graph(net: M.OxyNet, T=M.T_WIN) -> dict:
    cfg, net = net.cfg, net.eval()
    g = Graph()
    if cfg.stem == 'shared':
        g.ops.append({'op': 'shared_stem_input', 'in': 'x', 'domain': 'domain', 'out': 'stem_in', 'n_slots': cfg.n_slots,
                      'slot_width': cfg.slot_width, 'pad': list(net.slot_pad),
                      'w': g.t('slot.w', net.slot_conv.weight.detach().double().numpy()),
                      'b': g.t('slot.b', net.slot_conv.bias.detach().double().numpy()),
                      'mask': g.t('slot.mask', net.domain_slot_mask.double().numpy())})
        g.conv('stem', 'stem_in', 'h', net.stem, relu=False)
        g.ops.append({'op': 'add_domain', 'in': 'h', 'domain': 'domain', 'out': 'h', 'table': g.t('domain.emb', net.domain_emb.weight.detach().double().numpy()), 'relu': True})
    else:
        g.conv('stem', 'x', 'h', net.stem, relu=True)
    cur = 'h'
    for i, b in enumerate(net.blocks):
        g.conv(f'b{i}.c1', cur, f'b{i}.a', b.c1, relu=True)
        g.conv(f'b{i}.c2', f'b{i}.a', f'b{i}.b', b.c2, relu=False)
        skip = cur
        if b.proj is not None:
            g.conv(f'b{i}.proj', cur, f'b{i}.s', b.proj, relu=False)
            skip = f'b{i}.s'
        g.ops.append({'op': 'add', 'in': [f'b{i}.b', skip], 'out': f'b{i}', 'relu': True})
        cur = f'b{i}'
    g.ops.append({'op': 'pool', 'in': cur, 'out': 'pooled'})
    g.dense('fc_emb', ['pooled'], 'emb', net.fc_emb, relu=True)
    g.ops.append({'op': 'sub', 'in': ['emb', 'emb_base'], 'out': 'dz'})
    g.dense('hr1', ['emb'], 'hr_h', net.hr1, relu=True)
    g.dense('hr2', ['hr_h'], 'hr_out', net.hr2)
    g.dense('spo2', ['emb', 'side'], 'spo2_out', net.spo2)
    g.dense('lowox', ['dz', 'side'], 'lowox_out', net.lowox)
    g.dense('sqi', ['emb'], 'sqi_out', net.sqi)
    return {
        'format': FORMAT,
        'config': cfg.to_dict(),
        'input': {'channels': M.CH_NAMES, 'T': T, 'fs': M.FS, 'side': M.SIDE_NAMES, 'domains': M.DOMAINS, 'emb': cfg.emb},
        'hr': {'lo': M.HR_LO, 'hi': M.HR_HI, 'bins': M.HR_BINS},
        'outputs': {
            'hr_logits': {'buf': 'hr_out'},
            'spo2': {'buf': 'spo2_out', 'idx': 0, 'shift': M.SPO2_CENTER, 'scale': M.SPO2_SCALE},
            'spo2_logvar': {'buf': 'spo2_out', 'idx': 1},
            'lowox_logit': {'buf': 'lowox_out', 'idx': 0},
            'sqi_logit': {'buf': 'sqi_out', 'idx': 0},
            'emb': {'buf': 'emb'},
        },
        'params': M.count_params(net, include_aux=False),
        'tensors': g.tensors,
        'ops': g.ops,
    }


# ---------------------------------------------------------------- ONNX
def export_onnx(net: M.OxyNet, path: Path, T=M.T_WIN):
    w = M.ExportWrapper(net.eval()).eval()
    cfg = net.cfg
    ex = (torch.zeros(2, cfg.in_ch, T), torch.zeros(2, dtype=torch.long), torch.zeros(2, cfg.side_dim), torch.zeros(2, cfg.emb))
    names_in = ['x', 'domain', 'side', 'emb_base']
    dyn = {n: {0: 'batch'} for n in names_in + M.OUT_NAMES}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(w, ex, str(path), input_names=names_in, output_names=M.OUT_NAMES, dynamic_axes=dyn, opset_version=17, dynamo=False)


ONNX_ATOL, ONNX_RTOL = 1e-5, 1e-6   # float32 spacing at |y|~100 (spo2 = 93 + 5 raw) is 7.6e-6, so a purely absolute 1e-5 would be about 1 ulp there


def onnx_max_diff(net: M.OxyNet, path: Path, n=6, seed=0):
    """-> (max abs diff, max of |diff| - rtol*|ref|); the second must be < ONNX_ATOL."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    g = torch.Generator().manual_seed(seed)
    cfg, worst, excess = net.cfg, 0.0, 0.0
    for b in (1, n):
        x = torch.randn(b, cfg.in_ch, M.T_WIN, generator=g)
        dom = torch.randint(0, cfg.n_domains, (b,), generator=g)
        side, eb = torch.randn(b, cfg.side_dim, generator=g), torch.randn(b, cfg.emb, generator=g)
        with torch.no_grad():
            ref = net(x, dom, side, eb)
        outs = sess.run(M.OUT_NAMES, {'x': x.numpy(), 'domain': dom.numpy(), 'side': side.numpy(), 'emb_base': eb.numpy()})
        for k, o in zip(M.OUT_NAMES, outs):
            d = np.abs(o - ref[k].numpy())
            worst = max(worst, float(d.max())); excess = max(excess, float((d - ONNX_RTOL * np.abs(ref[k].numpy())).max()))
    return worst, excess


# ---------------------------------------------------------------- fixtures
def dt_pattern(n):
    """Deterministic irregular frame-time deltas (seconds), mirrored in tests/cnn_parity.mjs."""
    i = np.arange(n)
    return (1.0 / 30.0) * (1.0 + 0.5 * (i % 11 == 0) + 3.0 * (i % 401 == 0))


def run(net, x, domain=None, side=None, emb_base=None):
    with torch.no_grad():
        o = net(torch.as_tensor(x), domain, side, emb_base)
    return {k: o[k].numpy() for k in M.OUT_NAMES}


def case(net, name, x, domain, side, emb_base):
    x = np.asarray(x, dtype=np.float32)
    o = run(net, x[None], torch.tensor([domain]), torch.as_tensor(side, dtype=torch.float32)[None], torch.as_tensor(emb_base, dtype=torch.float32)[None])
    return {'name': name, 'domain': domain, 'x': b64(x), 'side': [float(v) for v in np.asarray(side, np.float32)],
            'emb_base': [float(v) for v in np.asarray(emb_base, np.float32)],
            'out': {k: (o[k][0].astype(float).tolist() if o[k].ndim > 1 else float(o[k][0])) for k in M.OUT_NAMES}}


def jnum(a):
    """float array -> list with NaN as None (valid JSON)."""
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def gather_buffer(lrgb, dt, te, L=450):
    """The CTX + WIN buffer ending at sample te, odd-reflected about the first sample before the recording start (windows.WindowSet.lrgb)."""
    ii = te - L + np.arange(L)
    below = ii < 0
    v = lrgb[np.where(below, -ii, ii)]
    out = np.where(below[:, None], 2 * lrgb[0][None] - v, v)
    return out, dt[np.where(below, 0, ii)]


def recorded_windows(rgb, dt=None, mask=None):
    """Causal per-window tensors of one recording exactly as the app builds them: windows end at te = 300, 360, ...; windows ending within
    the first 60 s only collect baseline statistics (built with no baseline), later ones use the baseline from those windows."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mask = None if mask is None else np.asarray(mask, bool)[None]
    dt = np.full(len(rgb), 1.0 / FE.FS) if dt is None else np.asarray(dt, dtype=np.float64)
    lrgb = FE.log_rgb(rgb)
    import oxydrop_ml as P
    tes = np.arange(P.WIN, len(rgb) + 1, P.HOP)
    cal = tes <= P.BASE_SEC * P.FS
    X, S, A = [], [], []
    for te in tes[cal]:
        lb, db = gather_buffer(lrgb, dt, te)
        x, s, a = FE.windows_to_tensor(lb[None], db[None], None, mask)
        X.append(x[0]); S.append(s[0]); A.append(a[0])
    base = FE.baseline_from_abs(np.array(A)) if len(A) else np.full(len(FE.BASE), np.nan)
    for te in tes[~cal]:
        lb, db = gather_buffer(lrgb, dt, te)
        x, s, a = FE.windows_to_tensor(lb[None], db[None], base[None], mask)
        X.append(x[0]); S.append(s[0]); A.append(a[0])
    return dict(x=np.array(X), side=np.array(S), abs=np.array(A), te=tes, calibrating=cal, base=base)


def make_fixture(net, seed=0, n_stream=2700):
    cfg = net.cfg
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(3):   # random phone inputs, random side and baseline embedding
        cases.append(case(net, f'random_phone_{i}', rng.normal(0, 1.5, (cfg.in_ch, M.T_WIN)), 0, rng.normal(0, 0.5, cfg.side_dim), np.abs(rng.normal(0, 1, cfg.emb))))
    c = np.zeros((cfg.in_ch, M.T_WIN))   # contact red/IR domain: slots 0 and 1 and the aux channels that exist for two wavelengths
    for ch in (0, 1, M.N_SLOTS + M.AUX_NAMES.index('dt'), M.N_SLOTS + M.AUX_NAMES.index('ror_RG_rel'), M.N_SLOTS + M.AUX_NAMES.index('beat_ok'),
               M.N_SLOTS + M.AUX_NAMES.index('dc_R_rel'), M.N_SLOTS + M.AUX_NAMES.index('dc_G_rel')):
        c[ch] = rng.normal(0, 1.0, M.T_WIN)
    cases.append(case(net, 'random_contact_0', c, 1, np.zeros(cfg.side_dim), np.zeros(cfg.emb)))
    cases.append(case(net, 'zeros_phone', np.zeros((cfg.in_ch, M.T_WIN)), 0, np.zeros(cfg.side_dim), np.zeros(cfg.emb)))
    cases.append(case(net, 'large_values', rng.normal(0, 2.0, (cfg.in_ch, M.T_WIN)), 0, rng.normal(0, 2, cfg.side_dim), rng.normal(0, 3, cfg.emb)))

    feat = None
    rec = ROOT / 'tests/fixtures/ppg-csv/Left/100001.csv'
    if rec.exists() and FE is not None:
        rgb = np.loadtxt(rec, delimiter=',', skiprows=1)[:n_stream]
        feat = {'record': 'Left/100001.csv', 'n_samples': int(len(rgb)), 'source': 'oxydrop_ml.features (FEAT-003)',
                'features_sha256': hashlib.sha256(Path(FE.__file__).read_bytes()).hexdigest(), 'variants': {}}
        noise = (120 + 6 * rng.standard_normal((2400, 3))).astype(np.float32)                      # no pulse at all (the beat detector still fires on noise)
        noise[1000:1090] = 255.0                                                                    # plus a clipped stretch
        contact = rgb.copy(); contact[:, 2] = 0.0                                                   # contact layout: [red, IR, 0] with the third wavelength masked
        variants = [('uniform', rgb, None, None, None), ('irregular_dt', rgb, dt_pattern(len(rgb)), None, None),
                    ('noise', noise.astype(np.float64), None, None, noise), ('contact_mask', contact, None, [True, True, False], None)]
        for vname, x_rgb, dt, mask, raw in variants:
            W = recorded_windows(x_rgb, dt, mask)
            K = len(W['te'])
            first_post = int(np.argmax(~W['calibrating']))
            keep = sorted({0, first_post, min(first_post + 8, K - 1), K - 1})
            X = torch.as_tensor(W['x'])
            with torch.no_grad():   # baseline embedding = mean of the calibration-window embeddings (the on-device recipe)
                e_cal = net(X[torch.as_tensor(W['calibrating'])], torch.full((K,), 1 if mask else 0)[torch.as_tensor(W['calibrating'])])['emb'].mean(0)
            v = {'te': W['te'].tolist(), 'calibrating': W['calibrating'].tolist(), 'base': jnum(W['base']),
                 'side': [jnum(r) for r in W['side']], 'abs': [jnum(r) for r in W['abs']], 'mask': mask, 'rgb_f32': None if raw is None else b64(raw),
                 'ch_mean': W['x'].mean(2).astype(float).tolist(), 'ch_absmean': np.abs(W['x']).mean(2).astype(float).tolist(),
                 'full_windows': [{'k': int(k), 'x': b64(W['x'][k])} for k in keep]}
            if vname in ('uniform', 'contact_mask'):   # full pipeline: features -> network, with the person's own baseline embedding
                dom = 1 if mask else 0
                v['model_cases'] = [case(net, f'{vname}_k{k}', W['x'][k], dom, W['side'][k], e_cal.numpy()) for k in keep[1:]]
                cases.extend(v['model_cases'])
            feat['variants'][vname] = v
    return {'format': FORMAT + '-fixture', 'seed': seed, 'tolerance': 1e-4, 'cases': cases, 'features': feat,
            'dt_pattern': 'dt[i] = (1/30) * (1 + 0.5*(i%11==0) + 3*(i%401==0))'}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', help='state_dict (.pt) saved by train.py with {"cfg":..., "state":...}; random init if omitted')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weights', default=os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'oxynet_single_network_check.json'),
                    help='single-network JSON used for the parity check only; the shipped members app/data/hybrid_R_s*.json are written by export_hybrid.py (a random-init export must never land in app/data)')
    ap.add_argument('--onnx', default=str(ROOT / 'app/data/oxynet.onnx'))
    ap.add_argument('--fixture', default=str(ROOT / 'tests/fixtures/cnn_ref.json'))
    a = ap.parse_args(argv)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location='cpu')
        net = M.OxyNet(M.OxyCfg(**ck['cfg'])); net.load_state_dict(ck['state'])
    else:
        torch.manual_seed(a.seed)
        net = M.OxyNet(M.student_cfg()); randomize_bn(net, a.seed)
    net.eval()
    graph = export_graph(net)
    graph['random_init'] = not a.ckpt
    Path(a.weights).parent.mkdir(parents=True, exist_ok=True)
    Path(a.weights).write_text(json.dumps(graph, separators=(',', ':')))
    export_onnx(net, Path(a.onnx))
    d, ex = onnx_max_diff(net, Path(a.onnx))
    print(f'params (no aux head) {graph["params"]}; weights json {Path(a.weights).stat().st_size / 1024:.0f} KB; onnx {Path(a.onnx).stat().st_size / 1024:.0f} KB; max|onnx-torch| = {d:.2e} (excess over rtol {ONNX_RTOL:g}: {ex:.2e})')
    assert ex < ONNX_ATOL, 'ONNX does not match torch'
    fx = make_fixture(net, a.seed)
    if a.ckpt:
        fx['ckpt_sha256'] = hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest()   # tests/cnn_parity.mjs compares it with the hash in app/data/hybrid_weights.json
    Path(a.fixture).parent.mkdir(parents=True, exist_ok=True)
    Path(a.fixture).write_text(json.dumps(fx, separators=(',', ':')))
    print(f'fixture {Path(a.fixture).stat().st_size / 1024:.0f} KB, {len(fx["cases"])} model cases')


if __name__ == '__main__':
    main()
