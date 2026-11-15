"""Tests for OxyNet (model.py), the exporter (export_js.py) and the reference input builder (input_ref.py).
Run from the repo root:  ml_venv/bin/python -m pytest python/oxydrop_ml/tests/test_model.py -q -s
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python'))
from oxydrop_ml import model as M  # noqa: E402
from oxydrop_ml import export_js as E  # noqa: E402
try:   # FEAT-003 reference window builder (absent only in a bare ml-model-cnn checkout)
    from oxydrop_ml import features as FE  # noqa: E402
except ImportError:  # pragma: no cover
    FE = None
needs_features = pytest.mark.skipif(FE is None, reason='python/oxydrop_ml/features.py (FEAT-003) not present in this checkout')

PARAM_BUDGET = 300_000


def make(cfg, seed=0):
    torch.manual_seed(seed)
    net = M.OxyNet(cfg)
    E.randomize_bn(net, seed)
    return net.eval()


def batch(cfg, B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, cfg.in_ch, M.T_WIN, generator=g), torch.tensor([0, 1, 0, 1][:B]),
            torch.randn(B, cfg.side_dim, generator=g), torch.randn(B, cfg.emb, generator=g))


# ------------------------------------------------------------------ shapes and budget
@pytest.mark.parametrize('cfg_fn', [M.student_cfg, M.teacher_cfg])
def test_output_shapes(cfg_fn):
    cfg = cfg_fn()
    net = make(cfg)
    x, dom, side, eb = batch(cfg)
    o = net(x, dom, side, eb, with_aux=True)
    B = x.shape[0]
    assert o['hr_logits'].shape == (B, 141)
    for k in ('spo2', 'spo2_logvar', 'lowox_logit', 'sqi_logit', 'aux_spo2'):
        assert o[k].shape == (B,), k
    assert o['emb'].shape == (B, cfg.emb)
    assert all(torch.isfinite(v).all() for v in o.values())
    assert 'aux_spo2' not in net(x, dom)


def test_parameter_budget():
    stu, tea = M.OxyNet(M.student_cfg()), M.OxyNet(M.teacher_cfg())
    n_s, n_t = M.count_params(stu), M.count_params(tea)
    print(f'\nstudent parameters: {n_s:,} (aux head included; {M.count_params(stu, include_aux=False):,} shipped), budget {PARAM_BUDGET:,}')
    print(f'teacher parameters: {n_t:,}')
    assert n_s < PARAM_BUDGET
    assert 1_000_000 <= n_t <= 3_000_000
    for stem in ('joint',):
        assert M.count_params(M.OxyNet(M.student_cfg(stem=stem))) < PARAM_BUDGET


def test_hr_decode():
    logits = torch.full((2, M.HR_BINS), -10.0)
    logits[0, 75 - M.HR_LO] = 10.0
    logits[1, 120 - M.HR_LO] = 10.0
    assert M.hr_from_logits(logits, 'argmax').tolist() == [75.0, 120.0]
    assert np.allclose(M.hr_from_logits(logits).numpy(), [75, 120], atol=1e-3)


# ------------------------------------------------------------------ multi-domain stem and baseline subtraction
def test_domain_mask_and_embedding():
    cfg = M.student_cfg()
    net = make(cfg)
    x, dom, side, eb = batch(cfg, 2)
    dom0, dom1 = torch.tensor([0, 0]), torch.tensor([1, 1])
    x2 = x.clone(); x2[:, 2] += 5.0                                   # slot 2 = B (phone) / empty (contact)
    assert torch.allclose(net(x, dom1)['spo2'], net(x2, dom1)['spo2'], atol=1e-6)     # contact domain ignores slot 2
    assert not torch.allclose(net(x, dom0)['spo2'], net(x2, dom0)['spo2'], atol=1e-4)  # phone domain uses it
    assert not torch.allclose(net(x, dom0)['hr_logits'], net(x, dom1)['hr_logits'], atol=1e-4)   # domain embedding is used


def test_baseline_embedding_only_affects_lowox():
    cfg = M.student_cfg()
    net = make(cfg)
    x, dom, side, eb = batch(cfg)
    a, b = net(x, dom, side, eb), net(x, dom, side, eb + 1.0)
    assert not torch.allclose(a['lowox_logit'], b['lowox_logit'], atol=1e-4)
    for k in ('hr_logits', 'spo2', 'spo2_logvar', 'sqi_logit', 'emb'):
        assert torch.equal(a[k], b[k]), k
    c = net(x, dom, side, a['emb'])                # baseline == own embedding: low-oxygen head sees only the side vector
    z = net.lowox(torch.cat([torch.zeros_like(a['emb']), side], 1))[:, 0]
    assert torch.allclose(c['lowox_logit'], z, atol=1e-5)


def test_causal_padding_is_left_only():
    net = make(M.student_cfg(causal=True, stem='joint'))
    for m in net.modules():
        if isinstance(m, M.Conv) and m.k > 1:
            assert m.pad[1] == 0 and m.pad[0] == m.dilation * (m.k - 1)
    # a causal stack must not look ahead: perturbing the last 10 samples leaves the early part of the feature map unchanged
    x = torch.randn(1, M.N_CH, M.T_WIN)
    feats = lambda v: net.blocks[1](net.blocks[0](net.stem(v)))   # noqa: E731  (stage 0, no stride; joint stem takes all channels directly)
    y = x.clone(); y[:, :, -10:] += 3
    assert torch.allclose(feats(x)[..., :M.T_WIN - 10], feats(y)[..., :M.T_WIN - 10], atol=1e-5)


# ------------------------------------------------------------------ export: independent numpy interpreter of the JSON op list
def np_tensor(g, name):
    t = g['tensors'][name]
    return np.frombuffer(__import__('base64').b64decode(t['b64']), dtype='<f4').reshape(t['shape']).astype(np.float64)


def np_conv(x, w, b, stride, dil, pad, relu):
    co, ci, k = w.shape
    xp = np.pad(x, ((0, 0), (pad[0], pad[1])))
    tout = (xp.shape[1] - dil * (k - 1) - 1) // stride + 1
    y = np.zeros((co, tout)) + b[:, None]
    for kk in range(k):
        seg = xp[:, kk * dil: kk * dil + (tout - 1) * stride + 1: stride]
        y += np.einsum('oi,it->ot', w[:, :, kk], seg)
    return np.maximum(y, 0) if relu else y


def np_forward(g, x, domain, side, emb_base):
    v = {'x': x, 'side': side, 'emb_base': emb_base}
    for op in g['ops']:
        o = op['op']
        if o == 'shared_stem_input':
            ns, ws = op['n_slots'], op['slot_width']
            w, b, mask = np_tensor(g, op['w']), np_tensor(g, op['b']), np_tensor(g, op['mask'])[domain]
            parts = [np_conv(x[s:s + 1], w, b, 1, 1, op['pad'], True) * mask[s] for s in range(ns)]
            v[op['out']] = np.concatenate(parts + [x[ns:]], 0)
        elif o == 'conv1d':
            v[op['out']] = np_conv(v[op['in']], np_tensor(g, op['w']), np_tensor(g, op['b']), op['stride'], op['dilation'], op['pad'], op['relu'])
        elif o == 'add_domain':
            y = v[op['in']] + np_tensor(g, op['table'])[domain][:, None]
            v[op['out']] = np.maximum(y, 0) if op['relu'] else y
        elif o == 'add':
            y = v[op['in'][0]] + v[op['in'][1]]
            v[op['out']] = np.maximum(y, 0) if op['relu'] else y
        elif o == 'pool':
            a = v[op['in']]
            v[op['out']] = np.concatenate([a.mean(1), a.max(1)])
        elif o == 'dense':
            z = np.concatenate([v[n].reshape(-1) for n in op['in']])
            y = np_tensor(g, op['w']) @ z + np_tensor(g, op['b'])
            v[op['out']] = np.maximum(y, 0) if op['relu'] else y
        elif o == 'sub':
            v[op['out']] = v[op['in'][0]] - v[op['in'][1]]
        else:
            raise AssertionError(o)
    out = {}
    for name, d in g['outputs'].items():
        a = v[d['buf']]
        out[name] = a if 'idx' not in d else d.get('shift', 0) + d.get('scale', 1) * a[d['idx']]
    return out


@pytest.mark.parametrize('kw', [{}, {'stem': 'joint'}, {'causal': True}, {'widths': (16, 24), 'depths': (1, 2)}])
def test_export_round_trip_numpy(kw):
    cfg = M.student_cfg(**kw)
    net = make(cfg)
    g = json.loads(json.dumps(E.export_graph(net)))      # through JSON like the app does
    x, dom, side, eb = batch(cfg, 3)
    with torch.no_grad():
        ref = net(x, dom, side, eb)
    worst = 0.0
    for i in range(3):
        o = np_forward(g, x[i].numpy().astype(np.float64), int(dom[i]), side[i].numpy().astype(np.float64), eb[i].numpy().astype(np.float64))
        for k in M.OUT_NAMES:
            worst = max(worst, float(np.abs(np.asarray(o[k]) - ref[k][i].numpy()).max()))
    print(f'\nexport round trip {kw or "default"}: max |numpy(folded) - torch| = {worst:.2e}')
    assert worst < 1e-4
    assert g['params'] == M.count_params(net, include_aux=False)
    assert not any('aux' in k for k in g['tensors'])    # training-only head is never exported


def test_bn_folding_is_exercised():
    net = make(M.student_cfg())
    bn = net.blocks[0].c1.bn
    assert not torch.allclose(bn.running_var, torch.ones_like(bn.running_var)) and not torch.allclose(bn.weight, torch.ones_like(bn.weight))


# ------------------------------------------------------------------ ONNX
def test_onnx_matches_torch(tmp_path):
    ort = pytest.importorskip('onnxruntime')   # noqa: F841
    net = make(M.student_cfg())
    p = tmp_path / 'oxynet.onnx'
    E.export_onnx(net, p)
    d, excess = E.onnx_max_diff(net, p)
    print(f'\nONNX vs torch: max abs diff {d:.2e}, excess over rtol {excess:.2e}')
    assert excess < 1e-5
    import onnx
    ops = {n.op_type for n in onnx.load(str(p)).graph.node}
    print('ONNX op types:', sorted(ops))
    assert 'LSTM' not in ops and 'Attention' not in ops


# ------------------------------------------------------------------ JS agreement for the non-default variants (skipped without node)
@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
@pytest.mark.parametrize('kw', [{'stem': 'joint'}, {'causal': True}])
def test_js_parity_variants(tmp_path, kw):
    cfg = M.student_cfg(**kw)
    net = make(cfg)
    (tmp_path / 'w.json').write_text(json.dumps(E.export_graph(net)))
    (tmp_path / 'f.json').write_text(json.dumps(E.make_fixture(net, 3)))
    r = subprocess.run(['node', str(ROOT / 'tests/cnn_parity.mjs'), str(tmp_path / 'w.json'), str(tmp_path / 'f.json')], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-600:])
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ input builder (FEAT-003 reference)
def synth_rgb(n, hr=72, seed=0, amp_r=3.0):
    t = np.arange(n) / 30
    rng = np.random.default_rng(seed)
    pulse = np.sin(2 * np.pi * hr / 60 * t)
    return np.stack([200 + amp_r * pulse, 120 + 2.0 * pulse, 60 + 0.5 * pulse], 1) + rng.normal(0, 0.1, (n, 3))


def test_layout_constants():
    assert M.N_CH == 10 and M.SIDE_DIM == 9 and M.N_SLOTS == 3
    assert cfg_defaults_match()


def cfg_defaults_match():
    c = M.OxyCfg()
    return c.in_ch == M.N_CH and c.side_dim == M.SIDE_DIM and c.n_slots == M.N_SLOTS and len(c.domain_slot_mask) == c.n_domains


@needs_features
def test_layout_matches_features_module():
    assert M.CH_NAMES == FE.CHANNELS and M.SIDE_NAMES == FE.SIDE
    import oxydrop_ml as P
    assert [P.DOMAIN['hoffman'], P.DOMAIN['openox']] == [0, 1] and len(M.DOMAINS) == 2


@needs_features
def test_network_runs_on_reference_tensors():
    rgb = synth_rgb(450)
    x, s = FE.window_to_tensor(rgb, None, None)
    assert x.shape == (10, 300) and s.shape == (9,)
    net = make(M.student_cfg())
    o = net(torch.as_tensor(x)[None], torch.tensor([0]), torch.as_tensor(s)[None])
    assert all(torch.isfinite(v).all() for v in o.values())


@needs_features
def test_js_fixture_features_present():
    fx = json.loads((ROOT / 'tests/fixtures/cnn_ref.json').read_text())
    assert fx['features'] is not None and set(fx['features']['variants']) >= {'uniform', 'irregular_dt', 'noise', 'contact_mask'}
