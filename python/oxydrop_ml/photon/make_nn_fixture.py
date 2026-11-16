"""Parity fixture for app/js/nn_runtime.js: tests/fixtures/nn_runtime_ref.json.

    cd python && python -m oxydrop_ml.photon.make_nn_fixture [--out ../tests/fixtures/nn_runtime_ref.json] [--seed 7]

Builds six small random float64 PyTorch networks that exercise every op of the runtime (Linear, Conv1d causal / same / valid / explicit padding, stride, dilation, groups,
ReLU / GELU / GELU-tanh / tanh / sigmoid, LayerNorm over channel / time / all, BatchNorm folded and raw, GRU with and without a carried hidden state, global avg / max pooling,
residual add, concat, slice, select_last), exports each to the 'nn-runtime-v1' JSON graph of app/js/nn_runtime.js (schema documented in its header) and stores random inputs
with the torch outputs. `NNGraph` below is also the exporter template for a winner: add one method call per module, then `graph.json()`.
Weights are random (this checks the arithmetic, not any trained model); weights of the 'gru_state' model are rounded to float32 and stored as b64 f32, 'cnn' as b64 f64, the rest as JSON numbers.
Run with torch in float64; the JS test tolerance is 1e-6.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
FORMAT = 'nn-runtime-v1'
REPO = Path(__file__).resolve().parents[3]


class NNGraph:
    """Collects tensors and ops in the 'nn-runtime-v1' schema. Each method takes the torch module (or tensors), records the op and returns the output buffer name."""

    def __init__(self, enc='data'):
        self.enc, self.inputs, self.tensors, self.ops, self.n = enc, {}, {}, [], 0

    def _t(self, name, x):
        a = x.detach().cpu().numpy().astype(np.float64)
        if self.enc == 'data':
            self.tensors[name] = dict(shape=list(a.shape), data=a.reshape(-1).tolist())
        elif self.enc == 'b64f64':
            self.tensors[name] = dict(shape=list(a.shape), dtype='f64', b64=base64.b64encode(a.astype('<f8').tobytes()).decode())
        else:
            self.tensors[name] = dict(shape=list(a.shape), dtype='f32', b64=base64.b64encode(a.astype('<f4').tobytes()).decode())
        return name

    def _out(self, p):
        self.n += 1
        return f'{p}{self.n}'

    def input(self, name, shape):
        self.inputs[name] = dict(shape=list(shape)); return name

    def linear(self, x, m: nn.Linear, per_step=False):
        o = self._out('lin'); k = o
        op = dict(op='linear', **{'in': x}, out=o, w=self._t(k + '.w', m.weight), per_step=per_step)
        if m.bias is not None:
            op['b'] = self._t(k + '.b', m.bias)
        self.ops.append(op); return o

    def conv1d(self, x, m: nn.Conv1d, padding):
        o = self._out('conv'); k = o
        op = dict(op='conv1d', **{'in': x}, out=o, w=self._t(k + '.w', m.weight), stride=m.stride[0], dilation=m.dilation[0], groups=m.groups, padding=padding)
        if m.bias is not None:
            op['b'] = self._t(k + '.b', m.bias)
        self.ops.append(op); return o

    def act(self, x, fn):
        o = self._out(fn); self.ops.append(dict(op='act', **{'in': x}, out=o, fn=fn)); return o

    def layernorm(self, x, m: nn.LayerNorm, dim):
        o = self._out('ln'); k = o
        self.ops.append(dict(op='layernorm', **{'in': x}, out=o, w=self._t(k + '.w', m.weight), b=self._t(k + '.b', m.bias), eps=m.eps, dim=dim)); return o

    def batchnorm(self, x, m: nn.BatchNorm1d, folded=True):
        """BatchNorm1d in eval mode. folded=True writes scale/shift (what an exporter should do), False writes the raw statistics (folded by the runtime at load)."""
        o = self._out('bn'); k = o
        if folded:
            sc = m.weight / torch.sqrt(m.running_var + m.eps); sh = m.bias - m.running_mean * sc
            self.ops.append(dict(op='batchnorm', **{'in': x}, out=o, scale=self._t(k + '.scale', sc), shift=self._t(k + '.shift', sh)))
        else:
            self.ops.append(dict(op='batchnorm', **{'in': x}, out=o, mean=self._t(k + '.mean', m.running_mean), var=self._t(k + '.var', m.running_var),
                                 w=self._t(k + '.w', m.weight), b=self._t(k + '.b', m.bias), eps=m.eps))
        return o

    def gru(self, x, m: nn.GRU, output='seq', h0=None, state_out=None):
        o = self._out('gru'); k = o
        op = dict(op='gru', **{'in': x}, out=o, w_ih=self._t(k + '.wih', m.weight_ih_l0), w_hh=self._t(k + '.whh', m.weight_hh_l0),
                  b_ih=self._t(k + '.bih', m.bias_ih_l0), b_hh=self._t(k + '.bhh', m.bias_hh_l0), output=output)
        if h0:
            op['h0'] = h0
        if state_out:
            op['state_out'] = state_out
        self.ops.append(op); return o

    def pool(self, x, mode):
        o = self._out('pool'); self.ops.append(dict(op='pool', **{'in': x}, out=o, mode=mode)); return o

    def select_last(self, x):
        o = self._out('last'); self.ops.append(dict(op='select_last', **{'in': x}, out=o)); return o

    def concat(self, xs, axis='flat'):
        o = self._out('cat'); self.ops.append(dict(op='concat', **{'in': list(xs)}, out=o, axis=axis)); return o

    def add(self, a, b):
        o = self._out('add'); self.ops.append(dict(op='add', **{'in': [a, b]}, out=o)); return o

    def slice(self, x, start, end):
        o = self._out('slice'); self.ops.append(dict(op='slice', **{'in': x}, out=o, start=start, end=end)); return o

    def json(self, outputs):
        return dict(format=FORMAT, inputs=self.inputs, tensors=self.tensors, ops=self.ops, outputs=list(outputs))


def rnd_bn(m, g):
    """random but fixed running statistics so folding is not the identity"""
    with torch.no_grad():
        m.running_mean.copy_(torch.randn(m.num_features, generator=g, dtype=torch.float64) * 0.5)
        m.running_var.copy_(torch.rand(m.num_features, generator=g, dtype=torch.float64) + 0.5)
        m.weight.copy_(torch.randn(m.num_features, generator=g, dtype=torch.float64) * 0.5 + 1)
        m.bias.copy_(torch.randn(m.num_features, generator=g, dtype=torch.float64) * 0.3)
    return m.eval()


def init(mod, g, round32=False):
    """deterministic random weights (float64); optionally rounded to float32 so a b64 f32 export is exact"""
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.4)
            if round32:
                p.copy_(p.float().double())
    return mod


def flat(t):
    return t.detach().reshape(-1).cpu().numpy().astype(np.float64).tolist()


def pad_same(x, k, d):
    tot = d * (k - 1); l = tot // 2
    return F.pad(x, (l, tot - l))


def build(seed=7):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(seed)
    R = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    models = []

    # ---- 1. TCN: causal convs with dilation, BN raw and folded, GELU, residual, causal read-out (output per time step and last)
    G = NNGraph('data'); x = G.input('x', [3, 40])
    c1 = init(nn.Conv1d(3, 8, 3, dilation=1), g); b1 = rnd_bn(nn.BatchNorm1d(8), g)
    c2 = init(nn.Conv1d(8, 8, 3, dilation=2), g); b2 = rnd_bn(nn.BatchNorm1d(8), g)
    c3 = init(nn.Conv1d(8, 4, 1), g); head = init(nn.Linear(4, 2), g)
    h = G.act(G.batchnorm(G.conv1d(x, c1, 'causal'), b1, folded=False), 'relu')
    h2 = G.act(G.batchnorm(G.conv1d(h, c2, 'causal'), b2, folded=True), 'gelu')
    r = G.add(h, h2)
    seq = G.conv1d(r, c3, 'valid'); out = G.linear(G.select_last(seq), head)
    X = R(3, 40)
    with torch.no_grad():
        t = X[None]
        a = torch.relu(b1(c1(F.pad(t, (2, 0)))))
        a2 = F.gelu(b2(c2(F.pad(a, (4, 0)))))
        s = c3(a + a2); o = head(s[0, :, -1])
    models.append(dict(name='tcn', graph=G.json([seq, out]), cases=[dict(inputs=dict(x=flat(X)), outputs={seq: flat(s), out: flat(o)})], causal=dict(output=seq, input='x', lookback=2 + 4)))

    # ---- 2. CNN: same padding with odd total pad (k=4, d=3), LayerNorm over channels, tanh-GELU, grouped strided dilated conv, explicit padding, avg+max pooling, sigmoid head, slice
    G = NNGraph('b64f64'); x = G.input('x', [2, 33])
    c1 = init(nn.Conv1d(2, 6, 4, dilation=3, padding='same'), g); ln = init(nn.LayerNorm(6), g)
    c2 = init(nn.Conv1d(6, 6, 3, stride=2, dilation=2, groups=2), g); c3 = init(nn.Conv1d(6, 5, 4, stride=3), g); fc = init(nn.Linear(10, 3), g)
    h = G.conv1d(x, c1, 'same'); h = G.layernorm(h, ln, 'channel'); h = G.act(h, 'gelu_tanh')
    h = G.act(G.conv1d(h, c2, 'valid'), 'tanh'); h = G.act(G.conv1d(h, c3, [2, 1]), 'relu')
    p = G.pool(h, 'avgmax'); lg = G.linear(p, fc); pr = G.act(lg, 'sigmoid'); sl = G.slice(pr, 1, 3)
    X = R(2, 33)
    with torch.no_grad():
        t = X[None]
        a = c1(t)                                                                # torch padding='same' (left = total // 2)
        assert torch.allclose(a, F.conv1d(pad_same(t, 4, 3), c1.weight, c1.bias, dilation=3))
        a = ln(a.transpose(1, 2)).transpose(1, 2); a = F.gelu(a, approximate='tanh')
        a = torch.tanh(c2(a)); a = torch.relu(c3(F.pad(a, (2, 1))))
        pp = torch.cat([a.mean(-1), a.amax(-1)], -1); o = torch.sigmoid(fc(pp))[0]
    models.append(dict(name='cnn', graph=G.json([pr, sl]), cases=[dict(inputs=dict(x=flat(X)), outputs={pr: flat(o), sl: flat(o[1:3])})]))

    # ---- 3. GRU sequence model: per-step linear, LayerNorm over time (torch LayerNorm(T)), sigmoid, GRU (seq and last), concat of last state with avg pooling, heads
    G = NNGraph('data'); x = G.input('x', [5, 12])
    l1 = init(nn.Linear(5, 7), g); ln = init(nn.LayerNorm(12), g); gr = init(nn.GRU(7, 6, batch_first=True), g); l2 = init(nn.Linear(6 + 7, 3), g)
    h = G.linear(x, l1, per_step=True); h = G.layernorm(h, ln, 'time'); h = G.act(h, 'sigmoid')
    sq = G.gru(h, gr, 'seq'); last = G.select_last(sq); avg = G.pool(h, 'avg'); cat = G.concat([last, avg]); y = G.linear(cat, l2)
    X = R(5, 12)
    with torch.no_grad():
        a = l1(X.T)                                   # [T, 7]
        a = ln(a.T).T                                 # LayerNorm(12) over time per channel
        a = torch.sigmoid(a)                          # [T, 7]
        so, hn = gr(a[None])                          # [1, T, 6]
        o = l2(torch.cat([so[0, -1], a.mean(0)]))
    models.append(dict(name='gru_seq', graph=G.json([sq, y]), cases=[dict(inputs=dict(x=flat(X)), outputs={sq: flat(so[0].T), y: flat(o)})]))

    # ---- 4. Streaming GRU: hidden state carried between two calls (window stream of one recording), weights rounded to float32 and stored as b64 f32
    G = NNGraph('b64f32'); x = G.input('x', [4, 6]); h0 = G.input('h0', [5])
    lin = init(nn.Linear(4, 4), g, True); gr = init(nn.GRU(4, 5, batch_first=True), g, True); hd = init(nn.Linear(5, 1), g, True)
    a = G.act(G.linear(x, lin, per_step=True), 'tanh'); G.n += 0
    last = G.gru(a, gr, 'last', h0=h0, state_out='h_out'); y = G.act(G.linear(last, hd), 'sigmoid')
    X1, X2 = R(4, 12), None
    with torch.no_grad():
        full = torch.tanh(lin(X1.T))[None]                       # [1, 12, 4]: two consecutive windows of 6 steps
        so1, h1 = gr(full[:, :6], torch.zeros(1, 1, 5)); so2, h2 = gr(full[:, 6:], h1)
        y1 = torch.sigmoid(hd(h1[0, 0])); y2 = torch.sigmoid(hd(h2[0, 0]))
    w1, w2 = X1[:, :6], X1[:, 6:]
    models.append(dict(name='gru_state', graph=G.json([y, 'h_out']), cases=[
        dict(inputs=dict(x=flat(w1), h0=[0.0] * 5), outputs={y: flat(y1), 'h_out': flat(h1)}),
        dict(inputs=dict(x=flat(w2), h0=flat(h1)), outputs={y: flat(y2), 'h_out': flat(h2)})]))

    # ---- 5. MLP on a vector: Linear, LayerNorm over all values, ReLU, Linear, tanh, Linear, sigmoid
    G = NNGraph('data'); x = G.input('x', [8])
    a1 = init(nn.Linear(8, 10), g); ln = init(nn.LayerNorm(10), g); a2 = init(nn.Linear(10, 6), g); a3 = init(nn.Linear(6, 2), g)
    h = G.act(G.layernorm(G.linear(x, a1), ln, 'all'), 'relu'); h = G.act(G.linear(h, a2), 'tanh'); y = G.act(G.linear(h, a3), 'sigmoid')
    X = R(8)
    with torch.no_grad():
        o = torch.sigmoid(a3(torch.tanh(a2(torch.relu(ln(a1(X)))))))
    models.append(dict(name='mlp', graph=G.json([y]), cases=[dict(inputs=dict(x=flat(X)), outputs={y: flat(o)})]))

    # ---- 6. Batch-norm + conv + concat over channels + max pooling (sanity for the remaining ops, two cases)
    G = NNGraph('data'); x = G.input('x', [3, 20])
    ca = init(nn.Conv1d(3, 4, 5, padding=2), g); cb = init(nn.Conv1d(3, 2, 1), g); bn = rnd_bn(nn.BatchNorm1d(6), g); fc = init(nn.Linear(6, 1), g)
    m1 = G.conv1d(x, ca, 2); m2 = G.conv1d(x, cb, 'valid'); m = G.concat([m1, m2], 'channel'); m = G.batchnorm(m, bn, folded=True); m = G.act(m, 'relu')
    p = G.pool(m, 'max'); y = G.linear(G.concat([p]), fc)
    cases = []
    for _ in range(2):
        X = R(3, 20)
        with torch.no_grad():
            m = torch.cat([ca(X[None]), cb(X[None])], 1); o = fc(torch.relu(bn(m)).amax(-1)[0])
        cases.append(dict(inputs=dict(x=flat(X)), outputs={y: flat(o)}))
    models.append(dict(name='bn_concat_max', graph=G.json([y]), cases=cases))
    return models


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(REPO / 'tests' / 'fixtures' / 'nn_runtime_ref.json')); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(argv)
    models = build(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(format=FORMAT, torch=torch.__version__, dtype='float64', tolerance=1e-6, seed=a.seed, models=models), separators=(',', ':')))
    print(f'wrote {a.out}: {len(models)} models, {sum(len(m["cases"]) for m in models)} cases')


if __name__ == '__main__':
    main()
