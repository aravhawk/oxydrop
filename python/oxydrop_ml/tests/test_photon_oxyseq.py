"""OxySeq: causality, gated adaptation, augmentation disjointness and the label-permutation null (no data files needed)."""
import inspect

import numpy as np
import torch

from oxydrop_ml.photon import oxyseq as O


def _net(tta=True, seed=0):
    torch.manual_seed(seed)
    net = O.OxySeq(tta=tta)
    with torch.no_grad():
        net.rho.fill_(1.0)                    # a large adaptation rate makes any leak visible
    return net.eval()


def test_strictly_causal():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((1, 80, O.ND + O.NS)), dtype=torch.float32)
    base = torch.zeros(1, 80); base[:, :26] = 1
    for tta in (True, False):
        net = _net(tta)
        with torch.no_grad():
            a = net(x, base)
            x2 = x.clone(); x2[:, 50:] = torch.tensor(rng.standard_normal((1, 30, O.ND + O.NS)) * 5, dtype=torch.float32)
            b = net(x2, base)
        for k in ('logit', 'dhat', 'qlogit', 'lin'):
            assert torch.allclose(a[k][:, :50], b[k][:, :50], atol=1e-6), k
            assert not torch.allclose(a[k][:, 50:], b[k][:, 50:])


def test_no_time_input():
    assert list(inspect.signature(O.OxySeq.forward).parameters) == ['self', 'x', 'base']
    x = O.inputs(dict(sd=np.ones(O.ND), lmu=np.zeros(O.NS), lsd=np.ones(O.NS)), np.zeros((5, O.ND)), np.ones((5, O.NS)))
    assert x.shape == (5, O.ND + O.NS)


def test_adaptation_frozen_when_gate_closed():
    net = _net(True)
    xd = torch.randn(1, 40, O.ND)
    gate = torch.zeros(1, 40); gate[:, :10] = 1
    with torch.no_grad():
        b = net.adapt(xd, gate)
    assert torch.all(b[:, 0] == 0)                                  # b_{-1} = 0: window 0 sees no offset
    assert torch.allclose(b[:, 11:], b[:, 10:11].expand_as(b[:, 11:]), atol=1e-6)   # gate closed after window 9: offsets held
    r = net.rate()
    ref = torch.zeros(O.ND)
    for k in range(10):                                            # closed form equals the recurrence
        ref = (1 - r) * ref + r * xd[0, k]
    assert torch.allclose(b[0, 10], ref, atol=1e-5)


def test_augmentation_independent_of_stress_harness():
    src = inspect.getsource(O)
    assert 'degrade_x' not in src.split('Run:')[1] and 'import stress' not in src and 'from . import stress' not in src
    x = np.ones((50, O.ND + O.NS), np.float32)
    g = np.random.default_rng(1)
    outs = [O.augment(x, g) for _ in range(20)]
    assert any(not np.allclose(o, x) for o in outs) and any(np.allclose(o, x) for o in outs)
    for o in outs:
        assert np.all(o[:, O.ND:] <= x[:, O.ND:] + 1e-6)                # augmentation only lowers the SNR


def test_null_permutes_labels_globally():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(3):
        t = np.arange(0, 200, 2)
        y = 97 - (t > 100) * rng.uniform(5, 15, len(t))
        recs.append(dict(rec=f'{i}L', sid=i, t=t, y=y, sbase=97.0, D=np.zeros((len(t), O.ND)), snr=np.ones((len(t), O.NS))))
    p = O.permute_labels(recs, 0)
    a = np.sort(np.concatenate([r['y'][r['t'] >= 60] for r in recs])); b = np.sort(np.concatenate([r['y'][r['t'] >= 60] for r in p]))
    assert np.allclose(a, b)
    assert not np.allclose(np.concatenate([r['y'] for r in recs]), np.concatenate([r['y'] for r in p]))
    assert all(np.allclose(r['y'][r['t'] < 60], q['y'][q['t'] < 60]) for r, q in zip(recs, p))
