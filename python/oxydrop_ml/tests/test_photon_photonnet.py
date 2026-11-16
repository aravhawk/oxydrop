"""PhotonNet (round-2 candidate R2A): architecture, physics-layer gradients, leakage guards, null labels, augmentation disjointness, score files
and the plain-ops export. Light tests only (no training run)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from oxydrop_ml.photon import flashgap as FG
from oxydrop_ml.photon import photonnet as PN
from oxydrop_ml.photon import photonnet_export as PE
from oxydrop_ml.photon import stress as ST

N_IN = len(FG.COLS)


def test_size_and_heads():
    net = PN.PhotonNet(N_IN, phys=True)
    nn_only = sum(p.numel() for n, p in net.named_parameters() if n.split('.')[0] in ('trunk', 'z_head', 'e_head', 'r_head', 's_head'))
    assert PN.n_params(net) < 100_000 and nn_only < 100_000
    x = torch.zeros(7, N_IN, dtype=torch.float64)
    o = net(x, torch.full((7,), 0.97, dtype=torch.float64))
    assert o['z'].shape == (7, 4) and o['mu'].shape == (7,)
    assert (o['sigma'] >= 0.5).all()
    assert (o['z'].abs() <= 2 * torch.as_tensor(PN.TI.P_SD) + 1e-12).all()
    assert PN.PhotonNet(N_IN, phys=False).dec is None


def test_physics_layer_trains_the_encoder():
    """The reconstruction loss through the PhotonTwin layer reaches the trunk weights (one jointly trained network)."""
    torch.manual_seed(0)
    net = PN.PhotonNet(N_IN, phys=True)
    x = torch.randn(16, N_IN, dtype=torch.float64)
    sb = torch.full((16,), 0.97, dtype=torch.float64)
    o = net(x, sb)
    y6 = torch.randn(16, 6, dtype=torch.float64) * 0.05
    loss = net.recon_nll(o['z'], sb, y6, torch.zeros(16, 3, dtype=torch.float64))
    loss.backward()
    assert net.trunk[0].weight.grad.abs().sum() > 0
    assert net.dec.eta_raw.grad.abs().sum() > 0


def _toy_data():
    rows = []
    for sid in ('1', '2', '3'):
        for h in 'LR':
            for t in range(60, 80, 2):
                rows.append(dict(rec=sid + h, t=t, sid=sid, y_spo2=80.0 + t / 10 + int(sid), sat_b=0.95 + int(sid) / 100))
    mt = pd.DataFrame(rows)
    return dict(mt=mt, X={'clean': np.zeros((len(mt), N_IN))})


def test_null_labels_touch_training_people_only():
    D = _toy_data()
    mt = D['mt']
    y0, sb0, m0 = PN._labels(D, None, '2')
    assert np.array_equal(y0, mt.y_spo2.values)
    te = (mt.sid == '2').values
    assert np.allclose(sb0[te], m0) and np.allclose(m0, np.mean([0.96, 0.96, 0.98, 0.98]))
    for k in range(PN.N_NULL):
        y, sb, m = PN._labels(D, k, '2')
        assert np.array_equal(y[te], mt.y_spo2.values[te])
        assert sorted(y[~te]) == sorted(mt.y_spo2.values[~te]) and not np.array_equal(y[~te], mt.y_spo2.values[~te])
        assert np.allclose(sb, m)
    assert not np.array_equal(PN._labels(D, 0, '2')[0], PN._labels(D, 1, '2')[0])


def test_fit_refuses_the_held_out_person():
    D = _toy_data()
    with pytest.raises(AssertionError):
        PN.fit(D, ['1', '2'], ['2'], 0, False, D['mt'].y_spo2.values, D['mt'].sat_b.values, steps=1, snaps=(1,))


def test_augmentation_is_disjoint_from_the_stress_test():
    keys = [FG.aug_key(s, ri, k) for s in PN.SEEDS for ri in range(12) for k in range(PN.K_AUG)]
    pairs = [(lv, PN.TI.STRESS_SEED) for lv in ST.LEVELS] + [('X1', PN.TI.STRESS_SEED)]
    assert FG.assert_disjoint(keys, pairs)
    src = open(PN.__file__).read()
    assert 'degrade_x' not in src and 'ST.degrade(' not in src


def test_score_files_contract():
    p = PN.SCORES / 'photonnet__L0.csv'
    if not p.exists():
        pytest.skip('score files not written yet')
    for lv in [f'L{k}' for k in ST.LEVELS] + ['X1']:
        d = pd.read_csv(PN.SCORES / f'photonnet__{lv}.csv')
        assert {'rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer', 'conf', 'spo2_hat'} <= set(d.columns)
        assert len(d) == 5644 and d.score.between(0, 1).all() and d.pred.isin([0, 1]).all()
        assert ((d.score > 0.5).astype(int) == d.pred).all()


def test_export_matches_reference():
    if not (PE.OUT.exists() and PE.REF.exists()):
        pytest.skip('export not written yet')
    M = json.loads(PE.OUT.read_text())
    ref = json.loads(PE.REF.read_text())
    assert M['format'] == PE.FORMAT and len(M['nets']) == 3 and len(ref['t']) == 2
    o = PE.forward_json(M, np.asarray(ref['features']))
    for k in ('mu', 'sigma', 'score', 'conf'):
        assert np.allclose(o[k], ref['expected'][k], atol=1e-9)
