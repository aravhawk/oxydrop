"""Light checks of the OxySSL candidate (no training): stream disjointness, pretraining disjointness, head/OxyNet equivalence, side-scale folding,
null permutation scope, and the contract format of the written score files."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from oxydrop_ml.photon import oxyssl as O

HAVE_DATA = os.path.exists(os.path.join(O.DATA_DIR, 'hoffman', 'meta.parquet')) and O.CK_INIT.exists()
COLS = ['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer']


def test_aug_stream_namespace():
    from oxydrop_ml.photon import flashgap as FG
    assert O.AUG_TAG >= O.EVAL_SEED_MAX and O.AUG_TAG != FG.AUG_TAG
    a = O.aug_rng(0, 0, 0).random(4); b = FG.aug_rng(0, 0, 0).random(4)
    assert not np.allclose(a, b)
    src = open(O.__file__).read()
    assert 'import degrade_x' not in src and 'from .stress import degrade_x' not in src


@pytest.mark.skipif(not HAVE_DATA, reason='needs the Hoffman shard and the stage-B checkpoint')
def test_pretrain_disjoint_from_hoffman():
    info = O.assert_pretrain_disjoint()
    assert info['hoffman_people'] == 6 and info['pretrain_rows'] > 0


@pytest.mark.skipif(not O.CK_INIT.exists(), reason='needs the stage-B checkpoint')
def test_top_equals_oxynet_and_folding():
    torch.manual_seed(0)
    net, _ = O.load_init()
    top = O.Top(net).eval()
    x = torch.randn(4, 10, 300); s = torch.randn(4, 9); eb = torch.rand(4, 64)
    with torch.no_grad():
        a = net(x, None, s, eb); b = top(net.encode(x), s, eb)
    for k in ('lowox_logit', 'sqi_logit', 'spo2', 'spo2_logvar', 'hr_logits'):
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    from oxydrop_ml.photon.oxyssl_export import assemble
    sc = np.linspace(0.5, 2.0, 9).astype(np.float32)
    full = assemble(net, top, sc)
    with torch.no_grad():
        c = full(x, None, s, eb); d = top(net.encode(x), s / torch.from_numpy(sc), eb)
    assert torch.allclose(c['lowox_logit'], d['lowox_logit'], atol=1e-5)


def test_null_permutation_only_touches_training_rows():
    meta = pd.DataFrame(dict(low=[0, 1, 0, 1, 1, 0.], spo2_ref=[99, 85, 97, 80, 82, 98.], has_spo2=[True] * 6, hr_ref=[60, 70, 80, 90, 100, 110.],
                             has_hr=[True] * 6))
    src = np.array([0, 1, 2, 3]); dst = np.array([3, 2, 1, 0])
    L = O.labels_of(meta, (src, dst))
    assert (L['low'][4:] == meta.low.values[4:]).all() and (L['spo2'][4:] == meta.spo2_ref.values[4:]).all()
    assert sorted(L['low'][:4]) == sorted(meta.low.values[:4]) and (L['spo2'][:4] == meta.spo2_ref.values[dst]).all()


def score_files():
    return sorted(p for p in O.SCORES.glob('oxyssl*__*.csv'))


@pytest.mark.skipif(not score_files(), reason='no oxyssl score files yet')
def test_score_files_contract():
    for p in score_files():
        d = pd.read_csv(p)
        assert list(d.columns[:8]) == COLS, p.name
        assert len(d) == 5644 and d.score.between(0, 1).all() and set(d.pred.unique()) <= {0, 1}, p.name
        assert d.answer.dtype == bool, p.name


@pytest.mark.skipif(not (O.RES / 'oxyssl_raw' / 'oxyssl_s0_selection.json').exists(), reason='no selection log yet')
def test_selection_nested_per_fold():
    sel = json.load(open(O.RES / 'oxyssl_raw' / 'oxyssl_s0_selection.json'))
    assert len(sel) == 6 and len({s['test'] for s in sel}) == 6
    for s in sel:
        assert s['lr'] in O.LR_GRID and s['epochs'] in O.EPOCH_GRID and s['q'] in O.Q_GRID
        assert len(s['grid']) == len(O.LR_GRID) * len(O.EPOCH_GRID)
