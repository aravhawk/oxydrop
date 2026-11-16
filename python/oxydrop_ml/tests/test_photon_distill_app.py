"""FEAT-006: the app file app/data/distill.json reproduces the trained export (results/photon/distill_weights.json) and the parity fixture is current."""
import json

import numpy as np

from oxydrop_ml.photon import distill_app_export as A


def test_app_file_reproduces_trained_reference():
    src = json.loads(A.SRC.read_text()); app = json.loads(A.OUT_APP.read_text())
    assert app['format'] == A.FORMAT and app['cols'] == src['cols'] and app['tau'] == src['tau'] and len(app['students']) == 3
    sc, sp, cf, an = A.forward(app, np.asarray(src['reference']['X'], float))
    assert np.max(np.abs(sc - np.asarray(src['reference']['score']))) < 1e-6 and list(an) == src['reference']['answer']


def test_fixture_matches_app_file_and_stress_csv_is_exact():
    A.stress_csv()
    app = json.loads(A.OUT_APP.read_text()); ref = json.loads(A.OUT_REF.read_text())
    for name, r in ref['recordings'].items():
        X = np.array([[np.nan if v is None else v for v in row] for row in r['X']], float)
        sc, sp, cf, an = A.forward(app, X)
        assert np.max(np.abs(sc - np.asarray(r['p']))) < 1e-12 and list(an) == r['answer'], name
