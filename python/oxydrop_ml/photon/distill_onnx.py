"""ONNX export of the 3 PhotonDistill students (round-3 R3-EXPORT). Alternate artefact only: the app keeps running them through app/js/nn_runtime.js.

    cd python && python -m oxydrop_ml.photon.distill_onnx [--src results/photon/distill_weights.json] [--out results/photon/onnx]

Each student file `distill_student<k>.onnx` (opset 17, float32) takes the RAW 39 features [n, 39] (NaN / inf allowed) and contains the whole student:
impute non-finite with the training median, clip to [lo, hi], standardise, Linear-ReLU-Linear-ReLU-Linear. Output `o` [n, 3] = (hyp logit, SpO2 head, abstain logit).
Ensemble outside the graph, as in the app: p = mean sigmoid(o0), SpO2 = mean(90 + 5 o1), conf = mean sigmoid(o2), answer = conf >= tau (tau in distill_manifest.json).

Parity check (written to results/photon/onnx/parity.json): the windows of tests/fixtures/ref_distill.json (3 fixture recordings, 1,190 windows) are run through
(a) the torch module in float64 (the python reference, same weights), (b) the torch module in float32 (what the ONNX graph computes), (c) onnxruntime on the .onnx files.
Reported: max |onnxruntime - torch float32| and max |onnxruntime - torch float64| on logits, on the ensemble probability, SpO2 and abstain confidence, and the
number of abstain-flag mismatches. Needs torch, onnx, onnxruntime (venv: `pip install onnx onnxruntime` if missing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'results' / 'photon' / 'distill_weights.json'
OUT = REPO / 'results' / 'photon' / 'onnx'
REF = REPO / 'tests' / 'fixtures' / 'ref_distill.json'
OPSET = 17


def torch_student(st, dtype):
    import torch
    from torch import nn

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=dtype)
            for k in ('med', 'lo', 'hi', 'mu', 'sd'):
                self.register_buffer(k, t(st['prep'][k]))
            self.l1, self.l2, self.lo_ = nn.Linear(len(st['w1'][0]), len(st['w1'])), nn.Linear(len(st['w2'][0]), len(st['w2'])), nn.Linear(len(st['wo'][0]), len(st['wo']))
            for lin, w, b in ((self.l1, 'w1', 'b1'), (self.l2, 'w2', 'b2'), (self.lo_, 'wo', 'bo')):
                lin.weight.data = t(st[w])
                lin.bias.data = t(st[b])

        def forward(self, x):
            x = torch.where(torch.isfinite(x), x, self.med)
            z = (torch.minimum(torch.maximum(x, self.lo), self.hi) - self.mu) / self.sd
            return self.lo_(torch.relu(self.l2(torch.relu(self.l1(z)))))

    return Student().to(dtype).eval()


def export_student(st, path):
    import torch
    m = torch_student(st, torch.float32)
    x = torch.zeros(2, len(st['prep']['med']), dtype=torch.float32)
    kw = dict(input_names=['x'], output_names=['o'], dynamic_axes={'x': {0: 'n'}, 'o': {0: 'n'}}, opset_version=OPSET)
    try:
        torch.onnx.export(m, (x,), str(path), dynamo=False, **kw)       # legacy TorchScript exporter: no onnxscript dependency
    except TypeError:
        torch.onnx.export(m, (x,), str(path), **kw)
    return m


def load_windows(ref):
    X, rec = [], []
    for name, R in ref['recordings'].items():
        X += [[np.nan if v is None else v for v in row] for row in R['X']]
        rec += [name] * len(R['X'])
    return np.asarray(X, np.float64), np.asarray(rec)


def ens(o, tau):
    sig = lambda v: 1 / (1 + np.exp(-np.clip(v, -30, 30)))
    p = np.mean([sig(a[:, 0]) for a in o], 0)
    sp = np.mean([90 + 5 * a[:, 1] for a in o], 0)
    cf = np.mean([sig(a[:, 2]) for a in o], 0)
    return p, sp, cf, cf >= tau


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(SRC))
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    import torch
    import onnx
    import onnxruntime as ort
    src = json.loads(Path(a.src).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tau = float(src['tau'])
    X, rec = load_windows(json.loads(REF.read_text()))
    ref = json.loads(REF.read_text())
    o64, o32, oort, files = [], [], [], []
    for k, st in enumerate(src['students']):
        path = out / f'distill_student{k}.onnx'
        m32 = export_student(st, path)
        onnx.checker.check_model(onnx.load(str(path)))
        files.append(path.name)
        m64 = torch_student(st, torch.float64)
        with torch.no_grad():
            o64.append(m64(torch.tensor(X, dtype=torch.float64)).numpy())
            o32.append(m32(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64))
        sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        oort.append(sess.run(['o'], {'x': X.astype(np.float32)})[0].astype(np.float64))
        # batch-size independence (the app feeds one window at a time)
        one = np.concatenate([sess.run(['o'], {'x': X[i:i + 1].astype(np.float32)})[0] for i in range(0, len(X), 97)])
        assert np.max(np.abs(one - oort[-1][::97])) < 1e-6, 'batch-1 and batch-n outputs differ'
    # non-finite / out-of-range inputs (the fixtures have none): NaN, +-inf and a huge value in a copy of the first 200 windows
    rng = np.random.default_rng(0)
    Xn = X[:200].copy()
    Xn[rng.random(Xn.shape) < 0.05] = np.nan
    Xn[rng.random(Xn.shape) < 0.01] = np.inf
    Xn[rng.random(Xn.shape) < 0.01] = -np.inf
    Xn[rng.random(Xn.shape) < 0.01] = 1e9
    nan_err = 0.0
    for k, st in enumerate(src['students']):
        sess = ort.InferenceSession(str(out / files[k]), providers=['CPUExecutionProvider'])
        with torch.no_grad():
            t64 = torch_student(st, torch.float64)(torch.tensor(Xn, dtype=torch.float64)).numpy()
        nan_err = max(nan_err, float(np.max(np.abs(sess.run(['o'], {'x': Xn.astype(np.float32)})[0] - t64))))
    P64, S64, C64, A64 = ens(o64, tau)
    P32, S32, C32, A32 = ens(o32, tau)
    PO, SO, CO, AO = ens(oort, tau)
    # the app's python reference (distill_app_export.forward over the app JSON) as a second, independent float64 anchor
    pref = np.concatenate([np.asarray(R['p']) for R in ref['recordings'].values()])
    aref = np.concatenate([np.asarray(R['answer']) for R in ref['recordings'].values()])
    e = lambda x, y: float(np.max(np.abs(x - y)))
    res = dict(
        opset=OPSET, files=files, n_windows=int(len(X)), recordings={n: int((rec == n).sum()) for n in ref['recordings']}, tau=tau,
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
        max_abs_logit_ort_vs_torch32=max(e(a, b) for a, b in zip(oort, o32)),
        max_abs_logit_ort_vs_torch64=max(e(a, b) for a, b in zip(oort, o64)),
        max_abs_p_ort_vs_torch32=e(PO, P32), max_abs_p_ort_vs_torch64=e(PO, P64), max_abs_p_ort_vs_app_reference=e(PO, pref),
        max_abs_spo2_ort_vs_torch64=e(SO, S64), max_abs_conf_ort_vs_torch64=e(CO, C64),
        abstain_mismatch_ort_vs_torch64=int((AO != A64).sum()), abstain_mismatch_ort_vs_app_reference=int((AO != aref).sum()),
        abstain_mismatch_torch32_vs_torch64=int((A32 != A64).sum()),
        min_abs_conf_minus_tau=float(np.min(np.abs(CO - tau))),
        n_nan_inputs=int((~np.isfinite(X)).sum()),
        nonfinite_test=dict(n_windows=200, n_nonfinite_entries=int((~np.isfinite(Xn)).sum()), max_abs_logit_ort_vs_torch64=nan_err),
    )
    (out / 'parity.json').write_text(json.dumps(res, indent=1))
    (out / 'distill_manifest.json').write_text(json.dumps(dict(
        format='oxydrop-distill-onnx-v1', opset=OPSET, cols=src['cols'], tau=tau, students=files, input='x [n, 39] float32 raw features (NaN allowed)',
        output='o [n, 3] float32 = (hyp logit, (SpO2-90)/5, abstain logit) per student',
        ensemble='p = mean sigmoid(o0); spo2 = mean(90 + 5 o1); conf = mean sigmoid(o2); answer = conf >= tau',
        note='alternate export; the app runs the same weights through app/js/nn_runtime.js (app/data/distill.json). Research prototype, not a medical device.'), indent=1))
    print(json.dumps(res, indent=1))
    assert res['max_abs_p_ort_vs_torch64'] < 1e-4 and res['abstain_mismatch_ort_vs_torch64'] <= 1 and nan_err < 1e-3
    return res


if __name__ == '__main__':
    main()
