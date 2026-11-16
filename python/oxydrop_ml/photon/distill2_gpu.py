"""distill2_gpu: the SAME distill2 nested pipeline (python/oxydrop_ml/photon/distill2.py: same teachers, same two-stage nested grid, same seeds 0-2,
same null draws 0-2, same augmentation and stress inputs from distill's feature pack) with the student network and every batch on a CUDA device.

Only change vs distill2: distill2.train is replaced by `train` below, which moves the network and the full design tensors to cuda (asserting
torch.cuda.is_available()). Batch order (CPU torch.Generator) and initialisation (torch.manual_seed on CPU before .to(device)) match the CPU code, so
the only differences are floating-point (cuDNN/cuBLAS) ones. Teachers (sklearn) stay on CPU, as in distill2.

Run on a Kaggle GPU kernel (kaggle/distill2_gpu/ in this repo is the kernel script):
    DISTILL_PACK=.../distill_pack.npz python -m oxydrop_ml.photon.distill2_gpu --model distill2_gpu_s0 --fold 0
    DISTILL_PACK=... python -m oxydrop_ml.photon.distill2_gpu --collect          # score files distill2_gpu*__L*.csv + distill2_gpu_fold_info.json
    DISTILL_PACK=... DISTILL_FIXTURE=... python -m oxydrop_ml.photon.distill2_gpu --export   # results/photon/distill2_gpu_weights.json
Models: distill2_gpu_s0..s2, distill2_gpu (3-seed ensemble), distill2_gpu_null / distill2_gpu_d1_null / distill2_gpu_d2_null.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CACHE
from . import distill2 as B
from . import distill_student as D
from . import flashgap as F

RES, SCORES, LEVELS = B.RES, B.SCORES, B.LEVELS
NAME = 'distill2_gpu'
MODELS = {
    f'{NAME}_s0': dict(seed=0, null=None), f'{NAME}_s1': dict(seed=1, null=None), f'{NAME}_s2': dict(seed=2, null=None),
    f'{NAME}_null': dict(seed=0, null=0), f'{NAME}_d1_null': dict(seed=0, null=1), f'{NAME}_d2_null': dict(seed=0, null=2),
}
SEEDS = [f'{NAME}_s{s}' for s in range(3)]


def device_info():
    import torch
    assert torch.cuda.is_available(), 'distill2_gpu must run on a CUDA device'
    return dict(device=torch.cuda.get_device_name(0), n_gpu=torch.cuda.device_count(), torch=torch.__version__, cuda=torch.version.cuda)


def train(Xs, ys, w, q, spo2, lam, T, H, wd, seed, checkpoints):
    """distill2.train with the network and all tensors on cuda (same loss, optimiser, batch order and initialisation)."""
    import torch
    assert torch.cuda.is_available()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    dev = torch.device('cuda', int(os.environ.get('DISTILL_GPU_ID', '0')))
    nn = torch.nn
    prep = F.Prep().fit(Xs)
    Z = torch.tensor(prep.transform(Xs), dtype=torch.float32, device=dev)
    Y = torch.tensor(ys, dtype=torch.float32, device=dev)
    Wt = torch.tensor(w, dtype=torch.float32, device=dev)
    tq = np.clip(q, 1e-4, 1 - 1e-4)
    Q = torch.sigmoid(torch.tensor(np.log(tq / (1 - tq)) / T, dtype=torch.float32, device=dev))
    S2 = torch.tensor((np.nan_to_num(spo2, nan=90.0) - 90) / 5, dtype=torch.float32, device=dev)
    pure = lam == 'pure'
    lamv = 1.0 if pure else float(lam)
    pos = float((w * (ys == 1)).sum()); neg = float((w * (ys == 0)).sum())
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 3)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=B.LR, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    cw = torch.where(Y == 1, torch.tensor(neg / max(pos, 1e-9), device=dev), torch.tensor(1.0, device=dev)) * Wt
    bce = nn.functional.binary_cross_entropy_with_logits
    out, n = {}, len(Z)
    for ep in range(1, max(checkpoints) + 1):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, B.BATCH):
            b = perm[i:i + B.BATCH]
            o = net(Z[b]); z = o[:, 0]
            l = lamv * (T ** 2) * (bce(z / T, Q[b], reduction='none') * cw[b]).sum()
            if not pure:
                l = l + (bce(z, Y[b], reduction='none') * cw[b]).sum()
            l = l + D.W_SPO2 * (nn.functional.huber_loss(o[:, 1], S2[b], reduction='none') * Wt[b]).sum()
            corr = ((z.detach() > 0).float() == Y[b]).float()
            l = l + D.W_ABS * (bce(o[:, 2], corr, reduction='none') * Wt[b]).sum()
            l = l / cw[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        if ep in checkpoints:
            sd = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
            out[ep] = D.Student(prep, dict(w1=sd['0.weight'], b1=sd['0.bias'], w2=sd['2.weight'], b2=sd['2.bias'], wo=sd['4.weight'], bo=sd['4.bias']))
    return out


def _install():
    B.train = train
    B.MODELS = MODELS
    B.CACHE2 = Path(os.environ.get('DISTILL2_GPU_CACHE', str(Path(CACHE) / 'photon_distill2_gpu')))


def run_fold(name, k):
    _install()
    t0 = time.time()
    B.run_fold(name, k)
    p = B.fold_path(name, k)
    with open(p, 'rb') as fh:
        d = pickle.load(fh)
    d['info'].update(device_info(), fold_wall_s=round(time.time() - t0, 1))
    with open(p, 'wb') as fh:
        pickle.dump(d, fh)


def collect(log=print):
    _install()
    D._pack()
    mt = F.meta(); n = len(mt)
    names = [m for m in MODELS if all(B.fold_path(m, k).exists() for k in range(6))]
    Xd, _ = D.features([D.eval_cond(l) for l in LEVELS], log=log)
    taus, files = {}, {}
    for name in names:
        folds = B.load_folds(name)
        for lv in ([0] if MODELS[name]['null'] is not None else LEVELS):
            sc, sp, cf, tau = D.score_raw(folds, Xd[D.eval_cond(lv)], n)
            taus[name] = tau
            df = D.frame(mt, sc, sp, cf, cf >= tau)
            files[D.write(name, lv, df).name] = dict(model=name, devices=sorted({fo['info'].get('device', '?') for fo in folds}),
                                                    fold_wall_s=[fo['info'].get('fold_wall_s') for fo in folds])
            log(f'  {name} {lv}: TM {D.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')
    if all(s in taus for s in SEEDS):
        mtau = np.mean([taus[s] for s in SEEDS], 0)
        for lv in LEVELS:
            tag = f'L{lv}' if not isinstance(lv, str) else lv
            ds = [pd.read_csv(SCORES / f'{s}__{tag}.csv') for s in SEEDS]
            sc = np.mean([d.score.values for d in ds], 0); sp = np.mean([d.spo2_hat.values for d in ds], 0); cf = np.mean([d.conf.values for d in ds], 0)
            df = D.frame(mt, sc, sp, cf, cf >= mtau)
            files[D.write(NAME, lv, df).name] = dict(model=NAME, ensemble_of=SEEDS)
            log(f'  {NAME} (ensemble) {lv}: TM {D.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')
    info = {m: [dict(sid=fo['sid'], **fo['info']) for fo in B.load_folds(m)] for m in names}
    (RES / f'{NAME}_fold_info.json').write_text(json.dumps(info, indent=1, default=str))
    return files


def export(out=None):
    """distill2.export with the distill2_gpu seed folds and the refit on all 6 people on the GPU."""
    _install()
    from . import distill_export as DE
    D._pack()
    mt = F.meta()
    rows = np.arange(len(mt)); y = mt.y.values.astype(int); s2 = mt.y_spo2.values.astype(float)
    students, taus, rec = [], [], []
    for s in range(3):
        folds = B.load_folds(f'{NAME}_s{s}')
        r = {k: B.modal([fo['info'][k] for fo in folds]) for k in ('teacher', 'lam', 'T', 'H', 'wd', 'epochs')}
        tau = float(np.median([fo['tau'] for fo in folds]))
        conds = D.train_conds(s)
        Xd, _ = D.features(conds, log=lambda *a: None)
        Xs, ys, w, qs = B.teachers(Xd, conds, rows, y)
        m = train(Xs, ys, w, qs[r['teacher']], np.tile(s2, len(conds)), r['lam'], r['T'], r['H'], r['wd'], s, (r['epochs'],))[r['epochs']]
        students.append(m); taus.append(tau); rec.append(dict(seed=s, teacher=r['teacher'], lam_kd=r['lam'], T=r['T'], hidden=r['H'], weight_decay=r['wd'],
                                                               epochs=r['epochs'], tau=tau))
    tw, Xw = DE.fixture_windows()
    sc, sp, cf, an = DE.predict_ens(students, taus, Xw)
    js = dict(format=DE.FORMAT, model=NAME, cols=F.COLS, hidden=[int(r['hidden']) for r in rec], recipe=rec, tau=float(np.mean(taus)),
              students=[DE.student_json(m) for m in students], trained_on=device_info(),
              reference=dict(fixture=str(DE.FIXTURE.relative_to(F.REPO)), t=DE._l(tw), X=DE._l(Xw), score=DE._l(sc), spo2_hat=DE._l(sp), conf=DE._l(cf),
                             answer=[bool(v) for v in an]))
    out = Path(out or RES / f'{NAME}_weights.json')
    out.write_text(json.dumps(js))
    print('wrote', out, rec, np.round(sc, 4).tolist(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=None)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--export', action='store_true')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    if a.collect:
        print(json.dumps(collect(), indent=1)); return
    if a.export:
        export(); return
    _install()
    for nm in ([a.model] if a.model else list(MODELS)):
        for k in ([a.fold] if a.fold is not None else range(6)):
            if not B.fold_path(nm, k).exists():
                run_fold(nm, k)


if __name__ == '__main__':
    import sys
    from oxydrop_ml.photon import distill2_gpu as _m
    sys.exit(_m.main())
