"""Round-3 GPU relaunch `multida_gpu`: the multida network (oxydrop_ml.photon.multida) ported to torch on CUDA, with a wider nested grid.

LICENCE NOTICE: trained on MTHS (CC BY-NC-ND 4.0): weights are for non-commercial research only, not shipped, not committed.

    cd python && MULTIDA_PACK=... MULTIDA_AUX=... MULTIDA_OUT=... python -m oxydrop_ml.photon.multida_gpu --models multida_gpu_s0   # on a Kaggle GPU
    cd python && python -m oxydrop_ml.photon.multida_gpu --ensemble                                                             # local, light (CSV mean)

Same front end (distill's 39 features), same data (Hoffman training people + MTHS + BUT PPG aux pack), same architecture
(39 -> 32 -> ReLU -> 32 -> ReLU -> 4 heads; per-domain Prep, only the Hoffman Prep deployed; gradient-reversal domain head GRL 0.1, 32 -> 16 -> 3),
same multi-task loss (distill teacher KD on Hoffman rows, T = 2), same Adam (lr 3e-3, batch 512, coupled L2 weight decay as torch.optim.Adam).
Differences from multida: torch/CUDA instead of numpy (torch default init per config, generator seeded by (seed, config)); all grid configs of one inner
fit are trained at once as a stacked batch (independent weights, same minibatch order); nested grid widened like distill2:
    lam_kd {8, 16, 32} x weight decay {1e-4, 1e-2} x w_aux {0.3, 1} x epochs {15, 30}  (24 configs), GRL 0.1 fixed.
Selection criterion and tau as multida/distill (0.5 clean + 0.5 augmented inner OOF TM over leave-one-inner-person-out on the 5 training people;
flashgap.select_tau). Ties: fewer epochs, less aux, less KD, smaller weight decay. Grid fixed before any multida_gpu score (after the CPU multida
results were seen; Amendment 8). Models: multida_gpu_s0/s1/s2, multida_gpu (3-seed ensemble), multida_gpu_null / _null_d1 / _null_d2 (as multida).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from .. import splits as SP
from . import distill_student as DS
from . import flashgap as F
from . import multida as M

LICENCE = ('Trained on MTHS (CC BY-NC-ND 4.0): non-commercial research use only; not for commercial use or redistribution as part of a product. '
           'BUT PPG CC BY 4.0. Hoffman et al. data per its licence.')
SCORES = F.SCORES
LEVELS = M.LEVELS
GRID_LAM = (8.0, 16.0, 32.0)
GRID_WD = (1e-4, 1e-2)
GRID_AUX = (0.3, 1.0)
GRID_EP = (15, 30)
OUT_DIR = M.OUT_DIR
MODELS = {
    'multida_gpu_s0': dict(seed=0, aux=True, null=None),
    'multida_gpu_s1': dict(seed=1, aux=True, null=None),
    'multida_gpu_s2': dict(seed=2, aux=True, null=None),
    'multida_gpu_null': dict(seed=0, aux=True, null=0),
    'multida_gpu_null_d1': dict(seed=0, aux=True, null=1),
    'multida_gpu_null_d2': dict(seed=0, aux=True, null=2),
}


def _dev():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_batch(H, A, cfgs, grl, epochs, seed, checkpoints):
    """Train len(cfgs) independent networks (cfg = (lam, wd, w_aux)) at once on the GPU. Returns {(cfg, epoch): multida.Net}."""
    import torch
    dev = _dev()
    prep = F.Prep().fit(H['X'])
    Zs, dom = [prep.transform(H['X'])], [np.zeros(len(H['X']), int)]
    n_h = len(H['X'])
    for d in (1, 2):
        m = A['dom'] == d
        Zs.append(F.Prep().fit(A['X'][m]).transform(A['X'][m])); dom.append(np.full(int(m.sum()), d))
    order = np.concatenate([np.nonzero(A['dom'] == d)[0] for d in (1, 2)])
    a_sp, a_dh = A['spo2'][order], A['dhr'][order]
    Z = np.vstack(Zs); D = np.concatenate(dom); n = len(Z)
    ish = (np.arange(n) < n_h).astype(float)
    y = np.concatenate([H['y'], np.where(np.isfinite(a_sp), (a_sp < 90).astype(float), 0.0)]).astype(float)
    has_y = np.concatenate([np.ones(n_h), np.isfinite(a_sp).astype(float)])
    pos = float((H['w'] * (H['y'] == 1)).sum()); neg = float((H['w'] * (H['y'] == 0)).sum())
    pw = neg / max(pos, 1e-9)
    C = len(cfgs)
    lam = np.array([c[0] for c in cfgs]); wd = np.array([c[1] for c in cfgs]); wa = np.array([c[2] for c in cfgs])
    wrow = np.vstack([np.concatenate([H['w'], np.full(n - n_h, a)]) for a in wa])                    # (C, n)
    cw = np.where(y == 1, pw, 1.0)[None] * wrow * has_y[None]
    q = np.concatenate([np.clip(H['q'], 1e-4, 1 - 1e-4), np.full(n - n_h, 0.5)])
    Q = 1 / (1 + np.exp(-np.clip(np.log(q / (1 - q)) / DS.T_KD, -30, 30)))
    sp = (np.concatenate([np.nan_to_num(H['spo2'], nan=90.0), np.nan_to_num(a_sp, nan=90.0)]) - 90) / 5
    w_sp = wrow * np.concatenate([np.isfinite(H['spo2']), np.isfinite(a_sp)]).astype(float)[None]
    dh = np.concatenate([np.zeros(n_h), np.nan_to_num(a_dh)]) / 10
    w_dh = np.concatenate([np.zeros(n_h), np.isfinite(a_dh).astype(float)])[None] * wa[:, None]
    dcount = np.bincount(D, minlength=3).astype(float)
    w_dom = (n / (len(dcount[dcount > 0]) * np.maximum(dcount, 1)))[D]
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    Zt, yt, isht, Qt, spt, dht, wdomt = T(Z), T(y), T(ish), T(Q), T(sp), T(dh), T(w_dom)
    Dt = torch.as_tensor(D, dtype=torch.long, device=dev)
    cwt, wrowt, wspt, wdht = T(cw), T(wrow), T(w_sp), T(w_dh)
    lamt, wdt = T(lam), T(wd)
    g = torch.Generator(device='cpu'); g.manual_seed(1000 + int(seed))

    def lin(i, o):
        b = 1 / np.sqrt(i)
        w = (torch.rand(C, o, i, generator=g) * 2 - 1) * b; bb = (torch.rand(C, o, generator=g) * 2 - 1) * b
        return w.to(dev).requires_grad_(), bb.to(dev).requires_grad_()
    P = {}
    P['w1'], P['b1'] = lin(Z.shape[1], M.HID); P['w2'], P['b2'] = lin(M.HID, M.HID); P['wo'], P['bo'] = lin(M.HID, M.NOUT)
    P['wd1'], P['bd1'] = lin(M.HID, 16); P['wd2'], P['bd2'] = lin(16, 3)
    opt = torch.optim.Adam(list(P.values()), lr=DS.LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    class GR(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, gr):
            return -grl * gr

    L = lambda x, w, b: torch.baddbmm(b[:, None, :], x, w.transpose(1, 2))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    huber = lambda a, b: torch.nn.functional.smooth_l1_loss(a, b, reduction='none', beta=1.0)
    Tk = DS.T_KD
    out, cps = {}, set(checkpoints)
    gnp = torch.Generator(device=dev); gnp.manual_seed(7 + int(seed))
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev, generator=gnp)
        for i in range(0, n, DS.BATCH):
            b = perm[i:i + DS.BATCH]
            x = Zt[b][None].expand(C, -1, -1)
            h = torch.relu(L(torch.relu(L(x, P['w1'], P['b1'])), P['w2'], P['b2']))
            o = L(h, P['wo'], P['bo'])                                                  # (C, B, 4)
            cwb, wrb, ishb = cwt[:, b], wrowt[:, b], isht[b][None]
            S = cwb.sum(1).clamp_min(1.0)                                               # (C,)
            z = o[..., 0]
            yb = yt[b][None].expand(C, -1)
            l = (cwb * bce(z, yb, reduction='none')).sum(1)
            l = l + lamt * Tk ** 2 * (cwb * ishb * bce(z / Tk, Qt[b][None].expand(C, -1), reduction='none')).sum(1)
            l = l + DS.W_SPO2 * (wspt[:, b] * huber(o[..., 1], spt[b][None].expand(C, -1))).sum(1)
            corr = ((z > 0).float() == yb).float().detach()
            l = l + DS.W_ABS * (wrb * ishb * bce(o[..., 2], corr, reduction='none')).sum(1)
            l = l + M.W_HR * (wdht[:, b] * huber(o[..., 3], dht[b][None].expand(C, -1))).sum(1)
            hd = torch.relu(L(GR.apply(h), P['wd1'], P['bd1']))
            dl = L(hd, P['wd2'], P['bd2'])
            ce = torch.nn.functional.cross_entropy(dl.reshape(-1, 3), Dt[b].repeat(C), reduction='none').view(C, -1)
            l = l + (wdomt[b][None] * ce).sum(1)
            opt.zero_grad(set_to_none=True)
            (l / S).sum().backward()
            with torch.no_grad():
                for p in P.values():
                    p.grad.add_(p * wdt.view(-1, *([1] * (p.dim() - 1))))                 # coupled L2 (torch Adam weight_decay)
            opt.step()
        if ep in cps:
            for ci, c in enumerate(cfgs):
                out[(c, ep)] = M.Net(prep, {k: P[k][ci].detach().double().cpu().numpy() for k in ('w1', 'b1', 'w2', 'b2', 'wo', 'bo')})
    return out


def fit_fold(Xd, conds, tr, y_tr, s_tr, t_tr, g_tr, spec, A):
    seed = spec['seed']
    cfgs = [(l, w, a) for l in GRID_LAM for w in GRID_WD for a in GRID_AUX]
    keys = [(c, e) for c in cfgs for e in GRID_EP]
    oof = {k: {c: (np.zeros(len(tr)), np.zeros(len(tr))) for c in conds} for k in keys}
    for ip in np.unique(g_tr):
        itr = np.nonzero(g_tr != ip)[0]; iva = np.nonzero(g_tr == ip)[0]
        H = M.hof_design(Xd, conds, tr[itr], y_tr[itr], s_tr[itr])
        for k, m in train_batch(H, A, cfgs, M.GRL, max(GRID_EP), seed, GRID_EP).items():
            for c in conds:
                p, _, a = m.predict(Xd[c][tr[iva]])
                oof[k][c][0][iva] = p; oof[k][c][1][iva] = a
    crit = {}
    for k, d in oof.items():
        clean = DS.tm(t_tr, y_tr, d['clean'][0] > 0.5)
        aug = np.mean([DS.tm(t_tr, y_tr, d[c][0] > 0.5) for c in conds if c != 'clean'])
        crit[k] = (0.5 * clean + 0.5 * aug, clean, aug)
    best = max(crit, key=lambda k: (round(crit[k][0], 6), -k[1], -k[0][2], -k[0][0], -k[0][1]))
    pc, ac = oof[best]['clean']
    tau, cert, cov, acc = F.select_tau(ac, (pc > 0.5).astype(int), y_tr)
    H = M.hof_design(Xd, conds, tr, y_tr, s_tr)
    model = train_batch(H, A, [best[0]], M.GRL, best[1], seed, [best[1]])[(best[0], best[1])]
    info = dict(lam=best[0][0], wd=best[0][1], w_aux=best[0][2], epochs=best[1], inner_crit=round(crit[best][0], 4),
                inner_tm_clean=round(crit[best][1], 4), inner_tm_aug=round(crit[best][2], 4),
                grid={f'lam={k[0][0]},wd={k[0][1]},aux={k[0][2]},ep={k[1]}': [round(v, 4) for v in crit[k]] for k in crit},
                tau=round(float(tau), 4), tau_certified=bool(cert), inner_coverage=round(float(cov), 3), inner_answered_acc=round(float(acc), 3))
    return model, tau, info


def fit_model(name, Xd, log=print):
    spec = MODELS[name]
    mt = F.meta()
    y, t, sid, s2 = mt.y.values.astype(int), mt.t.values.astype(float), mt.sid.values.astype(str), mt.y_spo2.values.astype(float)
    conds = DS.train_conds(spec['seed'])
    folds = []
    for f in SP.loso(sid):
        tr, te = f.train_idx, f.test_idx
        SP.assert_no_leak(sid, tr, te)
        y_tr, s_tr = y[tr].copy(), s2[tr].copy()
        if spec['null'] is not None:
            p = DS.null_perm(len(tr), f.test_group, spec['null'])
            y_tr, s_tr = y_tr[p], s_tr[p]
        t0 = time.time()
        model, tau, info = fit_fold(Xd, conds, tr, y_tr, s_tr, t[tr], sid[tr], spec, M.aux_for(spec, f.test_group))
        info['seconds'] = round(time.time() - t0, 1)
        log(f'{name} fold {f.test_group}: lam {info["lam"]} wd {info["wd"]} aux {info["w_aux"]} ep {info["epochs"]} crit {info["inner_crit"]:.3f}; '
            f'tau {tau:.3f}; {info["seconds"]} s', flush=True)
        folds.append(dict(sid=f.test_group, te=te, model=model, tau=tau, info=info))
    path = OUT_DIR / 'models' / f'{name}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(dict(licence_notice=LICENCE, folds=folds), fh)
    return folds


def run_models(names, levels=LEVELS, log=print):
    import torch
    os.environ.setdefault(DS.PACK_ENV, os.environ.get('MULTIDA_PACK', ''))
    DS._pack()
    mt = F.meta(); n = len(mt)
    seeds = sorted({MODELS[m]['seed'] for m in names})
    conds = list(dict.fromkeys(sum([DS.train_conds(s) for s in seeds], []) + [DS.eval_cond(l) for l in levels]))
    Xd, _ = DS.features(conds, log=log)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'
    log('device', dev, flush=True)
    info = {}
    for name in names:
        t0 = time.time()
        folds = fit_model(name, Xd, log=log)
        lvs = [0] if MODELS[name]['null'] is not None else levels
        for lv in lvs:
            sc, sp, cf, tau = DS.score_raw(folds, Xd[DS.eval_cond(lv)], n)
            df = DS.frame(mt, sc, sp, cf, cf >= tau)
            DS.write(name, lv, df)
            log(f'  {name} {lv}: TM {DS.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}', flush=True)
        info[name] = dict(device=dev, wall_s=round(time.time() - t0, 1), folds=[dict(sid=f['sid'], **{k: v for k, v in f['info'].items() if k != 'grid'}) for f in folds],
                          licence_notice=LICENCE)
        (SCORES.parent / f'{name}_fold_info.json').write_text(json.dumps(info[name], indent=1, default=str))
    return info


def ensemble(levels=LEVELS, log=print):
    """Light, local: mean of the 3 seed score files; answer = conf >= mean of the seeds' fold taus (taus from the pulled fold-info files)."""
    taus = []
    base = None
    for s in range(3):
        fi = json.loads((SCORES.parent / f'multida_gpu_s{s}_fold_info.json').read_text())
        tau_of = {str(f['sid']): f['tau'] for f in fi['folds']}
        d0 = pd.read_csv(SCORES / f'multida_gpu_s{s}__L0.csv')
        taus.append(d0.sid.astype(str).map(tau_of).values.astype(float))
        base = d0 if base is None else base
    mtau = np.mean(taus, 0)
    assert np.isfinite(mtau).all()
    for lv in levels:
        tag = f'L{lv}' if not isinstance(lv, str) else lv
        ds = [pd.read_csv(SCORES / f'multida_gpu_s{s}__{tag}.csv') for s in range(3)]
        for d in ds:
            assert (d.rec.values == base.rec.values).all() and (d.t.values == base.t.values).all()
        sc = np.mean([d.score.values for d in ds], 0); sp = np.mean([d.spo2_hat.values for d in ds], 0); cf = np.mean([d.conf.values for d in ds], 0)
        df = base[['rec', 't', 'sid', 'y_spo2', 'y']].copy()
        df['score'] = sc; df['pred'] = (sc > 0.5).astype(int); df['answer'] = cf >= mtau; df['spo2_hat'] = sp; df['conf'] = cf
        DS.write('multida_gpu', lv, df)
        log(f'  multida_gpu (ensemble) {lv}: TM {DS.tm(df.t.values, df.y.values, df.pred.values):.3f}, answered {df.answer.mean():.2f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--levels', nargs='*', default=None)
    ap.add_argument('--ensemble', action='store_true')
    a = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings('ignore')
    lv = LEVELS if a.levels is None else [int(x) if x.isdigit() else x for x in a.levels]
    if a.ensemble:
        ensemble(lv); return
    run_models(a.models or list(MODELS), lv, log=print)


if __name__ == '__main__':
    main()
