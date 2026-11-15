"""Self-supervised (SimCLR-style contrastive) pretraining of the OxyNet encoder on the unlabeled PPG corpus (FEAT-003).

    python -m oxydrop_ml.ssl_pretrain --steps 2500 --out ckpt/ssl_full.pt [--no-openox] [--data DIR]
    python -m oxydrop_ml.ssl_pretrain --probe ckpt/ssl_full.pt            # linear-probe HR sanity check on phone data (host only)

Corpus: OpenOximetry contact PPG (red, IR), PhysioNet PTT-PPG, moshekam exercise PPG, apnoea neck oximeter, BIDMC (all band-passed AC, scale 100, float16
[N, 300, 2]). It contains no Hoffman, MTHS or BUT PPG person. The OpenOx people held out for OpenOx-specific tasks (`train.openox_holdout`) are removed too, so
no outer test subject of ANY task in this project contributes unlabeled signal (PREREGISTRATION.md rule 5). `--no-openox` drops OpenOx entirely
(for the "no OpenOx data" ablation).

Method: two random views (amplitude, time shift, speed warp, noise, wander, motion burst, time masking, channel dropout; augment.augment_ssl_view) of the
same window go through OxyNet.encode + fc_emb (the 64-d embedding the heads use) and a small projection MLP; NT-Xent with temperature 0.2 over the 2B views.
Half of the batches are presented as a phone-like input (domain 0, third slot = random mix of the two wavelengths) so the domain-0 stem also gets pretrained.
Outputs: the checkpoint {'cfg','state','ssl':{...}}, a loss curve CSV and a manifest entry with the SHA-256 of the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = 'oxydrop_ml'

from . import DATA_DIR, WIN
from .augment import augment_ssl_view
from .model import OxyNet, OxyCfg, N_CH

TEMP = 0.2
PRETRAIN_FORBIDDEN_PREFIX = ('hof_', 'mths_', 'but_')


def select_windows(meta: pd.DataFrame, exclude_subjects=(), no_openox=False):
    """Row indices of the corpus that SSL may use: never an excluded subject (outer test people / OpenOx held-out people), never a Hoffman/MTHS/BUT person,
    and no OpenOx row when no_openox. Returns an int array."""
    subj = meta.subject.astype(str).values
    ok = ~pd.Series(subj).str.startswith(PRETRAIN_FORBIDDEN_PREFIX).values
    ok &= ~np.isin(subj, list(exclude_subjects))
    if no_openox:
        ok &= (meta.src.values != 'openox')
    return np.nonzero(ok)[0]


def make_view_batch(ac, rng, scale, phone_p=0.5):
    """ac [B, 300, 2] float -> two views x1, x2 [B, 10, 300] float32 plus domain [B] (same domain for both views)."""
    B = ac.shape[0]
    a = np.transpose(np.asarray(ac, np.float32), (0, 2, 1)) / scale      # [B, 2, 300]
    dom = (rng.random(B) >= phone_p).astype(np.int64)                    # 0 phone-like, 1 contact
    mix = rng.uniform(0.2, 0.8, (B, 1, 1)).astype(np.float32)
    views = []
    for _ in range(2):
        v = augment_ssl_view(a, rng)
        x = np.zeros((B, N_CH, WIN), np.float32)
        x[:, 0:2] = v
        third = mix * v[:, 0:1] + (1 - mix) * v[:, 1:2]
        x[:, 2:3] = third * (dom[:, None, None] == 0)
        views.append(x)
    return views[0], views[1], dom


class Projector(nn.Module):
    def __init__(self, emb, out=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, out))

    def forward(self, e):
        return F.normalize(self.net(e), dim=-1)


def nt_xent(z1, z2, temp=TEMP):
    z = torch.cat([z1, z2], 0)
    n = z1.shape[0]
    sim = z @ z.T / temp
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    loss = F.cross_entropy(sim, tgt)
    acc = (sim.argmax(1) == tgt).float().mean()
    return loss, acc


def embed(net, x, dom):
    return F.relu(net.fc_emb(net.encode(x, dom)))


def pretrain(ac_path_dir=None, steps=2500, bs=192, lr=2e-3, seed=0, no_openox=False, exclude=(), device='cpu', log=print, cfg=None, curve_path=None, max_rows=None):
    d = ac_path_dir or os.path.join(DATA_DIR, 'pretrain')
    info = json.load(open(os.path.join(d, 'info.json')))
    ac = np.memmap(os.path.join(d, 'ac.f16'), dtype=np.float16, mode='r', shape=(info['n_windows'], WIN, 2))
    meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    rows = select_windows(meta, exclude, no_openox)
    if max_rows and len(rows) > max_rows:
        rows = np.random.default_rng(seed).choice(rows, max_rows, replace=False)
    used = meta.iloc[rows]
    assert not used.subject.astype(str).str.startswith(PRETRAIN_FORBIDDEN_PREFIX).any()
    assert not np.isin(used.subject.values, list(exclude)).any()
    log(f'ssl corpus rows {len(rows)} of {len(meta)}; sources {used.src.value_counts().to_dict()}; subjects {used.subject.nunique()}')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sample = np.asarray(ac[np.sort(rng.choice(rows, min(4000, len(rows)), replace=False))], np.float32)
    scale = float(max(sample.std(), 1e-3))
    net = OxyNet(cfg or OxyCfg()).to(device)
    proj = Projector(net.cfg.emb).to(device)
    params = list(net.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + np.cos(np.pi * (i - warm) / max(1, steps - warm))) * 0.98 + 0.02)
    curve = []
    t0 = time.time()
    run_l, run_a = [], []
    for step in range(steps):
        net.train(); proj.train()
        r = np.sort(rng.choice(rows, bs, replace=False))
        x1, x2, dom = make_view_batch(ac[r], rng, scale)
        dom_t = torch.from_numpy(dom).to(device)
        z1 = proj(embed(net, torch.from_numpy(x1).to(device), dom_t))
        z2 = proj(embed(net, torch.from_numpy(x2).to(device), dom_t))
        loss, acc = nt_xent(z1, z2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); sched.step()
        run_l.append(float(loss)); run_a.append(float(acc))
        if (step + 1) % 25 == 0 or step == steps - 1:
            curve.append(dict(step=step + 1, loss=float(np.mean(run_l)), top1=float(np.mean(run_a)), seconds=time.time() - t0))
            run_l, run_a = [], []
            if (step + 1) % 100 == 0 or step == steps - 1:
                log(f"  ssl step {step + 1}/{steps} loss {curve[-1]['loss']:.3f} top1 {curve[-1]['top1']:.3f} ({curve[-1]['seconds']:.0f}s)")
    if curve_path:
        os.makedirs(os.path.dirname(os.path.abspath(curve_path)), exist_ok=True)
        pd.DataFrame(curve).to_csv(curve_path, index=False)
    meta_out = dict(rows=int(len(rows)), subjects=int(used.subject.nunique()), sources={k: int(v) for k, v in used.src.value_counts().items()}, steps=steps, bs=bs, lr=lr,
                    seed=seed, no_openox=bool(no_openox), n_excluded_subjects=len(list(exclude)), scale=scale, temperature=TEMP,
                    first_loss=curve[0]['loss'], last_loss=curve[-1]['loss'], last_top1=curve[-1]['top1'], seconds=time.time() - t0)
    return net, meta_out, curve


def save(net, meta_out, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'cfg': net.cfg.to_dict(), 'state': net.state_dict(), 'ssl': meta_out}, path)
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------- linear-probe sanity check
def probe(ckpt_path, root=None, seed=0):
    """Linear (ridge) probe of the frozen 64-d embedding for HR on PHONE data: fit on MTHS + BUT windows, test on Hoffman windows (people never seen by SSL;
    this is a sanity check that the representation carries pulse-rate information, not a result). Compared with a random-init encoder and the train-mean
    predictor. Returns a dict."""
    from sklearn.linear_model import Ridge
    from . import metrics as M
    from .train import Pool, Norm, HP

    pool = Pool(datasets=('hoffman', 'mths', 'butppg'), root=root)
    hp = HP()
    rng = np.random.default_rng(seed)
    tr = np.nonzero(np.isin(pool.ds, ('mths', 'butppg')) & pool.has_hr)[0]
    te = np.nonzero((pool.ds == 'hoffman') & pool.eval & pool.has_hr)[0]
    te = rng.choice(te, min(2500, len(te)), replace=False)
    nm = Norm.fit(pool, tr, hp, rng)
    out = {}
    for name, path in (('ssl', ckpt_path), ('random_init', None)):
        if path:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            net = OxyNet(OxyCfg(**ck['cfg'])); net.load_state_dict(ck['state'])
        else:
            torch.manual_seed(seed); net = OxyNet(OxyCfg())
        net.eval()

        def feats(rows):
            E = []
            with torch.no_grad():
                for a in range(0, len(rows), 512):
                    x, s, dom = pool.inputs(rows[a:a + 512], hp=hp)
                    x, s = nm.apply(x, s, dom)
                    E.append(embed(net, torch.from_numpy(x), torch.from_numpy(dom)).numpy())
            return np.concatenate(E)
        Xtr, Xte = feats(tr), feats(te)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, pool.hr[tr])
        p = m.predict((Xte - mu) / sd)
        h = M.hr_metrics(p, pool.hr[te])
        out[name] = dict(mae=h['mae'], within5=h['within5_full'])
    base = M.hr_metrics(np.full(len(te), float(np.mean(pool.hr[tr]))), pool.hr[te])
    out['train_mean'] = dict(mae=base['mae'], within5=base['within5_full'])
    out['n_train'] = int(len(tr)); out['n_test'] = int(len(te))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Contrastive SSL pretraining of the OxyNet encoder on the unlabeled PPG corpus.')
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--bs', type=int, default=192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-openox', action='store_true', help='drop OpenOximetry windows (ablation)')
    ap.add_argument('--data', default=None, help='corpus directory (default OXY_CACHE/data/pretrain)')
    ap.add_argument('--out', default=None, help='checkpoint path')
    ap.add_argument('--curve', default=None, help='loss-curve CSV path')
    ap.add_argument('--manifest', default=None, help='JSON manifest to update with this run (key = --tag)')
    ap.add_argument('--tag', default='ssl_full')
    ap.add_argument('--hash-dir', default=None, help='with --probe --manifest: also record the SHA-256 of every *.pt in this directory')
    ap.add_argument('--holdout', default=None, help='JSON list of OpenOx subject ids to exclude (default: train.openox_holdout of the shard)')
    ap.add_argument('--max-rows', type=int, default=None, help='subsample the corpus (smoke tests)')
    ap.add_argument('--probe', default=None, help='run the HR linear-probe sanity check on this checkpoint and exit')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args(argv)
    if a.probe:
        r = probe(a.probe)
        print(json.dumps(r, indent=1))
        if a.manifest:
            m = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {}
            m['linear_probe_hr'] = dict(checkpoint=os.path.basename(a.probe), sha256=hashlib.sha256(open(a.probe, 'rb').read()).hexdigest(),
                                        task='ridge on the frozen 64-d embedding (phone domain): fit MTHS+BUT windows, test 2,500 Hoffman windows (no person shared)', **r)
            if a.hash_dir:
                m['checkpoints_sha256'] = {f: hashlib.sha256(open(os.path.join(a.hash_dir, f), 'rb').read()).hexdigest() for f in sorted(os.listdir(a.hash_dir))
                                           if f.endswith('.pt') and not f.startswith(('smoke', 'exp'))}
            json.dump(m, open(a.manifest, 'w'), indent=1)
        return
    if a.holdout:
        hold = json.load(open(a.holdout))
    else:
        from .train import openox_holdout
        hold = openox_holdout(pd.read_parquet(os.path.join(DATA_DIR, 'openox', 'meta.parquet'), columns=['subject', 'monk', 'has_hr']))
    net, mo, curve = pretrain(a.data, a.steps, a.bs, a.lr, a.seed, a.no_openox, hold, a.device, curve_path=a.curve, max_rows=a.max_rows)
    if a.out:
        sha = save(net, mo, a.out)
        mo['sha256'] = sha
        print('saved', a.out, sha)
    if a.manifest:
        m = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {}
        m[a.tag] = mo
        json.dump(m, open(a.manifest, 'w'), indent=1)
    print(json.dumps(mo))


if __name__ == '__main__':
    main()
