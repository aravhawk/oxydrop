"""SSL3 (round-3 candidate R3-SSL): OxyNet encoder pretrained self-supervised (contrastive NT-Xent + masked reconstruction) on public,
non-OpenOx, non-MIMIC PPG, then fine-tuned END TO END (encoder unfrozen, small LR) leave-one-person-out on Hoffman with freshly initialised
multi-task heads (hypoxia, SpO2 + log-variance, HR, quality/abstain) and the FlashGap distant-flash augmentation prior. The deployed model is ONE
OxyNet (the architecture app/js/cnn.js runs); see ssl3_export.py.

    python -m oxydrop_ml.photon.ssl3 bundle                 # local, light: write the public pretraining subset for the Kaggle dataset
    python -m oxydrop_ml.photon.ssl3 pretrain --sec 600     # GPU: SSL pretraining -> CACHE/ssl3/ssl3_pre.pt
    python -m oxydrop_ml.photon.ssl3 prep                   # OxyNet tensors of every Hoffman recording version (clean, aug<k>, L1..L5, X1)
    python -m oxydrop_ml.photon.ssl3 fit --variant s0       # nested LOSO fine-tuning; s<k> = seed k, n<d> = null draw d (labels permuted)
    python -m oxydrop_ml.photon.ssl3 ensemble --seeds 0 1 2

What is learned from round-2 OxySSL (frozen encoder, null 0.643 FAILED): there the heads started from an OpenOx-supervised stage B that already
ranked hypoxia, so a permuted-label fit kept that ranking. Here NOTHING supervised precedes Hoffman: the pretraining is label-free on people who are
not in Hoffman, and the HR / SpO2 / hypoxia / quality heads are re-initialised in every fit. fc_emb (the 64-d embedding the contrastive loss used) is
kept from pretraining as part of the encoder.

Pretraining corpus (provenance: data/pretrain.py, data/butppg, data/mths): PhysioNet PTT-PPG, moshekam exercise PPG, the apnoea neck oximeter (contact
PPG, band-passed AC, [300, 2]), BUT PPG and MTHS smartphone PPG (ready OxyNet tensors [10, 300], phone domain). OpenOximetry and BIDMC (a MIMIC-II
derivative) are excluded, so the corpus may run on Kaggle. assert_pretrain_disjoint checks that no pretraining subject is a Hoffman person.
Objectives: two GPU-augmented views of each window (amplitude / per-channel gain, circular time shift, white noise, baseline wander, one span masked to 0,
slot dropout, aux-channel jitter); NT-Xent (temperature 0.2) on the projected fc_emb embedding + masked reconstruction (a small conv decoder on the
last encoder feature map rebuilds the standardised waveform slots of the masked span of view 1). Contact windows are shown half as contact (domain 1)
and half as phone-like (domain 0, third slot = random mix of the two wavelengths), as in ssl_pretrain.py.

Fine-tuning protocol (ROUND2.md / ROUND3.md):
  * LOSO by person, 6 outer folds. Per outer fold, inner leave-one-person-out over the 5 training people selects the number of epochs
    (EPOCH_GRID, checkpoints of one run) on 0.5 TM(clean inner OOF) + 0.5 TM(augmented inner OOF), ties to fewer epochs, and the abstain threshold
    on sigmoid(quality) (inner-OOF quantile in Q_GRID with the best answered bal. acc., ties to higher coverage). The held-out person is never used.
  * Learning rates are fixed before any score (LR_HEAD 1e-3, LR_ENC 2e-4, cosine), BatchNorm in train mode; emb_base (mean embedding of the
    recording's first 8 label-free baseline windows) is recomputed in eval mode every EB_REFRESH steps and at every checkpoint.
  * Training data: clean training people + K_AUG distant-flash copies per training recording from the FlashGap prior (flashgap.sample_params /
    optics_gain / flash_augment) with SSL3's own stream SeedSequence([AUG_TAG 'OXS3', 0, rec_idx, k]); no stress.degrade call, X1 never imported
    for training. Tested on stress.degrade(level, 0) / degrade_x('X1', 0) of the held-out person's whole recording, not refit per level.
  * Null: labels (low, SpO2, HR jointly) permuted globally across the training windows of each fold, 3 draws, same nested pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from torch import nn

from .. import CACHE, DATA_DIR
from .. import metrics as M
from ..model import OxyNet, OxyCfg, HR_LO, HR_BINS, SPO2_CENTER, SPO2_SCALE, N_CH

REPO = Path(__file__).resolve().parents[3]
RES = REPO / 'results' / 'photon'
SCORES = RES / 'scores'
RAW = RES / 'ssl3_raw'
WORK = Path(os.environ.get('SSL3_CACHE', os.path.join(CACHE, 'ssl3')))
PRE_CK = WORK / 'ssl3_pre.pt'
NAME = 'ssl3'
PUB_SRC = ('ptt_ppg', 'moshekam', 'apnoea')           # public, non-OpenOx, non-MIMIC parts of data/pretrain

AUG_TAG = 0x4F585333            # 'OXS3'
assert AUG_TAG >= 1_000_000
SMOKE = os.environ.get('SSL3_SMOKE') == '1'            # tiny CPU code-path check only; never writes score files
K_AUG = 1 if SMOKE else 3
AUG_SEED = 0
LEVELS = [0, 1, 2, 3, 4, 5]
EVAL = [0, 'X1'] if SMOKE else LEVELS + ['X1']
EPOCH_GRID = (1, 2) if SMOKE else (1, 2, 4, 6)
Q_GRID = (0.0, 0.1, 0.2, 0.3)
LR_HEAD, LR_ENC = 1e-3, 2e-4
BS = 256
WD = 1e-4
BASE_ROWS = 8
EB_REFRESH = 20
DROPOUT = 0.1
W_LOW, W_SPO2, W_HR, W_SQI = 1.0, 0.5, 0.3, 0.3
HEADS = ('hr1', 'hr2', 'spo2', 'lowox', 'sqi', 'aux_spo2')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vname(v):
    return v if isinstance(v, str) else f'L{v}'


def log_(*s):
    print(time.strftime('%H:%M:%S'), *s, flush=True)


# ------------------------------------------------------------------------------------------------ provenance
def hoffman_people():
    return set(pd.read_parquet(os.path.join(DATA_DIR, 'hoffman', 'meta.parquet'), columns=['subject']).subject.astype(str))


def assert_pretrain_disjoint():
    """Every pretraining subject (public contact corpus + BUT PPG + MTHS) is disjoint from the Hoffman people; no OpenOx / BIDMC row is used."""
    hof = hoffman_people()
    pm = pd.read_parquet(WORK / 'pub_meta.parquet')
    assert set(pm.src.unique()) <= set(PUB_SRC), 'only public non-OpenOx non-MIMIC sources'
    subj = set(pm.subject.astype(str))
    for d in ('butppg', 'mths'):
        subj |= set(pd.read_parquet(os.path.join(DATA_DIR, d, 'meta.parquet'), columns=['subject']).subject.astype(str))
    assert not (subj & hof), 'pretraining people overlap Hoffman'
    assert not any(s.startswith('hof') for s in subj)
    return dict(pretrain_subjects=len(subj), hoffman_people=len(hof), contact_rows=len(pm))


def bundle():
    """Local, light (chunked memmap copy): the PUB_SRC rows of data/pretrain/ac.f16 -> WORK/pub_ac.f16.npy [N, 300, 2] + pub_meta.parquet."""
    WORK.mkdir(parents=True, exist_ok=True)
    d = os.path.join(DATA_DIR, 'pretrain')
    info = json.load(open(os.path.join(d, 'info.json')))
    meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    ac = np.memmap(os.path.join(d, 'ac.f16'), dtype=np.float16, mode='r', shape=(info['n_windows'], info['win'], 2))
    rows = np.nonzero(meta.src.isin(PUB_SRC).values & ~meta.subject.astype(str).str.startswith(('hof_', 'mths_', 'but_')).values)[0]
    out = np.lib.format.open_memmap(WORK / 'pub_ac.f16.npy', mode='w+', dtype=np.float16, shape=(len(rows), info['win'], 2))
    for a in range(0, len(rows), 20000):
        out[a:a + 20000] = ac[rows[a:a + 20000]]
    out.flush()
    meta.iloc[rows].reset_index(drop=True).to_parquet(WORK / 'pub_meta.parquet')
    json.dump(dict(scale=info['scale'], n=len(rows)), open(WORK / 'pub_info.json', 'w'))
    print('bundle', len(rows), meta.iloc[rows].src.value_counts().to_dict(), assert_pretrain_disjoint())


# ------------------------------------------------------------------------------------------------ pretraining (GPU)
def encode_map(net, x, dom):
    """OxyNet.encode up to the last residual block (feature map [B, C, T/8])."""
    cfg = net.cfg
    B, _, T = x.shape
    slots = x[:, :cfg.n_slots].reshape(B * cfg.n_slots, 1, T)
    h = TF.relu(net.slot_conv(TF.pad(slots, net.slot_pad))).reshape(B, cfg.n_slots, cfg.slot_width, T)
    m = net.domain_slot_mask[dom]
    h = (h * m[:, :, None, None]).reshape(B, cfg.n_slots * cfg.slot_width, T)
    z = torch.cat([h, x[:, cfg.n_slots:]], dim=1)
    z = TF.relu(net.stem(z, relu=False) + net.domain_emb(dom)[:, :, None])
    for b in net.blocks:
        z = b(z)
    return z


class SSLHead(nn.Module):
    def __init__(self, emb, cmap, n_slots=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(emb, 128), nn.ReLU(), nn.Linear(128, 64))
        self.dec = nn.Sequential(nn.Conv1d(cmap, 64, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2), nn.Conv1d(64, 48, 3, padding=1), nn.ReLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv1d(48, 32, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                                 nn.Conv1d(32, n_slots, 5, padding=2))


def gpu_view(x, dom, g, mask_span=True):
    """x [B, 10, T] -> augmented copy + mask [B, T] (True = masked)."""
    B, C, T = x.shape
    r = lambda *s: torch.rand(*s, device=x.device, generator=g)
    v = x.clone()
    sl = v[:, :3]
    sl = sl * (torch.exp((r(B, 1, 1) - 0.5) * 1.4)) * torch.exp((r(B, 3, 1) - 0.5) * 0.4)
    tt = torch.arange(T, device=x.device)[None, None].float()
    wander = (r(B, 1, 1) * 0.3) * torch.sin(2 * np.pi * (r(B, 1, 1) * 0.3 / 30.0) * tt + r(B, 1, 1) * 6.28)
    std = sl.std(-1, keepdim=True) + 1e-3
    sl = sl + wander * std + torch.randn(sl.shape, device=x.device, generator=g) * std * r(B, 1, 1) * 0.3
    drop = (r(B, 3, 1) < 0.1) & (dom[:, None, None] == 0)
    drop[:, 0] = False
    sl = sl * (~drop)
    v[:, :3] = sl
    v[:, 3:] = v[:, 3:] * torch.exp((r(B, 1, 1) - 0.5) * 0.4) + torch.randn(v[:, 3:].shape, device=x.device, generator=g) * 0.01
    sh = (r(B) * 61).long() - 30
    idx = (tt[0, 0].long()[None] - sh[:, None]) % T
    v = torch.gather(v, 2, idx[:, None].expand(B, C, T))
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    if mask_span:
        ln = (30 + r(B) * 60).long(); st = (r(B) * (T - ln)).long()
        mask = (tt[0, 0][None] >= st[:, None]) & (tt[0, 0][None] < (st + ln)[:, None])
        v = v * (~mask)[:, None]
    return v, mask


def pretrain(sec=600, max_steps=40000, bs=384, bs_phone=128, lr=2e-3, seed=0, w_rec=1.0):
    torch.manual_seed(seed)
    info = json.load(open(WORK / 'pub_info.json'))
    log_('disjoint', assert_pretrain_disjoint(), 'device', DEV)
    ac = np.load(WORK / 'pub_ac.f16.npy', mmap_mode='r')
    ac = torch.from_numpy(np.ascontiguousarray(ac[:2000] if SMOKE else ac)).to(DEV)            # [N, 300, 2] f16
    if SMOKE:
        bs, bs_phone = 32, 16
    ph = torch.from_numpy(np.concatenate([np.load(os.path.join(DATA_DIR, d, 'x.f16.npy')) for d in ('butppg', 'mths')])).to(DEV)
    ph = torch.nan_to_num(ph)
    scale = float(info['scale'])
    net = OxyNet(OxyCfg()).to(DEV)
    with torch.no_grad():
        cmap = encode_map(net, torch.zeros(2, N_CH, 300, device=DEV), torch.zeros(2, dtype=torch.long, device=DEV)).shape[1]
    head = SSLHead(net.cfg.emb, cmap).to(DEV)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=DEV.type == 'cuda')
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    t0 = time.time(); step = 0; curve = []
    N, NP = ac.shape[0], ph.shape[0]
    while step < max_steps and time.time() - t0 < sec:
        frac = min(1.0, (time.time() - t0) / sec)
        for gr in opt.param_groups:
            gr['lr'] = lr * (min(1.0, step / 300)) * 0.5 * (1 + np.cos(np.pi * frac))
        i = torch.randint(0, N, (bs,), device=DEV, generator=g)
        a = ac[i].float().transpose(1, 2) / scale                                          # [bs, 2, 300]
        domc = (torch.rand(bs, device=DEV, generator=g) >= 0.5).long()
        xc = torch.zeros(bs, N_CH, 300, device=DEV)
        xc[:, 0:2] = a
        mix = torch.rand(bs, 1, 1, device=DEV, generator=g) * 0.6 + 0.2
        xc[:, 2:3] = (mix * a[:, 0:1] + (1 - mix) * a[:, 1:2]) * (domc == 0)[:, None, None]
        j = torch.randint(0, NP, (bs_phone,), device=DEV, generator=g)
        x = torch.cat([xc, ph[j].float()])
        dom = torch.cat([domc, torch.zeros(bs_phone, dtype=torch.long, device=DEV)])
        v1, m1 = gpu_view(x, dom, g, True)
        v2, _ = gpu_view(x, dom, g, False)
        with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == 'cuda'):
            z1 = encode_map(net, v1, dom); z2 = encode_map(net, v2, dom)
            e1 = TF.relu(net.fc_emb(torch.cat([z1.mean(-1), z1.amax(-1)], 1)))
            e2 = TF.relu(net.fc_emb(torch.cat([z2.mean(-1), z2.amax(-1)], 1)))
            rec = head.dec(z1)
        p1 = TF.normalize(head.proj(e1.float()), dim=-1); p2 = TF.normalize(head.proj(e2.float()), dim=-1)
        from ..ssl_pretrain import nt_xent
        l_c, acc = nt_xent(p1, p2)
        tgt = x[:, :3]
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, keepdim=True) + 1e-3)
        rec = TF.interpolate(rec.float(), size=300, mode='linear', align_corners=False)
        slot_ok = (net.domain_slot_mask[dom] > 0) & (x[:, :3].abs().amax(-1) > 0)
        mm = m1[:, None, :] & slot_ok[:, :, None]
        l_r = ((rec - tgt) ** 2 * mm).sum() / mm.sum().clamp(min=1)
        loss = l_c + w_rec * l_r
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(opt); scaler.update(); step += 1
        if step % 200 == 0:
            curve.append(dict(step=step, sec=round(time.time() - t0, 1), nt_xent=float(l_c), acc=float(acc), rec=float(l_r)))
            if step % 1000 == 0:
                log_('pretrain', curve[-1])
    net = net.cpu().eval()
    meta = dict(steps=step, sec=round(time.time() - t0, 1), bs=bs, bs_phone=bs_phone, lr=lr, w_rec=w_rec, sources=list(PUB_SRC) + ['butppg', 'mths'],
                contact_rows=int(N), phone_rows=int(NP), samples_seen=int(step * (bs + bs_phone)))
    torch.save(dict(cfg=net.cfg.to_dict(), state=net.state_dict(), ssl3=meta), PRE_CK)
    pd.DataFrame(curve).to_csv(WORK / 'pretrain_curve.csv', index=False)
    log_('pretrain done', meta)
    return meta


def load_pre():
    ck = torch.load(PRE_CK, map_location='cpu', weights_only=False)
    cfg = {k: (tuple(v) if k in ('widths', 'depths') else v) for k, v in ck['cfg'].items()}
    cfg['dropout'] = DROPOUT
    net = OxyNet(OxyCfg(**cfg))
    net.load_state_dict(ck['state'])
    return net, ck


# ------------------------------------------------------------------------------------------------ Hoffman tensors
def shard():
    d = os.path.join(DATA_DIR, 'hoffman')
    meta = pd.read_parquet(os.path.join(d, 'meta.parquet'))
    recs = pd.read_parquet(os.path.join(d, 'recs.parquet'))
    n = json.load(open(os.path.join(d, 'info.json')))['n_samples']
    series = np.memmap(os.path.join(d, 'series.f32'), dtype=np.float32, mode='r', shape=(n, 3))
    dtv = np.memmap(os.path.join(d, 'dt.f16'), dtype=np.float16, mode='r', shape=(n,))
    return meta, recs, series, dtv


def aug_draw(rec_idx, k):
    from . import flashgap as FG
    rng = np.random.default_rng(np.random.SeedSequence([AUG_TAG, AUG_SEED, int(rec_idx), int(k)]))
    p = FG.sample_params(rng)
    a, m = FG.optics_gain(p)
    p['a_c'] = [float(v) for v in a]; p['m_c'] = [float(v) for v in m]
    return p, rng


def version_rgb(rgb, version, rec_idx):
    if version == 'clean':
        return rgb, None
    if version.startswith('aug'):
        from . import flashgap as FG
        p, rng = aug_draw(rec_idx, int(version[3:]))
        return FG.flash_augment(rgb, p, rng), p
    from . import stress as S
    if version == 'X1':
        return S.degrade_x(rgb, 'X1', 0), None
    return S.degrade(rgb, int(version[1:]), 0), None


def versions():
    return ['clean'] + [f'aug{k}' for k in range(K_AUG)] + [vname(l) for l in EVAL if l != 0]


def _prep_one(v):
    torch.set_num_threads(1)
    from .. import features as FE
    from .oxyssl import gather
    out = WORK / f'hof_{v}.npz'
    if out.exists():
        return v, None
    meta, recs, series, dtv = shard()
    X = np.zeros((len(meta), N_CH, 300), np.float16); S_ = np.zeros((len(meta), 9), np.float32); params = []
    for ri, r in recs.iterrows():
        rows = np.nonzero(meta.rec.values == r.rec)[0]
        off, ln = int(r.offset), int(r.length)
        rgb = np.expm1(np.asarray(series[off:off + ln], np.float64))
        rgbv, p = version_rgb(rgb, v, ri)
        if p is not None:
            params.append(p)
        lrgb = FE.log_rgb(rgbv); dt = np.asarray(dtv[off:off + ln], np.float64)
        ends = meta.end_idx.values[rows] - off
        buf, dtb = gather(lrgb, dt, ends)
        _, _, absf = FE.windows_to_tensor(buf, dtb, None, None)
        base = FE.baseline_from_abs(absf[meta.in_baseline.values[rows].astype(bool)])
        x, s, _ = FE.windows_to_tensor(buf, dtb, np.repeat(base[None], len(ends), 0), None)
        X[rows] = np.nan_to_num(x).astype(np.float16); S_[rows] = np.nan_to_num(s)
    np.savez(out, X=X, S=S_)
    return v, params


def prep(jobs=4):
    from multiprocessing import Pool
    from .stress import D_RHO_MM, AMBIENT, LEVELS as SL
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(jobs) as pool:
        res = pool.map(_prep_one, versions())
    params = [p for _, ps in res if ps for p in ps]
    lv = {(float(D_RHO_MM[l]), float(AMBIENT[l])) for l in SL if l != 0}
    assert all((float(p['d_rho_mm']), float(p['leak'])) not in lv for p in params)
    if params:
        json.dump(params, open(WORK / 'aug_params.json', 'w'))
    log_(f'prep done {time.time() - t0:.0f}s')


# ------------------------------------------------------------------------------------------------ fine-tuning (GPU)
class Bank:
    def __init__(self):
        vs = versions()
        z = [np.load(WORK / f'hof_{v}.npz') for v in vs]
        self.vi = {v: i for i, v in enumerate(vs)}
        self.X = torch.from_numpy(np.stack([a['X'] for a in z])).to(DEV)                # [V, N, 10, 300] f16
        self.S = np.stack([a['S'] for a in z])                                           # [V, N, 9]


class Data:
    def __init__(self, bank, meta, vers, rows, labels, s_scale):
        V = np.array([bank.vi[v] for v in vers])
        self.v = np.repeat(V, len(rows)); self.idx = np.tile(rows, len(vers))
        self.S = torch.from_numpy((bank.S[self.v, self.idx] / s_scale).astype(np.float32)).to(DEV)
        rec = meta.rec.values; inb = meta.in_baseline.values.astype(bool)
        keys = [(v, r) for v in V for r in pd.unique(rec[rows])]
        gid = {k: i for i, k in enumerate(keys)}
        self.g = torch.from_numpy(np.array([gid[(v, r)] for v, r in zip(self.v, rec[self.idx])], np.int64)).to(DEV)
        self.gb_v = torch.from_numpy(np.array([v for v, _ in keys], np.int64)).to(DEV)
        self.gb_r = torch.from_numpy(np.stack([np.nonzero((rec == r) & inb)[0][:BASE_ROWS] for _, r in keys]).astype(np.int64)).to(DEV)
        self.is_clean = np.isin(self.v, [bank.vi['clean']])
        L = labels
        t = lambda a: torch.from_numpy(np.asarray(a)).to(DEV)
        self.y_np = L['low'][self.idx]
        self.y = t(self.y_np.astype(np.float32)); self.spo2 = t(np.nan_to_num(L['spo2'][self.idx], nan=93.0).astype(np.float32))
        self.hsp = t(L['has_spo2'][self.idx] & np.isfinite(L['spo2'][self.idx])); self.hhr = t(L['has_hr'][self.idx] & np.isfinite(L['hr'][self.idx]))
        self.hrt = t(hr_target(np.nan_to_num(L['hr'][self.idx], nan=80.0)).astype(np.float32))
        self.vt = t(self.v); self.it = t(self.idx)
        self.bank = bank

    def x(self, b):
        return self.bank.X[self.vt[b], self.it[b]].float()

    @torch.no_grad()
    def emb_base(self, net):
        was = net.training; net.eval()
        G, K = self.gb_r.shape
        xb = self.bank.X[self.gb_v[:, None].expand(G, K).reshape(-1), self.gb_r.reshape(-1)].float()
        E = []
        for a in range(0, len(xb), 2048):
            E.append(net(xb[a:a + 2048])['emb'])
        net.train(was)
        return torch.cat(E).reshape(G, K, -1).mean(1)


def labels_of(meta, perm=None):
    L = dict(low=meta.low.values.astype(np.float32), spo2=meta.spo2_ref.values.astype(np.float32), has_spo2=meta.has_spo2.values.astype(bool),
             hr=meta.hr_ref.values.astype(np.float32), has_hr=meta.has_hr.values.astype(bool))
    if perm is not None:
        src, dst = perm
        for k in L:
            v = L[k].copy(); v[src] = L[k][dst]; L[k] = v
    return L


def hr_target(hr):
    b = np.clip(np.round(hr - HR_LO), 0, HR_BINS - 1)
    ar = np.arange(HR_BINS)[None]
    t = np.exp(-0.5 * ((ar - b[:, None]) / 1.5) ** 2)
    return t / t.sum(1, keepdims=True)


def fresh_net(pre_state, cfg, seed):
    torch.manual_seed(seed)
    net = OxyNet(cfg)
    st = {k: v for k, v in pre_state.items() if k.split('.')[0] not in HEADS}
    missing, _ = net.load_state_dict(st, strict=False)
    assert all(k.split('.')[0] in HEADS for k in missing)
    return net.to(DEV)


def train(pre, D: Data, epochs_ckpt, seed, on_ckpt=None):
    pre_state, cfg = pre
    net = fresh_net(pre_state, cfg, seed)
    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    enc = [p for n, p in net.named_parameters() if n.split('.')[0] not in HEADS]
    hd = [p for n, p in net.named_parameters() if n.split('.')[0] in HEADS]
    opt = torch.optim.AdamW([dict(params=enc, lr=LR_ENC, base=LR_ENC), dict(params=hd, lr=LR_HEAD, base=LR_HEAD)], weight_decay=WD)
    n = len(D.y_np)
    npos = max(D.y_np.sum(), 1.0); pw = torch.tensor((n - npos) / npos, dtype=torch.float32, device=DEV)
    E = max(epochs_ckpt); steps_ep = int(np.ceil(n / BS)); total = E * steps_ep; step = 0
    for ep in range(1, E + 1):
        net.train()
        order = torch.randperm(n, generator=gen).to(DEV)
        for a in range(0, n, BS):
            if step % EB_REFRESH == 0:
                EB = D.emb_base(net)
            b = order[a:a + BS]
            if len(b) < 8:
                continue
            for gr in opt.param_groups:
                gr['lr'] = gr['base'] * 0.5 * (1 + np.cos(np.pi * step / total))
            o = net(D.x(b), None, D.S[b], EB[D.g[b]])
            y = D.y[b]
            l_low = TF.binary_cross_entropy_with_logits(o['lowox_logit'], y, pos_weight=pw)
            m = D.hsp[b]
            l_sp = (0.5 * (torch.exp(-o['spo2_logvar'][m]) * ((D.spo2[b][m] - o['spo2'][m]) / SPO2_SCALE) ** 2 + o['spo2_logvar'][m])).mean() if m.any() else 0.0
            mh = D.hhr[b]
            l_hr = -(D.hrt[b][mh] * TF.log_softmax(o['hr_logits'][mh], -1)).sum(-1).mean() if mh.any() else 0.0
            corr = ((o['lowox_logit'].detach() > 0).float() == y).float()
            l_sqi = TF.binary_cross_entropy_with_logits(o['sqi_logit'], corr)
            loss = W_LOW * l_low + W_SPO2 * l_sp + W_HR * l_hr + W_SQI * l_sqi
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        if ep in epochs_ckpt and on_ckpt is not None:
            on_ckpt(ep, net)
    return net


@torch.no_grad()
def predict(net, D: Data):
    net.eval()
    EB = D.emb_base(net)
    out = {k: [] for k in ('score', 'conf', 'spo2')}
    for a in range(0, len(D.y_np), 2048):
        b = torch.arange(a, min(a + 2048, len(D.y_np)), device=DEV)
        o = net(D.x(b), None, D.S[b], EB[D.g[b]])
        out['score'].append(torch.sigmoid(o['lowox_logit'])); out['conf'].append(torch.sigmoid(o['sqi_logit'])); out['spo2'].append(o['spo2'])
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def tm(t, y, pred):
    return M.time_matched(t, y, pred)['tm_bacc']


def bacc(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    if y.min() == y.max():
        return np.nan
    return 0.5 * (((pred == 1) & (y == 1)).sum() / (y == 1).sum() + ((pred == 0) & (y == 0)).sum() / (y == 0).sum())


def side_scale(bank, rows):
    s = bank.S[bank.vi['clean']][rows]
    sc = np.maximum(s.std(0), 0.05).astype(np.float32); sc[8] = 1.0
    return sc


def fit_fold(pre, bank, meta, test_sid, seed, labels, train_vers):
    subj = meta.subject.values
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    tr_people = sorted(set(subj[lab]) - {test_sid})
    tr_rows = np.nonzero(lab & (subj != test_sid))[0]
    t_all = meta.t_start.values.astype(float)
    oof = {}
    for ip in tr_people:
        itr = tr_rows[subj[tr_rows] != ip]; ite = tr_rows[subj[tr_rows] == ip]
        sc = side_scale(bank, itr)
        Dtr = Data(bank, meta, train_vers, itr, labels, sc)
        Dte = Data(bank, meta, train_vers, ite, labels, sc)

        def keep(ep, net):
            pr = predict(net, Dte)
            oof.setdefault(ep, []).append(dict(pr, y=Dte.y_np, t=t_all[Dte.idx], clean=Dte.is_clean))
        train(pre, Dtr, EPOCH_GRID, seed, keep)
    res = []
    for ep, parts in oof.items():
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        c = d['clean']; pred = d['score'] > 0.5
        obj = 0.5 * tm(d['t'][c], d['y'][c], pred[c]) + 0.5 * tm(d['t'][~c], d['y'][~c], pred[~c]) if (~c).any() else tm(d['t'], d['y'], pred)
        res.append((ep, obj, d))
    res.sort(key=lambda r: (-np.round(r[1], 4), r[0]))
    ep, obj, d = res[0]
    pred = d['score'] > 0.5
    best_q = None
    for q in Q_GRID:
        thr = float(np.quantile(d['conf'], q)) if q > 0 else -np.inf
        ans = d['conf'] >= thr
        cand = (np.round(bacc(d['y'][ans], pred[ans]), 4), ans.mean())
        if best_q is None or cand > best_q[0]:
            best_q = (cand, q, thr)
    _, q, thr = best_q
    sel = dict(test=test_sid, epochs=int(ep), inner_obj=float(obj), q=q, thr=float(thr), inner_answered_bacc=float(best_q[0][0]),
               inner_cov=float(best_q[0][1]), grid={str(k): round(float(o), 4) for k, o, _ in res})
    sc = side_scale(bank, tr_rows)
    net = train(pre, Data(bank, meta, train_vers, tr_rows, labels, sc), (ep,), seed)
    te_rows = np.nonzero(lab & (subj == test_sid))[0]
    out = {}
    for lv in EVAL:
        v = 'clean' if lv == 0 else vname(lv)
        out[vname(lv)] = dict(rows=te_rows, **predict(net, Data(bank, meta, [v], te_rows, labels_of(meta), sc)))
    log_(f'  fold {test_sid}: ep {ep} inner {obj:.3f} q {q} thr {thr:.3f} grid {sel["grid"]}')
    return sel, out


def score_frame(meta, rows, score, pred, answer, conf, spo2):
    from .evalkit import eval_windows
    ref = eval_windows()
    m = meta.iloc[rows]
    d = pd.DataFrame(dict(rec=m.rec.str.replace('hof_', '', regex=False).values, t=m.t_start.astype(int).values, low=m.low.astype(int).values,
                          score=score, pred=pred.astype(int), answer=answer.astype(bool), conf=conf, spo2_hat=spo2))
    out = ref.merge(d, on=['rec', 't'], how='left', validate='one_to_one')
    assert out.score.notna().all(), 'every evaluated window must be scored'
    assert (out.y.values == out.low.values).all()
    return out[['rec', 't', 'sid', 'y_spo2', 'y', 'score', 'pred', 'answer', 'conf', 'spo2_hat']]


def pre_state():
    net, ck = load_pre()
    return net.state_dict(), net.cfg


def run_variant(name, seed=0, perm_draw=None, aug=True, write=True):
    pre = pre_state()
    bank = Bank()
    meta = shard()[0]
    lab = meta.has_low.values.astype(bool) & meta['eval'].values.astype(bool)
    people = sorted(set(meta.subject.values[lab]))
    assert not (set(people) & set(pd.read_parquet(WORK / 'pub_meta.parquet').subject.astype(str)))
    train_vers = ['clean'] + ([f'aug{k}' for k in range(K_AUG)] if aug else [])
    sels, raw = [], {}
    t0 = time.time()
    for sid in (people[:1] if SMOKE else people):
        if perm_draw is not None:
            src = np.nonzero(lab & (meta.subject.values != sid))[0]
            rng = np.random.default_rng(np.random.SeedSequence([AUG_TAG, 7, int(perm_draw), people.index(sid)]))
            labels = labels_of(meta, (src, rng.permutation(src)))
        else:
            labels = labels_of(meta)
        sel, out = fit_fold(pre, bank, meta, sid, seed, labels, train_vers)
        sels.append(sel)
        for k, v in out.items():
            raw.setdefault(k, []).append(dict(v, thr=sel['thr']))
    frames = {}
    for k, parts in raw.items():
        rows = np.concatenate([p['rows'] for p in parts]); sc = np.concatenate([p['score'] for p in parts])
        cf = np.concatenate([p['conf'] for p in parts]); sp = np.concatenate([p['spo2'] for p in parts])
        ans = np.concatenate([p['conf'] >= p['thr'] for p in parts])
        frames[k] = pd.DataFrame(dict(score=sc, ans=ans)) if SMOKE else score_frame(meta, rows, sc, sc > 0.5, ans, cf, sp)
    if write and not SMOKE:
        SCORES.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
        for k, df in frames.items():
            if perm_draw is not None and k != 'L0':
                continue
            df.to_csv(SCORES / f'{name}__{k}.csv', index=False, float_format='%.6g')
        json.dump(sels, open(RAW / f'{name}_selection.json', 'w'), indent=1)
    l0 = frames['L0']
    if SMOKE:
        log_('smoke ok', {k: len(v) for k, v in frames.items()}, sels)
        return sels, frames
    log_(f'{name}: L0 TM {tm(l0.t.values, l0.y.values, l0.pred.values):.3f} ({time.time() - t0:.0f}s)')
    return sels, frames


def run(level: int = 0, seed: int = 0) -> pd.DataFrame:
    """Contract entry point: out-of-fold score frame of seed `seed` at `level` (int or 'X1')."""
    _, frames = run_variant(f'{NAME}_s{seed}', seed=seed, write=False)
    return frames[vname(level)]


def ensemble(seeds, name=NAME):
    meta = shard()[0]
    r1 = meta.drop_duplicates('rec')
    sid_of = pd.Series(r1.subject.values, index=r1.rec.str.replace('hof_', '', regex=False).values)
    thr = {}
    for s in seeds:
        sel = json.load(open(RAW / f'{name}_s{s}_selection.json'))
        thr[s] = {x['test']: x['thr'] for x in sel}
    for lv in EVAL:
        k = vname(lv)
        fs = [pd.read_csv(SCORES / f'{name}_s{s}__{k}.csv') for s in seeds]
        o = fs[0].copy()
        o['score'] = np.mean([f.score.values for f in fs], 0); o['conf'] = np.mean([f.conf.values for f in fs], 0)
        o['spo2_hat'] = np.mean([f.spo2_hat.values for f in fs], 0)
        subj = o.rec.map(sid_of).values
        t_mean = np.array([np.mean([max(thr[s][x], 0.0) for s in seeds]) for x in subj])
        o['pred'] = (o.score > 0.5).astype(int); o['answer'] = o.conf.values >= t_mean
        o.to_csv(SCORES / f'{name}__{k}.csv', index=False, float_format='%.6g')
        log_(f'ensemble {k}: TM {tm(o.t.values, o.y.values, o.pred.values):.3f} answered {o.answer.mean():.2f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('bundle')
    p = sp.add_parser('pretrain'); p.add_argument('--sec', type=float, default=600); p.add_argument('--steps', type=int, default=40000)
    p = sp.add_parser('prep'); p.add_argument('--jobs', type=int, default=4)
    p = sp.add_parser('fit'); p.add_argument('--variant', nargs='+', default=['s0'])
    p = sp.add_parser('ensemble'); p.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    if a.cmd == 'bundle':
        bundle()
    elif a.cmd == 'pretrain':
        pretrain(a.sec, a.steps)
    elif a.cmd == 'prep':
        prep(a.jobs)
    elif a.cmd == 'fit':
        for v in a.variant:
            if v.startswith('s'):
                run_variant(f'{NAME}_{v}', seed=int(v[1:]))
            elif v.startswith('n'):
                d = int(v[1:])
                run_variant(f'{NAME}_null' if d == 0 else f'{NAME}_d{d}_null', seed=0, perm_draw=d)
            elif v == 'noaug':
                run_variant(f'{NAME}_noaug', seed=0, aug=False)
    elif a.cmd == 'ensemble':
        ensemble(a.seeds)


if __name__ == '__main__':
    main()
