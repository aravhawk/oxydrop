"""OxySeq (round-2 candidate R2D): an end-to-end trained causal sequence model of hypoxia over the window stream of one recording.

Inputs (strictly causal, no elapsed time): per 10 s window (2 s hop) the 19 chain v1 FUSION deltas vs the person's label-free baseline minute and the
per-window spectral pulse SNR of R, G, B (oxytrack.features: causal per-window features computed from the per-frame RGB). Scaling constants (delta SD,
log-SNR mean/SD) come from the training people of the fold.

Network (one module, trained jointly, plain ops only: Linear, sigmoid, ReLU, one GRU layer):
  lin      per-window linear hypoxia logit from the raw inputs (auxiliary head; it drives the adaptation gate)
  TTA      label-free per-person test-time adaptation of the 19 deltas: b_k = (1 - a_k) b_{k-1} + a_k x_k, a_k = r * g_k, r = 0.5 sigmoid(rho)
           learned per channel (the adaptation rate is a trained parameter), g_k = 1 inside the label-free first minute or when the per-window
           head calls the window clearly normal (sigmoid(lin) < 0.10), else 0 (offsets frozen, so a desaturation cannot be absorbed as drift).
           The GRU sees x_k - b_{k-1}. Variant oxyseq_notta fixes r = 0.
  core     Linear(22 -> H) + ReLU -> GRU(H) (unidirectional, so causal) -> Linear(H + 22 -> 3) heads:
           hypoxia logit (score = sigmoid), SpO2 desaturation vs baseline (spo2_hat = mu_b + 5 dhat), quality logit (abstain head, trained to
           predict whether the hypoxia head is right on that window).
Clock-leak guards: elapsed time is never an input; training uses random crops (random length 32..128 windows at random offsets, GRU state reset at
the crop start), so a hidden state that counts windows cannot learn the shared protocol's timeline; blank control (oxyseq_blank__L0) and globally
permuted-label nulls (oxyseq_null__L0, 3 draws) are run through the same pipeline.
Augmentation prior (declared; independent torch generator seeded 1000 + seed; never stress.py (level, seed) pairs, never degrade_x): with
probability 0.5 per training crop a signal-retention factor f ~ U(0.05, 1) is drawn; deltas become f x + sqrt(1 - f^2) e (e AR(1) with phi 0.8 in
SD units, the 5-window overlap) and log10 SNR drops by 2 log10(1/f) (the pulse power falls as f^2). This is a feature-space model of a weaker,
noisier pulse (distant flash, low light), not the harness physics.
Nested protocol: outer LOSO by person (both hands); inner leave-one-training-person-out (splits.inner_group_folds k=5) picks the number of training
steps (checkpoints of one run) by TM bal. acc. at the nested threshold (baselines.common.choose_thr, tuned), the hypoxia threshold, and the abstain
threshold on the quality probability (smallest tau whose answered inner-OOF windows reach acc and bal. acc. >= 0.90, else the best answered bal.
acc. with coverage >= 0.20) and the 80 % SpO2 band half-width. Final: 3 seeds trained on the 5 outer-training people with the chosen steps; the
shipped score is the 3-seed ensemble (mean probabilities); seed-mean TM reported in the log. Not refit per stress level.
Run: cd python && python -m oxydrop_ml.photon.oxyseq
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

from .. import metrics as M
from .. import splits as S
from ..baselines import common as BC
from . import oxytrack as OT

torch.set_num_threads(2)
RES, SCORES = OT.RES, OT.SCORES
THR, WIN, CAL = OT.THR, OT.WIN, OT.CAL
ND, NS = 19, 3
H = 16
GATE_P = 0.10
CKPTS = (25, 50, 100, 200)        # v2: v1 grid (100, 200, 400) had the inner choice at its lower edge (100) in 6/6 folds
SEEDS = (0, 1, 2)
LR, WD, BATCH = 3e-3, 1e-4, 16
CROP = (32, 128)
AUG_P, AUG_F, AUG_PHI = 0.5, (0.05, 1.0), 0.8
Z80 = OT.Z80


class OxySeq(nn.Module):
    def __init__(self, h=H, tta=True):
        super().__init__()
        self.tta = tta
        self.lin = nn.Linear(ND + NS, 1)
        self.rho = nn.Parameter(torch.full((ND,), -3.0))
        self.inp = nn.Linear(ND + NS, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h + ND + NS, 3)

    def rate(self):
        return 0.5 * torch.sigmoid(self.rho) if self.tta else torch.zeros_like(self.rho)

    def adapt(self, xd, gate):
        """Causal gated EMA offsets b_{k-1} (B, L, ND); b_{-1} = 0. Closed form of the linear recurrence in float64."""
        a = (self.rate()[None, None, :] * gate[..., None]).double()
        logA = torch.cumsum(torch.log1p(-a), 1)
        b = torch.exp(logA) * torch.cumsum(a * xd.double() * torch.exp(-logA), 1)
        b = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], 1)
        return b.float()

    def forward(self, x, base):
        lin = self.lin(x)[..., 0]
        xd, xs = x[..., :ND], x[..., ND:]
        if self.tta:
            g = torch.maximum(base, (torch.sigmoid(lin) < GATE_P).float()).detach()
            xd = xd - self.adapt(xd, g)
        xt = torch.cat([xd, xs], -1)
        h, _ = self.gru(F.relu(self.inp(xt)))
        o = self.head(torch.cat([h, xt], -1))
        return dict(logit=o[..., 0], dhat=o[..., 1], qlogit=o[..., 2], lin=lin)


# ------------------------------------------------------------------------------------------------ data
def scaler(train):
    D = np.vstack([r['D'][OT.eval_mask(r)] for r in train])
    ls = np.log10(np.vstack([r['snr'] for r in train]))
    return dict(sd=D.std(0) + 1e-9, lmu=ls.mean(0), lsd=ls.std(0) + 1e-9,
                mu_b=float(np.mean([r['sbase'] for r in train])), snr_med=np.median(np.vstack([r['snr'] for r in train]), 0))


def inputs(sc, D, snr):
    xd = np.clip(np.asarray(D, float) / sc['sd'], -8, 8)
    xs = (np.log10(np.asarray(snr, float)) - sc['lmu']) / sc['lsd']
    return np.concatenate([xd, xs], 1).astype(np.float32)


def tensors(sc, r, blank=False):
    n = len(r['t'])
    if blank:
        D = np.zeros((n, ND)); snr = np.tile(sc['snr_med'], (n, 1))
    else:
        D, snr = r['D'], r['snr']
    m = OT.eval_mask(r)
    y = np.where(m, (np.nan_to_num(r['y'], nan=100.0) < THR).astype(np.float32), 0.0)
    d = np.where(m, np.nan_to_num(r['y'] - r['sbase']) / 5.0, 0.0) if np.isfinite(r.get('sbase', np.nan)) else np.zeros(n)
    return dict(x=inputs(sc, D, snr), base=(r['t'] + WIN <= CAL).astype(np.float32), y=y.astype(np.float32), d=d.astype(np.float32),
                m=m.astype(np.float32))


def augment(x, gen):
    """Declared training prior (weaker, noisier pulse). x (L, 22) numpy; gen numpy Generator independent of stress.py."""
    if gen.random() >= AUG_P:
        return x
    f = gen.uniform(*AUG_F)
    L = len(x)
    w = gen.standard_normal((L, ND))
    w[1:] *= np.sqrt(1 - AUG_PHI ** 2)
    e = lfilter([1.0], [1.0, -AUG_PHI], w, axis=0).astype(np.float32)      # stationary AR(1), unit variance
    x = x.copy()
    x[:, :ND] = f * x[:, :ND] + np.sqrt(1 - f * f) * e
    x[:, ND:] = x[:, ND:] - 2.0 * np.log10(1.0 / f)
    return x


def batch(data, gen, aug=True):
    L = int(gen.integers(CROP[0], CROP[1] + 1))
    lens = np.array([len(d['x']) for d in data], float)
    idx = gen.choice(len(data), BATCH, p=lens / lens.sum())
    out = {k: [] for k in ('x', 'base', 'y', 'd', 'm')}
    for i in idx:
        d = data[i]
        n = len(d['x']); LL = min(L, n)
        s = int(gen.integers(0, n - LL + 1))
        for k in out:
            v = d[k][s:s + LL]
            if k == 'x' and aug:
                v = augment(v, gen)
            if LL < L:
                v = np.concatenate([v, np.zeros((L - LL,) + v.shape[1:], v.dtype)])
            out[k].append(v)
    return {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}


def train(data, seed, ckpts=None, tta=True, log=None):
    """Train one network on a list of tensor dicts (training people only). Returns {step: state_dict} at the checkpoints."""
    ckpts = CKPTS if ckpts is None else ckpts
    torch.manual_seed(seed)
    gen = np.random.default_rng(1000 + seed)
    net = OxySeq(tta=tta)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    y = np.concatenate([d['y'][d['m'] > 0] for d in data])
    pw = torch.tensor(float((1 - y).sum() / max(y.sum(), 1)))
    out = {}
    for step in range(1, max(ckpts) + 1):
        b = batch(data, gen)
        o = net(b['x'], b['base'])
        m = b['m']; nm = m.sum().clamp_min(1)
        l_h = (F.binary_cross_entropy_with_logits(o['logit'], b['y'], pos_weight=pw, reduction='none') * m).sum() / nm
        l_l = (F.binary_cross_entropy_with_logits(o['lin'], b['y'], pos_weight=pw, reduction='none') * m).sum() / nm
        l_d = (F.smooth_l1_loss(o['dhat'], b['d'], reduction='none') * m).sum() / nm
        corr = ((o['logit'] > 0).float() == b['y']).float().detach()
        l_q = (F.binary_cross_entropy_with_logits(o['qlogit'], corr, reduction='none') * m).sum() / nm
        loss = l_h + 0.5 * l_l + 0.2 * l_d + 0.2 * l_q
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpts:
            out[step] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return out


def predict(state, td, tta=True):
    net = OxySeq(tta=tta); net.load_state_dict(state); net.eval()
    with torch.no_grad():
        o = net(torch.from_numpy(td['x'])[None], torch.from_numpy(td['base'])[None])
    return dict(score=torch.sigmoid(o['logit'][0]).numpy().astype(float), dhat=o['dhat'][0].numpy().astype(float) * 5.0,
                q=torch.sigmoid(o['qlogit'][0]).numpy().astype(float))


def choose_tau(q, correct, y, pred):
    """Abstain threshold on the quality probability: smallest tau (on inner-OOF quantiles) whose answered windows reach acc and bal. acc. >= 0.90."""
    best = (-1.0, 0.0)
    for tau in np.unique(np.quantile(q, np.arange(0.0, 0.81, 0.02))):
        a = q >= tau
        if a.sum() < 20 or len(np.unique(y[a])) < 2:
            continue
        acc = correct[a].mean()
        bacc = 0.5 * (pred[a][y[a] == 1].mean() + (1 - pred[a][y[a] == 0]).mean())
        if acc >= BC.BAND_TARGET and bacc >= BC.BAND_TARGET:
            return float(tau)
        if a.mean() >= BC.BAND_MIN_COVERAGE and bacc > best[0]:
            best = (bacc, float(tau))
    return best[1]


# ------------------------------------------------------------------------------------------------ nested LOSO
def permute_labels(tr_recs, draw):
    """Null: SpO2 labels permuted globally across all training eval windows (hypoxia label and desaturation move together)."""
    rng = np.random.default_rng(5000 + draw)
    ms = [OT.eval_mask(r) for r in tr_recs]
    allv = np.concatenate([r['y'][m] - r['sbase'] for r, m in zip(tr_recs, ms)])
    ally = np.concatenate([r['y'][m] for r, m in zip(tr_recs, ms)])
    p = rng.permutation(len(ally))
    ally, allv = ally[p], allv[p]
    out, i = [], 0
    for r, m in zip(tr_recs, ms):
        n = int(m.sum()); r2 = dict(r); yy = r['y'].copy()
        yy[m] = ally[i:i + n]; r2['y'] = yy; r2['sbase'] = float(np.mean(ally[i:i + n] - allv[i:i + n]))
        out.append(r2); i += n
    return out


def nested(recs, tta=True, seeds=None, null_draw=None, log=print):
    seeds = SEEDS if seeds is None else seeds
    sids = sorted({r['sid'] for r in recs.values()})
    folds = []
    for p in sids:
        t0 = time.time()
        tr = [r for r in recs.values() if r['sid'] != p]
        if null_draw is not None:
            tr = permute_labels(tr, null_draw)
        rows_sid = np.concatenate([np.full(OT.eval_mask(r).sum(), r['sid']) for r in tr])
        rows_y = np.concatenate([(r['y'][OT.eval_mask(r)] < THR).astype(int) for r in tr])
        inner = list(S.inner_group_folds(rows_sid, k=5, seed=0, balance=rows_y))
        oof = {c: [] for c in CKPTS}
        Y, T, SP = [], [], []
        for fi in inner:
            va = set(int(s) for s in fi.test_group)
            itr = [r for r in tr if r['sid'] not in va]; iva = [r for r in tr if r['sid'] in va]
            sc = scaler(itr)
            st = train([tensors(sc, r) for r in itr], seed=0, tta=tta)
            for r in iva:
                m = OT.eval_mask(r); td = tensors(sc, r)
                for c in CKPTS:
                    pr = predict(st[c], td, tta)
                    oof[c].append((pr['score'][m], pr['q'][m], sc['mu_b'] + pr['dhat'][m]))
                Y.append((r['y'][m] < THR).astype(int)); T.append(r['t'][m]); SP.append(r['y'][m])
        Y, T, SP = np.concatenate(Y), np.concatenate(T), np.concatenate(SP)
        ch = {}
        for c in CKPTS:
            s = np.concatenate([o[0] for o in oof[c]])
            ch[c] = BC.choose_thr(s, Y, T, 'tm_bacc', 'tune', 0.5)
        c = max(CKPTS, key=lambda k: (ch[k][1], -k))
        thr = float(ch[c][0])
        s = np.concatenate([o[0] for o in oof[c]]); q = np.concatenate([o[1] for o in oof[c]]); hat = np.concatenate([o[2] for o in oof[c]])
        pred = (s > thr).astype(int)
        tau = choose_tau(q, (pred == Y).astype(float), Y, pred)
        half = float(np.quantile(np.abs(SP - hat), 0.80))
        sc = scaler(tr)
        data = [tensors(sc, r) for r in tr]
        states = [train(data, seed=sd, ckpts=(c,), tta=tta)[c] for sd in seeds]
        rate = [float((0.5 * torch.sigmoid(st['rho'])).mean()) if tta else 0.0 for st in states]
        folds.append(dict(sid=p, steps=c, thr=thr, tau=tau, half=half, inner_tm=float(ch[c][1]), trace={k: float(v[1]) for k, v in ch.items()},
                          inner_cover=float((q >= tau).mean()), sc=sc, states=states, rate=rate))
        log(f'  fold {p}: steps={c} thr={thr:.3f} tau={tau:.3f} inner TM {ch[c][1]:.3f} (trace ' +
            ', '.join(f'{k}:{v[1]:.3f}' for k, v in ch.items()) + f') inner cover {folds[-1]["inner_cover"]:.2f} rate {np.mean(rate):.3f} ({time.time() - t0:.0f} s)', flush=True)
    return folds


def score(folds, recs, tta=True, blank=False, seeds_idx=None):
    """Contract DataFrame on the held-out people (ensemble of the fold's seeds, or the seed subset seeds_idx)."""
    rows = []
    for fo in folds:
        for r in [r for r in recs.values() if r['sid'] == fo['sid']]:
            td = tensors(fo['sc'], r, blank=blank)
            sts = fo['states'] if seeds_idx is None else [fo['states'][i] for i in seeds_idx]
            prs = [predict(st, td, tta) for st in sts]
            s = np.mean([p['score'] for p in prs], 0); q = np.mean([p['q'] for p in prs], 0); dh = np.mean([p['dhat'] for p in prs], 0)
            m = OT.eval_mask(r)
            hat = fo['sc']['mu_b'] + dh[m]
            rows.append(pd.DataFrame(dict(rec=r['rec'], t=r['t'][m], sid=r['sid'], y_spo2=r['y'][m], y=(r['y'][m] < THR).astype(int), score=s[m],
                                          pred=(s[m] > fo['thr']).astype(int), answer=q[m] >= fo['tau'], spo2_hat=hat, spo2_lo=hat - fo['half'],
                                          spo2_hi=hat + fo['half'], conf=q[m])))
    return pd.concat(rows, ignore_index=True).sort_values(['sid', 'rec', 't']).reset_index(drop=True)


_FOLDS = {}


def feats(level=0):
    """Cached causal features (oxytrack.features cache, stress seed 0) loaded directly, so a remote worker needs no raw recordings."""
    tag = OT.level_tag(level)
    ps = sorted(OT.CDIR.glob(f'feat_{tag}_s0_*.npz'))
    if len(ps) != 12:
        return OT.features(level, 0)
    out = {}
    for p in ps:
        z = np.load(p)
        out[p.stem.split('_')[-1]] = dict(t=z['t'], D=z['D'], snr=z['snr'])
    return out


def run(level=0, seed=0, model='oxyseq'):
    """Contract entry point: out-of-fold ensemble scores at stress `level` (models fit once on clean training people)."""
    recs0 = OT.rec_table(feats(0), OT.window_table())
    if model not in _FOLDS:
        _FOLDS[model] = nested(recs0, tta=(model != 'oxyseq_notta'))
    return score(_FOLDS[model], OT.rec_table(feats(level) if seed == 0 else OT.features(level, seed), OT.window_table()), tta=(model != 'oxyseq_notta'))


# ------------------------------------------------------------------------------------------------ main
def _f(x, nd=3):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def write_log(st):
    L = ['# OxySeq log (round-2 candidate R2D)', '',
         'Generated by `python -m oxydrop_ml.photon.oxyseq`. Out-of-fold numbers (outer LOSO by person, both hands); every choice nested on the 5 '
         'training people (inner leave-one-training-person-out, seed 0 for the inner runs). Fit on clean training people plus the declared feature-space '
         'augmentation prior (module docstring); not refit per stress level. Stress levels are simulations (stress.py), applied to the held-out person.', '',
         f'Architecture: per-window linear gate head + gated learned-rate TTA of the 19 deltas + Linear(22->{H}) ReLU + GRU({H}) + Linear({H}+22->3) '
         f'heads (hypoxia, SpO2 desaturation, quality/abstain). Training: Adam lr {LR}, wd {WD}, batch {BATCH} random crops of {CROP[0]}..{CROP[1]} '
         f'windows, steps chosen from {CKPTS} by inner CV; 3 seeds {SEEDS} for the final fit; shipped score = 3-seed ensemble.', '',
         '## Disclosures', '',
         '* Designed after round-1 outer results were seen (ROUND2.md, Amendment 7). The configuration (architecture, crop lengths, loss weights, '
         'augmentation prior) was fixed before the first run; see the development history for the one later change.',
         '* The inner selection runs one seed (0); the chosen steps, hypoxia threshold and abstain tau are applied to the 3-seed ensemble.',
         '* The augmentation is a feature-space prior, not a raw-RGB physics simulation (time budget); it is independent of stress.py.',
         '* The TTA gate uses the label-free first-minute indicator (t + 10 s <= 60 s) as oxytrack does; it gates the adaptation only and is not a '
         'network input.', '',
         '## Development history (disclosed researcher degree of freedom)', '',
         '* v1 (steps grid 100, 200, 400): the inner selection chose 100, the grid minimum, in 6/6 folds (inner TM fell with more steps in every '
         'fold). Outer results were then seen: L0 TM 0.708, blank 0.500, L1 0.694, L2 0.656, L3 0.533, L4 0.503, L5 0.498, X1 0.508. Its nulls and notta were not run. Log and run output kept as `oxyseq_log_v1.md`, `oxyseq_run_v1.txt`; its score files were removed because v2 '
         'chose the same 100 steps in 6/6 folds and reproduced them (identical pred and answer on every window, scores within 5e-6).',
         '* v2 (this file): the only change is the steps grid extended downward to (25, 50, 100, 200) because the v1 choice sat on the grid edge '
         '(an inner-CV signal); the change was nevertheless made after the v1 outer numbers were seen, so v2 carries that selection.', '']
    L += ['## Results (computed with oxytrack.tm_summary / paired_vs; official numbers in results/photon/TABLE.md)', '',
          '| file | TM bal. acc. [95 % CI] | AUROC | coverage | acc. answered | SpO2 MAE | diff vs chain_v1 [95 % CI] | people better | worst person |',
          '|---|---|---|---|---|---|---|---|---|']
    for k, v in st['rows'].items():
        L.append(f'| {k} | {_f(v["tm"])} [{_f(v.get("lo"))}, {_f(v.get("hi"))}] | {_f(v["auroc"])} | {_f(v["cov"])} | {_f(v["acc_ans"])} | {_f(v.get("mae"), 2)} | '
                 f'{v.get("pd", "")} | {v.get("nb", "")} | {_f(v.get("worst"))} |')
    L += ['']
    for k, v in st.get('notes', {}).items():
        L.append(f'* {k}: {v}')
    L += ['', '## Nested choices per outer fold', '']
    for model, folds in st['folds'].items():
        L += [f'### {model}', '', '| held-out person | steps | inner TM by steps | threshold | abstain tau (quality prob) | inner coverage | 80 % band half-width | '
              'mean learned TTA rate per seed |', '|---|---|---|---|---|---|---|---|']
        for fo in folds:
            L.append(f'| {fo["sid"]} | {fo["steps"]} | ' + ', '.join(f'{k}:{v:.3f}' for k, v in fo['trace'].items()) +
                     f' | {fo["thr"]:.3f} | {fo["tau"]:.3f} | {fo["inner_cover"]:.2f} | {fo["half"]:.2f} | ' + ', '.join(f'{x:.3f}' for x in fo['rate']) + ' |')
        L.append('')
    (RES / 'oxyseq_log.md').write_text('\n'.join(L) + '\n')


def summarize(df, ref):
    s = OT.tm_summary(df)
    out = dict(tm=s['tm_bacc'], auroc=s['tm_auroc'], cov=s['coverage'], acc_ans=s['acc_answered'], mae=s.get('mae', np.nan),
               worst=float(np.nanmin(list(s['per'].values()))))
    b = M.person_bootstrap(df.sid.values, lambda i: M.time_matched(df.t.values[i], df.y.values[i], df.pred.values[i])['tm_bacc'], B=500, seed=0)
    if isinstance(b, dict):
        out['lo'], out['hi'] = b.get('lo'), b.get('hi')
    if ref is not None:
        pr = OT.paired_vs(df, ref, B=500)
        out['pd'] = f'{pr["diff"]:+.3f} [{pr["lo"]:+.3f}, {pr["hi"]:+.3f}]'; out['nb'] = f'{pr["n_better"]}/{pr["n"]}'
    return out


def chain_ref(level):
    p = SCORES / f'chain_v1__{OT.level_tag(level)}.csv'
    return pd.read_csv(p) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', nargs='*', default=['0', '1', '2', '3', '4', '5', 'X1'])
    ap.add_argument('--skip', nargs='*', default=[], help='parts to skip: notta null')
    a = ap.parse_args(argv)
    levels = [int(x) if x.isdigit() else x for x in a.levels]
    t0 = time.time()
    wdf = OT.window_table()
    recs0 = OT.rec_table(feats(0), wdf)
    st = dict(rows={}, folds={}, notes={})
    print('oxyseq nested', flush=True)
    folds = nested(recs0, tta=True)
    st['folds']['oxyseq'] = folds
    for lv in levels:
        recs = recs0 if lv == 0 else OT.rec_table(feats(lv), wdf)
        df = score(folds, recs)
        OT.write_scores(df, 'oxyseq', lv)
        st['rows'][f'oxyseq__{OT.level_tag(lv)}'] = summarize(df, chain_ref(lv))
        if lv == 0:
            seed_tm = [OT.tm_summary(score(folds, recs, seeds_idx=[i]))['tm_bacc'] for i in range(len(SEEDS))]
            st['notes']['L0 single seeds'] = ', '.join(f'{x:.3f}' for x in seed_tm) + f' (seed mean {np.mean(seed_tm):.3f}); oxyseq__L0 is the ensemble'
            dfb = score(folds, recs0, blank=True)
            OT.write_scores(dfb, 'oxyseq_blank', 0)
            st['rows']['oxyseq_blank__L0'] = summarize(dfb, None)
        print(f'{OT.level_tag(lv)} ' + str({k: round(v["tm"], 3) for k, v in st['rows'].items()}) + f' ({time.time() - t0:.0f} s)', flush=True)
        write_log(st)
    if 'null' not in a.skip:
        tms = []
        for draw in range(3):
            fnull = nested(recs0, tta=True, seeds=(0,), null_draw=draw)
            df = score(fnull, recs0)
            if draw == 0:
                OT.write_scores(df, 'oxyseq_null', 0)
            tms.append(OT.tm_summary(df)['tm_bacc'])
            st['notes']['null draws (TM bal. acc.)'] = ', '.join(f'{x:.3f}' for x in tms) + f'; mean {np.mean(tms):.3f} (requirement <= 0.55); oxyseq_null__L0.csv is draw 0; null runs use the full nested pipeline with 1 final seed'
            write_log(st)
            print(f'null draw {draw}: {tms[-1]:.3f} ({time.time() - t0:.0f} s)', flush=True)
    if 'notta' not in a.skip:
        print('oxyseq_notta nested', flush=True)
        fn = nested(recs0, tta=False)
        st['folds']['oxyseq_notta'] = fn
        for lv in levels:
            recs = recs0 if lv == 0 else OT.rec_table(feats(lv), wdf)
            df = score(fn, recs, tta=False)
            OT.write_scores(df, 'oxyseq_notta', lv)
            st['rows'][f'oxyseq_notta__{OT.level_tag(lv)}'] = summarize(df, chain_ref(lv))
        write_log(st)
        print(f'notta done ({time.time() - t0:.0f} s)', flush=True)
    json.dump({k: {kk: (vv if isinstance(vv, str) else float(vv)) for kk, vv in v.items() if vv is not None} for k, v in st['rows'].items()},
              open(RES / 'oxyseq_summary.json', 'w'), indent=1)
    print(f'done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
