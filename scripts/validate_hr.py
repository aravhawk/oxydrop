"""Validate heart-rate estimators on real phone-PPG datasets and synthetic stress tests.

A = OLD peak-interval HR (copied from features.win_feats, trailing 30 s of the causal band-passed channel)
B = naive spectral argmax (largest FFT peak 0.7-3.5 Hz, primary channel, estimation window)
C = NEW robust estimator (oxydrop.hr.best_channel_estimate; 'valid' = accepted by the gate)

Primary channel rule (all estimators): green if std_G > 0.3*std_R else red (std over the last 10 s, as in win_feats).
Datasets: BUT PPG v2.0 (10 s clips, ECG HR + expert quality), Hoffman 2022 (full length, oximeter pulse),
MTHS (finger video, oximeter HR at 1 Hz).  Run one stage at a time:  python scripts/validate_hr.py [but|long|synth|all]
Outputs: results/hr_validation/*.csv
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
from scipy.signal import butter, sosfilt, find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from oxydrop import hr as HR  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'hr_validation')
os.makedirs(OUT, exist_ok=True)
FS = 30
SOS = butter(3, [0.7, 4.0], btype='band', fs=FS, output='sos')   # same causal filter as features.SOS_CARDIAC
BUT = '/home/aravhawk/.t3/scratch/veritas/oxy_push/data/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0'
HOF = '/home/aravhawk/.t3/scratch/veritas/_feas_photon/repo'
MTHS = '/home/aravhawk/.t3/scratch/veritas/oxy_push/data/mths/repo/MTHS/Data'
SEED = 20261114
REC_FLO = {'v': None}  # lower HR band edge (Hz) of the recommended configuration; set from rec.pkl for the long/synth stages
CRASHES = {'n': 0}   # channel-windows where hr.py as written raises StopIteration


# ---------------------------------------------------------------- estimators
def old_hr(a30, a10_sd):
    """OLD logic copied from features.win_feats (S1 cardio): beat detection on -g, trailing window a30."""
    g = a30[:, 1] if a10_sd[1] > a10_sd[0] * 0.3 else a30[:, 0]
    x = -g
    pk, _ = find_peaks(x, distance=int(FS * 0.33), prominence=np.std(x) * 0.3)
    ibi = np.array([])
    if len(pk) >= 6:
        ibi = np.diff(pk) / FS
        ibi = ibi[(ibi > 0.3) & (ibi < 1.6)]
    return 60 / np.median(ibi) if len(ibi) >= 5 else np.nan


def naive_hr(x, fs=FS):
    x = np.asarray(x, float)
    xc = x - x.mean()
    P = np.abs(np.fft.rfft(xc * np.hanning(len(xc)), n=4096))
    fr = np.fft.rfftfreq(4096, 1 / fs)
    b = (fr >= 0.7) & (fr <= 3.5)
    return 60 * fr[b][np.argmax(P[b])]


def estimate_hr_patched(x, fs=FS, f_lo=None):
    """hr.estimate_hr with ONE fix: the parabolic refinement of the spectral peak is applied only when kb is a true local
    maximum of the spectrum, and the offset is clamped to +-0.5 bin. (As written, a harmonic-sum maximum that sits on a spectral
    slope gives an unbounded offset, e.g. hr = -35 or +30 bpm, and a negative f0 passes the 'agree' test.)"""
    f_lo = HR.F_LO if f_lo is None else f_lo
    x = np.asarray(x, float)
    out = dict(hr=np.nan, hr_acf=np.nan, sqi=0.0, acf=0.0)
    n = len(x)
    if n < HR.MIN_SEC * fs or np.std(x) < 1e-9:
        return out
    xc = x - x.mean()
    P = np.abs(np.fft.rfft(xc * np.hanning(n), n=HR.NFFT))
    df = fs / HR.NFFT
    ks = np.arange(int(np.ceil(f_lo / df)), int(np.floor(HR.F_HI / df)) + 1)
    kb = ks[np.argmax(P[ks] + 0.7 * P[2 * ks])]
    kf = float(kb)
    if 0 < kb < len(P) - 1 and P[kb] >= P[kb - 1] and P[kb] >= P[kb + 1]:
        a, b, c = P[kb - 1], P[kb], P[kb + 1]
        d = a - 2 * b + c
        if d < 0:
            kf = kb + float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    f0 = kf * df
    P2 = P ** 2
    tot = P2[int(np.ceil(0.65 / df)):int(np.floor(4.0 / df)) + 1].sum() + 1e-18
    w = int(round(0.10 / df))
    near = P2[max(kb - w, 0):kb + w + 1].sum() + P2[max(2 * kb - w, 0):2 * kb + w + 1].sum()
    sqi = float(near / tot)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    r = np.fft.irfft(np.abs(np.fft.rfft(xc, n=m)) ** 2)[:n]
    r = r / (r[0] + 1e-18)
    lo, hi = int(np.ceil(fs / 3.0)), min(int(np.floor(fs / f_lo)), n - 2)
    seg = r[lo - 1:hi + 2]
    pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not pk:
        out.update(hr=60 * f0, sqi=sqi)
        return out
    rmax = max(seg[i] for i in pk)
    # second fix: with all candidate peaks negative, 0.85*rmax > rmax and hr.py's next(...) raises StopIteration
    lag_i = next((i for i in pk if seg[i] >= 0.85 * rmax), max(pk, key=lambda i: seg[i]))
    a, b, c = seg[lag_i - 1], seg[lag_i], seg[lag_i + 1]
    d = a - 2 * b + c
    lag = (lo - 1) + lag_i + (0.5 * (a - c) / d if d != 0 else 0.0)
    out.update(hr=60 * f0, hr_acf=60 * fs / lag, sqi=sqi, acf=float(seg[lag_i]))
    return out


def channel_ests(acs, primary, fixed=False, f_lo=None):
    """Per-channel raw outputs of hr.estimate_hr (or the patched copy) in best_channel_estimate's order (gate re-applied in select())."""
    sd = acs.std(0)
    order = [primary] + [int(c) for c in np.argsort(-sd) if c != primary]
    fn = (lambda x, fs: estimate_hr_patched(x, fs, f_lo)) if fixed else HR.estimate_hr
    res = []
    for c in order:
        try:
            e = fn(acs[:, c], FS)
        except StopIteration:                      # hr.py as written crashes here (counted in CRASHES)
            CRASHES['n'] += 1
            e = dict(hr=np.nan, hr_acf=np.nan, sqi=0.0, acf=0.0)
        res.append((c,) + tuple(e[k] for k in ('hr', 'hr_acf', 'sqi', 'acf')))
    return res


def select(ch, sqi_min=None, acf_min=None, agree=None, fixed=False):
    """Re-implementation of best_channel_estimate for arbitrary gate thresholds -> (hr, valid, channel)."""
    sqi_min = HR.SQI_MIN if sqi_min is None else sqi_min
    acf_min = HR.ACF_MIN if acf_min is None else acf_min
    agree = HR.AGREE if agree is None else agree
    for c, h, ha, s, a in ch:
        if np.isfinite(h) and np.isfinite(ha) and (h > 0 or not fixed) and (abs(ha - h) / h < agree) and s >= sqi_min and a >= acf_min:
            return h, True, c
    c, h, ha, s, a = ch[0]
    return h, False, c


def eval_window(ac, te, L):
    """ac: (N,3) causal band-passed log RGB. Window = last L s before sample te; A uses trailing 30 s (its design)."""
    n = int(L * FS)
    w = ac[te - n:te]
    a10 = ac[te - 10 * FS:te]
    a30 = ac[max(0, te - 30 * FS):te]
    sd10 = a10.std(0) + 1e-9
    sdw = w.std(0)
    primary = 1 if sdw[1] > 0.3 * sdw[0] else 0
    rowA = old_hr(a30, sd10)
    rowB = naive_hr(w[:, primary])
    chF = channel_ests(w, primary, fixed=True)
    chR = channel_ests(w, primary, fixed=True, f_lo=REC_FLO['v']) if REC_FLO['v'] is not None else chF
    return rowA, rowB, channel_ests(w, primary), chF, chR


def bandpass_from_rgb(rgb, mirror=False):
    lg = np.log(np.clip(rgb, 1e-3, None) + 1.0)
    if mirror:
        lg = np.vstack([lg[::-1], lg])
    lg = lg - lg[0]
    return np.stack([sosfilt(SOS, lg[:, c]) for c in range(3)], 1)


# ---------------------------------------------------------------- metrics
def metrics(est, ref, valid=None, name=''):
    est = np.asarray(est, float); ref = np.asarray(ref, float)
    n = len(est)
    ok = np.isfinite(est) & np.isfinite(ref)
    d = dict(subset=name, n=n, coverage=ok.mean() if n else np.nan)
    if valid is not None:
        valid = np.asarray(valid, bool)
        d['valid_frac'] = (valid & ok).mean() if n else np.nan
    for tag, m in (('all', ok), ('acc', ok & valid if valid is not None else None)):
        if m is None:
            continue
        e = est[m]; r = ref[m]; err = np.abs(e - r)
        d[f'n_{tag}'] = int(m.sum())
        d[f'MAE_{tag}'] = err.mean() if len(err) else np.nan
        d[f'medAE_{tag}'] = np.median(err) if len(err) else np.nan
        d[f'w5_{tag}'] = (err <= 5).mean() if len(err) else np.nan
        d[f'w10_{tag}'] = (err <= 10).mean() if len(err) else np.nan
        d[f'dbl_{tag}'] = (np.abs(e - 2 * r) < 0.12 * 2 * r).mean() if len(err) else np.nan
        d[f'half_{tag}'] = (np.abs(e - 0.5 * r) < 0.12 * 0.5 * r).mean() if len(err) else np.nan
    return d


def table(df, subsets, est_cols):
    """est_cols: {label: (est_col, valid_col or None)}"""
    rows = []
    for sname, mask in subsets.items():
        d = df[mask]
        for lab, (ec, vc) in est_cols.items():
            r = metrics(d[ec].values, d['ref'].values, d[vc].values if vc else None, sname)
            r['estimator'] = lab
            rows.append(r)
    t = pd.DataFrame(rows)
    first = ['subset', 'estimator', 'n', 'coverage', 'valid_frac']
    return t[[c for c in first if c in t] + [c for c in t if c not in first]]


# ---------------------------------------------------------------- BUT PPG
def load_but():
    q = pd.read_csv(f'{BUT}/quality-hr-ann.csv', encoding='utf-8-sig')
    info = pd.read_csv(f'{BUT}/subject-info.csv', encoding='utf-8-sig')
    info.columns = ['ID', 'Gender', 'Age', 'Height', 'Weight', 'EarFinger', 'Motion', 'BP', 'Glyc', 'SpO2']
    q = q.merge(info[['ID', 'EarFinger', 'Motion']], on='ID')
    clips = []
    for _, r in q.iterrows():
        rid = int(r.ID)
        hdr = open(f'{BUT}/{rid}/{rid}_PPG.hea').read().split('\n')
        if int(hdr[0].split()[1]) != 3:
            continue
        sig = []
        for ln in hdr[1:4]:
            tok = ln.split(); g, rest = tok[2].split('(')[0], tok[2].split('(')[1]
            sig.append((float(g), float(rest.split(')')[0])))
        raw = np.fromfile(f'{BUT}/{rid}/{rid}_PPG.dat', dtype='<i2')
        if len(raw) < 900:
            continue
        x = raw[:900].reshape(-1, 3).astype(float)
        rgb = np.clip(np.stack([(x[:, c] - sig[c][1]) / sig[c][0] for c in range(3)], 1), 1e-3, None)
        clips.append((rid, r.HR, r.Quality, r.EarFinger, r.Motion, rgb))
    return clips


def stage_but():
    t0 = time.time()
    clips = load_but()
    print('BUT clips', len(clips), flush=True)
    rows, chs, wins = [], [], []
    chk = 0
    for rid, ref, qual, ef, mot, rgb in clips:
        ac = bandpass_from_rgb(rgb, mirror=True)           # mirrored pre-pad (as t11), scored window = last 10 s
        te = len(ac)
        A, B, ch, chF, _ = eval_window(ac, te, 10)
        h, v, c = select(ch)
        hF, vF, cF = select(chF, fixed=True)
        wins.append(ac[te - 300:te])
        if chk < 300:                                      # cross-check re-implementation vs the real function (where it does not crash)
            wn = ac[te - 300:te]
            try:
                e, cc = HR.best_channel_estimate(wn, 1 if wn.std(0)[1] > 0.3 * wn.std(0)[0] else 0)
                assert bool(e['valid']) == v and cc == c and (np.isclose(e['hr'], h) or (np.isnan(h) and np.isnan(e['hr']))), (rid, e, h, v)
            except StopIteration:
                pass
            chk += 1
        rows.append(dict(ID=rid, subj=rid // 1000, ref=ref, q=qual, ef=ef, motion=mot, A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_ch=c,
                         C_sqi=ch[0][3], C_acf=ch[0][4], P_hr=hF, P_valid=vF, P_ch=cF))
        chs.append(chF)
    D = pd.DataFrame(rows)
    # primary-channel raw gate stats for sweep
    D.to_csv(f'{OUT}/but_clips.csv', index=False)
    print('hr.py StopIteration crashes (channel-windows):', CRASHES['n'], '| negative/odd hr from hr.py as written:', int(((D.C_hr < 39) | (D.C_hr > 181)).sum()), flush=True)
    pd.to_pickle((D, chs, wins), '/home/aravhawk/.t3/scratch/hrval_tmp/but_cache.pkl')
    print('done est', time.time() - t0, flush=True)
    print(D.groupby(['ef', 'motion', 'q']).size())
    return D, chs, wins


def subsets_but(D):
    # Ear/finger coding: ef=1 is the FINGER class (median RGB ~ (238,52,42) = flash-lit fingertip with saturated red; ef=0 is dark,
    # median (78,20,15) = ear). NOTE earlier pivot/opus/t11 picked the majority class (ef=0) as 'finger'; that was the ear class.
    # Motion is a string code in subject-info.csv: '0' = none, 1-7 = scripted motions, '2;3' = combined.
    rest = D.motion.astype(str) == '0'
    return {'all clips': D.ID > 0, 'expert-good': D.q == 1, 'expert-bad': D.q == 0,
            'finger at rest': (D.ef == 1) & rest, 'finger at rest, good': (D.ef == 1) & rest & (D.q == 1),
            'ear': D.ef == 0, 'ear, good': (D.ef == 0) & (D.q == 1), 'motion': ~rest, 'motion, good': ~rest & (D.q == 1)}


def sweep_but(D, chs, f_lo):
    rows = []
    sqis = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    acfs = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    ref = D.ref.values; good = (D.q == 1).values; bad = ~good
    for s_, a_, g_ in itertools.product(sqis, acfs, [0.06, 0.10, 0.15]):
        sel = [select(ch, s_, a_, g_, fixed=True) for ch in chs]
        h = np.array([x[0] for x in sel], float); v = np.array([x[1] for x in sel], bool)
        err = np.abs(h - ref); acc = v & np.isfinite(h)
        wrong = err > 10
        e = err[acc]; hh = h[acc]; rr = ref[acc]
        rows.append(dict(F_LO=f_lo, SQI_MIN=s_, ACF_MIN=a_, AGREE=g_, accept_all=acc.mean(), accept_good=acc[good].mean(), accept_bad=acc[bad].mean(),
                         n_acc=int(acc.sum()), MAE_acc=e.mean() if len(e) else np.nan, w5_acc=(e <= 5).mean() if len(e) else np.nan,
                         w10_acc=(e <= 10).mean() if len(e) else np.nan,
                         dbl_acc=(np.abs(hh - 2 * rr) < 0.24 * rr).mean() if len(e) else np.nan,
                         half_acc=(np.abs(hh - 0.5 * rr) < 0.06 * rr).mean() if len(e) else np.nan,
                         wrong10_acc=(e > 10).mean() if len(e) else np.nan,
                         reject_wrong10=(~acc[wrong]).mean(), reject_bad=(~acc[bad]).mean(),
                         MAE_acc_good=err[acc & good].mean() if (acc & good).any() else np.nan,
                         w5_acc_good=(err[acc & good] <= 5).mean() if (acc & good).any() else np.nan))
    return pd.DataFrame(rows)


def pick(S):
    c = S[S.accept_good >= 0.70].copy()
    c['score'] = c.reject_bad + c.reject_wrong10
    return c.sort_values(['score', 'MAE_acc'], ascending=[False, True])


# ---------------------------------------------------------------- long recordings (Hoffman, MTHS)
def long_windows(name, rec_id, ac, ref_1hz, L, te_start, step):
    rows = []
    N = len(ac)
    for te in range(int(te_start * FS), N + 1, int(step * FS)):
        t1 = te // FS
        seg = ref_1hz[max(0, t1 - L):t1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < max(3, 0.5 * L):
            continue
        A, B, ch, chF, chR = eval_window(ac, te, L)
        h, v, c = select(ch)
        hF, vF, cF = select(chF, fixed=True)
        rows.append(dict(ds=name, rec=rec_id, L=L, t_end=te / FS, ref=float(np.median(seg)), A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_ch=c,
                         C_sqi=ch[0][3], C_acf=ch[0][4], P_hr=hF, P_valid=vF))
        rows[-1]['chs'] = chR         # recommended-config channel outputs (patched, recommended f_lo), kept for gating
    return rows


def stage_long():
    allr = []
    # Hoffman
    t0 = time.time()
    for side in ('Left', 'Right'):
        for sid in range(100001, 100007):
            rgb = pd.read_csv(f'{HOF}/data/ppg-csv/{side}/{sid}.csv').values.astype(float)
            g = pd.read_csv(f'{HOF}/data/gt/{sid}.csv', encoding='utf-8-sig')
            cols = [c for c in ['Pulse 1', 'Pulse 2', 'Pulse 4', 'Pulse 5'] if c in g]
            P = g[cols].replace(0, np.nan).values.astype(float)
            ref = np.nanmedian(P, axis=1)
            ac = bandpass_from_rgb(rgb)
            for L in (30, 10):
                allr += long_windows('hoffman', f'{sid}{side[0]}', ac, ref, L, 30, 6)
        print('hoffman', side, time.time() - t0, flush=True)
    # MTHS
    for f in sorted(os.listdir(MTHS)):
        if not f.startswith('signal_'):
            continue
        i = f[7:-4]
        rgb = np.load(f'{MTHS}/{f}').astype(float)
        lab = np.load(f'{MTHS}/label_{i}.npy').astype(float)
        ref = lab[:, 0].copy(); ref[(ref < 30) | (ref > 220)] = np.nan
        ac = bandpass_from_rgb(rgb)
        allr += long_windows('mths', i, ac, ref, 10, 15, 3)
        allr += long_windows('mths', i, ac, ref, 30, 30, 6)
    print('long done', time.time() - t0, flush=True)
    D = pd.DataFrame(allr)
    pd.to_pickle(D, '/home/aravhawk/.t3/scratch/hrval_tmp/long_cache.pkl')
    D.drop(columns='chs').to_csv(f'{OUT}/long_windows.csv', index=False)
    return D


def long_tables(D, rec):
    rows = []
    for (ds, L), d in D.groupby(['ds', 'L']):
        for el, ec, vc in (('A old peak', 'A_hr', None), ('B naive FFT', 'B_hr', None), ('C hr.py as written', 'C_hr', 'C_valid'),
                           ('C patched, default gate', 'P_hr', 'P_valid')):
            r = metrics(d[ec].values, d.ref.values, d[vc].values if vc else None, f'{ds} L={L}s'); r['estimator'] = el; rows.append(r)
        sel = [select(ch, fixed=True, sqi_min=rec['sqi_min'], acf_min=rec['acf_min'], agree=rec['agree']) for ch in d.chs]
        r = metrics([x[0] for x in sel], d.ref.values, [x[1] for x in sel], f'{ds} L={L}s'); r['estimator'] = 'C patched, recommended gate'; rows.append(r)
    t = pd.DataFrame(rows)
    first = ['subset', 'estimator', 'n', 'coverage', 'valid_frac']
    return t[[c for c in first if c in t] + [c for c in t if c not in first]]


# ---------------------------------------------------------------- synthetic stress tests
def synth_trial(rng, n_total=40.0):
    f0 = rng.uniform(50, 110) / 60
    h2 = rng.uniform(0.4, 1.2)          # dicrotic / second-harmonic amplitude relative to fundamental
    h3 = rng.uniform(0.0, 0.3)
    snr_db = rng.uniform(0, 15)
    src = 60.0
    t = np.arange(0, n_total, 1 / src)
    # slight HRV: random-walk frequency
    fm = f0 * (1 + 0.03 * np.cumsum(rng.standard_normal(len(t))) / np.sqrt(src * 10))
    fm = np.clip(fm, 0.6 * f0, 1.4 * f0)
    ph = 2 * np.pi * np.cumsum(fm) / src
    ph2 = rng.uniform(0, 2 * np.pi)
    s = np.sin(ph) + h2 * np.sin(2 * ph + ph2) + h3 * np.sin(3 * ph + rng.uniform(0, 2 * np.pi))
    s = s / s.std()
    # respiratory amplitude modulation
    s = s * (1 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.2, 0.35) * t + rng.uniform(0, 6.28)))
    noise = rng.standard_normal(len(t))
    noise = noise / noise.std() * 10 ** (-snr_db / 20)
    motion = np.zeros_like(t)
    has_motion = rng.random() < 0.5
    if has_motion:
        for _ in range(rng.integers(1, 3)):
            t0 = rng.uniform(5, n_total - 4); dur = rng.uniform(1, 3)
            m = (t > t0) & (t < t0 + dur)
            nz = rng.standard_normal(len(t))
            lowp = np.convolve(nz, np.ones(30) / 30, mode='same')       # ~2 Hz lowpassed noise at 60 fps
            motion[m] += (lowp[m] / lowp.std()) * rng.uniform(2, 6)
    drift = rng.uniform(1, 6) * np.sin(2 * np.pi * rng.uniform(0.03, 0.15) * t + rng.uniform(0, 6.28))
    steps = np.zeros_like(t)
    for _ in range(rng.integers(0, 4)):
        steps[t > rng.uniform(3, n_total - 3)] += rng.uniform(-6, 6) * (rng.random() < 0.7)
    sig = s + noise
    # AC amplitude in 8-bit grey levels (peak ~ amp); G smaller than R on a saturating finger flash
    ampG = rng.uniform(0.6, 4.0); ampR = ampG * rng.uniform(0.5, 2.5)
    dcG = rng.uniform(40, 160); dcR = rng.uniform(180, 250); dcB = rng.uniform(20, 80)
    gain = np.where(rng.random() < 0.3, 1.0, 1.0)
    chs = []
    for dc, amp, w in ((dcR, ampR, 1.0), (dcG, ampG, 1.0), (dcB, ampG * 0.4, 0.5)):
        x = dc + amp * (sig + w * 0.0) + amp * (motion + drift + steps) * rng.uniform(0.7, 1.3) * 1.0
        chs.append(x)
    X = np.stack(chs, 1)
    X = np.round(np.clip(X, 0, 255))                                  # 8-bit quantisation at the source rate
    # 30 fps sampling with +-4 ms timestamp jitter, linear interpolation, jitter unknown to the estimator
    nk = int((n_total - 0.05) * FS)
    tk = np.arange(nk) / FS + rng.uniform(-0.004, 0.004, nk)
    tk = np.clip(tk, 0, t[-1])
    Y = np.stack([np.interp(tk, t, X[:, c]) for c in range(3)], 1)
    return f0 * 60, dict(f0=f0 * 60, h2=h2, h3=h3, snr_db=snr_db, motion=has_motion, ampG=ampG, ampR=ampR), Y


def stage_synth(n=300):
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(n):
        ref, meta, rgb = synth_trial(rng)
        ac = bandpass_from_rgb(rgb)
        te = len(ac)
        A, B, ch, chF, chR = eval_window(ac, te, 10)
        h, v, c = select(ch)
        hF, vF, cF = select(chF, fixed=True)
        rows.append(dict(trial=k, ref=ref, **{kk: vv for kk, vv in meta.items() if kk != 'f0'}, A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_sqi=ch[0][3], C_acf=ch[0][4],
                         P_hr=hF, P_valid=vF))
        rows[-1]['chs'] = chR
    D = pd.DataFrame(rows)
    D.drop(columns='chs').to_csv(f'{OUT}/synthetic_trials.csv', index=False)
    return D


# ---------------------------------------------------------------- main
EST = {'A old peak': ('A_hr', None), 'B naive FFT': ('B_hr', None), 'C hr.py as written': ('C_hr', 'C_valid'),
       'C patched, default gate': ('P_hr', 'P_valid')}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    pc = '/home/aravhawk/.t3/scratch/hrval_tmp/'
    rec = None
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    if stage in ('but', 'all'):
        D, chs, wins = stage_but()
        t = table(D, subsets_but(D), EST)
        t.to_csv(f'{OUT}/summary_but.csv', index=False)
        print(t.round(3).to_string())
    if stage in ('flo', 'all'):
        D, chs, wins = pd.read_pickle(pc + 'but_cache.pkl')
        sd = np.array([w.std(0) for w in wins])
        prim = np.where(sd[:, 1] > 0.3 * sd[:, 0], 1, 0)
        sweeps, rows = [], []
        for f_lo in (0.65, 0.75, 0.85, 0.95):
            chF = [channel_ests(w, int(pr), fixed=True, f_lo=f_lo) for w, pr in zip(wins, prim)]
            S = sweep_but(D, chF, f_lo)
            sweeps.append(S)
            sel = [select(ch, fixed=True) for ch in chF]
            D3 = D.assign(Q_hr=[x[0] for x in sel], Q_valid=[x[1] for x in sel])
            tt = table(D3, subsets_but(D3), {f'C patched f_lo={f_lo}, default gate': ('Q_hr', 'Q_valid')})
            rows.append(tt)
            print('f_lo', f_lo, 'done', flush=True)
        S = pd.concat(sweeps); S.to_csv(f'{OUT}/but_gate_sweep.csv', index=False)
        pd.concat(rows).to_csv(f'{OUT}/summary_but_flo_default_gate.csv', index=False)
        cand = pick(S)
        cand.groupby('F_LO').head(5).to_csv(f'{OUT}/but_gate_sweep_top.csv', index=False)
        print(cand.groupby('F_LO').head(3).to_string())
        r0 = cand.iloc[0]
        rec = dict(f_lo=float(r0.F_LO), sqi_min=float(r0.SQI_MIN), acf_min=float(r0.ACF_MIN), agree=float(r0.AGREE))
        print('RECOMMENDED', rec)
        pd.to_pickle(rec, pc + 'rec.pkl')
        chR = [channel_ests(w, int(pr), fixed=True, f_lo=rec['f_lo']) for w, pr in zip(wins, prim)]
        sel = [select(ch, fixed=True, sqi_min=rec['sqi_min'], acf_min=rec['acf_min'], agree=rec['agree']) for ch in chR]
        D2 = D.assign(Cr_hr=[x[0] for x in sel], Cr_valid=[x[1] for x in sel])
        sub = subsets_but(D2); sub['subjects even'] = (D2.subj % 2 == 0); sub['subjects odd'] = (D2.subj % 2 == 1)
        t2 = table(D2, sub, {'C recommended (patched, f_lo, gate)': ('Cr_hr', 'Cr_valid')})
        t2.to_csv(f'{OUT}/summary_but_recommended.csv', index=False)
        D2.to_csv(f'{OUT}/but_clips.csv', index=False)
        print(t2.round(3).to_string())
    if stage in ('long', 'all'):
        rec = rec or pd.read_pickle(pc + 'rec.pkl'); REC_FLO['v'] = rec['f_lo']
        D = stage_long()
        t = long_tables(D, rec)
        t.to_csv(f'{OUT}/summary_long.csv', index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
        print(t.round(3).to_string())
    if stage in ('synth', 'all'):
        rec = rec or pd.read_pickle(pc + 'rec.pkl'); REC_FLO['v'] = rec['f_lo']
        D = stage_synth()
        selr = [select(ch, fixed=True, sqi_min=rec['sqi_min'], acf_min=rec['acf_min'], agree=rec['agree']) for ch in D.chs]
        D['R_hr'] = [x[0] for x in selr]; D['R_valid'] = [x[1] for x in selr]
        D.drop(columns='chs').to_csv(f'{OUT}/synthetic_trials.csv', index=False)
        sub = {'all trials': D.trial >= 0, 'no motion': ~D.motion, 'motion': D.motion, 'dicrotic h2>=0.8': D.h2 >= 0.8, 'dicrotic h2<0.8': D.h2 < 0.8,
               'SNR<5 dB': D.snr_db < 5, 'SNR>=10 dB': D.snr_db >= 10}
        t = table(D, sub, dict(EST, **{'C patched, recommended gate': ('R_hr', 'R_valid')}))
        t.to_csv(f'{OUT}/summary_synthetic.csv', index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
        print(t.round(3).to_string())


if __name__ == '__main__':
    main()
