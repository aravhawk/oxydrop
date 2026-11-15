"""Validate heart-rate estimators on real phone-PPG datasets and synthetic stress tests.

A = OLD peak-interval HR (copied from features.win_feats, trailing 30 s of the causal band-passed channel)
B = naive spectral argmax (largest FFT peak 0.7-3.5 Hz, primary channel, estimation window)
C = NEW robust estimator (oxydrop.hr.best_channel_estimate, exactly the shipped code; 'valid' = accepted by the gate)
The gate (F_LO, SQI_MIN, ACF_MIN, AGREE) is NOT tuned here: it comes from the subject-level cross-fit in scripts/hr_crossfit.py.

Primary channel rule (all estimators): green if std_G > 0.3*std_R else red (std over the last 10 s, as in win_feats).
Datasets: BUT PPG v2.0 (10 s clips, ECG HR + expert quality), Hoffman 2022 (full length, oximeter pulse),
MTHS (finger video, oximeter HR at 1 Hz).  Run one stage at a time:  python scripts/validate_hr.py [but|long|synth|all]   (then scripts/hr_crossfit.py for the held-out gate selection)
Env overrides for the dataset and cache paths: OXY_BUT, OXY_HOF, OXY_MTHS, OXY_HRTMP.
Outputs: results/hr_validation/*.csv
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.signal import butter, sosfilt, find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from oxydrop import hr as HR  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'hr_validation')
os.makedirs(OUT, exist_ok=True)
FS = 30
SOS = butter(3, [0.7, 4.0], btype='band', fs=FS, output='sos')   # same causal filter as features.SOS_CARDIAC
BUT = os.environ.get('OXY_BUT', '/home/aravhawk/.t3/scratch/veritas/oxy_push/data/butppg/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0')
HOF = os.environ.get('OXY_HOF', '/home/aravhawk/.t3/scratch/veritas/_feas_photon/repo')
MTHS = os.environ.get('OXY_MTHS', '/home/aravhawk/.t3/scratch/veritas/oxy_push/data/mths/repo/MTHS/Data')
SEED = 20261114
TMP = os.environ.get('OXY_HRTMP', '/home/aravhawk/.t3/scratch/hrval_tmp')   # per-recording feature caches (kept out of the repo)


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


def channel_ests(acs, primary, f_lo=None):
    """Per-channel raw outputs of the shipped hr.estimate_hr in best_channel_estimate's order (gate re-applied in select())."""
    sd = acs.std(0)
    order = [primary] + [int(c) for c in np.argsort(-sd) if c != primary]
    res = []
    for c in order:
        e = HR.estimate_hr(acs[:, c], FS, f_lo)
        res.append((c,) + tuple(e[k] for k in ('hr', 'hr_acf', 'sqi', 'acf')))
    return res


def select(ch, sqi_min=None, acf_min=None, agree=None):
    """Re-implementation of best_channel_estimate for arbitrary gate thresholds -> (hr, valid, channel)."""
    for c, h, ha, s, a in ch:
        if HR.gate_ok(dict(hr=h, hr_acf=ha, sqi=s, acf=a), sqi_min, acf_min, agree):
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
    return rowA, rowB, channel_ests(w, primary)


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
    rows, wins = [], []
    for rid, ref, qual, ef, mot, rgb in clips:
        ac = bandpass_from_rgb(rgb, mirror=True)           # mirrored pre-pad (as t11), scored window = last 10 s
        te = len(ac)
        A, B, ch = eval_window(ac, te, 10)
        h, v, c = select(ch)
        w = ac[te - 300:te]
        e, cc = HR.best_channel_estimate(w, 1 if w.std(0)[1] > 0.3 * w.std(0)[0] else 0)   # the exact shipped function must agree with select()
        assert bool(e['valid']) == v and cc == c and (np.isclose(e['hr'], h) or (np.isnan(h) and np.isnan(e['hr']))), (rid, e, h, v)
        wins.append(w)
        rows.append(dict(ID=rid, subj=rid // 1000, ref=ref, q=qual, ef=ef, motion=mot, A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_ch=c,
                         C_sqi=ch[0][3], C_acf=ch[0][4]))
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/but_clips.csv', index=False)
    os.makedirs(TMP, exist_ok=True)
    pd.to_pickle((D, None, wins), TMP + '/but_cache.pkl')
    print('done est', time.time() - t0, flush=True)
    print(D.groupby(['ef', 'motion', 'q']).size())
    return D


def subsets_but(D):
    # Ear/finger coding: ef=1 is the FINGER class (median RGB ~ (238,52,42) = flash-lit fingertip with saturated red; ef=0 is dark,
    # median (78,20,15) = ear). NOTE earlier pivot/opus/t11 picked the majority class (ef=0) as 'finger'; that was the ear class.
    # Motion is a string code in subject-info.csv: '0' = none, 1-7 = scripted motions, '2;3' = combined.
    rest = D.motion.astype(str) == '0'
    return {'all clips': D.ID > 0, 'expert-good': D.q == 1, 'expert-bad': D.q == 0,
            'finger at rest': (D.ef == 1) & rest, 'finger at rest, good': (D.ef == 1) & rest & (D.q == 1),
            'ear': D.ef == 0, 'ear, good': (D.ef == 0) & (D.q == 1), 'motion': ~rest, 'motion, good': ~rest & (D.q == 1)}


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
        A, B, ch = eval_window(ac, te, L)
        h, v, c = select(ch)
        rows.append(dict(ds=name, rec=rec_id, L=L, t_end=te / FS, ref=float(np.median(seg)), A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_ch=c,
                         C_sqi=ch[0][3], C_acf=ch[0][4]))
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
    pd.to_pickle(D, TMP + '/long_cache.pkl')
    D.to_csv(f'{OUT}/long_windows.csv', index=False)
    return D


def long_tables(D):
    rows = []
    for (ds, L), d in D.groupby(['ds', 'L']):
        for el, ec, vc in (('A old peak', 'A_hr', None), ('B naive FFT', 'B_hr', None), ('C v2 robust (shipped gate)', 'C_hr', 'C_valid')):
            r = metrics(d[ec].values, d.ref.values, d[vc].values if vc else None, f'{ds} L={L}s'); r['estimator'] = el; rows.append(r)
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
        A, B, ch = eval_window(ac, te, 10)
        h, v, c = select(ch)
        rows.append(dict(trial=k, ref=ref, **{kk: vv for kk, vv in meta.items() if kk != 'f0'}, A_hr=A, B_hr=B, C_hr=h, C_valid=v, C_sqi=ch[0][3], C_acf=ch[0][4]))
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/synthetic_trials.csv', index=False)
    return D


# ---------------------------------------------------------------- summary in the format of HR_SUMMARY.csv
def one_estimator(est, ref, accept=None):
    """n = number of clips in the subset, coverage = share of them for which this estimator outputs a number (accepted ones for 'v2 accepted')."""
    est = np.asarray(est, float); ref = np.asarray(ref, float)
    have = np.isfinite(est) & np.isfinite(ref)
    if accept is not None:
        have &= np.asarray(accept, bool)
    e = est[have]; r = ref[have]; err = np.abs(e - r)
    nz = len(err) > 0
    return dict(n=len(est), n_scored=int(have.sum()), coverage=have.mean() if len(est) else np.nan, MAE=err.mean() if nz else np.nan,
                medAE=np.median(err) if nz else np.nan, within5=(err <= 5).mean() if nz else np.nan, within10=(err <= 10).mean() if nz else np.nan,
                doubling=(np.abs(e - 2 * r) < 0.24 * r).mean() if nz else np.nan, halving=(np.abs(e - 0.5 * r) < 0.06 * r).mean() if nz else np.nan)


def hr_summary(D, subsets):
    rows = []
    for sname, mask in subsets.items():
        d = D[mask]
        for lab, col, acc in (('old peak-interval', 'A_hr', None), ('naive FFT peak', 'B_hr', None), ('v2 all clips (no gate)', 'C_hr', None),
                              ('v2 accepted (gated)', 'C_hr', d.C_valid.values if len(d) else None)):
            r = one_estimator(d[col].values, d.ref.values, acc); r.update(subset=sname, estimator=lab); rows.append(r)
    t = pd.DataFrame(rows)
    return t[['subset', 'estimator'] + [c for c in t if c not in ('subset', 'estimator')]]


# ---------------------------------------------------------------- main
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    if stage in ('but', 'all'):
        D = stage_but()
        S = hr_summary(D, subsets_but(D))
        S.to_csv(f'{OUT}/HR_SUMMARY.csv', index=False)
        print(S.round(3).to_string())
        print(f"shipped gate: F_LO {HR.F_LO} SQI_MIN {HR.SQI_MIN} ACF_MIN {HR.ACF_MIN} AGREE {HR.AGREE}")
    if stage in ('long', 'all'):
        D = stage_long()
        t = long_tables(D)
        t.to_csv(f'{OUT}/summary_long.csv', index=False)
        print(t.round(3).to_string())
    if stage in ('synth', 'all'):
        D = stage_synth()
        sub = {'all trials': D.trial >= 0, 'no motion': ~D.motion, 'motion': D.motion, 'dicrotic h2>=0.8': D.h2 >= 0.8, 'dicrotic h2<0.8': D.h2 < 0.8,
               'SNR<5 dB': D.snr_db < 5, 'SNR>=10 dB': D.snr_db >= 10}
        t = hr_summary(D, sub)
        t.to_csv(f'{OUT}/summary_synthetic.csv', index=False)
        print(t.round(3).to_string())


if __name__ == '__main__':
    main()
