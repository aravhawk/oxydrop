"""Gradient-boosted trees on handcrafted features (XGBoost and LightGBM), classification (P1) and regression (P3).

Feature sets
  fusion19_v1 / fusion19_v2   the 19 FUSION features of the shipped chains (python/oxydrop/chain.py), calibration-relative (minus the person's own
                              first-60 s median), v1 = peak-interval HR / v2 = gated robust-HR features. Built in chain_v1.py / chain_v2.py.
  hand26                      the 26-feature recipe of /home/aravhawk/.t3/scratch/veritas/multi/RESULTS.md section 3 (multi/scripts/vlib.py `feats`, copied here so
                              the repo is self-contained): per 10 s window, causal Butterworth 0.7-4 Hz with warm start, first 2 s dropped, DC levels and differences,
                              log AC amplitudes, AC ratios, covariance slopes, channel correlations, spectral peak SNR/HR, peak-amplitude ratios.
                              `hand26` = absolute values, `hand26_cal` = minus the mean over the windows fully inside the person's label-free baseline span.
Hyperparameters (and for P1 the decision threshold) are selected by the nested CV of common.run_cls on the outer training people only.
"""
import os

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .. import CACHE
from .. import windows as W
from . import common as C

FS, WIN, DROP = 30, 300, 60
_SOS = butter(3, [0.7 / (FS / 2), 4.0 / (FS / 2)], btype='band', output='sos')
_ZI = sosfilt_zi(_SOS)
_HANN = np.hanning(WIN - DROP)[:, None]
_FR = np.fft.rfftfreq(2048, 1 / FS)
_BAND = (_FR >= 0.8) & (_FR <= 3.5)
_KB = np.where(_BAND)[0]

FEATS26 = (['dcR', 'dcG', 'dcB', 'dcRG', 'dcRB', 'dcGB', 'lacR', 'lacG', 'lacB', 'rrRG', 'rrRB', 'rrGB', 'slRG', 'slRB', 'slGB', 'cRG', 'cRB', 'cGB',
            'hr', 'snr', 'prRG', 'prRB', 'prGB', 'paR', 'paG', 'paB'])
RATIO6 = ['rrRG', 'rrRB', 'rrGB', 'prRG', 'prRB', 'prGB']      # "ratio features" of the ridge baseline (log AC ratios, time and spectral domain)


def feats26(win_log):
    """win_log (300, 3) log(rgb + 1) of one window -> vector of the 26 features, in FEATS26 order (multi/scripts/vlib.py `feats`)."""
    ac = sosfilt(_SOS, win_log, axis=0, zi=_ZI[:, :, None] * win_log[0][None, None, :])[0][DROP:]
    dc = win_log.mean(0)
    sd = ac.std(0) + 1e-9
    Cv = np.cov(ac.T)
    cc = np.corrcoef(ac.T)
    sp = np.abs(np.fft.rfft(ac * _HANN, n=2048, axis=0))
    k = _KB[np.argmax(sp[_BAND].sum(1))]
    a = sp[max(k - 3, 0):k + 4].max(0) + 1e-9
    return np.array([dc[0], dc[1], dc[2], dc[0] - dc[1], dc[0] - dc[2], dc[1] - dc[2], np.log(sd[0]), np.log(sd[1]), np.log(sd[2]),
                     np.log(sd[0] / sd[1]), np.log(sd[0] / sd[2]), np.log(sd[1] / sd[2]),
                     Cv[0, 1] / (Cv[1, 1] + 1e-12), Cv[0, 2] / (Cv[2, 2] + 1e-12), Cv[1, 2] / (Cv[2, 2] + 1e-12), cc[0, 1], cc[0, 2], cc[1, 2],
                     _FR[k], np.log(sp[k].sum() / (sp[_BAND].sum() + 1e-9)), np.log(a[0] / a[1]), np.log(a[0] / a[2]), np.log(a[1] / a[2]),
                     np.log(a[0]), np.log(a[1]), np.log(a[2])])


def hand26_for(ws, chunk=1024):
    """Absolute hand26 features of every window of the shard (cached next to the baselines results cache). NaN where a channel is constant."""
    os.makedirs(C.CACHE_B, exist_ok=True)
    p = C.CACHE_B / f'hand26_{ws.ds}_{len(ws)}.npy'
    if p.exists():
        return np.load(p)
    import warnings
    out = np.full((len(ws), len(FEATS26)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for a in range(0, len(ws), chunk):
            idx = np.arange(a, min(len(ws), a + chunk))
            buf = ws.lrgb(idx)[:, -WIN:, :].astype(np.float64)
            for j in range(len(idx)):
                out[a + j] = feats26(buf[j])
    out[~np.isfinite(out).all(1)] = np.nan
    np.save(p, out)
    return out


def calibrate(meta, H):
    """Subtract the person's own label-free baseline: mean over the windows of the same base_key that lie fully inside the baseline span (`in_baseline`)."""
    Hc = np.full_like(H, np.nan)
    keys = meta.base_key.values; inb = meta.in_baseline.values.astype(bool)
    for k in np.unique(keys):
        rows = np.nonzero(keys == k)[0]
        b = rows[inb[rows] & np.isfinite(H[rows]).all(1)]
        if len(b) >= 2:
            Hc[rows] = H[rows] - H[b].mean(0)
    return Hc


# ------------------------------------------------------------------------------------------------ models
class _Pred:
    def __init__(self, f, v=None):
        self.score = f
        if v is not None:
            self.valid = v


def xgb_cls(hp, seed=0, n_jobs=4):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=hp['n'], max_depth=hp['depth'], learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         tree_method='hist', n_jobs=n_jobs, random_state=seed, verbosity=0, **{k: v for k, v in hp.items() if k == 'scale_pos_weight'})


def lgb_cls(hp, seed=0, n_jobs=4):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=hp['n'], num_leaves=hp['leaves'], learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          min_child_samples=20, class_weight='balanced', n_jobs=n_jobs, random_state=seed, verbose=-1)


GRID_XGB = [dict(depth=d, n=n) for d in (2, 4) for n in (100, 300)]
GRID_LGB = [dict(leaves=l, n=n) for l in (4, 8) for n in (100, 300)]


def tree_cls_spec(ctx, lib, fkey, name=None):
    """Spec for common.run_cls: gradient-boosted classifier on ctx.F[fkey] (rows x features), class-balanced, nested grid, tuned threshold."""
    X = ctx.F[fkey]; y = ctx.y

    def fit(hp, rows):
        ok = np.isfinite(X[rows]).all(1) if lib == 'lgbm' else np.ones(len(rows), bool)
        r = rows[ok]
        if lib == 'xgb':
            pos = max(int(y[r].sum()), 1); h = dict(hp, scale_pos_weight=(len(r) - pos) / pos)
            m = xgb_cls(h)
        else:
            m = lgb_cls(hp)
        m.fit(X[r], y[r])
        return _Pred(lambda q: m.predict_proba(X[q])[:, 1])
    return C.Spec(name or f'{lib}_{fkey}', fit, GRID_XGB if lib == 'xgb' else GRID_LGB, thr_mode='tune',
                  note=f'{"XGBoost" if lib == "xgb" else "LightGBM"} on {fkey}; hp grid {(GRID_XGB if lib == "xgb" else GRID_LGB)} and threshold by inner CV on training people')


# regression (P3)
def xgb_reg(hp, seed=0, n_jobs=4):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=hp['n'], max_depth=hp['depth'], learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=10.0, tree_method='hist', n_jobs=n_jobs, random_state=seed, verbosity=0)


def lgb_reg(hp, seed=0, n_jobs=4):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=hp['n'], num_leaves=hp['leaves'], max_depth=3, learning_rate=0.05, min_child_samples=30, subsample=0.7, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=10.0, n_jobs=n_jobs, random_state=seed, verbose=-1)
