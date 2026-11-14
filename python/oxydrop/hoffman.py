"""Loader for the UW Hoffman 2022 smartphone-oximetry dataset (MIT): https://github.com/ubicomplab/oximetry-phone-cam-data
Expected layout: <root>/data/ppg-csv/{Left,Right}/<id>.csv (per-frame mean R,G,B at 30 fps) and <root>/data/gt/<id>.csv (1 Hz reference oximeters)."""
import numpy as np, pandas as pd

WIN, STEP = 10, 2


def load(root):
    recs = []
    for side in ('Left', 'Right'):
        for sid in range(100001, 100007):
            rgb = pd.read_csv(f'{root}/data/ppg-csv/{side}/{sid}.csv').values.astype(float)
            g = pd.read_csv(f'{root}/data/gt/{sid}.csv', encoding='utf-8-sig')
            cols = [c for c in ['SpO2 1', 'SpO2 2', 'SpO2 4', 'SpO2 5'] if c in g]
            gt = g[cols].replace(0, np.nan).mean(axis=1).values          # mean of the working reference oximeters, 1 Hz
            n = min(len(gt), len(rgb) // 30)
            y = {}
            for s in range(0, n - WIN + 1, STEP):
                seg = gt[s:s + WIN]
                if np.isfinite(seg).sum() >= 5:
                    y[s] = float(np.nanmean(seg))                        # window label = mean reference SpO2; hypoxic if < 90
            recs.append(dict(rec=f'{sid}{side[0]}', sid=sid, rgb=rgb, gt=gt, y=y))
    return recs
