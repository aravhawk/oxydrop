"""Training range of the window amplitude, for the generic low-AC guard of the hybrid engine (app/js/hybrid_engine.js OOD_LAC_MIN).

    source scripts/ml_env.sh && python scripts/ood_range.py     # writes results/ml/ood_ac_range.csv

lac = log(std of the band-passed log-intensity of a 10 s window), about the AC/DC ratio of the pulse (python/oxydrop_ml/features.py ABS 'lac_*').
The guard uses the largest of the three channels. The cut is the 0.1th percentile of this number over all windows of the six Hoffman people (the only
data with low-oxygen labels the network was trained on); no label and no phone model is involved. Aggregates only.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(os.environ.get('VERITAS', '/home/aravhawk/.t3/scratch/veritas'), 'ml_cache/data')
rows = []
for ds in ('hoffman', 'mths', 'butppg'):
    m = pd.read_parquet(f'{D}/{ds}/meta.parquet', columns=['abs_lac_R', 'abs_lac_G', 'abs_lac_B'])
    x = np.nanmax(m.values, axis=1); x = x[np.isfinite(x)]
    for q in (0.1, 1, 5, 50):
        v = float(np.quantile(x, q / 100)); rows.append(dict(dataset=ds, n_windows=len(x), percentile=q, lac_max=round(v, 3), ac_ratio_percent=round(100 * float(np.exp(v)), 4)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, 'results/ml/ood_ac_range.csv'), index=False)
print(out.to_string(index=False))
