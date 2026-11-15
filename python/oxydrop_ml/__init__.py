"""OxyDrop ML: data pipeline, evaluation harness and (in sibling modules) the multi-task CNN.

Every evaluation runs through this package: `splits` (LOSO, nested folds), `metrics` (HR, SpO2, classification, time-matched,
selective, person bootstrap, train-vs-test gap), `features` (causal window tensor, mirrored in app/js), `windows` (shard I/O)
and `data/*` (dataset builders that write into the cache directory, never into git).
"""
import os

FS = 30                      # uniform grid (Hz) for every dataset
WIN_S = 10                   # window length (s)
HOP_S = 2                    # hop between windows (s)
WIN = WIN_S * FS             # 300 samples
HOP = HOP_S * FS             # 60 samples
CTX = 150                    # warm-up context (5 s) before each window for the causal filters; stored with the window
BASE_SEC = 60                # label-free baseline span (first minute of a recording)

# Cache root: shared, outside git. Override with OXY_CACHE.
CACHE = os.environ.get('OXY_CACHE', '/home/aravhawk/.t3/scratch/veritas/ml_cache')
DATA_DIR = os.path.join(CACHE, 'data')

DATASETS = ('hoffman', 'mths', 'butppg', 'openox')
DOMAIN = {'hoffman': 0, 'mths': 0, 'butppg': 0, 'openox': 1, 'pretrain': 1}   # 0 = phone camera, 1 = contact red/IR PPG
