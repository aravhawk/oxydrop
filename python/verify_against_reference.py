"""Check that this clean package reproduces the original research reference outputs (tests/fixtures/ref_all_12rec_python.json).
Usage: python verify_against_reference.py <hoffman_repo_root>"""
import os; sys_path = os.path.dirname(os.path.abspath(__file__))
import sys, json, numpy as np
sys.path.insert(0, sys_path)
from oxydrop import chain, hoffman
root = sys.argv[1]
ref = json.load(open(__file__.rsplit('/', 1)[0] + '/../tests/fixtures/ref_all_12rec_python.json'))
W = chain.load_weights(__file__.rsplit('/', 1)[0] + '/tiny_chain_weights.json')
worst = 0
for r in hoffman.load(root):
    o = chain.infer(r['rgb'], W); R = ref[r['rec']]
    assert [a[0] for a in o] == R['t'] and [a[2] for a in o] == R['d'], r['rec']
    worst = max(worst, np.abs(np.array([a[1] for a in o]) - np.array(R['p'])).max())
    print(r['rec'], 'ok')
print('max |dp| vs reference =', worst)
