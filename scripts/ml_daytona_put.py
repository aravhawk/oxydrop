#!/usr/bin/env python3
"""Small add-ons to scripts/ml_daytona.py (which is left unchanged): upload a local file to a sandbox, and a light `up` without the 228 MB data pack.

    source scripts/ml_env.sh
    python scripts/ml_daytona_put.py uplite --tag D9 [--cpu 4 --mem 4 --disk 3]   # sandbox + code + numpy/pandas/sklearn/torch-cpu, no data pack
    python scripts/ml_daytona_put.py put --tag D9 --src local/file.npz --dst data/file.npz    # dst is relative to the sandbox work dir (~/w)

The sandbox state file is the one of ml_daytona.py (ml_cache/dt/<tag>.json), so `bg`, `tail`, `pull`, `down` of ml_daytona.py work on it.
Credentials come from the environment (DAYTONA_API_KEY) and are never printed.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_daytona as MD  # noqa: E402


def uplite(a):
    D, Img, Res = MD.sdk()
    sb = D.create(Img(image='python:3.12-slim', resources=Res(cpu=a.cpu, memory=a.mem, disk=a.disk),
                      labels={'proj': 'veritas-ml-train', 'owner': a.owner}), timeout=600)
    sb.set_autostop_interval(0)
    json.dump(dict(id=sb.id, tag=a.tag), open(MD.state_path(a.tag), 'w'))
    print('sandbox', sb.id, flush=True)
    ex = lambda c, t=1800: sb.process.exec(c, timeout=t)
    print(ex('nproc; free -g | head -2; df -h / | tail -1').result, flush=True)
    ex(f'mkdir -p {MD.W}/data {MD.W}/runs')
    sb.fs.upload_file(MD.code_tar(), f'{MD.HOME}/code.tgz')
    ex(f'cd {MD.W} && tar xzf {MD.HOME}/code.tgz -C {MD.W}')
    t0 = time.time()
    r = ex('pip install -q numpy scipy pandas scikit-learn torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 2>&1 | tail -3', 1700)
    print(r.result, 'pip', round(time.time() - t0), 's', flush=True)
    print(ex(f'cd {MD.W}/python && python -c "import torch,pandas,sklearn;print(torch.__version__)"').result, flush=True)


def put(a):
    D, sb, st = MD.get(a.tag)
    dst = f'{MD.W}/{a.dst.lstrip("/")}'
    sb.process.exec(f'mkdir -p {os.path.dirname(dst)}', timeout=60)
    with open(a.src, 'rb') as fh:
        sb.fs.upload_file(fh.read(), dst)
    print('put', a.src, '->', dst, os.path.getsize(a.src))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='sub', required=True)
    p = sp.add_parser('uplite'); p.add_argument('--tag', required=True); p.add_argument('--cpu', type=int, default=4)
    p.add_argument('--mem', type=int, default=4); p.add_argument('--disk', type=int, default=3); p.add_argument('--owner', default='feat009')
    p = sp.add_parser('put'); p.add_argument('--tag', required=True); p.add_argument('--src', required=True); p.add_argument('--dst', required=True)
    a = ap.parse_args()
    {'uplite': uplite, 'put': put}[a.sub](a)


if __name__ == '__main__':
    main()
