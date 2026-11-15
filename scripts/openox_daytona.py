#!/usr/bin/env python3
"""Create a Daytona sandbox for the OpenOximetry SaO2 test (FEAT-007). OpenOximetry is PhysioNet credentialed data: Daytona only, never Kaggle, never git.

    source scripts/ml_env.sh                                   # exports DAYTONA_API_KEY without printing it
    python scripts/openox_daytona.py up --tag oxs              # sandbox + OpenOx shard (no tensor cache) + checkpoints + code + pip install
    python scripts/ml_daytona.py bg   --tag oxs --name infer -- python -m oxydrop_ml.openox_spo2 infer --ckpt-dir ../ckpt --out ../runs/openox_spo2
    python scripts/ml_daytona.py bg   --tag oxs --name analyze -- python -m oxydrop_ml.openox_spo2 analyze --out ../runs/openox_spo2 --res ../runs/aggregates
    python scripts/ml_daytona.py pull --tag oxs --path runs/aggregates --path runs/openox_spo2/lag.json     # aggregates only; per-window files stay in the sandbox
    python scripts/ml_daytona.py down --tag oxs

Only derived aggregates are pulled into the repository. The per-window prediction table is pulled to ml_cache/dt/<tag>/ (outside git) only with an explicit --path.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_daytona as D          # noqa: E402

CK = os.path.join(D.VERITAS, 'ml_cache', 'ckpt')
RUNS = os.path.join(D.VERITAS, 'ml_cache', 'runs', 'hyb_final')


def pack_openox():
    out = os.path.join(D.DT, 'pack_openox.tgz')
    subprocess.run(['tar', 'czf', out, '-C', D.DATA, '--exclude=x.f16.npy', '--exclude=s.f32.npy', 'openox'], check=True)
    return out


def ckpt_tar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for fam in ('R', 'P'):
            for s in (0, 1, 2):
                t.add(os.path.join(RUNS, f'{fam}_s{s}.pt'), arcname=f'ckpt/{fam}_s{s}.pt')
        t.add(os.path.join(CK, 'stageB_full.pt'), arcname='ckpt/stageB_full.pt')
    return buf.getvalue()


def up(a):
    Dt, Img, Res = D.sdk()
    sb = Dt.create(Img(image='python:3.12-slim', resources=Res(cpu=a.cpu, memory=a.mem, disk=a.disk), labels={'proj': 'veritas-ml-train', 'owner': 'feat007'}), timeout=600)
    sb.set_autostop_interval(0)
    json.dump(dict(id=sb.id, tag=a.tag), open(D.state_path(a.tag), 'w'))
    print('sandbox created', flush=True)
    ex = lambda c, t=1800: sb.process.exec(c, timeout=t)
    ex(f'mkdir -p {D.W}/data {D.W}/ckpt {D.W}/runs')
    p = pack_openox() if not os.path.exists(os.path.join(D.DT, 'pack_openox.tgz')) else os.path.join(D.DT, 'pack_openox.tgz')
    t0 = time.time()
    sb.fs.upload_file(open(p, 'rb').read(), f'{D.HOME}/pack_openox.tgz')
    print('uploaded OpenOx shard', round(time.time() - t0), 's', flush=True)
    sb.fs.upload_file(ckpt_tar(), f'{D.HOME}/ckpt.tgz')
    sb.fs.upload_file(D.code_tar(), f'{D.HOME}/code.tgz')
    print(ex(f'cd {D.W} && tar xzf {D.HOME}/pack_openox.tgz -C data && tar xzf {D.HOME}/ckpt.tgz -C {D.W} && tar xzf {D.HOME}/code.tgz -C {D.W} && rm {D.HOME}/*.tgz && df -h / | tail -1 && ls {D.W}/ckpt').result)
    r = ex('pip install -q numpy scipy pandas pyarrow scikit-learn torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 2>&1 | tail -3', 1700)
    print(r.result)
    print(ex(f'cd {D.W}/python && OXY_CACHE={D.W} python -c "import torch,pandas,scipy;print(torch.__version__)"').result)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='sub', required=True)
    p = sp.add_parser('up'); p.add_argument('--tag', required=True); p.add_argument('--cpu', type=int, default=4); p.add_argument('--mem', type=int, default=4); p.add_argument('--disk', type=int, default=5)
    a = ap.parse_args()
    {'up': up}[a.sub](a)


if __name__ == '__main__':
    main()
