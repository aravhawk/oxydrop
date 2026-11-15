#!/usr/bin/env python3
"""Run OxyDrop ML jobs on a Daytona sandbox (the only place OpenOximetry-derived tensors may go besides the Surface host; never Kaggle).

    source scripts/ml_env.sh                                  # exports DAYTONA_API_KEY without printing it
    python scripts/ml_daytona.py up   --tag A [--cpu 4 --mem 4 --disk 5]   # create sandbox, upload code + compact data pack, pip install
    python scripts/ml_daytona.py bg   --tag A --name job1 -- python -m oxydrop_ml.loso_run run --task p1 --out runs/x   # background job
    python scripts/ml_daytona.py tail --tag A [--name job1]   # last log lines of one job (or the list of jobs)
    python scripts/ml_daytona.py pull --tag A --path runs [--path ckpt]   # download directories as tar (to ml_cache/dt/<tag>/)
    python scripts/ml_daytona.py down --tag A                 # delete the sandbox

The data pack (ml_cache/dt/pack_data.tgz, built by `python scripts/ml_daytona.py pack`) holds the compact shards (series, dt, meta, recs, info; no tensor caches)
and the pretraining corpus. The sandbox state is kept in ml_cache/dt/<tag>.json. Free-tier limits seen: 4 vCPU per sandbox, 10 GiB memory and 30 GiB disk in total.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VERITAS = os.environ.get('VERITAS', '/home/aravhawk/.t3/scratch/veritas')
DT = os.path.join(VERITAS, 'ml_cache', 'dt')
DATA = os.path.join(VERITAS, 'ml_cache', 'data')
HOME = '/home/daytona'
W = f'{HOME}/w'


def sdk():
    from daytona import Daytona, CreateSandboxFromImageParams, Resources
    return Daytona(), CreateSandboxFromImageParams, Resources


def state_path(tag):
    return os.path.join(DT, f'{tag}.json')


def get(tag):
    D = sdk()[0]
    st = json.load(open(state_path(tag)))
    return D, D.get(st['id']), st


def pack():
    os.makedirs(DT, exist_ok=True)
    out = os.path.join(DT, 'pack_data.tgz')
    subprocess.run(['tar', 'czf', out, '-C', DATA, '--exclude=x.f16.npy', '--exclude=s.f32.npy', '--exclude=ref.parquet', '--exclude=draws.parquet',
                    'hoffman', 'mths', 'butppg', 'openox', 'pretrain'], check=True)
    return out


def code_tar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        src = os.path.join(ROOT, 'python', 'oxydrop_ml')
        for dp, dn, fn in os.walk(src):
            dn[:] = [d for d in dn if d not in ('__pycache__', 'tests', '.pytest_cache')]
            for f in fn:
                if f.endswith(('.py', '.json', '.md')):
                    full = os.path.join(dp, f)
                    t.add(full, arcname=os.path.join('python', os.path.relpath(full, os.path.join(ROOT, 'python'))))
    return buf.getvalue()


def up(a):
    D, Img, Res = sdk()
    sb = D.create(Img(image='python:3.12-slim', resources=Res(cpu=a.cpu, memory=a.mem, disk=a.disk), labels={'proj': 'veritas-ml-train', 'owner': 'feat006'}), timeout=600)
    sb.set_autostop_interval(0)          # background jobs do not count as activity: without this the sandbox stops after 15 minutes
    json.dump(dict(id=sb.id, tag=a.tag), open(state_path(a.tag), 'w'))
    print('sandbox', sb.id, flush=True)
    ex = lambda c, t=1800: sb.process.exec(c, timeout=t)
    r = ex('nproc; cat /sys/fs/cgroup/cpu.max; free -g | head -2; df -h / | tail -1')
    print(r.result)
    ex(f'mkdir -p {W}/data {W}/ckpt {W}/runs')
    t0 = time.time()
    sb.fs.upload_file(open(os.path.join(DT, 'pack_data.tgz'), 'rb').read(), f'{HOME}/pack_data.tgz')
    print('uploaded data pack', round(time.time() - t0), 's', flush=True)
    sb.fs.upload_file(code_tar(), f'{HOME}/code.tgz')
    ex(f'cd {W} && tar xzf {HOME}/pack_data.tgz -C data && tar xzf {HOME}/code.tgz -C {W} && rm {HOME}/pack_data.tgz && df -h / | tail -1')
    r = ex('pip install -q numpy scipy pandas pyarrow scikit-learn torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 2>&1 | tail -3', 1700)
    print(r.result)
    print(ex(f'cd {W}/python && OXY_CACHE={W} python -c "import torch,pandas,scipy,sklearn;print(torch.__version__, torch.get_num_threads())"').result)


def update_code(a):
    D, sb, st = get(a.tag)
    sb.fs.upload_file(code_tar(), f'{HOME}/code.tgz')
    print(sb.process.exec(f'cd {W} && tar xzf {HOME}/code.tgz -C {W} && echo ok', timeout=120).result)


def bg(a):
    D, sb, st = get(a.tag)
    cl = a.cmdline[1:] if a.cmdline and a.cmdline[0] == '--' else a.cmdline
    cmd = ' '.join(cl)
    sc = f'cd {W}/python && export OXY_CACHE={W} OMP_NUM_THREADS={a.threads} MKL_NUM_THREADS={a.threads} PYTHONUNBUFFERED=1 && ({cmd}) > {W}/{a.name}.log 2>&1; echo EXIT $? >> {W}/{a.name}.log'
    sb.fs.upload_file(sc.encode(), f'{W}/{a.name}.sh')
    sb.process.exec(f'nohup bash {W}/{a.name}.sh > /dev/null 2>&1 &', timeout=60)
    print('started', a.name)


def tail(a):
    D, sb, st = get(a.tag)
    if a.name:
        print(sb.process.exec(f'tail -n {a.n} {W}/{a.name}.log', timeout=60).result)
    else:
        print(sb.process.exec(f'ls -la {W}/*.log; uptime', timeout=60).result)


def pull(a):
    D, sb, st = get(a.tag)
    os.makedirs(os.path.join(DT, a.tag), exist_ok=True)
    for p in a.path:
        name = p.strip('/').replace('/', '_')
        sb.process.exec(f'cd {W} && tar czf {HOME}/{name}.tgz {p}', timeout=600)
        data = sb.fs.download_file(f'{HOME}/{name}.tgz')
        out = os.path.join(DT, a.tag, f'{name}.tgz')
        open(out, 'wb').write(data)
        print('pulled', out, len(data))


def down(a):
    D, sb, st = get(a.tag)
    sb.delete()
    os.remove(state_path(a.tag))
    print('deleted')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='sub', required=True)
    p = sp.add_parser('pack')
    p = sp.add_parser('up'); p.add_argument('--tag', required=True); p.add_argument('--cpu', type=int, default=4); p.add_argument('--mem', type=int, default=4); p.add_argument('--disk', type=int, default=5)
    p = sp.add_parser('code'); p.add_argument('--tag', required=True)
    p = sp.add_parser('bg'); p.add_argument('--tag', required=True); p.add_argument('--name', required=True); p.add_argument('--threads', type=int, default=4); p.add_argument('cmdline', nargs=argparse.REMAINDER)
    p = sp.add_parser('tail'); p.add_argument('--tag', required=True); p.add_argument('--name', default=None); p.add_argument('-n', type=int, default=15)
    p = sp.add_parser('pull'); p.add_argument('--tag', required=True); p.add_argument('--path', action='append', required=True)
    p = sp.add_parser('down'); p.add_argument('--tag', required=True)
    a = ap.parse_args()
    {'pack': lambda a: print(pack()), 'up': up, 'code': update_code, 'bg': bg, 'tail': tail, 'pull': pull, 'down': down}[a.sub](a)


if __name__ == '__main__':
    main()
