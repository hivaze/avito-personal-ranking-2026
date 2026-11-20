"""verify_where V8: late-training trajectories from the training logs (val_ndcg / holdout_ndcg per eval step), relative to the last step,
averaged per group: 8 references kd05w192, 8 kd05t25w192, arms A/B/C (4 each). Read-only. Output out/v8_traj.txt"""
import re, numpy as np
from pathlib import Path
R = Path('<repo>'); OUT = R / 'lab/nn_x/w12/moe/why/verify_where/out/v8_traj.txt'
pat = re.compile(r'^step (\d+)/98350 loss \S+ val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)')
def traj(f):
    d = {}
    for ln in open(f):
        mm = pat.match(ln)
        if mm: d[int(mm.group(1))] = (float(mm.group(2)), float(mm.group(3)))
    return d
G = {'ref kd05w192': [R / f'nn_x/w5/logs/w5_e10_kd05w192_s{s}.log' for s in range(951, 959)],
     'kd05t25w192': [R / f'nn_x/w5/logs/w5_e10_kd05t25w192_s{s}.log' for s in range(951, 959)],
     'A moeffn': [R / f'lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192moeffn_s{s}.log' for s in range(951, 955)],
     'B moevo': [R / f'lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192moevo_s{s}.log' for s in range(951, 955)],
     'C b3ff1': [R / f'lab/nn_x/w12/moe/out/logs/w5_e10_kd05w192b3ff1_s{s}.log' for s in range(951, 955)]}
steps = [80000, 82500, 85000, 87500, 90000, 92500, 95000, 97500, 98350]
L = ['mean over runs of (metric at step - metric at 98350), x1e5; val | holdout']
L.append(f'{"group":14s} ' + ' '.join(f'{s:>13d}' for s in steps))
for g, fs in G.items():
    T = [traj(f) for f in fs if f.exists()]
    T = [t for t in T if all(s in t for s in steps)]
    v = np.array([[t[s][0] - t[98350][0] for s in steps] for t in T]) * 1e5; h = np.array([[t[s][1] - t[98350][1] for s in steps] for t in T]) * 1e5
    L.append(f'{g:14s} ' + ' '.join(f'{a:+6.0f}|{b:+6.0f}' for a, b in zip(v.mean(0), h.mean(0))) + f'   (n={len(T)})')
    # holdout step-to-step noise: sd of successive differences over the last 6 evals
    hh = np.array([[t[s][1] for s in steps[3:]] for t in T]); vv = np.array([[t[s][0] for s in steps[3:]] for t in T])
    L.append(f'{"":14s} sd of successive holdout diffs (87500-98350) {np.diff(hh, axis=1).std() * 1e5:.0f}e-5; val {np.diff(vv, axis=1).std() * 1e5:.0f}e-5; argmax holdout over steps>=85000: {[max((s for s in steps[2:]), key=lambda s: t[s][1]) for t in T]}')
txt = '\n'.join(L); print(txt); OUT.write_text(txt + '\n')
