"""u20wit_verify extras: (a) args of the six U20 runs differ only where declared; (b) selection metric / steps; (c) teacher match on XH."""
import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '4'
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path('<repo>'); L = ROOT / 'nn_x/w5/logs'; ST = ROOT / 'nn_x/w12/next/u20wit/cache'
J = {(a, s): json.load(open(L / f'w5_e10_u20{a}_s{s}.json')) for a in 'abc' for s in (951, 952)}
ref = J['c', 951]['args']
for (a, s), j in J.items():
    d = {k: (ref.get(k), j['args'].get(k)) for k in set(ref) | set(j['args']) if ref.get(k) != j['args'].get(k)}
    h = j['history']; last = h[-1]
    best = [x for x in h if x['step'] == j['best_step']]
    print(f'u20{a}_s{s}: args differing from u20c_s951: {d}')
    print(f'    select_on {j["args"].get("select_on")} select_min_step {j["args"].get("select_min_step")} epochs {j["args"].get("epochs")} lr {j["args"].get("lr")} holdout {j["args"].get("holdout")} cooldown_mix {j["args"].get("cooldown_mix")} '
          f'| evals {len(h)} last step {last["step"]} best_step {j["best_step"]} best val {best[0]["val_ndcg"] if best else None:.5f} hold {best[0]["holdout_ndcg"] if best else None:.5f} | last val {last["val_ndcg"]:.5f} hold {last["holdout_ndcg"]:.5f} | minutes {j["train_minutes"]:.0f}')
# xf0 teacher members
import torch
for s in (951, 952, 953):
    ck = torch.load(ROOT / f'nn_x/w5/ckpt/w5_e10_xf0_s{s}.pt', map_location='cpu', weights_only=False); g = ck['args']
    print(f'teacher member xf0_s{s}: xfold {g.get("xfold")} unlab {g.get("xfold_unlab")} holdout {g.get("holdout")} kd_file {g.get("kd_file")!r} d {g.get("d")}')
print(open(ROOT / 'nn_x/w12/kd/out/dump_xf0.log').read()[:600] if (ROOT / 'nn_x/w12/kd/out/dump_xf0.log').exists() else 'no dump_xf0.log')
# teacher match on XH (cross-user holdout feeds: users B never saw unlabelled... B saw NO feed of these users unlabelled; all labelled for A and B)
f = pl.read_parquet(ROOT / 'data/flat/train/feeds.parquet', columns=['M', 'cand_off']); Mtr = f['M'].to_numpy().astype(np.int64)
xf_ = np.load(ST / 'subxh/src_fidx.npy'); rows = np.load(ST / 'subxh/rows.npy'); M = Mtr[xf_]; off = np.concatenate([[0], np.cumsum(M)[:-1]])
T = np.asarray(np.load(ROOT / 'nn_x/w12/kd/out/teacher_xf0.npy', mmap_mode='r')[rows]).astype(np.float64)
def fc(a, b):
    out = np.empty(len(M))
    for i, (o, m) in enumerate(zip(off, M)): out[i] = np.corrcoef(a[o:o + m], b[o:o + m])[0, 1]
    return out
c = {a: np.mean([fc(np.load(ST / f'scxh/w5_e10_u20{a}_s{s}__tr.npy').astype(np.float64), T) for s in (951, 952)], 0) for a in 'abc'}
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
print(f'XH (n {len(M)}) within-feed corr to teacher_xf0 (IN-SAMPLE teacher on these feeds): A {c["a"].mean():.5f} B {c["b"].mean():.5f} C {c["c"].mean():.5f}; B - A {np.mean(c["b"] - c["a"]):+.5f} +- {se(c["b"] - c["a"]):.5f}')
