"""Read-out of a re-cool arm of the lifelong lane against the harness references (CPU, read-only). Mirrors
nn_x/w12/kd/arm_read8.py rc:<arm> (16 members = K and L re-cools of seeds 951-958; train-only predictions of the SELECTED
checkpoints; pool = member-equal mean of per-feed z; NDCG per feed = src/metrics.py; plain per-feed SE) and adds the BUCKET of
feeds the extension reaches (capped feeds with >= 1 extension event, k0 arrays: val 1,132, holdout 2,159) and its complement.
  python read_hx.py <arm> [ref arm = ref]          e.g. read_hx.py hx ; read_hx.py ref2 (the harness's A/A null)
Proposed rule (NOTES.txt section 6; the lead declares the binding one): PASS iff bucket pooled z >= 2.5 AND both bucket deltas >= 0
AND the whole-set deltas val >= -1 SE and holdout >= -1 SE.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/lab/src")
from metrics import ndcg_per_feed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import DATA_OUT, Flat  # noqa: E402

arm = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "ref"
W = "<repo>/lab/nn_x/w5/preds"
F = Flat()
hold, _ = F.holdout()
fv = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
Mv, ov = fv["M"].to_numpy().astype(np.int64), fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float32)
Mh, oh = np.load(f"{W}/holdout/holdout_meta.npy").astype(np.int64)
yh = np.load(f"{W}/holdout/holdout_y.npy").astype(np.float32)
assert (Mh == F.M["train"][hold]).all()


def z(off, M, x):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, weights=x, minlength=len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x ** 2, minlength=len(M)) / M - mu ** 2, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


def members(a):
    out = []
    for st in "kl":
        for x in range(951, 959):
            v = f"w5_{st}cd_{a}_s{x}"
            if os.path.exists(f"{W}/nn_{v}_trainonly_val.npy") and os.path.exists(f"{W}/holdout/nn_{v}_holdout.npy"):
                out.append(v)
    return out


A, R = members(arm), members(ref)
pairs = [(a_, a_.replace(f"cd_{arm}_", f"cd_{ref}_")) for a_ in A if a_.replace(f"cd_{arm}_", f"cd_{ref}_") in R]
assert pairs, "no complete (arm, reference) pair"
A, R = [p[0] for p in pairs], [p[1] for p in pairs]


def pool(names):
    zv = np.mean([z(ov, Mv, np.load(f"{W}/nn_{v}_trainonly_val.npy").astype(np.float64)) for v in names], 0)
    zh = np.mean([z(oh, Mh, np.load(f"{W}/holdout/nn_{v}_holdout.npy").astype(np.float64)) for v in names], 0)
    return ndcg_per_feed(zv, yv, ov, Mv), ndcg_per_feed(zh, yh, oh, Mh)


av, ah = pool(A)
rv, rh = pool(R)
dv, dh = av - rv, ah - rh
bv = (np.load(f"{DATA_OUT}/val/ext_n.npy") > 0) & np.load(f"{DATA_OUT}/val/ext_capped.npy")
bt = (np.load(f"{DATA_OUT}/train/ext_n.npy") > 0) & np.load(f"{DATA_OUT}/train/ext_capped.npy")
bh = bt[hold]
se = lambda d: float(d.std() / np.sqrt(len(d)))
f = lambda d: f"{d.mean():+.5f} +- {se(d):.5f} (z {d.mean() / se(d):+.2f})"
print(f"== {arm} ({len(A)} members) vs {ref} ({len(R)} members), 16-pool train-only, paired per feed")
print(f"  val     all {f(dv)} | bucket ({bv.sum()}) {f(dv[bv])} | rest {f(dv[~bv])}")
print(f"  holdout all {f(dh)} | bucket ({bh.sum()}) {f(dh[bh])} | rest {f(dh[~bh])}")
zb = (dv[bv].mean() / se(dv[bv]) + dh[bh].mean() / se(dh[bh])) / np.sqrt(2)
za = (dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2)
ok = zb >= 2.5 and dv[bv].mean() >= 0 and dh[bh].mean() >= 0 and dv.mean() >= -se(dv) and dh.mean() >= -se(dh)
print(f"  pooled z: bucket {zb:+.2f}, all feeds {za:+.2f} (T1117 form)")
print(f"  PROPOSED RULE (bucket pooled z >= 2.5, both bucket deltas >= 0, whole-set deltas >= -1 SE): {'PASS' if ok else 'no pass'}")
