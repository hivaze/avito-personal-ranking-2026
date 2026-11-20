"""RESUME re-check (2026-11-15 03:35 UTC): an independent SAMPLE re-derivation of numbers cached by the first run, before
they are trusted in the report. Light: memory-mapped reads, 2 threads.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/recheck_sample.py

1 holdout rule -> 30,000 feeds / 4,630,966 rows / shares (facts of report section 1)
2 teacher_mean11 / teacher_mean25 on 500 random holdout feeds: finite, == centred mean of the members' own holdout files
3 W16 16-pool (train-only), val: selected vs @last (report: 0.40144 / 0.40157, -0.00013 +- 0.00016), written here with
  an independent pooling loop
4 the logs of 3 random final members: 'holdout: 30000 ... excluded', train feeds 3121208, steps/epoch 9835
Writes logs/recheck_sample.txt.
"""
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = "<repo>"
OUT = open(f"{HERE}/logs/recheck_sample.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
n = len(M)
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
nrows = int(M[hold].sum())
P(f"1 holdout: {len(hold)} feeds, {nrows} rows; {len(hold) / n:.5%} of {n} train feeds; {nrows / M.sum():.4%} of candidates; "
  f"{len(hold) / len(cand):.2%} of the {len(cand)} draw-set feeds; {len(hold) / (p > 0.5).sum():.2%} of the {(p > 0.5).sum()} feeds with p_random > 0.5")
assert len(hold) == 30000 and nrows == 4630966

# 2 teacher sample
rng = np.random.default_rng(20261115)
pick = np.sort(rng.choice(30000, 500, replace=False))
hoff = np.concatenate([[0], np.cumsum(M[hold])])
K = f"{R}/nn_x/w12/kd/out"
r6 = [f"{R}/nn_x/r6/preds/holdout/nn_r6_cd_m02_r{r}_holdout.npy" for r in range(101, 108)]
base = [f"{R}/nn_x/w5/preds/holdout/nn_w5_e10_base_s{s}_holdout.npy" for s in range(951, 955)]
gru = [f"{R}/preds/holdout/nn_v23_gru_pre2s_s{s}_holdout.npy" for s in range(501, 507)]
pop1 = [f"{R}/nn_x/abl/preds/holdout/nn_abl_e10_pop1_s{s}_holdout.npy" for s in range(951, 959)]
for name, files in (("teacher_mean11", r6 + base), ("teacher_mean25", r6 + base + gru + pop1)):
    tm = np.load(f"{K}/{name}.npy", mmap_mode="r")
    mm = [np.load(x, mmap_mode="r") for x in files]
    mx, nonfin, const = 0.0, 0, 0
    for i in pick:
        fidx = hold[i]
        t = np.asarray(tm[off[fidx]:off[fidx] + M[fidx]], np.float64)
        m = np.mean([np.asarray(x[hoff[i]:hoff[i + 1]], np.float64) for x in mm], 0)
        nonfin += int((~np.isfinite(t)).sum())
        const += int(t.std() < 1e-6)
        mx = max(mx, float(np.abs((t - t.mean()) - (m - m.mean())).max()))
    P(f"2 {name}: rows {tm.shape[0]}; 500 random holdout feeds: non-finite {nonfin}, constant feeds {const}, "
      f"max abs diff (centred) to the mean of the {len(files)} members' own holdout files {mx:.2e}")
    assert tm.shape[0] == int(M.sum()) and nonfin == 0 and const == 0 and mx < 1e-4

# 3 W16 pool, val, selected vs @last
vf = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet", columns=["M", "cand_off"])
Mv = vf["M"].to_numpy().astype(np.int64)
ov = vf["cand_off"].to_numpy().astype(np.int64)
yv = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float32)
fid = np.repeat(np.arange(len(Mv)), Mv)


def z(s):
    s = s.astype(np.float64)
    mu = np.bincount(fid, weights=s) / Mv
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=(s - mu[fid]) ** 2) / Mv, 1e-12))
    return (s - mu[fid]) / sd[fid]


names = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
res = {}
for tag in ("", "@last"):
    acc = np.zeros(len(yv))
    for v in names:
        acc += z(np.load(f"{R}/nn_x/w5/preds/nn_{v}{tag}_trainonly_val.npy"))
    res[tag] = ndcg_per_feed((acc / len(names)).astype(np.float32), yv, ov, Mv)
d = res[""] - res["@last"]
P(f"3 W16 train-only 16-pool val: selected {res[''].mean():.5f} / @last {res['@last'].mean():.5f}; selected - @last "
  f"{d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f} (n {len(d)})")

# 4 member logs
allm = names + [f"abl_e10_{a}_s{s}" for a in ("pop1kd05t25", "pop1kd05w192") for s in range(951, 955)]
for v in rng.choice(allm, 3, replace=False):
    d_ = f"{R}/nn_x/{'w5' if v.startswith('w5') else 'abl'}/logs/{v}.log"
    t = open(d_).read()
    m = re.search(r"train feeds (\d+), steps/epoch (\d+), total steps (\d+)", t)
    P(f"4 {v}: 'holdout: 30000 random-like train feeds excluded from training' {'holdout: 30000 random-like train feeds excluded from training' in t}; "
      f"train feeds {m.group(1)}, steps/epoch {m.group(2)}, total {m.group(3)}; '--val_mix' value in args {re.search(r'.val_mix.: ([0-9.]+)', t).group(1)}")
P("done")
