"""Item 2(b) + 2(c) arithmetic. CPU only (memory-mapped reads), about 2 minutes, 12 threads.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/teacher_holdout.py

A  holdout identity (train.py rule re-derived here: p_random > 0.9, M >= 50, rng 12345, 30,000 feeds) and data shares.
B  teacher_mean11.npy / teacher_mean25.npy on the 4,630,966 holdout rows: present (finite, non-constant per feed), and equal
   to the per-feed-centred mean of the 11 / 25 members' OWN saved holdout predictions (files written by each member's
   training run or holdout_eval.py: predictions of a model that never trained on these feeds).
C  provenance: every member checkpoint has args.holdout == 30000 (and, for re-cools, so has the snapshot it started from).
D  out-of-sample signature: teacher NDCG on the holdout feeds vs on 30,000 exploration-like TRAINING feeds drawn by the same
   rule (in-sample for the members): the holdout must be clearly lower.
E  steps per epoch with and without the holdout feeds (the exact bucket arithmetic of data.SplitData.batches) for the
   final recipe (cand_budget 32768, max_feeds 512) and the resulting schedule (total steps, t_cool, re-cool form).
Writes logs/teacher_holdout.txt / .json in this directory.
"""
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

torch.set_num_threads(12)
HERE = os.path.dirname(os.path.abspath(__file__))
R = "<repo>"
K = f"{R}/nn_x/w12/kd/out"
OUT = open(f"{HERE}/logs/teacher_holdout.txt", "w")
RES = {}


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet", columns=["user_id", "M", "cand_off"])
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
n = len(M)
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(n)).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[hold], M[hold])])
ish = np.zeros(n, bool)
ish[hold] = True
y = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r")
RES["A"] = dict(train_feeds=int(n), train_cands=int(M.sum()), holdout_feeds=len(hold), holdout_rows=int(len(rows)),
                share_feeds=len(hold) / n, share_cands=len(rows) / int(M.sum()), p09_M50_feeds=int(len(cand)),
                p05_feeds=int((p > 0.5).sum()), share_of_p09=len(hold) / len(cand), share_of_p05=len(hold) / int((p > 0.5).sum()),
                pool_without_holdout=int(((p > 0.5) & ~ish).sum()), pool_with_holdout=int((p > 0.5).sum()),
                holdout_mean_M=float(M[hold].mean()), train_mean_M=float(M.mean()), holdout_p_min=float(p[hold].min()))
P("A holdout identity:", json.dumps(RES["A"]))
# the stored holdout meta of the w5 harness is the same holdout
Mh, oh = np.load(f"{R}/nn_x/w5/preds/holdout/holdout_meta.npy")
yh = np.load(f"{R}/nn_x/w5/preds/holdout/holdout_y.npy")
assert (Mh == M[hold]).all() and (np.asarray(y[rows]).astype(np.float32) == yh.astype(np.float32)).all()
P(f"  stored holdout_meta / holdout_y of nn_x/w5 == this holdout ({len(rows)} rows): True")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
fid = np.repeat(np.arange(len(Mh)), Mh)


def cen(s):
    return s - np.bincount(fid, weights=s)[fid] / Mh[fid]


MEMB = dict(
    r6=[(f"{R}/nn_x/r6/ckpt/r6_cd_m02_r{r}.pt", f"{R}/nn_x/r6/preds/holdout/nn_r6_cd_m02_r{r}_holdout.npy") for r in range(101, 108)],
    base=[(f"{R}/nn_x/w5/ckpt/w5_e10_base_s{s}.pt", f"{R}/nn_x/w5/preds/holdout/nn_w5_e10_base_s{s}_holdout.npy") for s in range(951, 955)],
    gru=[(f"{R}/nn/ckpt/v23_gru_pre2s_s{s}.pt", f"{R}/preds/holdout/nn_v23_gru_pre2s_s{s}_holdout.npy") for s in range(501, 507)],
    pop1=[(f"{R}/nn_x/abl/ckpt/abl_e10_pop1_s{s}.pt", f"{R}/nn_x/abl/preds/holdout/nn_abl_e10_pop1_s{s}_holdout.npy") for s in range(951, 959)])
TEACH = {"mean11": MEMB["r6"] + MEMB["base"], "mean25": MEMB["r6"] + MEMB["base"] + MEMB["gru"] + MEMB["pop1"]}
# D: an in-sample comparison set = 30,000 exploration-like TRAINING feeds by the same rule (not in the holdout)
rest = np.setdiff1d(cand, hold)
ins = np.sort(np.random.default_rng(777).choice(rest, 30000, replace=False))
rows_in = np.concatenate([np.arange(o, o + m) for o, m in zip(off[ins], M[ins])])
Mi = M[ins]
oi = np.concatenate([[0], np.cumsum(Mi)[:-1]])
yi = np.asarray(y[rows_in]).astype(np.float32)
RES["B"], RES["D"] = {}, {}
for t, members in TEACH.items():
    tm = np.load(f"{K}/teacher_{t}.npy", mmap_mode="r")
    assert tm.shape == (int(M.sum()),), tm.shape
    th = np.asarray(tm[rows], np.float64)
    nonfin_all = 0
    for i in range(0, tm.shape[0], 50_000_000):
        nonfin_all += int((~np.isfinite(np.asarray(tm[i:i + 50_000_000]))).sum())
    sd_feed = np.sqrt(np.bincount(fid, weights=cen(th) ** 2) / Mh)
    hs = [np.load(h).astype(np.float64) for _, h in members]
    assert all(len(x) == len(rows) for x in hs)
    mm = np.mean(hs, 0)
    a, b = cen(th), cen(mm)
    nd_t = float(ndcg_per_feed(th.astype(np.float32), yh.astype(np.float32), oh, Mh).mean())
    nd_m = float(ndcg_per_feed(mm.astype(np.float32), yh.astype(np.float32), oh, Mh).mean())
    ti = np.asarray(tm[rows_in], np.float32)
    nd_in = float(ndcg_per_feed(ti, yi, oi, Mi).mean())
    RES["B"][t] = dict(members=len(members), nonfinite_all_rows=nonfin_all, nonfinite_holdout_rows=int((~np.isfinite(th)).sum()),
                       holdout_feeds_with_constant_teacher=int((sd_feed < 1e-6).sum()), teacher_sd_holdout=float(a.std()),
                       corr_centred=float(np.corrcoef(a, b)[0, 1]), max_abs_diff=float(np.abs(a - b).max()),
                       rms_diff=float(np.sqrt(((a - b) ** 2).mean())), ndcg_teacher_holdout=nd_t, ndcg_member_mean_holdout=nd_m)
    RES["D"][t] = dict(ndcg_holdout_out_of_sample=nd_t, ndcg_30000_training_exploration_feeds_in_sample=nd_in, gap=nd_in - nd_t)
    P(f"B teacher_{t}: {json.dumps(RES['B'][t])}")
    P(f"D teacher_{t}: {json.dumps(RES['D'][t])}")
    del tm, th, hs
# C provenance
RES["C"] = []
for grp, members in MEMB.items():
    for ck, _ in members:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        a = c["args"]
        row = dict(ckpt=os.path.relpath(ck, R), holdout=a.get("holdout"), select_on=a.get("select_on"), init=str(a.get("init", ""))[-60:],
                   kd_file=str(a.get("kd_file", "")), val_mix=a.get("val_mix", 0))
        if a.get("init"):
            ip = a["init"] if str(a["init"]).startswith("/") else f"{os.path.dirname(ck)}/{a['init']}.pt"
            if os.path.exists(ip):
                row["init_holdout"] = torch.load(ip, map_location="cpu", weights_only=False)["args"].get("holdout")
        RES["C"].append(row)
bad = [r for r in RES["C"] if r["holdout"] != 30000 or r.get("init_holdout", 30000) != 30000 or r["kd_file"] or r["val_mix"]]
P(f"C provenance: {len(RES['C'])} member checkpoints; holdout == 30000 for all (and for the init snapshot of every re-cool): {not bad}; "
  f"with a teacher file or val labels: {sum(bool(r['kd_file']) or bool(r['val_mix']) for r in RES['C'])}")
for r in RES["C"]:
    P("   ", json.dumps(r))


# E steps per epoch (exact SplitData.batches arithmetic)
def n_batches(feed_ids, cand_budget=32768, max_feeds=512, mpad=16):
    Mp = (M[feed_ids] + mpad - 1) // mpad * mpad
    tot = 0
    for mp, c in zip(*np.unique(Mp, return_counts=True)):
        B = int(max(1, min(max_feeds, cand_budget // mp)))
        tot += -(-int(c) // B)
    return tot


allf = np.arange(n)
s0, s1 = n_batches(allf[~ish]), n_batches(allf)
sh = n_batches(hold)
E = {}
for nm, s, ep, cf in (("E10 held-out (today)", s0, 10, 0.2), ("E10 full data", s1, 10, 0.2), ("re-cool held-out (today)", s0, 2, 1.0),
                      ("re-cool full data", s1, 2, 1.0)):
    total = int(s * ep)
    E[nm] = dict(steps_per_epoch=s, total=total, t_cool=int(round((1 - cf) * total)), cooldown_steps=total - int(round((1 - cf) * total)),
                 expected_pool_steps=0.02 * (total - int(round((1 - cf) * total))))
pool0, pool1 = allf[(p > 0.5) & ~ish], allf[p > 0.5]
E["pool"] = dict(pool_batches_without=n_batches(pool0), pool_batches_with=n_batches(pool1),
                 mean_feeds_per_pool_batch_without=len(pool0) / n_batches(pool0), mean_feeds_per_pool_batch_with=len(pool1) / n_batches(pool1))
E["holdout_only_batches"] = sh
RES["E"] = E
P("E steps:", json.dumps(E, indent=1))
json.dump(RES, open(f"{HERE}/logs/teacher_holdout.json", "w"), indent=1)
P("done")
