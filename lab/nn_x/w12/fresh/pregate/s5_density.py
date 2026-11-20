"""Step 5: THE DENSITY QUESTION (label-free; DECLARATION.txt F + AMENDMENT 2). No model is trained on train rows.

Train-row version of the fresh-exposure counts: a pseudo-period inside train whose SOURCE is a per-day random subsample of
train feeds at the val / test density (other users only, rule A1); the query rows are the candidates of the sampled feeds.
  period L: days 20709, 20710 | 20713, 20714, 20715 (cold = no candidate impression in any train feed of days <= 20708)
  period X: days 20703, 20705 | 20707, 20708, 20709, 20710 (the real calendar pattern; pooled rows only)
  populations: U = all feeds of the day, R = p_random > 0.9, RM = p_random > 0.9 with the val + test M histogram.
Writes out/s5_density.json; prints zero share / mean / PSI against val, public-like and private-like rows.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s5_density.py
"""
import json
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import BOUND, FLAT, OUT, PSI_EDGES, PSI_LABELS, WIN, count, day_of, psi

t0 = time.time()
SEED = 7
ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
ts = ftr["timestamp"].to_numpy().astype(np.int64); uid = ftr["user_id"].to_numpy().astype(np.int64)
M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64)
day = day_of(ts)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p_random = np.zeros(len(ftr)); p_random[pr["fidx"]] = np.nan_to_num(pr["p_random"], nan=0.0)
assert len(pr["fidx"]) == len(ftr)
items = np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")
# real val / test side
cm = np.load(f"{OUT}/cnt_main.npy"); meta = np.load(f"{OUT}/rowmeta.npz")
part, cold_vt, day_vt = meta["part"], meta["cold"], meta["day"]
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet"); fte = pl.read_parquet(f"{FLAT}/test/feeds.parquet")
M_vt = np.concatenate([fv["M"].to_numpy(), fte["M"].to_numpy()]).astype(np.int64)
mq = np.unique(np.quantile(M_vt, np.linspace(0, 1, 21)[1:-1]))
mbin_vt = np.digitize(M_vt, mq); mshare = np.bincount(mbin_vt, minlength=len(mq) + 1) / len(M_vt)
print(f"val + test M: mean {M_vt.mean():.1f}; train M mean {M.mean():.1f}; random-like (p_random > 0.9) train M mean {M[p_random > 0.9].mean():.1f}", flush=True)
TOD_BOUND = (BOUND + 10800) % 86400  # 11:32 MSK on the second test day


def sample(days, sizes, pop, rng):
    out = []
    for d, n in zip(days, sizes):
        m = day == d
        if pop in ("R", "RM"):
            m &= p_random > 0.9
        idx = np.flatnonzero(m)
        if pop == "RM":
            b = np.digitize(M[idx], mq); pick = []
            tgt = np.floor(mshare * n).astype(int); tgt[np.argsort(-(mshare * n - tgt))[:n - tgt.sum()]] += 1
            for k in range(len(mshare)):
                pool = idx[b == k]
                assert len(pool) >= tgt[k], f"RM: day {d} M-bin {k} has {len(pool)} < {tgt[k]} random-like feeds"
                pick.append(rng.choice(pool, tgt[k], replace=False))
            out.append(np.sort(np.concatenate(pick)))
        else:
            out.append(np.sort(rng.choice(idx, n, replace=False)))
    return np.concatenate(out)


def build(fe):
    """Counts [rows, 6] for the candidates of the sampled train feeds fe (sorted), sources = the same feeds."""
    Ms = M[fe]
    rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in fe])
    it = np.asarray(items[rows]).astype(np.int64)
    gid = np.repeat(np.arange(len(fe)), Ms)
    F = dict(ts=ts[fe], day=day[fe], eff_user=np.where(uid[fe] == 0, -1 - np.arange(len(fe)), uid[fe]).astype(np.int64))
    c = count(it, gid, F, it, gid)
    return c, it, gid


def stat(x):
    return dict(zero=float((x == 0).mean()), mean=float(x.mean()), ge1=float((x >= 1).mean()), ge2=float((x >= 2).mean()), ge5=float((x >= 5).mean()),
                n=int(len(x)), hist=(np.histogram(x, PSI_EDGES)[0] / max(len(x), 1)).round(5).tolist())


res = {"psi_bins": PSI_LABELS}
# ---------------------------------------------------------------- period L: cold / warm
DL = [20709, 20710, 20713, 20714, 20715]; NL = [7542, 6878, 7367, 7578, 7600]
cut = off[np.searchsorted(day, 20709)]  # feeds are sorted by time: rows of days <= 20708 are a prefix
seen = np.zeros(111_400_000, np.bool_); seen[np.asarray(items[:cut])] = True
print(f"period L: pre-period rows {cut:,} (days <= 20708), distinct items seen {int(seen.sum()):,}; {time.time() - t0:.0f}s", flush=True)
real = {"val": part == 0, "public-like": part == 1, "private-like": part == 2}
res["L"] = {}
for pop in ("U", "R", "RM"):
    rng = np.random.default_rng(SEED)
    fe = sample(DL, NL, pop, rng)
    c, it, gid = build(fe)
    cold = ~seen[it]
    d_r = day[fe][gid]; tod = ((ts[fe] + 10800) % 86400)[gid]
    seg = {"val-like": d_r <= 20710, "public-like": (d_r == 20713) | ((d_r == 20714) & (tod < TOD_BOUND)),
           "private-like": ((d_r == 20714) & (tod >= TOD_BOUND)) | (d_r == 20715), "day1": d_r == 20709}
    r = {"feeds": int(len(fe)), "rows": int(len(it)), "mean_M": float(M[fe].mean()), "cold_share": {k: float(cold[m].mean()) for k, m in seg.items()},
         "share_of_train_feeds": float(len(fe) / len(ftr))}
    print(f"\n=== period L, population {pop}: {len(fe)} feeds ({100 * len(fe) / len(ftr):.2f}% of train feeds), {len(it):,} rows, mean M {M[fe].mean():.1f}; "
          f"cold share val-like {r['cold_share']['val-like']:.3f} public-like {r['cold_share']['public-like']:.3f} private-like {r['cold_share']['private-like']:.3f}"
          f"  (real: val {cold_vt[part == 0].mean():.3f} public {cold_vt[part == 1].mean():.3f} private {cold_vt[part == 2].mean():.3f})")
    print(f"{'window':6s} {'subset':5s} | {'TRAIN val-like zero/mean':>26s} {'VAL zero/mean':>16s} {'PSI':>6s} | {'TRAIN pub-like':>16s} {'PUBLIC':>14s} {'PSI':>6s} | {'TRAIN priv-like':>16s} {'PRIVATE':>14s} {'PSI':>6s}")
    for w, wn in enumerate(WIN):
        for sub in ("cold", "warm", "all"):
            ms = cold if sub == "cold" else (~cold if sub == "warm" else np.ones(len(cold), bool))
            mv = cold_vt if sub == "cold" else (~cold_vt if sub == "warm" else np.ones(len(cold_vt), bool))
            line = f"{wn:6s} {sub:5s} |"
            for k, rk in (("val-like", "val"), ("public-like", "public-like"), ("private-like", "private-like")):
                a = c[seg[k] & ms, w]; b = cm[real[rk] & mv, w]
                p = psi(a, b)
                r[f"{wn}|{sub}|{k}"] = dict(train=stat(a), real=stat(b), psi=p)
                line += f" {(a == 0).mean():10.4f}/{a.mean():6.3f}  {(b == 0).mean():8.4f}/{b.mean():6.3f} {p:6.3f} |"
            print(line)
    # clean position-matched read: first day, as-of window (only earlier feeds of the same day exist in both)
    for sub in ("cold", "warm", "all"):
        ms = cold if sub == "cold" else (~cold if sub == "warm" else np.ones(len(cold), bool))
        mv = cold_vt if sub == "cold" else (~cold_vt if sub == "warm" else np.ones(len(cold_vt), bool))
        a = c[seg["day1"] & ms, 0]; b = cm[(day_vt == 20718) & mv, 0]
        r[f"a_day1|{sub}"] = dict(train=stat(a), real=stat(b), psi=psi(a, b))
        print(f"  first-day as-of (a), {sub}: train zero {(a == 0).mean():.4f} mean {a.mean():.3f} | val day 20718 zero {(b == 0).mean():.4f} mean {b.mean():.3f} | PSI {psi(a, b):.3f}")
    res["L"][pop] = r
del seen
# ---------------------------------------------------------------- period X: real calendar pattern, pooled rows
DX = [20703, 20705, 20707, 20708, 20709, 20710]; NX = [7542, 6878, 6050, 7367, 7578, 7600]
REALDAY = [20718, 20720, 20722, 20723, 20724, 20725]
res["X"] = {}
for pop in ("U", "R", "RM"):
    rng = np.random.default_rng(SEED)
    fe = sample(DX, NX, pop, rng)
    c, it, gid = build(fe)
    d_r = day[fe][gid]; tod = ((ts[fe] + 10800) % 86400)[gid]
    seg = {"val-like": d_r <= 20705, "public-like": (d_r == 20707) | ((d_r == 20708) & (tod < TOD_BOUND)),
           "private-like": ((d_r == 20708) & (tod >= TOD_BOUND)) | (d_r >= 20709)}
    r = {"feeds": int(len(fe)), "rows": int(len(it)), "mean_M": float(M[fe].mean()), "share_of_train_feeds": float(len(fe) / len(ftr))}
    print(f"\n=== period X (real calendar pattern), population {pop}: {len(fe)} feeds ({100 * len(fe) / len(ftr):.2f}% of train feeds), {len(it):,} rows, mean M {M[fe].mean():.1f}; ALL rows pooled")
    print(f"{'window':6s} | {'TRAIN val-like zero/mean':>26s} {'VAL zero/mean':>16s} {'PSI':>6s} | {'TRAIN pub-like':>16s} {'PUBLIC':>14s} {'PSI':>6s} | {'TRAIN priv-like':>16s} {'PRIVATE':>14s} {'PSI':>6s}")
    for w, wn in enumerate(WIN):
        line = f"{wn:6s} |"
        for k in ("val-like", "public-like", "private-like"):
            rk = "val" if k == "val-like" else k
            a = c[seg[k], w]; b = cm[real[rk], w]
            p = psi(a, b)
            r[f"{wn}|all|{k}"] = dict(train=stat(a), real=stat(b), psi=p)
            line += f" {(a == 0).mean():10.4f}/{a.mean():6.3f}  {(b == 0).mean():8.4f}/{b.mean():6.3f} {p:6.3f} |"
        print(line)
    print("  per day, share with count >= 1 (percent) train / real:")
    for w, wn in enumerate(WIN):
        line = f"   {wn:4s}"
        for dx, dr in zip(DX, REALDAY):
            a = (c[d_r == dx, w] >= 1).mean(); b = (cm[day_vt == dr, w] >= 1).mean()
            r[f"day|{wn}|{dr}"] = [float(a), float(b)]
            line += f" | {100 * a:5.1f} / {100 * b:5.1f}"
        print(line)
    res["X"][pop] = r
# ---------------------------------------------------------------- noise floor: PSI between two independent subsamples of the same population
print("\n=== noise floor: PSI between two independent RM subsamples of period X (seeds 7 and 8), val-like rows, pooled")
c1, _, g1 = build(sample(DX, NX, "RM", np.random.default_rng(SEED)))
fe2 = sample(DX, NX, "RM", np.random.default_rng(SEED + 1)); c2, _, g2 = build(fe2)
res["noise_floor"] = {wn: psi(c1[:, w], c2[:, w]) for w, wn in enumerate(WIN)}
print("  " + " | ".join(f"{wn} {v:.4f}" for wn, v in res["noise_floor"].items()))
json.dump(res, open(f"{OUT}/s5_density.json", "w"), indent=1)
print(f"\nDONE {time.time() - t0:.0f}s")
