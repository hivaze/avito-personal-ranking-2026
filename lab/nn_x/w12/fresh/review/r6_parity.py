"""CHECK 6: can a train-row version with the val / test distribution be built? My own construction (label-free PSI part;
plus a descriptive within-feed AUC with TRAIN labels, no model).

Pseudo-period X' (real calendar offsets 0, 2 | 4, 5, 6, 7): train days 20702, 20704 | 20706, 20707, 20708, 20709 (the
builder used 20703 ...). Sources = random-like train feeds (p_random > 0.9) drawn per day at the real feed numbers:
  S-R  : plain random-like sample;   S-RM : random-like sample re-weighted to the val + test feed-size deciles.
Queries: (i) the sampled feeds themselves, (ii) 20,000 production-ordered train feeds (p_random < 0.1) of the same days,
(iii) 20,000 train feeds drawn from ALL feeds of the same days. Counts: sources with ts < t, other users only.
"""
import json
import time

import numpy as np
import polars as pl

from rlib import BOUND, MSK, OUT, ROOT, psi

t0 = time.time()
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
ts = f["timestamp"].to_numpy().astype(np.int64); uid = f["user_id"].to_numpy().astype(np.int64)
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
day = (ts + MSK) // 86400
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
p_random = np.zeros(len(f)); p_random[pr["fidx"]] = np.nan_to_num(pr["p_random"], nan=0.0)
p_known = np.zeros(len(f), bool); p_known[pr["fidx"]] = ~np.isnan(pr["p_random"])
items = np.load(f"{ROOT}/data/flat/train/cand_item_id.npy", mmap_mode="r")
ytr = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
# real side
C = np.load(f"{OUT}/my_counts.npy"); meta = np.load(f"{OUT}/my_rowmeta.npz")
rts, rday, rsp = meta["ts"], meta["day"], meta["sp"]
real = {"val": rsp == 0, "public-like": (rsp == 1) & (rts < BOUND), "private-like": (rsp == 1) & (rts >= BOUND)}
Mvt = np.concatenate([pl.read_parquet(f"{ROOT}/data/flat/{s}/feeds.parquet")["M"].to_numpy() for s in ("val", "test")]).astype(np.int64)
edges = np.unique(np.quantile(Mvt, np.linspace(0, 1, 11)[1:-1]))
share_vt = np.bincount(np.digitize(Mvt, edges), minlength=len(edges) + 1) / len(Mvt)
DAYS = [20702, 20704, 20706, 20707, 20708, 20709]; NS = [7542, 6878, 6050, 7367, 7578, 7600]
REALDAY = [20718, 20720, 20722, 20723, 20724, 20725]
TOD_BOUND = (BOUND + MSK) % 86400
rng = np.random.default_rng(31)


def sample_sources(matchM):
    out = []
    for d, n in zip(DAYS, NS):
        idx = np.flatnonzero((day == d) & (p_random > 0.9))
        if not matchM:
            out.append(rng.choice(idx, n, replace=False)); continue
        b = np.digitize(M[idx], edges)
        w = share_vt[b] / np.maximum(np.bincount(b, minlength=len(share_vt))[b], 1)
        out.append(rng.choice(idx, n, replace=False, p=w / w.sum()))
    return np.sort(np.concatenate(out))


def rows_of(fe):
    r = np.concatenate([np.arange(off[x], off[x] + M[x]) for x in fe])
    g = np.repeat(fe, M[fe])
    uk = np.where(uid[g] == 0, -1 - g, uid[g])
    return pl.DataFrame({"g": g, "uk": uk, "ts": ts[g], "item": np.asarray(items[r]).astype(np.int64), "row": r})


def asof(q, s):
    """as-of count of every query row against the source rows (other feed, other user, earlier tick)."""
    j = q.lazy().join(s.lazy().select(pl.col("item"), pl.col("g").alias("g2"), pl.col("uk").alias("uk2"), pl.col("ts").alias("ts2")), on="item", how="inner") \
        .filter((pl.col("g") != pl.col("g2")) & (pl.col("uk") != pl.col("uk2")) & (pl.col("ts2") < pl.col("ts"))) \
        .group_by("row").agg(pl.len().alias("a"), (pl.col("ts2") >= pl.col("ts") - 86400).sum().alias("a24")).collect()
    out = q.join(j, on="row", how="left").fill_null(0)
    return out["a"].to_numpy().astype(np.int64), out["a24"].to_numpy().astype(np.int64)


def seg_of(g):
    d = day[g]; tod = (ts[g] + MSK) % 86400
    return {"val": (d == DAYS[0]) | (d == DAYS[1]), "public-like": (d == DAYS[2]) | ((d == DAYS[3]) & (tod < TOD_BOUND)),
            "private-like": ((d == DAYS[3]) & (tod >= TOD_BOUND)) | (d == DAYS[4]) | (d == DAYS[5])}


def auc_within(g, x, y):
    """mean over feeds (with a positive and a negative) of the within-feed AUC of x for y, ties 0.5."""
    o = np.argsort(g, kind="stable"); g, x, y = g[o], x[o], y[o]
    b = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True]); out = []
    for a_, b_ in zip(b[:-1], b[1:]):
        yy = y[a_:b_]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        xx = x[a_:b_]; p = xx[yy > 0]; n = xx[yy == 0]
        out.append(((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()))
    out = np.array(out)
    return float(out.mean()), float(out.std() / np.sqrt(len(out))), int(len(out))


res = {"days": DAYS, "real_days": REALDAY}
print(f"val + test M mean {Mvt.mean():.1f}; train M mean {M.mean():.1f}; random-like {M[p_random > 0.9].mean():.1f}; production-ordered (p < 0.1) {M[p_random < 0.1].mean():.1f}; "
      f"share of train feeds random-like {100 * (p_random > 0.9).mean():.1f}%, production-ordered {100 * (p_random < 0.1).mean():.1f}%, p unknown {100 * (~p_known).mean():.1f}%")
for tag, matchM in (("S-RM", True), ("S-R", False)):
    src_f = sample_sources(matchM)
    S = rows_of(src_f)
    print(f"\n=== sources {tag}: {len(src_f)} feeds, {len(S)} rows, mean M {M[src_f].mean():.1f} | {time.time() - t0:.0f}s", flush=True)
    res[tag] = {"mean_M": float(M[src_f].mean())}
    pools = {"(i) the sampled feeds": src_f}
    if matchM:
        inper = np.isin(day, DAYS)
        notsrc = np.ones(len(f), bool); notsrc[src_f] = False
        pools["(ii) production-ordered feeds"] = np.sort(rng.choice(np.flatnonzero(inper & (p_random < 0.1) & notsrc), 20000, replace=False))
        pools["(iii) all train feeds"] = np.sort(rng.choice(np.flatnonzero(inper & notsrc), 20000, replace=False))
    for pn, qf in pools.items():
        Q = S if pn.startswith("(i)") else rows_of(qf)
        a, a24 = asof(Q, S)
        g = Q["g"].to_numpy(); sg = seg_of(g); d = day[g]
        r = {"feeds": int(len(qf)), "rows": int(len(Q)), "mean_M": float(M[qf].mean())}
        line = f"  queries {pn} [{len(qf)} feeds, mean M {M[qf].mean():.1f}]: as-of coverage per day train / real: "
        for dd, rd in zip(DAYS, REALDAY):
            ct = float((a[d == dd] >= 1).mean()); cr = float((C[rday == rd, 0] >= 1).mean())
            r[f"cov_day{rd}"] = [ct, cr]; line += f"{100 * ct:.1f}/{100 * cr:.1f} "
        print(line)
        for k in ("val", "public-like", "private-like"):
            pa = psi(a[sg[k]], C[real[k], 0]); p24 = psi(a24[sg[k]], C[real[k], 1])
            r[f"psi_a|{k}"] = pa; r[f"psi_a24|{k}"] = p24
            r[f"zero_a|{k}"] = [float((a[sg[k]] == 0).mean()), float((C[real[k], 0] == 0).mean())]
            print(f"    {k:13s}: PSI a {pa:.3f}  a24 {p24:.3f} | zero share a train {(a[sg[k]] == 0).mean():.3f} real {(C[real[k], 0] == 0).mean():.3f} | mean a train {a[sg[k]].mean():.3f} real {C[real[k], 0].mean():.3f}")
        # descriptive, TRAIN labels: within-feed AUC of the as-of count for the click
        y = np.asarray(ytr[Q["row"].to_numpy()]).astype(np.float64)
        au = auc_within(g, a.astype(np.float64), y)
        r["auc_asof_train_click"] = au
        print(f"    within-feed AUC of the as-of count for the TRAIN click: {au[0]:.4f} +- {au[1]:.4f} ({au[2]} feeds)", flush=True)
        res[tag][pn] = r
json.dump(res, open(f"{OUT}/r6_parity.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
