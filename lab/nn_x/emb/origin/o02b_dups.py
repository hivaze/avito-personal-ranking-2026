"""O2b: who are the bit-identical vectors? Pair-level comparison of
  DUP   pairs of distinct train items with bit-identical vectors (from out/hash_train.npy),
  NULL  random pairs of distinct train candidate items in the same category,
  TWIN  (val table) top-1 neighbour pairs with cos >= 0.995 that are not bit-identical (from o02a kNN),
on: same category, posting time (from f4, data/feats/life_f4_age_knots.npy) equal within one 128-s slot / 1 h / 1 day,
co-candidacy in at least one feed, co-exposure to the same (non-zero) user, |mean f5 difference|.
Bit-identical vectors for distinct item_ids imply the item tower has no item-ID input and that every input of the two
listings coincides; if those pairs are same-slot / same-user batches, they are one seller's multi-listings."""
import json, sys, time
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/feats_v2/life_")
from post_time import age_hours

T0 = time.time()
R = "<repo>"
OUT = f"{R}/nn_x/emb/origin/out"
res = {}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l(split, n):
    return np.load(f"{R}/data/flat/{split}/{n}.npy", mmap_mode="r")


knots = np.load(f"{R}/data/feats/life_f4_age_knots.npy")


def appearances(split, rows_mask_lookup):
    """candidate appearances (row, fidx, user, tpost, f5, cat) of table rows flagged in rows_mask_lookup (bool, N)."""
    f = pl.read_parquet(f"{R}/data/flat/{split}/feeds.parquet")
    ts = f["timestamp"].to_numpy().astype(np.float64); us = f["user_id"].to_numpy()
    E = l(split, "cand_emb_idx"); out = []
    CH = 50_000_000
    for s in range(0, len(E), CH):
        e = np.asarray(E[s:s + CH]); m = (e >= 0); m[m] = rows_mask_lookup[e[m]]
        idx = np.nonzero(m)[0]
        fi = np.asarray(l(split, "cand_fidx")[s:s + CH])[idx]
        a = age_hours(np.asarray(l(split, "cand_f4")[s:s + CH])[idx], knots)
        out.append(pl.DataFrame({"row": e[idx].astype(np.int64), "fidx": fi.astype(np.int64), "user": us[fi].astype(np.int64),
                                 "tpost": ts[fi] - a * 3600, "f5": np.asarray(l(split, "cand_f5")[s:s + CH])[idx].astype(np.float64),
                                 "cat": np.asarray(l(split, "cand_category_id")[s:s + CH])[idx].astype(np.int64)}))
    return pl.concat(out)


def pair_stats(pairs, app, tag):
    """pairs: DataFrame(a, b) of table rows. app: appearances."""
    per = app.group_by("row").agg(pl.col("tpost").median(), pl.col("f5").mean(), pl.col("cat").mode().first())
    p = pairs.join(per.rename({"row": "a", "tpost": "ta", "f5": "fa", "cat": "ca"}), on="a", how="inner") \
             .join(per.rename({"row": "b", "tpost": "tb", "f5": "fb", "cat": "cb"}), on="b", how="inner")
    dt = (p["ta"] - p["tb"]).abs().to_numpy()
    okt = np.isfinite(dt)
    fa = app.select("row", "fidx").unique()
    cof = p.select("a", "b").join(fa.rename({"row": "a"}), on="a").join(fa.rename({"row": "b"}), on=["b", "fidx"]) \
           .select("a", "b").unique().height
    ua = app.filter(pl.col("user") > 0).select("row", "user").unique()
    cou = p.select("a", "b").join(ua.rename({"row": "a"}), on="a").join(ua.rename({"row": "b"}), on=["b", "user"]) \
           .select("a", "b").unique().height
    r = {"pairs_with_both_seen_as_candidates": p.height,
         "same_category": float((p["ca"] == p["cb"]).mean()),
         "tpost_same_128s_slot": float((dt[okt] < 128).mean()), "tpost_within_1h": float((dt[okt] < 3600).mean()),
         "tpost_within_1d": float((dt[okt] < 86400).mean()), "tpost_absdiff_median_h": float(np.median(dt[okt]) / 3600),
         "co_candidates_in_a_feed": cof / max(p.height, 1), "co_exposed_same_user": cou / max(p.height, 1),
         "abs_f5_diff_median": float((p["fa"] - p["fb"]).abs().median()),
         "f5_equal_within_0.005": float(((p["fa"] - p["fb"]).abs() < 0.005).mean())}
    log(tag, json.dumps(r))
    return r


# ---------------- train DUP pairs
H = np.load(f"{OUT}/hash_train.npy")[:, 0]
o = np.argsort(H, kind="stable"); hs = H[o]
starts = np.r_[0, np.nonzero(hs[1:] != hs[:-1])[0] + 1]; ends = np.r_[starts[1:], len(hs)]
sz = ends - starts
g = np.nonzero(sz > 1)[0]
a = o[starts[g]]
pa, pb = [], []
for k in range(1, 6):  # first member paired with up to 5 others
    gg = g[sz[g] > k]
    pa.append(o[starts[gg]]); pb.append(o[starts[gg] + k])
dup = pl.DataFrame({"a": np.concatenate(pa).astype(np.int64), "b": np.concatenate(pb).astype(np.int64)})
log("dup pairs", dup.height, "groups", len(g))
# NULL: random same-category pairs among train candidate items (sample 2M candidate rows -> rows by category)
rng = np.random.default_rng(0)
NT = 81557944
E = l("train", "cand_emb_idx"); C = l("train", "cand_category_id")
smp = np.sort(rng.choice(len(E), 3_000_000, replace=False))
er = np.asarray(E[smp]); cr = np.asarray(C[smp]); k = er >= 0
d0 = pl.DataFrame({"row": er[k].astype(np.int64), "cat": cr[k].astype(np.int64)}).unique("row").sample(fraction=1.0, shuffle=True, seed=1)
d0 = d0.with_columns(pl.int_range(pl.len()).over("cat").alias("r"))
null = d0.filter(pl.col("r") % 2 == 0).with_columns((pl.col("r") + 1).alias("r")).join(d0, on=["cat", "r"]) \
         .select(pl.col("row").alias("a"), pl.col("row_right").alias("b")).head(dup.height * 2)
log("null pairs", null.height)
need = np.zeros(NT, bool)
need[dup["a"].to_numpy()] = True; need[dup["b"].to_numpy()] = True
need[null["a"].to_numpy()] = True; need[null["b"].to_numpy()] = True
app = appearances("train", need)
log("appearances", app.height)
res["train_dup"] = pair_stats(dup, app, "DUP")
res["train_null_samecat"] = pair_stats(null, app, "NULL")
# group-level: members' category agreement
# ---------------- val TWIN vs DUP
NV = 2516711
Hv = np.load(f"{OUT}/hash_val.npy")[:, 0]
I = np.load(f"{OUT}/knn_val_idx.npy"); D = np.load(f"{OUT}/knn_val_dot.npy")
mv = np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))
nv = np.linalg.norm(np.asarray(mv).astype(np.float32), axis=1)
c1 = D[:, 0] / nv / nv[I[:, 0]]
ident1 = Hv == Hv[I[:, 0]]
res["val_top1_neighbour"] = {"share_bit_identical_twin": float(ident1.mean()),
                             "share_cos_ge_0.999_not_identical": float(((c1 >= 0.999) & ~ident1).mean()),
                             "share_cos_ge_0.995_not_identical": float(((c1 >= 0.995) & ~ident1).mean()),
                             "share_cos_ge_0.99": float((c1 >= 0.99).mean()),
                             "top1_cos_q": [float(v) for v in np.quantile(c1, [.01, .1, .25, .5, .75, .9, .99])]}
log("val top1", res["val_top1_neighbour"])
twin_rows = np.nonzero((c1 >= 0.995) & ~ident1)[0]
twin = pl.DataFrame({"a": twin_rows.astype(np.int64), "b": I[twin_rows, 0].astype(np.int64)})
dupv_rows = np.nonzero(ident1)[0]
dupv = pl.DataFrame({"a": dupv_rows.astype(np.int64), "b": I[dupv_rows, 0].astype(np.int64)})
needv = np.zeros(NV, bool)
for df in (twin, dupv):
    needv[df["a"].to_numpy()] = True; needv[df["b"].to_numpy()] = True
appv = appearances("val", needv)
res["val_dup"] = pair_stats(dupv, appv, "VAL DUP")
res["val_twin_0.995"] = pair_stats(twin, appv, "VAL TWIN")
json.dump(res, open(f"{OUT}/o02b.json", "w"), indent=1)
log("done")
