"""Follow-ups (1) s4 item-to-item graph and (2) region-normalised popularity: BUILD (label-free) features for val and the
fixed 30K train holdout. Definitions: nn_x/codes/followup/PREDECLARED.txt (written 05:47:47 UTC before this script).
CPU only, 12 threads. Outputs: nn_x/codes/followup/out/{val,hold}_<feature>.npy (+ val_home_f2.npy, val_region.npy, meta).
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json
import numpy as np, polars as pl, numba as nb
from pathlib import Path

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; OUT = ROOT / "nn_x/codes/followup/out"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOG = open(OUT / "f01_build.log", "w")


def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


D0 = 20702
SH = 27  # item ids < 2^27
day_of = lambda ts: ((ts.astype(np.int64) + 10800) // 86400 - D0).astype(np.int8)

# ------------------------------------------------------------------ feeds
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64); offtr = ftr["cand_off"].to_numpy().astype(np.int64)
utr = ftr["user_id"].to_numpy().astype(np.int64); nftr = len(ftr)
dtr = day_of(ftr["timestamp"].to_numpy())
ukey = np.where(utr > 0, utr, 10_000_000 + np.arange(nftr))      # user-0 feeds = separate users
_, uid_tr = np.unique(ukey, return_inverse=True); uid_tr = uid_tr.astype(np.int32)
log(f"train feeds {nftr}, distinct user keys {uid_tr.max()+1}, days {np.unique(dtr).tolist()}")

# holdout feeds (nn/holdout_eval.py)
pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mtr < 50] = 0.0
cand = np.where((pr > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh_ref, oh_ref = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert (Mh_ref == Mtr[hold]).all(), "holdout selection mismatch"
hrows = np.concatenate([np.arange(o, o + m) for o, m in zip(offtr[hold], Mtr[hold])])
yh_ref = np.load(ROOT / "preds/holdout/holdout_y.npy")
assert (np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[hrows] == yh_ref).all(), "holdout y mismatch"
log(f"holdout: {len(hold)} feeds, {len(hrows)} rows, days {np.unique(dtr[hold], return_counts=True)}")

# ------------------------------------------------------------------ home region per feed (K2 vote at the modal f2 level)
k2 = np.load(ROOT / "nn_x/codes/features/out/k2_item_comp.npy").astype(np.int64)
k2 = k2[np.argsort(k2[:, 0])]


def comp_of(items):
    i = np.searchsorted(k2[:, 0], items); i = np.minimum(i, len(k2) - 1)
    return np.where(k2[i, 0] == items, k2[i, 1], -1).astype(np.int32)


@nb.njit(parallel=True, cache=True)
def home_region(off, M, f2, comp):
    n = len(M); R = np.full(n, -1, np.int32); mode = np.zeros(n, np.float32); nvote = np.zeros(n, np.int32)
    for g in nb.prange(n):
        o = off[g]; m = M[g]
        v = np.sort(f2[o:o + m])
        best = 0.0; bestc = 0; i = 0
        while i < m:
            j = i
            while j < m and v[j] == v[i]:
                j += 1
            if v[i] != 0 and ((j - i) > bestc or ((j - i) == bestc and v[i] > best)):
                best = v[i]; bestc = j - i
            i = j
        mode[g] = best
        if bestc == 0:
            continue
        cs = np.empty(m, np.int32); k = 0
        for r in range(o, o + m):
            if f2[r] == best and comp[r] >= 0:
                cs[k] = comp[r]; k += 1
        if k == 0:
            continue
        cs = np.sort(cs[:k]); bc = -1; bn = 0; i = 0
        while i < k:
            j = i
            while j < k and cs[j] == cs[i]:
                j += 1
            if (j - i) > bn:
                bn = j - i; bc = cs[i]
            i = j
        R[g] = bc; nvote[g] = bn
    return R, mode, nvote


f2tr = np.load(FLAT / "train/cand_f2.npy"); citr = np.load(FLAT / "train/cand_item_id.npy").astype(np.int64)
comptr = comp_of(citr)
Rtr, modetr, nvtr = home_region(offtr, Mtr, f2tr, comptr)
log(f"train feeds with a home region {np.mean(Rtr >= 0):.4f}; distinct regions {len(np.unique(Rtr[Rtr >= 0]))}; "
    f"candidate rows with a K2 comp {np.mean(comptr >= 0):.4f}")
del f2tr, comptr
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64); nfv = len(fv)
dv = day_of(fv["timestamp"].to_numpy())
f2v = np.load(FLAT / "val/cand_f2.npy"); civ = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)
Rv, modev, nvv = home_region(ov, Mv, f2v, comp_of(civ))
log(f"val feeds with a home region {np.mean(Rv >= 0):.4f}; holdout feeds with a home region {np.mean(Rtr[hold] >= 0):.4f}")
np.save(OUT / "val_region.npy", Rv); np.save(OUT / "val_home_f2.npy", modev); np.save(OUT / "hold_region.npy", Rtr[hold])

# ------------------------------------------------------------------ train histories
hf = np.load(FLAT / "train/hist_fidx.npy")
ht = np.load(FLAT / "train/hist_event_type.npy"); hs = np.load(FLAT / "train/hist_event_source.npy")
hi = np.load(FLAT / "train/hist_item_id.npy")
assert hi.max() < 2 ** SH
log(f"train history events {len(hf):,}; e1 share {np.mean(ht == 1):.4f}; s4 share {np.mean(hs == 4):.4f}")

# (1) s4 edges: B = e1/s4 view; A = nearest older e1 event in the same snapshot with item != B
e1 = np.flatnonzero(ht == 1)
s4k = np.flatnonzero(hs[e1] == 4)                       # positions in the e1 sequence
log(f"e1 events {len(e1):,}; e1&s4 events {len(s4k):,}")
A = np.full(len(s4k), -1, np.int64)
todo = np.arange(len(s4k)); step = 1
while len(todo) and step <= 8:
    k = s4k[todo] + step
    ok = k < len(e1)
    kk = np.minimum(k, len(e1) - 1)
    same = ok & (hf[e1[kk]] == hf[e1[s4k[todo]]])
    diff = same & (hi[e1[kk]] != hi[e1[s4k[todo]]])
    A[todo[diff]] = hi[e1[kk[diff]]]
    todo = todo[same & ~diff]; step += 1
bi = e1[s4k]
has = A >= 0
log(f"s4 views with an older e1 anchor: {has.mean():.4f} ({has.sum():,})")
fB = hf[bi[has]]
EU = pl.DataFrame({"u": uid_tr[fB], "key": (A[has] << SH) | hi[bi[has]].astype(np.int64), "day": dtr[fB]})
EU = EU.group_by(["u", "key"]).agg(pl.col("day").min())
log(f"unique (user, A, B) triples {len(EU):,}")
E = EU.group_by(["key", "day"]).agg(pl.len().alias("cnt").cast(pl.Int32))
E = E.with_columns((pl.col("key") // (2 ** SH)).alias("A"), (pl.col("key") % (2 ** SH)).alias("B")).drop("key")
nE = E.group_by(["A", "B"]).agg(pl.col("cnt").sum())
log(f"distinct directed edges {len(nE):,}; edges with c>=2 {(nE['cnt'] >= 2).sum():,}; distinct anchors {nE['A'].n_unique():,}; "
    f"distinct targets {nE['B'].n_unique():,}; c quantiles {np.quantile(nE['cnt'].to_numpy(), [.5, .9, .99]).tolist()}")
E.write_parquet(OUT / "s4_edges_by_day.parquet")
del EU, A, bi, e1, s4k

# (2) region popularity: unique (user, region, item) over history events of any type
Rrow = Rtr[hf]
P = pl.DataFrame({"u": uid_tr[hf], "R": Rrow.astype(np.int16), "item": hi, "day": dtr[hf]})
del Rrow
P = P.filter(pl.col("R") >= 0).group_by(["u", "R", "item"]).agg(pl.col("day").min())
log(f"unique (user, region, item) with a region: {len(P):,}")
TR = P.group_by(["R", "item", "day"]).agg(pl.len().alias("cnt").cast(pl.Int32))
del P
TR.write_parquet(OUT / "region_item_by_day.parquet")
log(f"(region, item, day) rows {len(TR):,}; regions {TR['R'].n_unique()}")
del hf, ht, hs, hi


# ------------------------------------------------------------------ feature computation
def hist_frames(split, fids):
    """history (feed-local index g, item, type, hpos) for the given feed ids of a split."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    H = f["H"].to_numpy().astype(np.int64)[fids]; ho = f["hist_off"].to_numpy().astype(np.int64)[fids]
    rows = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, H)]) if H.sum() else np.zeros(0, np.int64)
    g = np.repeat(np.arange(len(fids)), H)
    it = np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r")[rows].astype(np.int64)
    ty = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")[rows]
    hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")[rows]
    return pl.DataFrame({"g": g, "item": it, "type": ty, "hpos": hp})


def s4_features(hist, cand, fday):
    """hist: g,item,type,hpos; cand: g,item,row; fday: per-feed day (None -> all edges)."""
    Ed = E if fday is None else E  # filtered after the join
    R32 = (hist.filter(pl.col("type") == 1).sort(["g", "hpos"]).unique(["g", "item"], keep="first", maintain_order=True)
           .with_columns(pl.int_range(pl.len()).over("g").alias("rk")).filter(pl.col("rk") < 32).select(["g", pl.col("item").alias("A")]))
    allA = hist.select(["g", pl.col("item").alias("A")]).unique()
    FD = pl.DataFrame({"g": np.arange(len(fday)), "fday": fday}) if fday is not None else None

    def filt(J):
        if FD is None:
            return J
        return J.join(FD, on="g").filter(pl.col("day") < pl.col("fday")).drop("fday")

    out = {}
    # directed
    J = filt(R32.join(Ed, on="A"))
    c = J.group_by(["g", "A", "B"]).agg(pl.col("cnt").sum().alias("c"))
    c = c.join(c.group_by(["g", "A"]).agg(pl.col("c").sum().alias("out")), on=["g", "A"])
    lk = c.group_by(["g", "B"]).agg((pl.col("c") / pl.col("out")).max().alias("s4_link"), pl.col("c").sum().alias("csum"))
    # symmetric
    U = pl.concat([Ed.select(["A", "B", "day", "cnt"]), Ed.select([pl.col("B").alias("A"), pl.col("A").alias("B"), "day", "cnt"])])
    Js = filt(R32.join(U, on="A"))
    cs = Js.group_by(["g", "A", "B"]).agg(pl.col("cnt").sum().alias("c"))
    cs = cs.join(cs.group_by(["g", "A"]).agg(pl.col("c").sum().alias("deg")), on=["g", "A"])
    sy = cs.group_by(["g", "B"]).agg((pl.col("c") / pl.col("deg")).max().alias("s4_sym"))
    # any history item
    Ja = filt(allA.join(Ed, on="A"))
    an = Ja.group_by(["g", "B"]).agg(pl.lit(1.0).alias("s4_link_any"))
    C = cand.join(lk, left_on=["g", "item"], right_on=["g", "B"], how="left") \
            .join(sy, left_on=["g", "item"], right_on=["g", "B"], how="left") \
            .join(an, left_on=["g", "item"], right_on=["g", "B"], how="left").sort("row")
    out["s4_link"] = C["s4_link"].fill_null(0).to_numpy().astype(np.float64)
    out["s4_sum"] = np.log1p(C["csum"].fill_null(0).to_numpy().astype(np.float64))
    out["s4_sym"] = C["s4_sym"].fill_null(0).to_numpy().astype(np.float64)
    out["s4_link_any"] = C["s4_link_any"].fill_null(0).to_numpy().astype(np.float64)
    return out


def region_features(cand, Rf, fday, m=5.0):
    """cand: g,item,row; Rf: per-feed region (-1 none); fday per-feed day or None."""
    FD = pl.DataFrame({"g": np.arange(len(Rf)), "R": Rf.astype(np.int16),
                       "fday": (fday if fday is not None else np.full(len(Rf), 99)).astype(np.int8)})
    T = TR
    NR = T.group_by(["R", "day"]).agg(pl.col("cnt").sum().alias("n"))
    TI = T.group_by(["item", "day"]).agg(pl.col("cnt").sum().alias("n"))
    C = cand.join(FD, on="g")
    a = C.filter(pl.col("R") >= 0).join(T, on=["R", "item"]).filter(pl.col("day") < pl.col("fday")).group_by("row").agg(pl.col("cnt").sum().alias("niR"))
    b = C.filter(pl.col("R") >= 0).join(TI, on="item").filter(pl.col("day") < pl.col("fday")).group_by("row").agg(pl.col("n").sum().alias("ni"))
    # pi_R per feed
    fr = FD.filter(pl.col("R") >= 0)
    nr = fr.join(NR, on="R").filter(pl.col("day") < pl.col("fday")).group_by("g").agg(pl.col("n").sum().alias("NR"))
    nt = fr.join(NR.group_by("day").agg(pl.col("n").sum().alias("nall")), how="cross").filter(pl.col("day") < pl.col("fday")) \
           .group_by("g").agg(pl.col("nall").sum().alias("N"))
    C = C.join(a, on="row", how="left").join(b, on="row", how="left").join(nr, on="g", how="left").join(nt, on="g", how="left").sort("row")
    niR = C["niR"].fill_null(0).to_numpy().astype(np.float64); ni = C["ni"].fill_null(0).to_numpy().astype(np.float64)
    NRv = C["NR"].fill_null(0).to_numpy().astype(np.float64); Nv = C["N"].fill_null(0).to_numpy().astype(np.float64)
    hasR = (C["R"].to_numpy() >= 0) & (NRv > 0) & (Nv > 0)
    pi = np.where(hasR, NRv / np.maximum(Nv, 1), 1.0)
    lift = np.where(hasR, np.log(((niR + m * pi) / (ni + m)) / pi), 0.0)
    cnt = np.where(hasR, np.log1p(niR), 0.0)
    return {"rp_lift": lift, "rp_cnt": cnt}, dict(rows_with_region=float(hasR.mean()), niR_pos=float(np.mean(niR[hasR] > 0)) if hasR.any() else 0.0)


# val
hv = hist_frames("val", np.arange(nfv))
cv = pl.DataFrame({"g": np.repeat(np.arange(nfv), Mv), "item": civ, "row": np.arange(len(civ))})
FV = s4_features(hv, cv, None)
RV, rinfo = region_features(cv, Rv, None)
for k, x in {**FV, **RV}.items():
    np.save(OUT / f"val_{k}.npy", x.astype(np.float32))
    log(f"val {k:12s}: nonzero share {np.mean(x != 0):.4f}, mean {x.mean():.4f}, max {x.max():.4f}")
log(f"val region info {rinfo}")
# holdout
hh = hist_frames("train", hold)
ch = pl.DataFrame({"g": np.repeat(np.arange(len(hold)), Mtr[hold]), "item": citr[hrows], "row": np.arange(len(hrows))})
FH = s4_features(hh, ch, dtr[hold].astype(np.int8))
RH, rinfo_h = region_features(ch, Rtr[hold], dtr[hold].astype(np.int8))
for k, x in {**FH, **RH}.items():
    np.save(OUT / f"hold_{k}.npy", x.astype(np.float32))
    log(f"hold {k:12s}: nonzero share {np.mean(x != 0):.4f}, mean {x.mean():.4f}, max {x.max():.4f}")
log(f"holdout region info {rinfo_h}")
np.save(OUT / "hold_rows.npy", hrows); np.save(OUT / "hold_fidx.npy", hold)
log("done")
