"""Signals (a): the same user's EARLIER labelled feeds. Leak-safe: only bank feeds with timestamp < feed timestamp - gap; user 0 excluded.
val: bank = train (all earlier by construction, min gap ~2.2 days). holdout: bank = train feeds of the same user strictly earlier
(gap 0 = any earlier; gap 2d = mimics the train->val distance). Output: bank_<split>[_g2].npz with a row-aligned float32 matrix + names."""
from hx import *
T = FLAT / "train"
ftr = pl.read_parquet(T / "feeds.parquet"); tu = ftr["user_id"].to_numpy(); tM = ftr["M"].to_numpy().astype(np.int64); toff = ftr["cand_off"].to_numpy(); tts = ftr["timestamp"].to_numpy()
V = load_val(); Hd = load_hold()
users = np.unique(np.r_[V["uid"], Hd["uid"]]); users = users[users != 0]
sel = np.nonzero(np.isin(tu, users))[0]; print("bank feeds", len(sel), "users", len(users), flush=True)
rows = np.concatenate([np.arange(toff[i], toff[i] + tM[i]) for i in sel])
ld = lambda n: np.load(T / f"cand_{n}.npy", mmap_mode="r")[rows]
pos = ld("pos").astype(np.float64)
B = pl.DataFrame({"bf": np.repeat(sel, tM[sel]).astype(np.int32), "u": np.repeat(tu[sel], tM[sel]).astype(np.int64), "ts0": np.repeat(tts[sel], tM[sel]).astype(np.int64),
                  "item": ld("item_id").astype(np.int64), "y": ld("target").astype(np.float64), "c": ld("category_id").astype(np.int32),
                  "exam": 1.0 / np.log2(pos + 2.0), "pos": pos})
B = B.with_columns((pl.col("exam") * (1 - pl.col("y"))).alias("skip"))
print("bank rows", B.height, "pos rate %.5f" % B["y"].mean(), flush=True)
Bfc = B.group_by(["bf", "u", "ts0", "c"]).agg(pl.len().alias("n"), pl.col("y").sum(), pl.col("skip").sum())
Bf = B.group_by(["bf", "u", "ts0"]).agg(pl.len().alias("n"), pl.col("y").sum())

def build(D, split, gap, tag):
    M, n = D["M"], D["n"]; fid = D["fid"]; src = FLAT / ("val" if split == "val" else "train"); rws = D["rows"]
    g = lambda nm: (np.load(src / f"cand_{nm}.npy") if rws is None else np.load(src / f"cand_{nm}.npy", mmap_mode="r")[rws])
    C = pl.DataFrame({"row": np.arange(len(fid)), "f": fid.astype(np.int32), "u": np.repeat(D["uid"], M).astype(np.int64), "ts": np.repeat(D["ts"], M).astype(np.int64),
                      "item": g("item_id").astype(np.int64), "c": g("category_id").astype(np.int32)})
    ui = (C.filter(pl.col("u") != 0).join(B.select("u", "item", "ts0", "y", "exam", "skip", "pos"), on=["u", "item"], how="inner").filter(pl.col("ts0") < pl.col("ts") - gap)
          .group_by("row").agg(pl.len().alias("ui_n"), pl.col("y").sum().alias("ui_y"), pl.col("skip").sum().alias("ui_skip"), pl.col("pos").min().alias("ui_minpos"),
                               ((pl.col("ts") - pl.col("ts0")).min() / 86400.0).alias("ui_gapd")))
    FC = C.filter(pl.col("u") != 0).select("f", "u", "ts", "c").unique()
    uc = (FC.join(Bfc, on=["u", "c"], how="inner").filter(pl.col("ts0") < pl.col("ts") - gap)
          .group_by(["f", "c"]).agg(pl.col("n").sum().alias("uc_n"), pl.col("y").sum().alias("uc_y"), pl.col("skip").sum().alias("uc_skip"), pl.len().alias("uc_feeds"),
                                    ((pl.col("ts") - pl.col("ts0")).min() / 86400.0).alias("uc_gapd")))
    FF = pl.DataFrame({"f": np.arange(n, dtype=np.int32), "u": D["uid"].astype(np.int64), "ts": D["ts"].astype(np.int64)}).filter(pl.col("u") != 0)
    uu = (FF.join(Bf, on="u", how="inner").filter(pl.col("ts0") < pl.col("ts") - gap)
          .group_by("f").agg(pl.col("n").sum().alias("u_n"), pl.col("y").sum().alias("u_y"), pl.len().alias("u_feeds"), ((pl.col("ts") - pl.col("ts0")).min() / 86400.0).alias("u_gapd")))
    C = C.join(ui, on="row", how="left").join(uc, on=["f", "c"], how="left").join(uu, on="f", how="left").sort("row")
    zc = ["ui_n", "ui_y", "ui_skip", "uc_n", "uc_y", "uc_skip", "uc_feeds", "u_n", "u_y", "u_feeds"]
    C = C.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in zc])
    C = C.with_columns(
        ((pl.col("uc_y") + 20 * (pl.col("u_y") + 1) / (pl.col("u_n") + 50)) / (pl.col("uc_n") + 20)).log().alias("uc_ctr_log"),          # the report's signal
        ((pl.col("uc_y") + 1) / (pl.col("u_y") + 2)).alias("uc_clickshare"), ((pl.col("uc_n") + 1) / (pl.col("u_n") + 50)).log().alias("uc_expo_log"),
        (pl.col("uc_skip") / (pl.col("uc_n") + 1)).alias("uc_skiprate"), ((pl.col("u_y") + 1) / (pl.col("u_n") + 50)).alias("u_ctr"),
        (pl.col("ui_n") > 0).cast(pl.Float64).alias("ui_shown"), (pl.col("ui_n") - pl.col("ui_y")).alias("ui_nonclick"))
    has = (C["u_feeds"].to_numpy() > 0)
    names = ["ui_shown", "ui_n", "ui_y", "ui_nonclick", "ui_skip", "ui_minpos", "ui_gapd", "uc_n", "uc_y", "uc_skip", "uc_feeds", "uc_gapd", "uc_ctr_log", "uc_clickshare", "uc_expo_log",
             "uc_skiprate", "u_n", "u_y", "u_feeds", "u_gapd", "u_ctr"]
    X = np.stack([C[c].to_numpy().astype(np.float32) for c in names], 1)
    # within-feed z views of the row-varying signals (0 on feeds without an earlier labelled feed)
    zn = ["ui_shown", "ui_skip", "uc_ctr_log", "uc_clickshare", "uc_expo_log", "uc_skiprate", "uc_y", "uc_n"]
    Z = np.stack([np.where(has, zfeed(np.nan_to_num(C[c].to_numpy().astype(np.float64)), D["off"], M), 0.0).astype(np.float32) for c in zn], 1)
    X = np.hstack([X, Z]); names += [c + "_fz" for c in zn]
    fh = np.bincount(fid, has, n) > 0; fp = np.bincount(fid, C["ui_n"].to_numpy() > 0, n) > 0; y = D["y"]; sh = C["ui_n"].to_numpy() > 0; ck = C["ui_y"].to_numpy() > 0
    print(f"{tag}: feeds with an earlier labelled feed {fh.mean():.4f}; feeds with a re-shown pair {fp.mean():.4f}; rows re-shown {sh.mean():.5f} (n {sh.sum()}); rows re-shown after a click {ck.sum()}", flush=True)
    print(f"   target rate re-shown {y[sh].mean():.4f} vs other rows of the same feeds {y[(~sh) & fp[fid]].mean():.4f}; mean base z of re-shown rows {D['base'][sh].mean():+.3f}; earlier-click rows target rate {y[ck].mean() if ck.sum() else float('nan'):.4f}")
    np.savez(OUT / f"bank_{tag}.npz", X=X, names=np.array(names), feed_has=fh, feed_pair=fp)

build(V, "val", 0, "val")
build(Hd, "hold", 0, "hold_g0")
build(Hd, "hold", 2 * 86400, "hold_g2")
# test coverage (bank = train only here; val users' labelled val feeds would add to it)
for sp in ("test",):
    f, M, off = feed_meta(sp); u = f["user_id"].to_numpy(); ts = f["timestamp"].to_numpy(); pub = ts < 1790497920
    intr = np.isin(u, np.unique(tu)) & (u != 0); inval = np.isin(u, V["uid"]) & (u != 0)
    print(f"test: user has train feeds: public {intr[pub].mean():.4f} private {intr[~pub].mean():.4f}; train or val: public {(intr | inval)[pub].mean():.4f} private {(intr | inval)[~pub].mean():.4f}")
