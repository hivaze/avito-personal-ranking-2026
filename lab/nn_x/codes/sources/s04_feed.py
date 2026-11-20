"""s04: association of each source with the recommendation feed itself, the candidate pool, and listing freshness.
(A) train consecutive same-user feed pairs A->B (gap <= 1 day), users sampled: for each (type, source) of B's history events
    whose item is NOT in A's history ("new" events): P(item was a candidate of A), P(item was a positive of A).
    And for A's positives / negatives found in B's history: distribution over (type, source) of their events.
(B) val/test history events by source: P(item is a candidate anywhere in the same split / in train), P(item was a candidate
    (positive) in one of the same user's TRAIN feeds), P(item reappears as a candidate in a LATER same-split feed of the user).
(C) listing age at the history snapshot, by source: posting time = candidate timestamp - age(f4) from any split's candidate
    appearance; age at feed time for events with hpos < 30 (recent). Fresh-listing surfaces (new-listing notifications,
    saved searches) should show young items.
"""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
sys.path.insert(0, "<repo>/feats_v2/life_")
from scommon import load_hist, load_cand, feeds, pshow, OUT, FLAT
from post_time import age_hours

KN = np.load("<repo>/data/feats/life_f4_age_knots.npy")


def part_A(n_users=60_000, seed=3):
    f = feeds("train").filter(pl.col("user_id") > 0)
    multi = f.group_by("user_id").len().filter(pl.col("len") >= 2)["user_id"].to_numpy()
    rng = np.random.default_rng(seed)
    us = rng.choice(multi, min(n_users, len(multi)), replace=False)
    fu = f.filter(pl.col("user_id").is_in(us)).sort(["user_id", "timestamp", "fidx"])
    fu = fu.with_columns(pl.col("fidx").shift(1).over("user_id").alias("A"), (pl.col("timestamp") - pl.col("timestamp").shift(1).over("user_id")).alias("gap"))
    pairs = fu.filter(pl.col("A").is_not_null() & (pl.col("gap") <= 86400)).select(pl.col("A").cast(pl.Int32), pl.col("fidx").alias("B"), "gap")
    print(f"(A) users {len(us)}, pairs (gap<=1d) {pairs.height}")
    fsel = np.unique(np.concatenate([pairs["A"].to_numpy(), pairs["B"].to_numpy()]))
    H, _ = load_hist("train", fsel)
    C, _ = load_cand("train", pairs["A"].to_numpy(), cols=("item_id", "target"))
    hA = H.select(pl.col("fidx").alias("A"), "item").unique()
    cA = C.select(pl.col("fidx").alias("A"), pl.col("item_id").alias("item"), "target").unique(subset=["A", "item"])
    hB = H.join(pairs, left_on="fidx", right_on="B").select("A", "fidx", "hpos", "item", "et", "es")
    hB = hB.join(hA.with_columns(pl.lit(True).alias("inA")), on=["A", "item"], how="left").with_columns(pl.col("inA").fill_null(False))
    new = hB.filter(~pl.col("inA")).join(cA, on=["A", "item"], how="left")
    t = new.group_by(["et", "es"]).agg(pl.len().alias("n_new_events"), pl.col("target").is_not_null().mean().alias("P_A_cand"),
                                      (pl.col("target") == 1).mean().alias("P_A_pos"), pl.col("hpos").mean().alias("hpos_mean")).sort(["et", "es"])
    t = t.filter((pl.col("n_new_events") >= 50) & (pl.col("es") > 0) | (pl.col("es") == 0) & pl.col("et").is_in([1, 4, 7]))
    print("(A1) B's new events (item not in A's history), by (type, source): P(item was an A candidate / A positive)")
    pshow(t, 80)
    t.write_csv(f"{OUT}/s04A1_newevents_feedorigin.csv")
    t0 = new.filter(~pl.col("et").is_in([1, 4, 7])).group_by("et").agg(pl.len().alias("n"), pl.col("target").is_not_null().mean().alias("P_A_cand"), (pl.col("target") == 1).mean().alias("P_A_pos")).sort("et")
    print("(A1b) same for source-less types")
    pshow(t0)
    # A's positives/negatives found in B: (type, source) of the OLDEST event on the item in B
    found = hB.join(cA, on=["A", "item"]).sort("hpos", descending=True).group_by(["A", "item"]).agg(pl.col("target").first(), pl.col("et").first().alias("et0"), pl.col("es").first().alias("es0"),
                                                                                                    (pl.col("et") == 1).any().alias("has_view"),
                                                                                                    ((pl.col("et") == 1) & (pl.col("es") == 1)).any().alias("has_view_s1"))
    for lab in [1, 0]:
        g = found.filter(pl.col("target") == lab)
        d = g.group_by(["et0", "es0"]).len().with_columns((pl.col("len") / g.height).alias("share")).sort("len", descending=True).head(15)
        print(f"(A2) A {'positives' if lab else 'negatives'} found in B: {g.height}; oldest event (type, source) distribution")
        pshow(d)
    return


def pool_items(split):
    return np.unique(np.load(f"{FLAT}/{split}/cand_item_id.npy", mmap_mode="r")[:])


def post_times(items_needed):
    """median posting time per item over candidate appearances in all splits (items restricted to items_needed)."""
    out = []
    for sp in ["val", "test", "train"]:
        f = pl.read_parquet(f"{FLAT}/{sp}/feeds.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.float64)
        it = np.load(f"{FLAT}/{sp}/cand_item_id.npy", mmap_mode="r")
        fi = np.load(f"{FLAT}/{sp}/cand_fidx.npy", mmap_mode="r")
        f4 = np.load(f"{FLAT}/{sp}/cand_f4.npy", mmap_mode="r")
        step = 40_000_000
        for i in range(0, it.shape[0], step):
            a = np.asarray(it[i:i + step]); m = np.isin(a, items_needed)
            if not m.any():
                continue
            tp = f[np.asarray(fi[i:i + step])[m]] - age_hours(np.asarray(f4[i:i + step])[m], KN) * 3600.0
            out.append(pl.DataFrame({"item": a[m], "tp": tp}))
    d = pl.concat(out).drop_nans().group_by("item").agg(pl.col("tp").median())
    return d


def part_BC():
    trpool = None
    for sp in ["val", "test"]:
        df, fs = load_hist(sp)
        own = pool_items(sp)
        if trpool is None:
            trpool = pool_items("train")
            print(f"train candidate pool: {len(trpool)} distinct items")
        df = df.with_columns(pl.Series("in_split_pool", np.isin(df["item"].to_numpy(), own)), pl.Series("in_train_pool", np.isin(df["item"].to_numpy(), trpool)))
        # same user's train feeds' candidates
        ftr = feeds("train").filter(pl.col("user_id").is_in(fs.filter(pl.col("user_id") > 0)["user_id"].unique().implode()))
        Ctr, _ = load_cand("train", ftr["fidx"].to_numpy(), cols=("item_id", "target"))
        Ctr = Ctr.join(ftr.select("fidx", "user_id"), on="fidx").group_by(["user_id", "item_id"]).agg(pl.col("target").max()).rename({"item_id": "item", "target": "tr_target"})
        df = df.join(Ctr, on=["user_id", "item"], how="left")
        # later same-split feeds of the same user: candidates
        C, _ = load_cand(sp, None, cols=("item_id",))
        C = C.join(fs.select("fidx", "user_id", pl.col("timestamp").alias("cts")), on="fidx").filter(pl.col("user_id") > 0).select("user_id", pl.col("item_id").alias("item"), "cts")
        later = df.filter(pl.col("user_id") > 0).select("fidx", "user_id", "item", "ts").unique().join(C, on=["user_id", "item"]).filter(pl.col("cts") > pl.col("ts")).select("fidx", "item").unique().with_columns(pl.lit(True).alias("later_cand"))
        df = df.join(later, on=["fidx", "item"], how="left").with_columns(pl.col("later_cand").fill_null(False))
        sb = df.filter(pl.col("et").is_in([1, 4, 7]) | (pl.col("es") > 0))
        g = sb.group_by(["es"]).agg(pl.len().alias("n"), pl.col("in_split_pool").mean(), pl.col("in_train_pool").mean(),
                                    pl.col("tr_target").is_not_null().mean().alias("was_user_train_cand"), (pl.col("tr_target") == 1).mean().alias("was_user_train_pos"),
                                    pl.col("later_cand").mean().alias("later_cand_same_user")).sort("es").filter(pl.col("es") <= 16)
        print(f"\n(B) {sp}: source-bearing events (types 1/4/7) by source")
        pshow(g)
        g.write_csv(f"{OUT}/s04B_pool_{sp}.csv")
        g1 = df.filter(pl.col("et") == 1).group_by("es").agg(pl.len().alias("n_views"), pl.col("in_split_pool").mean(), pl.col("in_train_pool").mean()).sort("es").filter(pl.col("es") <= 16)
        print("(B') type-1 views only")
        pshow(g1)
        # (C) age at feed time for recent events
        rec = df.filter((pl.col("hpos") < 30) & (pl.col("et").is_in([1, 4, 7])))
        tp = post_times(rec["item"].unique().to_numpy())
        rec = rec.join(tp, on="item", how="left").with_columns(((pl.col("ts") - pl.col("tp")) / 3600.0).alias("age_h"))
        c = rec.group_by(["et", "es"]).agg(pl.len().alias("n"), pl.col("age_h").is_not_null().mean().alias("dated"),
                                           pl.col("age_h").median().alias("age_med_h"), (pl.col("age_h") < 24).mean().alias("lt24h_of_all"),
                                           (pl.col("age_h").filter(pl.col("age_h").is_not_null()) < 24).mean().alias("lt24h_of_dated"),
                                           (pl.col("age_h").filter(pl.col("age_h").is_not_null()) < 3).mean().alias("lt3h_of_dated")).sort(["et", "es"]).filter((pl.col("es") <= 16) & (pl.col("n") >= 40))
        print(f"(C) {sp}: listing age at feed time for events with hpos<30, by (type, source)")
        pshow(c, 60)
        c.write_csv(f"{OUT}/s04C_age_{sp}.csv")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "A"):
        part_A()
    if which in ("all", "BC"):
        part_BC()
