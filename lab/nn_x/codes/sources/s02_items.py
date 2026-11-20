"""s02: item-level behaviour by source (val, test, train sample 100K feeds).
(a) follow-up after the oldest type-1 view of an item, by the view's source: share of items that get a newer contact (2/3),
    message (5/6/8), type 4, type 7, type 9, type 10, or another type-1 view.
(b) type-4 / type-7 events: source of the nearest OLDER and nearest NEWER type-1 view of the same item; P(no view at all).
(c) cross-source item paths: for items viewed (type 1) from >=2 sources, ordered pair (first source -> later source).
(d) previous-view similarity: for each type-1 view, cosine (item embeddings) with the immediately older type-1 view in the
    same history and same-category rate; also max cosine to the previous 5 views; split by the previous view's source.
"""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, OUT, FLAT

SRC_SHOW = list(range(0, 17))


def emb_cos(split, df):
    """cos to previous (older) type-1 view, same-cat flag, max cos to previous 5 views."""
    import json
    meta = json.load(open(f"<repo>/data/competition/{split}/item_embeddings/data.meta"))
    E = np.memmap(f"<repo>/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    v = df.filter(pl.col("et") == 1).sort(["fidx", "hpos"], descending=[False, True])  # older first
    fidx = v["fidx"].to_numpy(); ei = v["emb"].to_numpy(); cat = v["cat"].to_numpy(); es = v["es"].to_numpy()
    n = len(v)
    X = np.zeros((n, 128), np.float32)
    ok = ei >= 0
    # gather in chunks
    idx = np.flatnonzero(ok)
    for i in range(0, len(idx), 2_000_000):
        j = idx[i:i + 2_000_000]
        X[j] = E[ei[j]].astype(np.float32)
    nr = np.linalg.norm(X, axis=1); nr[nr == 0] = 1
    X /= nr[:, None]
    prev_ok = np.zeros(n, bool); prev_ok[1:] = fidx[1:] == fidx[:-1]
    cos1 = np.full(n, np.nan, np.float32)
    m = prev_ok.copy(); m[1:] &= ok[1:] & ok[:-1]; m[0] = False
    cos1[m] = (X[1:][m[1:]] * X[:-1][m[1:]]).sum(1)
    samecat = np.full(n, np.nan, np.float32)
    samecat[1:][prev_ok[1:]] = (cat[1:][prev_ok[1:]] == cat[:-1][prev_ok[1:]]).astype(np.float32)
    prev_src = np.full(n, -1, np.int16); prev_src[1:][prev_ok[1:]] = es[:-1][prev_ok[1:]]
    cmax = np.full(n, np.nan, np.float32)
    for k in range(1, 6):
        mk = np.zeros(n, bool); mk[k:] = (fidx[k:] == fidx[:-k]) & ok[k:] & ok[:-k]
        c = np.full(n, np.nan, np.float32)
        c[k:][mk[k:]] = (X[k:][mk[k:]] * X[:-k][mk[k:]]).sum(1)
        cmax = np.fmax(cmax, c)
    return v.select("fidx", "es").with_columns(pl.Series("cos_prev", cos1).fill_nan(None), pl.Series("samecat_prev", samecat).fill_nan(None), pl.Series("prev_src", prev_src), pl.Series("cos_max5", cmax).fill_nan(None))


def run(split, fsel=None):
    df, fs = load_hist(split, fsel)
    import numpy as _np
    # attach embedding idx
    from scommon import _rows
    H = fs["H"].to_numpy(); rows = _rows(fs["hist_off"].to_numpy(), H)
    df = df.with_columns(pl.Series("emb", _np.load(f"{FLAT}/{split}/hist_emb_idx.npy", mmap_mode="r")[rows]))
    print(f"\n===== {split}: {fs.height} feeds, {df.height} events =====")
    d = df.sort(["fidx", "item", "hpos"], descending=[False, False, True])  # per item, older first
    d = d.with_columns(pl.int_range(pl.len()).over(["fidx", "item"]).alias("k"))
    # (a) follow-up after oldest type-1 view
    first_view = d.filter(pl.col("et") == 1).group_by(["fidx", "item"]).agg(pl.col("hpos").max().alias("h0"), pl.col("es").sort_by("hpos", descending=True).first().alias("src0"))
    later = d.join(first_view, on=["fidx", "item"]).filter(pl.col("hpos") < pl.col("h0"))
    fl = later.group_by(["fidx", "item"]).agg(
        pl.col("et").is_in([2, 3]).any().alias("contact"), pl.col("et").is_in([5, 6, 8]).any().alias("msg"),
        (pl.col("et") == 4).any().alias("t4"), (pl.col("et") == 7).any().alias("t7"), (pl.col("et") == 9).any().alias("t9"),
        (pl.col("et") == 10).any().alias("t10"), (pl.col("et") == 1).any().alias("review"),
        ((pl.col("et") == 1) & (pl.col("es") != pl.col("src0"))).any().alias("review_other_src"))
    a = first_view.join(fl, on=["fidx", "item"], how="left").fill_null(False)
    ta = a.group_by("src0").agg(pl.len().alias("n_items"), *[pl.col(c).mean() for c in ["contact", "msg", "t4", "t7", "t9", "t10", "review", "review_other_src"]]).sort("src0").filter(pl.col("src0") <= 16)
    print("(a) follow-up rates after the oldest type-1 view of an item, by that view's source")
    pshow(ta)
    ta.write_csv(f"{OUT}/s02a_followup_{split}.csv")
    # (e) what PRECEDES each event of a source-bearing type on the same item (older events), by (type, source)
    pre = d.with_columns(
        (pl.col("et").is_in([2, 3]).cast(pl.Int32).cum_sum().over(["fidx", "item"]) - pl.col("et").is_in([2, 3]).cast(pl.Int32)).alias("n_contact_before"),
        (pl.col("et").is_in([5, 6, 8]).cast(pl.Int32).cum_sum().over(["fidx", "item"]) - pl.col("et").is_in([5, 6, 8]).cast(pl.Int32)).alias("n_msg_before"),
        ((pl.col("et") == 4).cast(pl.Int32).cum_sum().over(["fidx", "item"]) - (pl.col("et") == 4).cast(pl.Int32)).alias("n_t4_before"),
        ((pl.col("et") == 1).cast(pl.Int32).cum_sum().over(["fidx", "item"]) - (pl.col("et") == 1).cast(pl.Int32)).alias("n_view_before"),
        (pl.len().over(["fidx", "item"]) - 1 - pl.col("k")).alias("n_after"))
    te = pre.filter(pl.col("et").is_in([1, 4, 7]) & (pl.col("es") <= 16)).group_by(["et", "es"]).agg(
        pl.len().alias("n"), (pl.col("k") == 0).mean().alias("first_event_of_item"), (pl.col("n_view_before") > 0).mean().alias("view_before"),
        (pl.col("n_contact_before") > 0).mean().alias("contact_before"), (pl.col("n_msg_before") > 0).mean().alias("msg_before"),
        (pl.col("n_t4_before") > 0).mean().alias("t4_before"), (pl.col("n_after") > 0).mean().alias("any_event_after")).sort(["et", "es"])
    print("(e) what precedes an event on the same item (older events in the same history), by (type, source)")
    pshow(te, 60)
    te.write_csv(f"{OUT}/s02e_preceding_{split}.csv")
    # (b) source of nearest older/newer type-1 view for type-4 / type-7 events
    d2 = d.with_columns(pl.when(pl.col("et") == 1).then(pl.col("es")).otherwise(None).alias("vsrc"))
    d2 = d2.with_columns(pl.col("vsrc").forward_fill().over(["fidx", "item"]).alias("v_older_incl"),
                         pl.col("vsrc").backward_fill().over(["fidx", "item"]).alias("v_newer_incl"))
    b = d2.filter(pl.col("et").is_in([4, 7]))
    tb = b.group_by(["et", "es"]).agg(pl.len().alias("n"), pl.col("v_older_incl").is_null().mean().alias("no_older_view"),
                                     pl.col("v_newer_incl").is_null().mean().alias("no_newer_view"),
                                     (pl.col("v_older_incl").is_null() & pl.col("v_newer_incl").is_null()).mean().alias("no_view_at_all"),
                                     (pl.col("v_older_incl") == pl.col("es")).mean().alias("older_view_same_src"),
                                     (pl.col("v_newer_incl") == pl.col("es")).mean().alias("newer_view_same_src"),
                                     pl.col("v_older_incl").drop_nulls().mode().first().alias("older_view_modal_src")).sort(["et", "es"]).filter(pl.col("es") <= 16)
    print("(b) type-4/7 events: source agreement with the nearest older / newer type-1 view of the same item (null-excluded rates are over all events)")
    pshow(tb)
    tb.write_csv(f"{OUT}/s02b_inherit_{split}.csv")
    # full cross-tab source(type4) vs source(nearest older view)
    xb = b.filter(pl.col("v_older_incl").is_not_null() & (pl.col("es") <= 8) & (pl.col("v_older_incl") <= 8)).group_by(["et", "es", "v_older_incl"]).len()
    for et in [4, 7]:
        x = xb.filter(pl.col("et") == et).with_columns((pl.col("len") / pl.col("len").sum().over("es")).alias("p")).pivot(values="p", index="es", on="v_older_incl", sort_columns=True).sort("es")
        print(f"(b2) type-{et}: P(source of nearest older view = col | event source = row)")
        pshow(x)
    # (c) cross-source item paths among type-1 views
    v = d.filter(pl.col("et") == 1)
    srcs = v.group_by(["fidx", "item"]).agg(pl.col("es").alias("seq"), pl.col("es").n_unique().alias("nu"))
    multi = srcs.filter(pl.col("nu") > 1)
    print(f"(c) items with type-1 views: {srcs.height}; viewed from >=2 distinct sources: {multi.height} ({multi.height / srcs.height:.4f})")
    pairs = multi.select(pl.col("seq").list.first().alias("first"), pl.col("seq").list.eval(pl.element().filter(pl.element() != pl.element().first())).list.first().alias("next"))
    pc = pairs.group_by(["first", "next"]).len().sort("len", descending=True)
    pshow(pc.head(25))
    pc.write_csv(f"{OUT}/s02c_paths_{split}.csv")
    # repeat views from the same source
    rep = v.group_by(["fidx", "item", "es"]).len().group_by("es").agg(pl.len().alias("n_item_src"), (pl.col("len") > 1).mean().alias("repeat_same_src")).sort("es").filter(pl.col("es") <= 16)
    print("(c2) share of (feed,item,source) with >1 type-1 view from the same source")
    pshow(rep)
    # (d) similarity to previous view
    ec = emb_cos(split, df)
    td = ec.group_by("es").agg(pl.len().alias("n"), pl.col("cos_prev").mean(), pl.col("cos_prev").median().alias("cos_prev_med"), (pl.col("cos_prev") > 0.9).mean().alias("cos_prev_gt09"),
                               pl.col("samecat_prev").mean(), pl.col("cos_max5").mean(), (pl.col("prev_src") == pl.col("es")).mean().alias("prev_same_src")).sort("es").filter(pl.col("es") <= 16)
    print("(d) type-1 views: similarity with the immediately older type-1 view (any source)")
    pshow(td)
    td.write_csv(f"{OUT}/s02d_prevsim_{split}.csv")
    # by (prev_src, src) for the main sources
    tdd = ec.filter(pl.col("es").is_in([1, 2, 4, 5, 6, 7, 8, 9]) & pl.col("prev_src").is_in([0, 1, 2, 4, 5, 6, 7])).group_by(["prev_src", "es"]).agg(pl.len().alias("n"), pl.col("cos_prev").mean().alias("cos"), pl.col("samecat_prev").mean().alias("samecat"))
    print("(d2) cos with the previous view, by (previous source -> this source)")
    pshow(tdd.pivot(values="cos", index="prev_src", on="es", sort_columns=True).sort("prev_src"))
    pshow(tdd.pivot(values="n", index="prev_src", on="es", sort_columns=True).sort("prev_src"))
    return


if __name__ == "__main__":
    for sp in sys.argv[1:] or ["val", "test", "train"]:
        run(sp, train_sample(100_000, seed=1) if sp == "train" else None)
