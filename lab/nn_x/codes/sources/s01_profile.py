"""s01: per-source profile on val, test and a 200K-feed random train sample.
share, event-type mix, hpos profile, category mix (top cats, entropy, JS vs all events), no-emb share,
user_id=0 share, feeds-with share, burstiness (same source in the adjacent newer event), repeated-item share,
user concentration. Writes out/s01_profile_<split>.csv and prints tables."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, OUT


def ent(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a[a > 0] * np.log2(a[a > 0] / b[a > 0])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def profile(split, fsel=None):
    df, fs = load_hist(split, fsel)
    nfeeds = fs.height
    N = df.height
    print(f"\n===== {split}: {nfeeds} feeds, {N} events =====")
    # adjacency: event at hpos h and the newer event hpos h-1 in the same feed (rows are ordered newest first per feed)
    es = df["es"].to_numpy(); et = df["et"].to_numpy(); fidx = df["fidx"].to_numpy()
    same_feed_next = np.zeros(N, bool); same_feed_next[1:] = fidx[1:] == fidx[:-1]   # row i has a newer neighbour i-1
    nb_src = np.full(N, -1, np.int16); nb_src[1:] = es[:-1]
    nb_src[~same_feed_next] = -1
    # among type-1/4/7 events only (source-bearing): nearest newer source-bearing event in same feed
    sb = np.isin(et, [1, 4, 7])
    idx_sb = np.flatnonzero(sb)
    nb_sb = np.full(N, -1, np.int16)
    prev = idx_sb[:-1]; cur = idx_sb[1:]
    ok = fidx[prev] == fidx[cur]
    nb_sb[cur[ok]] = es[prev[ok]]
    df = df.with_columns(pl.Series("nb", nb_src), pl.Series("nb_sb", nb_sb))
    # repeated items and cross-source items within a feed
    it = df.group_by(["fidx", "item"]).agg(pl.len().alias("n_it"), pl.col("es").filter(pl.col("es") > 0).n_unique().alias("n_src_it"))
    df = df.join(it, on=["fidx", "item"], how="left")
    cat_all = df.group_by("cat").len().sort("cat")
    p_all = np.zeros(54); p_all[cat_all["cat"].to_numpy()] = cat_all["len"].to_numpy(); p_all /= p_all.sum()
    t1 = df.filter(pl.col("et") == 1).group_by("cat").len()
    p_t1 = np.zeros(54); p_t1[t1["cat"].to_numpy()] = t1["len"].to_numpy(); p_t1 /= p_t1.sum()
    user0 = df.filter(pl.col("user_id") == 0)
    n_user0_ev = user0.height
    nf_user0 = int((fs["user_id"] == 0).sum())
    rows = []
    for s, g in df.group_by("es", maintain_order=False):
        s = s[0]
        n = g.height
        tm = g.group_by("et").len()
        tmd = dict(zip(tm["et"].to_list(), tm["len"].to_list()))
        c = g.group_by("cat").len()
        pc = np.zeros(54); pc[c["cat"].to_numpy()] = c["len"].to_numpy(); pc /= pc.sum()
        top = np.argsort(-pc)[:3]
        hp = g["hpos"].to_numpy()
        g1 = g.filter(pl.col("et") == 1)
        hp1 = g1["hpos"].to_numpy() if g1.height else np.array([np.nan])
        nbv = g["nb"].to_numpy(); nbs = g["nb_sb"].to_numpy()
        per_feed = g.group_by("fidx").len()["len"].to_numpy()
        per_feed_sorted = np.sort(per_feed)[::-1]
        top1pct = per_feed_sorted[: max(1, int(np.ceil(0.01 * nfeeds)))].sum() / n
        rows.append(dict(
            src=s, n=n, share=n / N,
            share_et1=tmd.get(1, 0) / n, share_et4=tmd.get(4, 0) / n, share_et7=tmd.get(7, 0) / n,
            share_other_et=1 - (tmd.get(1, 0) + tmd.get(4, 0) + tmd.get(7, 0)) / n,
            hpos_mean=float(hp.mean()), hpos_med=float(np.median(hp)), hpos_lt8=float((hp < 8).mean()),
            hpos1_mean=float(np.nanmean(hp1)),
            feeds_with=g["fidx"].n_unique() / nfeeds,
            ev_per_feed_with=n / g["fidx"].n_unique(), top1pct_feeds_share=top1pct,
            user0_ev_share=g.filter(pl.col("user_id") == 0).height / n,
            noemb=float(g["noemb"].mean()),
            cat_top=",".join(f"{t}:{pc[t]:.2f}" for t in top), cat_ent=ent(pc), js_vs_t1=js(pc, p_t1),
            burst_adj=float((nbv[nbv >= 0] == s).mean()) if (nbv >= 0).any() else np.nan,
            burst_sb=float((nbs[nbs >= 0] == s).mean()) if (nbs >= 0).any() else np.nan,
            rep_item=float((g["n_it"].to_numpy() > 1).mean()),
            multi_src_item=float((g["n_src_it"].to_numpy() > 1).mean()),
        ))
    out = pl.DataFrame(rows).sort("src")
    # burst lift: P(neighbour same src | src) / P(neighbour src = s)
    sb_share = df.filter(pl.col("nb_sb") >= 0).group_by("nb_sb").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("base_sb")).rename({"nb_sb": "src"}).select("src", "base_sb")
    out = out.join(sb_share.with_columns(pl.col("src").cast(pl.Int8)), on="src", how="left").with_columns((pl.col("burst_sb") / pl.col("base_sb")).alias("burst_lift_sb"))
    # share of source among user0 feeds' events vs logged-in
    u0 = user0.group_by("es").len().with_columns((pl.col("len") / max(n_user0_ev, 1)).alias("share_in_user0")).select(pl.col("es").alias("src"), "share_in_user0")
    lg = df.filter(pl.col("user_id") != 0).group_by("es").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("share_in_logged")).select(pl.col("es").alias("src"), "share_in_logged")
    out = out.join(u0, on="src", how="left").join(lg, on="src", how="left")
    print(f"user0 feeds {nf_user0} ({nf_user0/nfeeds:.4f}), user0 events {n_user0_ev}")
    pshow(out.select("src", "n", "share", "share_et1", "share_et4", "share_et7", "share_other_et", "hpos_mean", "hpos_med", "hpos_lt8", "hpos1_mean", "feeds_with", "ev_per_feed_with", "top1pct_feeds_share", "noemb"))
    pshow(out.select("src", "n", "cat_top", "cat_ent", "js_vs_t1", "burst_adj", "burst_sb", "base_sb", "burst_lift_sb", "rep_item", "multi_src_item", "user0_ev_share", "share_in_user0", "share_in_logged"))
    print("category entropy all events %.3f, type-1 %.3f" % (ent(p_all), ent(p_t1)))
    out.write_csv(f"{OUT}/s01_profile_{split}.csv")
    # type x source counts and P(source | type)
    ts = df.filter(pl.col("et").is_in([1, 4, 7])).group_by(["et", "es"]).len().sort(["et", "es"])
    ts = ts.with_columns((pl.col("len") / pl.col("len").sum().over("et")).alias("p_src_given_type"))
    piv = ts.pivot(values="p_src_given_type", index="es", on="et").sort("es")
    print("P(source | type) for source-bearing types 1/4/7:")
    pshow(piv)
    ts.write_csv(f"{OUT}/s01_type_source_{split}.csv")
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["val", "test", "train"]
    for sp in which:
        if sp == "train":
            profile("train", train_sample(200_000, seed=0))
        else:
            profile(sp)
