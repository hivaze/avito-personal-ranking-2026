"""s05: context checks for the source mapping (val + test, train sample for drift).
(1) category-controlled follow-up: messaging / contact follow-up after a type-1 view, by source x category (cats 10,11,12,13,2,3,48,29)
    -> is the s7 'no messaging' signature a category property or a surface property?
(2) platform proxies: per source, O/E co-presence with type-10 events and with user_id = 0, within activity strata;
    share of feeds with H < 50 (new / light users) among feeds having the source.
(3) feed hour-of-day profile of each source among the newest 5 events (val+test), vs all-source baseline.
(4) train day drift: daily share of each source among type-1/4/7 events at hpos < 20 in a 300K-feed train sample.
"""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, OUT

S = list(range(0, 17))


def strata_oe(pres_src, flag, nsb):
    """O/E of (source present & flag) within deciles of nsb."""
    q = np.quantile(nsb, np.linspace(0, 1, 11)[1:-1]); st = np.searchsorted(q, nsb, side="right")
    O = 0.0; E = 0.0
    for s in np.unique(st):
        m = st == s
        O += (pres_src[m] & flag[m]).sum(); E += pres_src[m].mean() * flag[m].mean() * m.sum()
    return O / max(E, 1e-9)


def main():
    v, fv = load_hist("val"); t, ft = load_hist("test")
    df = pl.concat([v, t.with_columns(pl.col("fidx") + 10_000_000)])
    fs = pl.concat([fv, ft.with_columns(pl.col("fidx") + 10_000_000)])
    # (1) category-controlled follow-up
    d = df.sort(["fidx", "item", "hpos"], descending=[False, False, True])
    fv1 = d.filter(pl.col("et") == 1).group_by(["fidx", "item"]).agg(pl.col("hpos").max().alias("h0"), pl.col("es").sort_by("hpos", descending=True).first().alias("src0"), pl.col("cat").first())
    later = d.join(fv1.select("fidx", "item", "h0"), on=["fidx", "item"]).filter(pl.col("hpos") < pl.col("h0")).group_by(["fidx", "item"]).agg(
        pl.col("et").is_in([5, 6, 8]).any().alias("msg"), pl.col("et").is_in([2, 3]).any().alias("contact"), (pl.col("et") == 10).any().alias("t10"))
    a = fv1.join(later, on=["fidx", "item"], how="left").fill_null(False)
    cats = [10, 11, 12, 13, 2, 3, 48, 29, 50]
    tab = a.filter(pl.col("cat").is_in(cats) & pl.col("src0").is_in([0, 1, 2, 4, 5, 6, 7])).group_by(["cat", "src0"]).agg(pl.len().alias("n"), pl.col("msg").mean(), pl.col("contact").mean())
    print("(1) messaging follow-up after the oldest view, by category (rows) x view source (cols)")
    pshow(tab.pivot(values="msg", index="cat", on="src0", sort_columns=True).sort("cat"))
    print("    contact follow-up")
    pshow(tab.pivot(values="contact", index="cat", on="src0", sort_columns=True).sort("cat"))
    print("    n items")
    pshow(tab.pivot(values="n", index="cat", on="src0", sort_columns=True).sort("cat"))
    # category-wide event mix for these categories
    cm = df.filter(pl.col("cat").is_in(cats)).group_by("cat").agg(pl.len().alias("n"), *[(pl.col("et") == e).mean().alias(f"t{e}") for e in range(1, 11)]).sort("cat")
    print("    event-type mix by category (all events)")
    pshow(cm)
    # (2) platform proxies
    sb = df.filter(pl.col("et").is_in([1, 4, 7]))
    nsb = sb.group_by("fidx").len().rename({"len": "nsb"})
    f2 = fs.select("fidx", "user_id", "H").join(nsb, on="fidx", how="left").fill_null(0).sort("fidx")
    fids = f2["fidx"].to_numpy()
    P = np.zeros((len(fids), 31), bool)
    pr = sb.group_by(["fidx", "es"]).len()
    P[np.searchsorted(fids, pr["fidx"].to_numpy()), pr["es"].to_numpy()] = True
    t10 = np.zeros(len(fids), bool); t10[np.searchsorted(fids, df.filter(pl.col("et") == 10)["fidx"].unique().to_numpy())] = True
    t9 = np.zeros(len(fids), bool); t9[np.searchsorted(fids, df.filter(pl.col("et") == 9)["fidx"].unique().to_numpy())] = True
    msg = np.zeros(len(fids), bool); msg[np.searchsorted(fids, df.filter(pl.col("et").is_in([5, 6, 8]))["fidx"].unique().to_numpy())] = True
    anon = f2["user_id"].to_numpy() == 0
    ns = f2["nsb"].to_numpy(); H = f2["H"].to_numpy()
    rows = []
    for s in S:
        p = P[:, s]
        rows.append(dict(src=s, feeds_with=float(p.mean()), oe_type10=strata_oe(p, t10, ns), oe_type9=strata_oe(p, t9, ns), oe_msg=strata_oe(p, msg, ns),
                         oe_anon=strata_oe(p, anon, ns), share_H_lt50=float((H[p] < 50).mean()) if p.any() else np.nan))
    print("(2) platform/login proxies: O/E co-presence within activity strata (1 = independent)")
    print(f"    base: feeds with type10 {t10.mean():.4f}, type9 {t9.mean():.4f}, messaging {msg.mean():.4f}, anonymous {anon.mean():.4f}, H<50 {(H < 50).mean():.4f}")
    out = pl.DataFrame(rows)
    pshow(out)
    out.write_csv(f"{OUT}/s05_platform_proxies.csv")
    # (3) hour-of-day profile of newest 5 events
    nw = df.filter((pl.col("hpos") < 5) & pl.col("et").is_in([1, 4, 7])).with_columns(((pl.col("ts") + 3 * 3600) % 86400 // (6 * 3600)).alias("hb"))
    hb = nw.group_by(["es", "hb"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("es")).alias("p"))
    base = nw.group_by("hb").len().with_columns((pl.col("len") / pl.col("len").sum()).alias("pb")).select("hb", "pb")
    hb = hb.join(base, on="hb").with_columns((pl.col("p") / pl.col("pb")).alias("lift")).filter(pl.col("es") <= 12)
    print("(3) newest-5 source-bearing events: feed hour-bucket (MSK 0-6,6-12,12-18,18-24) lift vs all sources")
    pshow(hb.pivot(values="lift", index="es", on="hb", sort_columns=True).sort("es"))
    # (4) train day drift
    tr, ftr = load_hist("train", train_sample(300_000, seed=4))
    tr = tr.filter((pl.col("hpos") < 20) & pl.col("et").is_in([1, 4, 7])).with_columns(((pl.col("ts") + 3 * 3600) // 86400).alias("day"))
    dd = tr.group_by(["day", "es"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("day")).alias("p")).filter(pl.col("es") <= 12)
    d0 = dd["day"].min()
    dd = dd.with_columns((pl.col("day") - d0).alias("d"))
    print("(4) train: share of each source among newest-20 source-bearing events, by feed day (d=0 is Sep 6 MSK)")
    pshow(dd.pivot(values="p", index="d", on="es", sort_columns=True).sort("d"))
    vv = df.filter((pl.col("hpos") < 20) & pl.col("et").is_in([1, 4, 7])).with_columns((pl.col("fidx") >= 10_000_000).alias("is_test")).group_by(["is_test", "es"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("is_test")).alias("p")).filter(pl.col("es") <= 12)
    pshow(vv.pivot(values="p", index="is_test", on="es", sort_columns=True))


if __name__ == "__main__":
    main()
