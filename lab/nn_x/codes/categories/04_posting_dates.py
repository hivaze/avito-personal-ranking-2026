"""04: listing posting dates per category (t_post = t_feed - age(f4); f4 = exact listing age, reports/feat_life_.md).
One row per unique item (val+test candidates). Prints, per category, the posting-date quantiles and the share posted in
narrow windows (bulk-import signature), and the monthly histogram for codes 52 / 53 and a few references."""
from common_cat import *
import datetime as dt
rows = []
for s in ["val", "test"]:
    m = meta(s)
    it = np.asarray(cand(s, "item_id")).astype(np.int64); c = np.asarray(cand(s, "category_id")).astype(np.int64)
    fi = np.asarray(cand(s, "fidx")).astype(np.int64); f4 = np.asarray(cand(s, "f4"), dtype=np.float64)
    tpost = m["ts"][fi] - age_hours(f4) * 3600
    rows.append(pl.DataFrame({"i": it, "c": c, "t": tpost}))
df = pl.concat(rows).drop_nulls().filter(pl.col("t").is_not_nan()).group_by("i").agg(pl.col("c").first(), pl.col("t").median())
log("unique items with finite age", df.height)
def d(x): return dt.datetime.utcfromtimestamp(float(x) + MSK).strftime("%Y-%m-%d")
out = []
for c in sorted(df["c"].unique().to_list()):
    t = np.sort(df.filter(pl.col("c") == c)["t"].to_numpy())
    q = np.quantile(t, [0.05, 0.25, 0.5, 0.75])
    # densest 30-day window share
    w = 30 * 86400
    j = np.searchsorted(t, t + w)
    k = int(np.argmax(j - np.arange(len(t))))
    out.append((c, len(t), d(q[0]), d(q[1]), d(q[2]), d(q[3]), (j[k] - k) / len(t), d(t[k])))
print("code n_items  q05 q25 q50 q75  densest30d_share  densest_window_start")
for r in out:
    print(f"{r[0]:3d} {r[1]:8d}  {r[2]} {r[3]} {r[4]} {r[5]}  {r[6]:.3f}  {r[7]}")
for c in [52, 53, 49, 31, 10, 48, 19]:
    t = df.filter(pl.col("c") == c)["t"].to_numpy()
    mon = pl.Series([dt.datetime.utcfromtimestamp(float(x) + MSK).strftime("%Y-%m") for x in t]).value_counts().sort("")
    top = mon.sort("count", descending=True).head(8)
    print(f"code {c}: top posting months", list(zip(top[""].to_list(), top["count"].to_list())))
