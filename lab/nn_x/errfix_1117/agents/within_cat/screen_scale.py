"""Is the within-category score spread too compressed/stretched in some segments? z = a*s + b*(s - mean_cat s)*1[seg]."""
from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"),
                   (pl.col("score") - pl.col("score").max().over(["fidx", "cat"])).alias("s_dmax"),
                   pl.col("s_ft").fill_nan(0), pl.col("s_to").fill_nan(0))
d = d.with_columns((pl.col("s_ft") - pl.col("s_ft").mean().over(["fidx", "cat"])).alias("sft_dm"),
                   (pl.col("s_to") - pl.col("s_to").mean().over(["fidx", "cat"])).alias("sto_dm"))
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy().astype(np.float64); s = d["score"].to_numpy().astype(np.float64)
cn = d["cat_n_in_feed"].to_numpy(); scn = d["same_cat_newest"].to_numpy(); cu = d["cat_user_rank"].to_numpy(); M = d["M"].to_numpy()
segs = {"all": np.ones(len(s), bool), "cn>=20": cn >= 20, "cn>=40": cn >= 40, "cn5-19": (cn >= 5) & (cn < 20), "cn<5": cn < 5,
        "scn": scn, "not_scn": ~scn, "cu1": cu == 1, "cu0": cu == 0, "M>=150": M >= 150}
for col in ["s_dm", "sft_dm", "sto_dm"]:
    x0 = d[col].to_numpy().astype(np.float64)
    for nm, m in segs.items():
        X = (x0 * m)[:, None]
        a, b, se, L = clogit(X, s, y, off, sz, l2=0)
        print("%-7s %-8s a=%.3f b=%+.4f z=%+.2f" % (col, nm, a, b[0], b[0] / se[0]))
