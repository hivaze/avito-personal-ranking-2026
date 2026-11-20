"""Behavioural hypotheses as segment x feature interaction rules: score + w * seg(feed) * g(item), cross-fitted."""
import json
from util import *

df = load()
c = Ctx(df)
s = c.base


def col(n):
    return df[n].to_numpy().astype(np.float64)


def nz(a, v):
    a = a.copy(); a[~np.isfinite(a)] = v; return a


def zf(a):
    """within-feed z-score (so the rule only re-orders inside the feed)."""
    a = nz(a, np.nanmean(a))
    m = np.bincount(c.fid, weights=a) / c.sz
    v = np.bincount(c.fid, weights=a * a) / c.sz - m * m
    return (a - m[c.fid]) / np.sqrt(np.maximum(v[c.fid], 1e-6))


nt, ns = col("newest_type"), col("newest_src")
SEG = {
    "search_session": ns == 2,
    "feed_session": ns == 1,
    "after_contact": np.isin(nt, [2, 3]),
    "after_fav": nt == 4,
    "after_order": np.isin(nt, [5, 6, 8]),
    "after_lead": nt == 7,
    "guest": col("guest") == 1,
    "short_hist": col("H") <= 20,
    "focused": col("n_hist_cats32") <= 2,
    "diverse": col("n_hist_cats32") >= 7,
    "night": (col("hour") <= 5),
    "small_feed": col("M") < 40,
}
G = {
    "h1": zf(nz(col("h1"), 0.4)),
    "hmax": zf(nz(col("hmax"), 0.5)),
    "scn": col("same_cat_newest"),
    "popularity": zf(np.log1p(col("it_hist_all_luo")) + col("xp_imp_c50")),
    "cold": (col("xp_imp_c50") == 0).astype(float),
    "fresh": zf(-col("f4")),
    "ucos": zf(nz(col("ucos"), 0.45)),
    "c_order": zf(nz(col("c_order"), 0.3)),
    "c_contact": zf(nz(col("c_contact"), 0.3)),
    "hd95": (col("n_hdup95") > 0).astype(float),
    "cur0": (col("cat_user_rank") == 0).astype(float),
    "f1": zf(col("f1")),
}
W = [-0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2]
rows = []
for sn, sm in SEG.items():
    for gn, g in G.items():
        x = sm.astype(float) * g
        r = c.crossfit(lambda w: s + w * x, W)
        rows.append((sn, gn, r["cf"], r["se"], r["z"], r["chosen"], r["ins_delta"], float(sm[c.off].mean())))
        print(f"{sn:15s} {gn:11s} feeds={sm[c.off].mean():.3f} cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}", flush=True)
print("variants:", len(SEG) * len(G) * len(W), "pairs:", len(SEG) * len(G))
json.dump(rows, open("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/segments.json", "w"))
