"""Build (feed, category)-level features for the category-intent model.

Train subset: random-like feeds (p_random > 0.5, M >= 50) + a uniform sample of M < 50 feeds
(p_random is unreliable there). Val and test: all feeds. Output: out/grp_<name>.npz
"""
import time
import numpy as np
from sklearn.decomposition import PCA
from catlib import *

CF = ["tte_uc_ctr", "tte_uc_imp", "tte_c_ctr", "cos_u", "maxsim32"]
NPCA = 16


def group_matrix(v, top_src, pca):
    v = build_groups(v)
    g = v["g_feed"]
    gn = v["g_n"].astype(np.float64)
    cols, names = [], []

    def add(n, x):
        names.append(n); cols.append(np.asarray(x, np.float32))

    def grk(x):  # within-feed percentile rank over the feed's categories
        return feed_pctrank(v["goff"], v["G"], np.asarray(x, np.float64))

    def ggap(x):
        mx = np.full(len(v["M"]), -np.inf); np.maximum.at(mx, g, x)
        return x - mx[g]

    M = v["M"].astype(np.float64); Hh = v["H"].astype(np.float64)
    add("cat", v["g_cat"]); add("M", M[g]); add("G", v["G"][g]); add("H", Hh[g])
    hour = ((v["timestamp"].astype(np.int64) + 3 * 3600) % 86400) / 3600.0
    add("hour", hour[g]); add("user0", (v["user_id"] == 0)[g])
    add("n", gn); add("share", gn / M[g]); add("n_rk", grk(gn))
    f3 = gmax(v, v["c_f3"]); add("f3", f3); add("f3_rk", grk(f3)); add("f3_gap", ggap(f3))
    f1x = gmax(v, v["c_f1"]); add("f1max", f1x); add("f1max_rk", grk(f1x)); add("f1max_gap", ggap(f1x))
    add("f1mean", gmean(v, v["c_f1"])); add("f1min", gmin(v, v["c_f1"]))
    add("f2mean", gmean(v, v["c_f2"])); add("f4mean", gmean(v, v["c_f4"])); add("f4max", gmax(v, v["c_f4"]))
    f5x = gmax(v, v["c_f5"]); add("f5mean", gmean(v, v["c_f5"])); add("f5max", f5x); add("f5max_rk", grk(f5x))
    hg = hist_group_feats(v, top_src)
    for i, n in enumerate(HIST_NAMES):
        add(n, hg[:, i])
    hc = hg[:, HIST_NAMES.index("h_cnt")]
    add("h_share", hc / np.maximum(Hh[g], 1)); add("h_exp10_rk", grk(hg[:, HIST_NAMES.index("h_exp10")]))
    add("h_cnt20_rk", grk(hg[:, HIST_NAMES.index("h_cnt20")]))
    nu = np.bincount(g, weights=(hc > 0), minlength=len(v["M"]))
    add("h_ncat_in_feed", nu[g])
    # history categories not shown in the feed: share of recent history that is outside the feed's categories
    add("tte_uc_ctr", gmax(v, np.nan_to_num(v["x_tte_uc_ctr"], nan=-1))); add("tte_uc_imp", gmax(v, np.nan_to_num(v["x_tte_uc_imp"], nan=-1)))
    add("tte_c_ctr", gmax(v, np.nan_to_num(v["x_tte_c_ctr"], nan=-1)))
    cu = np.nan_to_num(v["x_cos_u"], nan=0.0); ms = np.nan_to_num(v["x_maxsim32"], nan=-1.0)
    add("cos_u_mean", gmean(v, cu)); cux = gmax(v, cu); add("cos_u_max", cux); add("cos_u_max_rk", grk(cux))
    add("maxsim_mean", gmean(v, ms)); msx = gmax(v, ms); add("maxsim_max", msx); add("maxsim_max_rk", grk(msx))
    up = pca.transform(v["user_emb"]).astype(np.float32)
    for k in range(NPCA):
        add(f"upca{k}", up[g, k])
    X = np.stack(cols, 1)
    d = dict(X=X, names=np.array(names), g_feed=g, g_cat=v["g_cat"], g_n=v["g_n"], goff=v["goff"], G=v["G"],
             feeds=v["feeds"], user_id=v["user_id"], cand_g=v["cand_g"], M=v["M"])
    if "g_y" in v:
        d["g_y"] = v["g_y"]; d["g_npos"] = v["g_npos"]
    return d


def main():
    t0 = time.time()
    import polars as pl
    vv = load_view("val", with_hist=True)
    src = vv["h_event_source"].astype(np.int64)
    top_src = np.argsort(-np.bincount(src, minlength=N_SRC))[:6].astype(np.int64)
    print("top sources", top_src)
    vt = load_view("test", with_hist=False)
    pca = PCA(NPCA, random_state=0).fit(np.vstack([vv["user_emb"], vt["user_emb"]]))
    print("pca evr", pca.explained_variance_ratio_.sum())
    del vv, vt
    # train subset
    z = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = z["p_random"]
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy()
    rng = np.random.default_rng(0)
    R = np.where((p > 0.5) & (M >= 50))[0]
    small = np.where(M < 50)[0]
    S = np.sort(rng.choice(small, 300_000, replace=False))
    sel = np.sort(np.concatenate([R, S]))
    is_R = np.isin(sel, R)
    print(f"train subset: R {len(R)} S {len(S)} total {len(sel)}", flush=True)
    for name, split, feeds in [("val", "val", None), ("test", "test", None), ("train", "train", sel)]:
        t1 = time.time()
        v = load_view(split, feeds=feeds, cand_feats=CF, with_hist=True)
        print(name, "loaded", time.time() - t1, flush=True)
        d = group_matrix(v, top_src, pca)
        if name == "train":
            d["is_R"] = is_R
            d["p_random"] = p[sel]
        np.savez(OUT / f"grp_{name}.npz", **d)
        print(name, d["X"].shape, "done", time.time() - t1, flush=True)
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()
