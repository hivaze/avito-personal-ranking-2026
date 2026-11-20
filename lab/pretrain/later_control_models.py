"""Has-later-feed gap of single-model scores 0.5(z(fe)+z(new)) on a window's train feeds, windowed vs background-only
control model (same config, no in-window snapshots). Usage: python pretrain/later_control_models.py 1 w1_v3 w1bg_v3"""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
PT = "<repo>/data/pretrain"
k = int(sys.argv[1]); tags = sys.argv[2:]
f = pl.read_parquet("<repo>/data/flat/train/feeds.parquet")
M_all = f["M"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy(); users = f["user_id"].to_numpy()
feeds = np.load(f"{PT}/win{k}/window_feeds.npy"); rows = np.load(f"{PT}/win{k}/cand_rows.npy")
M = M_all[feeds]; off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
y = np.load("<repo>/data/flat/train/cand_target.npy", mmap_mode="r")[rows].astype(float)
wf = pl.DataFrame({"fidx": feeds, "u": users[feeds], "t": ts[feeds]})
g = wf.filter(pl.col("u") != 0).group_by("u").agg(pl.col("t").max().alias("tmax"), pl.col("t").min().alias("tmin"))
wf = wf.join(g, on="u", how="left").sort("fidx")
later = ((wf["u"] != 0) & (wf["tmax"].fill_null(0) > wf["t"])).to_numpy()
first = ((wf["u"] != 0) & (wf["tmin"].fill_null(-1) == wf["t"])).to_numpy()   # earliest in-window feed (own snapshot in corpus)
rnd = np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"][feeds] > 0.9
for t in tags:
    L = lambda key: np.load(f"{PT}/winscores/pre_{t}_{key}_win{k}.npy")
    for name, x in [("score", 0.5 * (feed_rank_z(off, M, L("fe"))[:, 1] + feed_rank_z(off, M, L("new"))[:, 1])), ("fe_noid", L("fe_noid"))]:
        pf = ndcg_per_feed(x, y, off, M)
        r = {}
        for sub, m in [("all", np.ones(len(feeds), bool)), ("random", rnd)]:
            a, b = pf[m & later], pf[m & ~later]
            r[sub] = dict(mean=round(pf[m].mean(), 4), has_later=round(a.mean(), 4), no_later=round(b.mean(), 4), gap=round(a.mean() - b.mean(), 4),
                          gap_se=round(float(np.sqrt(a.var() / len(a) + b.var() / len(b))), 4),
                          first_feed=round(pf[m & first].mean(), 4), later_feeds_of_multi=round(pf[m & ~first & (wf["u"] != 0).to_numpy()].mean(), 4))
        print(t, name, json.dumps(r), flush=True)
