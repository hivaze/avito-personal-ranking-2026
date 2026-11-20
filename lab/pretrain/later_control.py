"""Leak check (2) with controls: NDCG of pre_final_train vs leak-free reference features, split by whether the feed's
user has another feed later in the same window. A leak shows up as a larger has-later gap for pre_ than for references."""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
PT = "<repo>/data/pretrain"
f = pl.read_parquet("<repo>/data/flat/train/feeds.parquet")
M_all = f["M"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy(); users = f["user_id"].to_numpy()
pr_all = np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"]
yall = np.load("<repo>/data/flat/train/cand_target.npy", mmap_mode="r")
out = {}
for k in map(int, sys.argv[1].split(",")):
    feeds = np.load(f"{PT}/win{k}/window_feeds.npy"); rows = np.load(f"{PT}/win{k}/cand_rows.npy")
    M = M_all[feeds]; off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    y = yall[rows].astype(float)
    wf = pl.DataFrame({"fidx": feeds, "u": users[feeds], "t": ts[feeds]})
    last = wf.filter(pl.col("u") != 0).group_by("u").agg(pl.col("t").max().alias("tmax"))
    wf = wf.join(last, on="u", how="left").sort("fidx")
    later = ((wf["u"] != 0) & (wf["tmax"].fill_null(0) > wf["t"])).to_numpy()
    rnd = pr_all[feeds] > 0.9
    res = {}
    for name, path in [("pre_final", "<repo>/preds/pre_final_train.npy"), ("pre_fe_noid", "<repo>/data/feats/train/pre_fe_noid.npy"),
                       ("pre_id_contrib", "<repo>/data/feats/train/pre_id_contrib.npy"),
                       ("maxsim32", "<repo>/data/feats/train/maxsim32.npy"), ("cos_u", "<repo>/data/feats/train/cos_u.npy"),
                       ("f1", "<repo>/data/feats/train/f1.npy")]:
        x = np.load(path, mmap_mode="r")[rows].astype(float)
        pf = ndcg_per_feed(x, y, off, M)
        r = {}
        for sub, m in [("all", np.ones(len(feeds), bool)), ("random", rnd)]:
            a, b = pf[m & later], pf[m & ~later]
            r[sub] = dict(has_later=round(a.mean(), 4), no_later=round(b.mean(), 4), gap=round(a.mean() - b.mean(), 4),
                          gap_se=round(float(np.sqrt(a.var() / len(a) + b.var() / len(b))), 4), n_has_later=int(len(a)))
        res[name] = r
        print(k, name, json.dumps(r), flush=True)
    out[k] = res
json.dump(out, open(f"<repo>/logs/pretrain_train_windows_later_control_{sys.argv[1].replace(',', '')}.json", "w"), indent=1)
