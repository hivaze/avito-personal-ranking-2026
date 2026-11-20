"""Within-feed correlation of the pre score with item-popularity / item-age proxies, val vs train windows (random-like feeds)."""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
R = "<repo>"
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"]
rng = np.random.default_rng(2)
proxies = ["it_hist_all_luo", "it_cand_cnt_train", "tte_it_imp", "it_cand_cnt_split", "maxsim32", "f1"]
def rows_for(split, feeds):
    f = pl.read_parquet(f"{R}/data/flat/{split}/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    if feeds is None: return None, M
    Mf = M[feeds]; return np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum()), Mf
sets = {"val": rows_for("val", None)}
for k in [1, 2, 3]:
    wf = np.load(f"{R}/data/pretrain/win{k}/window_feeds.npy"); wf = np.sort(rng.permutation(wf[pr[wf] > 0.9])[:30000])
    sets[f"train_W{k}"] = rows_for("train", wf)
out = {}
for name, (rows, M) in sets.items():
    sp = "val" if name == "val" else "train"
    o = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    L = lambda n: np.asarray(np.load(f"{R}/data/feats/{sp}/{n}.npy", mmap_mode="r")[rows] if rows is not None else np.load(f"{R}/data/feats/{sp}/{n}.npy"), np.float32)
    zp = L("pre_fe_z")
    r = {}
    for p in proxies:
        x = L(p); x = np.log1p(np.maximum(x, 0)) if p.startswith(("it_", "tte_it_imp")) else x
        zx = feed_rank_z(o, M, np.nan_to_num(x, nan=0.0))[:, 1]
        r[p] = round(float(np.corrcoef(zp, zx)[0, 1]), 3)
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open(f"{R}/logs/pretrain_diag_confound.json", "w"), indent=1)
