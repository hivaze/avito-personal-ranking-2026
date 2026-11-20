"""Cold-bin (it_hist_all_luo == 0) analysis: pre score level and label rate by pre quintile, val vs train windows,
and the background-only control (no in-window data) on W1."""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z
R = "<repo>"
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"]
rng = np.random.default_rng(4)
def rows_for(split, feeds):
    f = pl.read_parquet(f"{R}/data/flat/{split}/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    if feeds is None: return None, M
    Mf = M[feeds]; return np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum()), Mf
sets = {"val": ("val", None)}
for k in [1, 3]:
    wf = np.load(f"{R}/data/pretrain/win{k}/window_feeds.npy"); sets[f"train_W{k}"] = ("train", np.sort(rng.permutation(wf[pr[wf] > 0.9])[:40000]))
out = {}
for name, (sp, feeds) in sets.items():
    rows, M = rows_for(sp, feeds)
    o = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    L = lambda n: np.asarray(np.load(f"{R}/data/feats/{sp}/{n}.npy", mmap_mode="r")[rows] if rows is not None else np.load(f"{R}/data/feats/{sp}/{n}.npy"), np.float32)
    y = np.asarray(np.load(f"{R}/data/flat/{sp}/cand_target.npy", mmap_mode="r")[rows] if rows is not None else np.load(f"{R}/data/flat/{sp}/cand_target.npy"), np.float64)
    ith = L("it_hist_all_luo"); cold = ith == 0
    scores = {"pre_fe_z": L("pre_fe_z")}
    if name == "train_W1":
        win_rows = np.load(f"{R}/data/pretrain/win1/cand_rows.npy"); pos = np.searchsorted(win_rows, rows)
        scores["w1bg_ctrl_z"] = feed_rank_z(o, M, np.load(f"{R}/data/pretrain/winscores/pre_w1bg_v3_fe_win1.npy", mmap_mode="r")[pos].astype(np.float32))[:, 1]
    r = dict(cold_share=round(float(cold.mean()), 3), pos_cold=round(float(y[cold].mean()), 4), pos_warm=round(float(y[~cold].mean()), 4))
    for sn, z in scores.items():
        q = np.quantile(z[cold], [0.2, 0.4, 0.6, 0.8])
        b = np.digitize(z[cold], q)
        r[sn] = dict(mean_z_cold=round(float(z[cold].mean()), 3), mean_z_warm=round(float(z[~cold].mean()), 3),
                     pos_by_quintile_in_cold=[round(float(y[cold][b == i].mean()), 4) for i in range(5)])
    out[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(out, open(f"{R}/logs/pretrain_diag_cold.json", "w"), indent=1)
