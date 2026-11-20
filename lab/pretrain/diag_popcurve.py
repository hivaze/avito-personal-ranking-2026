"""Mean within-feed z of the pre score (and of maxsim32 for reference) by candidate history-count bin, val vs train windows."""
import sys, json
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z
R = "<repo>"
pr = np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"]
rng = np.random.default_rng(3)
edges = [0.5, 10, 100, 1000]
def rows_for(split, feeds):
    f = pl.read_parquet(f"{R}/data/flat/{split}/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    if feeds is None: return None, M
    Mf = M[feeds]; return np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum()), Mf
sets = {"val": ("val", rows_for("val", None)), "test": ("test", rows_for("test", None))}
for k in [1, 3]:
    wf = np.load(f"{R}/data/pretrain/win{k}/window_feeds.npy"); wf = np.sort(rng.permutation(wf[pr[wf] > 0.9])[:30000])
    sets[f"train_W{k}"] = ("train", rows_for("train", wf))
out = {}
for name, (sp, (rows, M)) in sets.items():
    o = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    L = lambda n: np.asarray(np.load(f"{R}/data/feats/{sp}/{n}.npy", mmap_mode="r")[rows] if rows is not None else np.load(f"{R}/data/feats/{sp}/{n}.npy"), np.float32)
    ith = L("it_hist_all_luo"); b = np.digitize(ith, edges)
    zp = L("pre_fe_z"); zn = feed_rank_z(o, M, L("pre_fe_noid"))[:, 1]; zm = L("maxsim32_z") if True else None
    zm = feed_rank_z(o, M, np.nan_to_num(L("maxsim32"), nan=0))[:, 1]
    row = {}
    for i, lab in enumerate(["0", "1-10", "10-100", "100-1000", ">1000"]):
        m = b == i
        if m.sum() < 500: continue
        row[lab] = dict(share=round(float(m.mean()), 3), z_pre=round(float(zp[m].mean()), 3), z_noid=round(float(zn[m].mean()), 3), z_maxsim=round(float(zm[m].mean()), 3))
    out[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(out, open(f"{R}/logs/pretrain_diag_popcurve.json", "w"), indent=1)
