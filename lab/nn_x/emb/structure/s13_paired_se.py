"""paired per-feed NDCG differences (val, common feeds with valid user and history) for the s05 scores; SE = sd/sqrt(n)."""
import sys, json, numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
z = np.load(OUT + "cache/s05_scores.npz"); both = z["both"]
f = pl.read_parquet(V + "feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
tg = np.load(V + "cand_target.npy").astype(np.float64)
rng = np.random.default_rng(0); jit = rng.random(len(tg)) * 1e-9
nd = {k: ndcg_per_feed(np.where(np.isfinite(z[k]), z[k], -1e9) + jit, tg, off, M)[both] for k in z.files if k != "both"}
pairs = [("cos_user", "cos_hist_rec5"), ("cos_user", "cos_seq_uhat"), ("cos_hist_rec5", "cos_seq_uhat"), ("z_user_Pz_hist_rec5_", "cos_hist_rec5"),
         ("z_user_Pz_hist_rec5_", "z_hist_rec5_Pz_seq_uhat_"), ("cos_user", "centered_cos_user"), ("cos_user", "whitened_cos_user"),
         ("cos_hist_rec5", "centered_cos_hist_rec5"), ("cos_hist_rec5", "whitened_cos_hist_rec5"), ("z_hist_rec5_Pz_seq_uhat_", "cos_hist_rec5")]
res = {"ndcg": {k: float(v.mean()) for k, v in nd.items()}, "n_feeds": int(both.sum())}
for a, b in pairs:
    d = nd[a] - nd[b]; res[f"{a} - {b}"] = [round(float(d.mean()), 5), round(float(d.std() / np.sqrt(len(d))), 5)]
print(json.dumps(res, indent=1)); json.dump(res, open(OUT + "s13_paired_se.json", "w"), indent=1)
