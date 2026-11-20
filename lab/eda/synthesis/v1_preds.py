"""V1: re-score the team's existing val predictions with the exact metric.
Checks: (a) current level of NN vs GBDT vs stackers, (b) pessimistic vs expected ties,
(c) stability across val days (Tue Sep 22 vs Thu Sep 24) and first-30%/last-70% by time,
(d) complementarity: per-feed z-score blends NN + GBDT, paired SEs.
"""
import sys, json
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

FLAT = Path("<repo>/data/flat")
PREDS = Path("<repo>/preds")
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
ts = f["timestamp"].to_numpy().astype(np.int64)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
fidx = np.load(FLAT / "val/cand_fidx.npy")
day = ((ts + 3 * 3600) // 86400)  # MSK day index
d0 = day.min()
tue = day == np.unique(day)[0]
order = np.argsort(ts, kind="stable"); first30 = np.zeros(len(M), bool); first30[order[: int(0.3 * len(M))]] = True
rng = np.random.default_rng(0)


def feed_z(s):
    s = np.asarray(s, np.float64)
    sums = np.add.reduceat(s, off); mu = sums / M
    d = s - np.repeat(mu, M)
    sd = np.sqrt(np.add.reduceat(d * d, off) / M)
    return d / np.repeat(np.where(sd > 0, sd, 1.0), M)


def pf(s):
    return ndcg_per_feed(s, y, off, M)


names = ["nn_final_val_oof", "nn_ens_stage1_val_oof", "nn_s2_nopos_aux_sim_mix_val_oof", "nn_k1_idcg_val_oof",
         "nn_final_trainonly_val", "nn_ens_stage1_trainonly_val", "nn_s2_nopos_aux_sim_trainonly_val", "nn_nopos_trainonly_val", "nn_nohist_trainonly_val",
         "cb_tr500k_pw_val", "cb_tr300k_none_val", "lgb_tr300k_none_val", "lgb_tr300k_feat_val", "xgb_tr1m_ndcg_val", "xgb_rand50_val",
         "stk_valonly_es_val", "stk_stk_offset_init_val", "stk2_s_scores_feats_val", "stk2_r_nnonly_val"]
res = {}
P = {}
ref = None
for n in names:
    p = PREDS / f"{n}.npy"
    if not p.exists():
        print("missing", n); continue
    s = np.load(p).astype(np.float64)
    P[n] = s
    v = pf(s)
    if ref is None:
        ref = v; refn = n
    d = v - ref
    # expected-tie version: shuffle-break ties with tiny noise relative to score spacing
    ve = pf(s + rng.random(len(s)) * 1e-12 * (1 + np.abs(s)))
    res[n] = dict(ndcg=v.mean(), se=v.std() / np.sqrt(len(v)), exp_tie=ve.mean(), tue=v[tue].mean(), thu=v[~tue].mean(),
                  first30=v[first30].mean(), last70=v[~first30].mean(), d_vs_ref=d.mean(), d_se=d.std() / np.sqrt(len(d)))
    r = res[n]
    print(f"{n:38s} {r['ndcg']:.5f} (exp {r['exp_tie']:.5f}) Tue {r['tue']:.4f} Thu {r['thu']:.4f} f30 {r['first30']:.4f} l70 {r['last70']:.4f}  "
          f"d vs {refn[:12]} {r['d_vs_ref']:+.5f}+-{r['d_se']:.5f}", flush=True)

# baselines
f1 = np.load(FLAT / "val/cand_f1.npy").astype(np.float64)
f5 = np.load(FLAT / "val/cand_f5.npy").astype(np.float64)
for nm, s in [("f1 (pess)", f1), ("f1+1e-3 f5", f1 + 1e-3 * f5), ("const", np.zeros_like(f1))]:
    v = pf(s); print(f"{nm:38s} {v.mean():.5f}  Tue {v[tue].mean():.4f} Thu {v[~tue].mean():.4f}")

# complementarity: z-blends of the best NN with each GBDT / stacker
print("\nBlends: z(nn_final) + w*z(other)")
base = pf(feed_z(P["nn_final_val_oof"]))
for other in ["cb_tr500k_pw_val", "xgb_tr1m_ndcg_val", "lgb_tr300k_none_val", "stk2_s_scores_feats_val", "stk_valonly_es_val", "nn_ens_stage1_trainonly_val"]:
    if other not in P: continue
    zo = feed_z(P[other])
    line = []
    for w in [0.1, 0.2, 0.35, 0.5]:
        v = pf(feed_z(P["nn_final_val_oof"]) + w * zo)
        d = v - base
        line.append(f"w{w}: {d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f}")
    print(f"  {other:30s} " + "  ".join(line))

# fine-tune gain over train-only by day (does it hold on the later day?)
if "nn_ens_stage1_val_oof" in P and "nn_ens_stage1_trainonly_val" in P:
    d = pf(P["nn_ens_stage1_val_oof"]) - pf(P["nn_ens_stage1_trainonly_val"])
    print(f"\nFine-tune gain (ens_stage1 ft - trainonly): all {d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f}  "
          f"Tue {d[tue].mean():+.5f}  Thu {d[~tue].mean():+.5f}  first30 {d[first30].mean():+.5f} last70 {d[~first30].mean():+.5f}")

# tie check on test predictions of the best model
t = np.load(PREDS / "nn_final_test.npy")
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = ft["M"].to_numpy(); offt = ft["cand_off"].to_numpy()
tf = np.load(FLAT / "test/cand_fidx.npy")
o = np.lexsort((t, tf)); st = t[o]; same = (np.diff(st) == 0) & (np.diff(tf[o]) == 0)
print(f"\nnn_final_test: within-feed tied adjacent pairs = {int(same.sum())}")
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, open("<repo>/eda/synthesis/v1_preds.json", "w"), indent=1)
