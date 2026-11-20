"""Post-declaration INFO only (not part of the rule): ensemble gain G3 - mean member per-feed NDCG on E vs L, and the
paired SE scale of member-vs-member deltas on E / L / all (power of a future dE/dL column)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[v] = "12"
import sys, json, itertools
import numpy as np, polars as pl
ROOT = "<repo>"; sys.path.insert(0, ROOT + "/src")
from metrics import ndcg_per_feed
hm = np.load(f"{ROOT}/preds/holdout/holdout_meta.npy"); hy = np.load(f"{ROOT}/preds/holdout/holdout_y.npy")
HM, HO = hm[0].astype(np.int64), hm[1].astype(np.int64)
ftr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet"); Mtr = ftr["M"].to_numpy()
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz"); p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[Mtr < 50] = 0; cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False)); assert (Mtr[hold] == HM).all()
ts = ftr["timestamp"].to_numpy().astype(np.int64)[hold]; E = ts < 1788901200; L = ~E
names = [x.strip() for x in open(f"{ROOT}/nn_x/pubdiag/build/g3_members.txt").read().replace("\n", ",").split(",") if x.strip()]
fid = np.repeat(np.arange(len(HM)), HM)
def fz(x):
    x = np.asarray(x, np.float64); c = np.bincount(fid, minlength=len(HM)).astype(float)
    mu = np.bincount(fid, weights=x, minlength=len(HM)) / c; v = np.bincount(fid, weights=(x - mu[fid]) ** 2, minlength=len(HM)) / c
    sd = np.sqrt(v); sd[sd < 1e-9] = 1; return (x - mu[fid]) / sd[fid]
nds, acc = [], 0
for n in names:
    s = np.load(f"{ROOT}/preds/holdout/nn_{n}_holdout.npy"); nds.append(ndcg_per_feed(s, hy, HO, HM)); acc = acc + fz(s)
nds = np.array(nds); g3 = ndcg_per_feed(acc / len(names), hy, HO, HM)
gain = g3 - nds.mean(0)
def ms(x): return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
out = {"ens_gain_E": ms(gain[E]), "ens_gain_L": ms(gain[L]), "ens_gain_all": ms(gain)}
eg, el = out["ens_gain_E"], out["ens_gain_L"]
out["ens_gain_E_minus_L"] = (eg[0] - el[0], float(np.hypot(eg[1], el[1])))
# paired SE of member-vs-member deltas (median over the 190 pairs)
se = {"E": [], "L": [], "all": []}
for i, j in itertools.combinations(range(len(names)), 2):
    d = nds[i] - nds[j]
    se["E"].append(ms(d[E])[1]); se["L"].append(ms(d[L])[1]); se["all"].append(ms(d)[1])
out["median_pairwise_member_delta_SE"] = {k: float(np.median(v)) for k, v in se.items()}
out["SE_ratio_E_over_all"] = out["median_pairwise_member_delta_SE"]["E"] / out["median_pairwise_member_delta_SE"]["all"]
out["SE_ratio_L_over_all"] = out["median_pairwise_member_delta_SE"]["L"] / out["median_pairwise_member_delta_SE"]["all"]
print(json.dumps(out, indent=1))
json.dump(out, open(f"{ROOT}/nn_x/tricks/hold_clean/out/ens_gain_info.json", "w"), indent=1)
