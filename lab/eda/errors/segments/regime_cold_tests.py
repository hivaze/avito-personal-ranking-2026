"""Two follow-ups (val, leak-masked):
A. Cold items: how much do the val-label components (nn_oof, stk) shift cold vs warm candidates relative to nn_tr,
   and does a CV-chosen cold offset (z + delta * cold) recover the cold-positive loss? (diagnostic of over-penalisation)
B. Intent-switch regime: can "positive is NOT in the f3-top category" be predicted from observable feed features (AUC),
   and does a regime-adaptive category boost help (nested CV)?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "16"); os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
d = pl.read_parquet(f"{OUT}/feed_table.parquet")
keep = ~d["leak"].to_numpy()
f = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy(); n = len(M)
fid = np.repeat(np.arange(n), M)
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
cat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
item = np.load(f"{FLAT}/val/cand_item_id.npy")
f3 = np.load(f"{FLAT}/val/cand_f3.npy").astype(np.float64)
Z = {k: feed_rank_z(off, M, np.load(f"{ROOT}/preds/{fn}.npy").astype(np.float32))[:, 1].astype(np.float64)
     for k, fn in [("nn_oof", "nn_final2_val_oof"), ("nn_tr", "nn_final2_trainonly_val"), ("stk", "stk2_res4_clean_val")]}
Z["blend"] = 0.5 * Z["nn_oof"] + 0.5 * Z["stk"]
seen = np.zeros(111_400_000, bool)
seen[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
seen[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
cold = (~seen[item]).astype(np.float64); del seen
users = f["user_id"].to_numpy(); users = np.where(users == 0, -np.arange(n) - 1, users)
feeds_idx = np.flatnonzero(keep)
gkf = list(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(feeds_idx, groups=users[feeds_idx]))
sel = np.flatnonzero(keep[fid]); Mk = M[keep]; offk = np.r_[0, np.cumsum(Mk)[:-1]]; yk = y[sel]
pos_all_cold = d["pos_cold_all"].to_numpy()[keep]
def nd(s): return ndcg_per_feed(s, yk, offk, Mk)
def paired(a, b):
    x = a - b; return f"{x.mean():+.4f} (SE {x.std(ddof=1) / np.sqrt(len(x)):.4f})"
base = {k: nd(Z[k][sel]) for k in Z}

print("## A. Cold vs warm candidates")
cs = cold[sel]
print(f"  cold share of candidates {cs.mean():.3f}; pos rate cold {yk[cs > 0].mean():.4f} warm {yk[cs == 0].mean():.4f}")
for k in ["nn_oof", "stk", "blend"]:
    dz = Z[k][sel] - Z["nn_tr"][sel]
    print(f"  mean z({k}) - z(nn_tr): cold {dz[cs > 0].mean():+.3f}  warm {dz[cs == 0].mean():+.3f}  "
          f"(cold positives {dz[(cs > 0) & (yk > 0)].mean():+.3f}, warm positives {dz[(cs == 0) & (yk > 0)].mean():+.3f})")
for k in ["nn_oof", "stk", "blend"]:
    grid = [0.0, 0.1, 0.2, 0.3]
    nds = np.stack([nd(Z[k][sel] + dl * cs) for dl in grid])
    out = np.zeros(len(Mk))
    for trf, vaf in gkf:
        out[vaf] = nds[int(np.argmax(nds[:, trf].mean(1))), vaf]
    print(f"  [{k}] + delta*cold (CV-chosen from {grid}): {out.mean():.4f} vs {k} {paired(out, base[k])}; in-sample per delta: "
          + ", ".join(f"{g}:{v:.4f}" for g, v in zip(grid, nds.mean(1)))
          + f" | pos_all_cold feeds {paired(out[pos_all_cold], base[k][pos_all_cold])}")

print("\n## B. Intent-switch regime (positive not in the f3-top category)")
ck = fid * 64 + cat
kk, inv = np.unique(ck, return_inverse=True); kfeed = kk // 64; nk = len(kk)
kpos = np.bincount(inv, y, nk) > 0
kf3 = np.bincount(inv, f3, nk) / np.bincount(inv, minlength=nk)
kcnt = np.bincount(inv, minlength=nk)
nnmax = np.full(nk, -np.inf); np.maximum.at(nnmax, inv, Z["nn_oof"])
# per feed: f3 top / second, share of f3-top category, NN margin between best category in f3-top and the rest
o = np.lexsort((-kf3, kfeed)); first = np.r_[0, np.flatnonzero(np.diff(kfeed[o])) + 1]
top_k = o[first]
second = np.where(np.r_[first[1:], len(o)] - first >= 2, o[np.minimum(first + 1, len(o) - 1)], -1)
f3_gap = kf3[top_k] - np.where(second >= 0, kf3[second], kf3[top_k] - 5)
top_share = kcnt[top_k] / M
# NN max in f3-top category minus NN max elsewhere
other_max = np.full(n, -np.inf); np.maximum.at(other_max, kfeed[~np.isin(np.arange(nk), top_k)], nnmax[~np.isin(np.arange(nk), top_k)])
nn_margin = nnmax[top_k] - np.where(np.isfinite(other_max), other_max, nnmax[top_k] - 5)
bp3 = np.full(n, -np.inf); np.maximum.at(bp3, kfeed[kpos], kf3[kpos])
offf3 = (np.bincount(kfeed, kf3 > bp3[kfeed], n) > 0)
ncat = np.bincount(kfeed, minlength=n)
dd = d
Xf = np.stack([f3_gap, top_share, np.log(ncat), np.log(M), np.log1p(dd["H"].to_numpy()), dd["hist_top_cat_share"].to_numpy(),
               dd["cand_cat_in_hist_share"].to_numpy(), dd["uid0"].to_numpy().astype(float), nn_margin,
               d["rho_nn_cb"].to_numpy()], 1)
names = ["f3_gap", "f3top_cand_share", "log_ncat", "logM", "logH", "hist_top_cat_share", "cand_cat_in_hist", "uid0", "nn_margin", "rho_nn_cb"]
Xf = (Xf - Xf[keep].mean(0)) / (Xf[keep].std(0) + 1e-9)
p = np.zeros(n)
for trf, vaf in gkf:
    lr = LogisticRegression(max_iter=500).fit(Xf[feeds_idx[trf]], offf3[feeds_idx[trf]])
    p[feeds_idx[vaf]] = lr.predict_proba(Xf[feeds_idx[vaf]])[:, 1]
print(f"  off-f3 base rate {offf3[keep].mean():.3f}; CV AUC from observable feed features {roc_auc_score(offf3[keep], p[keep]):.3f}")
for nm, c in zip(names, lr.coef_[0]):
    print(f"     coef {nm:20s} {c:+.3f}")
for nm, i in [("f3_gap alone", 0), ("f3top_cand_share alone", 1), ("nn_margin alone", 8)]:
    print(f"     AUC {nm}: {roc_auc_score(offf3[keep], -Xf[keep, i]):.3f}")
pk = p[keep]; q = np.quantile(pk, [1 / 3, 2 / 3]); tb = np.searchsorted(q, pk)
print("  NDCG by predicted-regime tertile (low P(off-f3) -> high):")
for t in range(3):
    m = tb == t
    print(f"     tertile {t}: n={m.sum()} true off-f3 rate {offf3[keep][m].mean():.3f}  nn_oof {base['nn_oof'][m].mean():.4f} "
          f"nn_tr {base['nn_tr'][m].mean():.4f} stk {base['stk'][m].mean():.4f}  stk-tr {paired(base['stk'][m], base['nn_tr'][m])}")
# regime-adaptive f3 boost: z_nn + w_t * z(f3) with w chosen per predicted tertile on train folds (nested)
f3z = feed_rank_z(off, M, f3.astype(np.float32))[:, 1].astype(np.float64)[sel]
grid = [-0.3, -0.15, 0.0, 0.15, 0.3]
for k in ["nn_oof", "blend"]:
    nds = np.stack([nd(Z[k][sel] + w * f3z) for w in grid])
    glob = np.zeros(len(Mk)); rout = np.zeros(len(Mk))
    pos_of = {fi: i for i, fi in enumerate(feeds_idx)}
    for trf, vaf in gkf:
        glob[vaf] = nds[int(np.argmax(nds[:, trf].mean(1))), vaf]
        # tertiles computed from CV probabilities (already OOF)
        for t in range(3):
            mt = trf[tb[trf] == t]; mv = vaf[tb[vaf] == t]
            rout[mv] = nds[int(np.argmax(nds[:, mt].mean(1))), mv]
    print(f"  [{k}] + w*z(f3): global-w CV {paired(glob, base[k])}; regime-routed w CV {paired(rout, base[k])}; "
          f"in-sample per w: " + ", ".join(f"{g}:{v:.4f}" for g, v in zip(grid, nds.mean(1))))
log("done")

# ---------------- C. regime-routed CB / stacker weight (CB beats NN in the f3-top & warm-positive cell) ----------------
print("\n## C. Regime-routed blend weights (tertiles of CV-predicted P(off-f3))")
Z["cb"] = feed_rank_z(off, M, np.load(f"{ROOT}/preds/cb_tr300k_none_val.npy").astype(np.float32))[:, 1].astype(np.float64)
for a, b, grid in [("nn_oof", "cb", [0.0, 0.1, 0.2, 0.3]), ("nn_oof", "stk", [0.0, 0.25, 0.5, 0.75])]:
    nds = np.stack([nd((1 - w) * Z[a][sel] + w * Z[b][sel]) for w in grid])
    glob = np.zeros(len(Mk)); rout = np.zeros(len(Mk)); chosen = []
    for trf, vaf in gkf:
        glob[vaf] = nds[int(np.argmax(nds[:, trf].mean(1))), vaf]
        for t in range(3):
            mt = trf[tb[trf] == t]; mv = vaf[tb[vaf] == t]
            j = int(np.argmax(nds[:, mt].mean(1))); rout[mv] = nds[j, mv]; chosen.append((t, grid[j]))
    print(f"  [{a}|{b}] global CV {paired(glob, base[a])}; regime-routed CV {paired(rout, base[a])}, vs global {paired(rout, glob)}; "
          f"chosen (tertile, w): {sorted(set(chosen))}")
    for t in range(3):
        m = tb == t
        print(f"     tertile {t}: in-sample NDCG per w " + ", ".join(f"{g}:{v:+.4f}" for g, v in zip(grid, nds[:, m].mean(1) - base[a][m].mean())))
