"""Category-level view of the cross-category error (val, leak-masked).

For each (feed, category): f3, NN max/mean z, CB max z, candidate share, history counts (all, newest 1/5/20,
decayed 0.5^(hpos/10), event_type 1). Measures how well each signal ranks the positive categories (top-1 hit, MRR),
split by regime (positive category is / is not f3-top), and whether a CV category-level LR on these signals adds to
the NN when used as a candidate-level boost z_nn + w * logit P(cat).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "16"); os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from sklearn.linear_model import LogisticRegression
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
f3 = np.load(f"{FLAT}/val/cand_f3.npy").astype(np.float64)
f1 = np.load(f"{FLAT}/val/cand_f1.npy").astype(np.float64)
Z = {k: feed_rank_z(off, M, np.load(f"{ROOT}/preds/{fn}.npy").astype(np.float32))[:, 1].astype(np.float64)
     for k, fn in [("nn_oof", "nn_final2_val_oof"), ("nn_tr", "nn_final2_trainonly_val"), ("cb", "cb_tr300k_none_val")]}
hfid = np.load(f"{FLAT}/val/hist_fidx.npy").astype(np.int64)
hcat = np.load(f"{FLAT}/val/hist_category_id.npy").astype(np.int64)
hpos = np.load(f"{FLAT}/val/hist_hpos.npy").astype(np.int64)
het = np.load(f"{FLAT}/val/hist_event_type.npy").astype(np.int64)
H = f["H"].to_numpy().astype(np.float64)

ck = fid * 64 + cat
kk, inv = np.unique(ck, return_inverse=True)
kfeed = kk // 64; nk = len(kk)
C = {}
C["pos"] = np.bincount(inv, y, nk) > 0
C["f3"] = np.bincount(inv, f3, nk) / np.bincount(inv, minlength=nk)
C["f1max"] = np.full(nk, -np.inf); np.maximum.at(C["f1max"], inv, f1)
for k in Z:
    mx = np.full(nk, -np.inf); np.maximum.at(mx, inv, Z[k]); C[f"{k}_max"] = mx
C["nn_oof_mean"] = np.bincount(inv, Z["nn_oof"], nk) / np.bincount(inv, minlength=nk)
C["cand_share"] = np.bincount(inv, minlength=nk) / M[kfeed]
hk = hfid * 64 + hcat
j = np.searchsorted(kk, hk); j = np.minimum(j, nk - 1); hit = kk[j] == hk
def hsum(w):
    return np.bincount(j[hit], w[hit], nk)
Hk = np.maximum(H[kfeed], 1)
C["h_all"] = hsum(np.ones(len(hk))) / Hk
C["h_last1"] = hsum((hpos == 0).astype(float))
C["h_last5"] = hsum((hpos < 5).astype(float)) / np.minimum(Hk, 5)
C["h_last20"] = hsum((hpos < 20).astype(float)) / np.minimum(Hk, 20)
C["h_dec10"] = hsum(0.5 ** (hpos / 10.0))
C["h_et1"] = hsum((het == 1).astype(float)) / Hk
C["h_et1_last20"] = hsum(((het == 1) & (hpos < 20)).astype(float))
C["h_inhist"] = (hsum(np.ones(len(hk))) > 0).astype(float)
log("category table", nk)

# regime per feed
fpos = np.bincount(kfeed, C["pos"], n) > 0
bp3 = np.full(n, -np.inf); np.maximum.at(bp3, kfeed[C["pos"]], C["f3"][C["pos"]])
f3_top_is_pos = np.bincount(kfeed, C["f3"] > bp3[kfeed], n) == 0
ncat = np.bincount(kfeed, minlength=n)
ok = keep & fpos & (ncat >= 2)
log("feeds with >=2 categories", ok.sum(), "f3-top regime share", f3_top_is_pos[ok].mean())


def cat_rank_quality(score, mask):
    """Pessimistic rank of the best positive category within the feed (0 = top), top-1 hit, MRR."""
    bp = np.full(n, -np.inf); np.maximum.at(bp, kfeed[C["pos"]], score[C["pos"]])
    above = np.bincount(kfeed, score > bp[kfeed], n)
    # pessimistic ties: count tied negative categories as above
    tie_neg = np.bincount(kfeed, (score == bp[kfeed]) & ~C["pos"], n)
    r = (above + tie_neg)[mask]
    return (r == 0).mean(), (1.0 / (1 + r)).mean()


sigs = ["f3", "f1max", "nn_oof_max", "nn_tr_max", "cb_max", "nn_oof_mean", "cand_share", "h_all", "h_last1", "h_last5",
        "h_last20", "h_dec10", "h_et1", "h_et1_last20"]
print("\n## Category ranking quality (feeds with >=2 categories). top1 = best-scoring category holds a positive; MRR of best positive category")
print("| signal | all top1 | all MRR | f3-top regime top1 | off-f3 regime top1 | off-f3 MRR |")
print("|---|---|---|---|---|---|")
for s in sigs:
    a = cat_rank_quality(C[s], ok); b = cat_rank_quality(C[s], ok & f3_top_is_pos); c = cat_rank_quality(C[s], ok & ~f3_top_is_pos)
    print(f"| {s} | {a[0]:.3f} | {a[1]:.3f} | {b[0]:.3f} | {c[0]:.3f} | {c[1]:.3f} |")

# ---------- CV category-level LR and candidate-level boost ----------
def frank(x):
    return feed_rank_z(np.r_[0, np.cumsum(ncat)[:-1]], ncat, x.astype(np.float32))
feats = []
names = []
for s in ["f3", "f1max", "nn_oof_max", "nn_tr_max", "nn_oof_mean", "cand_share", "h_all", "h_last1", "h_last5",
          "h_last20", "h_dec10", "h_et1", "h_et1_last20", "h_inhist"]:
    rz = frank(C[s]); feeds_ok = np.isfinite(rz).all(1)
    feats += [rz[:, 0], rz[:, 1]]; names += [f"{s}_rk", f"{s}_z"]
X_all = np.stack(feats, 1)
X_all = np.nan_to_num(X_all)
sets = {
    "nn_only": [i for i, nm in enumerate(names) if nm.startswith("nn_")],
    "nn+f": [i for i, nm in enumerate(names) if nm.startswith(("nn_", "f3", "f1max", "cand_share"))],
    "nn+f+hist": list(range(len(names))),
}
users = f["user_id"].to_numpy(); users = np.where(users == 0, -np.arange(n) - 1, users)
feeds_idx = np.flatnonzero(keep)
gkf = list(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(feeds_idx, groups=users[feeds_idx]))
krow_keep = keep[kfeed]
sel = np.flatnonzero(keep[fid]); Mk = M[keep]; offk = np.r_[0, np.cumsum(Mk)[:-1]]; yk = y[sel]
base = {k: ndcg_per_feed(Z[k][sel], yk, offk, Mk) for k in Z}
reg = f3_top_is_pos[keep]
def paired(a, b):
    x = a - b; return f"{x.mean():+.4f} (SE {x.std(ddof=1) / np.sqrt(len(x)):.4f})"
print("\n## CV category-intent LR (category level, 5-fold GroupKFold by user), used as boost z_nn_oof + w * logit P(cat)")
for sname, cols in sets.items():
    logit = np.zeros(nk)
    for trf, vaf in gkf:
        trk = np.isin(kfeed, feeds_idx[trf]); vak = np.isin(kfeed, feeds_idx[vaf])
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X_all[trk][:, cols], C["pos"][trk])
        logit[vak] = X_all[vak][:, cols] @ lr.coef_[0] + lr.intercept_[0]
    q = cat_rank_quality(logit, ok); qo = cat_rank_quality(logit, ok & ~f3_top_is_pos)
    print(f"  [{sname}] category top1 {q[0]:.3f} MRR {q[1]:.3f} | off-f3 top1 {qo[0]:.3f}")
    lc = logit[inv][sel]
    for w in [0.1, 0.25, 0.5]:
        s = ndcg_per_feed(Z["nn_oof"][sel] + w * lc, yk, offk, Mk)
        print(f"     w={w}: {s.mean():.4f} vs nn_oof {paired(s, base['nn_oof'])}  | f3-top feeds {paired(s[reg], base['nn_oof'][reg])}"
              f"  off-f3 feeds {paired(s[~reg], base['nn_oof'][~reg])}")
log("done")
