# A residual model on top of the pool's score: per-feed softmax, linear in 32 candidate signals, fitted on 4 of
# the 5 user folds and scored on the fifth.
# Run as (docstring): python s04b_clogit.py <config>   (configs: base, cand, pre, nd, age)
# Outputs: lab/solvers/out/errors/s04b_<config>.log, merged in ../results/pass2_residual_models.txt.
# Result for the full signal set: -0.00058 +- 0.00049 on val, and +0.00466 +- 0.00114 on feeds whose
# positives are all cold, paid for on the other feeds.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/errors/s04b_clogit.py
"""Conditional-logit (per-feed softmax) residual over z(v11), linear in features, nested user folds (nn/finetune.py folds).
Features: z(v11) + within-feed z of each candidate signal + raw (standardised) signal. L2-regularised, LBFGS (torch CPU).
Usage: python s04b_clogit.py <config>"""
import os, sys
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import GroupKFold

torch.set_num_threads(16)
cfg = sys.argv[1]
L2 = float(os.environ.get("L2", "1e-4"))
f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
Z = np.load(f"{SCR}/out/attrs.npz")


def ld(nm):
    for d in (f"{ROOT}/data/feats/val", f"{ROOT}/data/feats_new/val", f"{ROOT}/eda/errors/synthesis/derived/val"):
        p = f"{d}/{nm}.npy"
        if os.path.exists(p):
            return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
    if os.path.exists(f"{ROOT}/preds/{nm}.npy"):
        return np.load(f"{ROOT}/preds/{nm}.npy").astype(np.float64)
    raise FileNotFoundError(nm)


SETS = {
    "base": [],
    "cand": ["f1", "f2", "f3", "f4", "f5", "maxsim32", "seq_max_all", "cos_u", "cos_h10", "cat_cnt_all", "cat_min_hpos",
             "it_hist_all_luo", "seq_cv_pop", "seq_cv_invd20", "it_tr_ctr", "it_tr_coec", "ctx_nd_max", "ctx_nd_c90", "ctx_nd_c95",
             "feed_cat_share", "centrality", "pre_fe", "pre_new", "pre_hmax32", "cold_item", "ag_age", "seq_argmax_hpos",
             "ctx_hn_rrw", "ctx_nd_knn_h", "seq_dcos_hl2", "cat_cnt_et1", "cat_cnt_et10"],
    "pre": ["pre_fe", "pre_new", "pre_hmax32"],
    "nd": ["ctx_nd_max", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_knn_h", "ctx_hn_rrw"],
    "age": ["f4", "ag_age", "cold_item"],
}
LOGF = {"cat_cnt_all", "it_hist_all_luo", "seq_cv_pop", "ctx_nd_c90", "ctx_nd_c95", "seq_argmax_hpos", "cat_min_hpos", "cat_cnt_et1", "cat_cnt_et10", "ag_age"}
cols = ["z_v11"]
feats = [fz(s, fid, M)]
for c in SETS[cfg]:
    x = Z[c].astype(np.float64) if c in Z.files else ld(c)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    if c in LOGF:
        x = np.log1p(np.maximum(x, 0))
    feats.append(fz(x, fid, M)); cols.append(c + "_fz")
X = np.stack(feats, 1)
X = np.clip(X, -8, 8)
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
fidt = torch.from_numpy(fid)
oof = np.zeros(len(y))
W = []
for k, (trf, hof) in enumerate(folds):
    trm = np.isin(fid, trf)
    Xt = torch.from_numpy(X[trm]).float()
    yt = torch.from_numpy(y[trm]).float()
    # remap feed ids to 0..nt-1
    _, fl = np.unique(fid[trm], return_inverse=True)
    fl = torch.from_numpy(fl)
    nf = int(fl.max()) + 1
    npos = torch.zeros(nf).index_add_(0, fl, yt)
    w = torch.zeros(X.shape[1], requires_grad=True)
    with torch.no_grad():
        w[0] = 1.0
    opt = torch.optim.LBFGS([w], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        sc = Xt @ w
        mx = torch.zeros(nf).scatter_reduce(0, fl, sc, reduce="amax", include_self=False)
        lse = torch.log(torch.zeros(nf).index_add_(0, fl, torch.exp(sc - mx[fl]))) + mx
        # listwise CE with per-feed normalised target (feeds weighted equally)
        ce = -(torch.zeros(nf).index_add_(0, fl, yt * sc) / npos.clamp(min=1) - lse)
        loss = ce.mean() + L2 * (w[1:] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wk = w.detach().numpy()
    W.append(wk)
    hm = np.isin(fid, hof)
    oof[hm] = X[hm] @ wk
base = ndcg_per_feed(s, y, off, M)
nd = ndcg_per_feed(oof, y, off, M)
d = nd - base
day = pf["day"].to_numpy()
sl = {"all": np.ones(n, bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1,
      "all_cold": pf["poscold"].to_numpy() == "all_cold"}
print(f"{cfg} L2={L2}: " + "  ".join(f"{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}" for k, m in sl.items()))
Wm = np.mean(W, 0)
print("  mean weights:", ", ".join(f"{c} {w:+.3f}" for c, w in sorted(zip(cols, Wm), key=lambda t: -abs(t[1]))[:14]))
np.save(f"{SCR}/out/clogit_{cfg}.npy", oof.astype(np.float32))
