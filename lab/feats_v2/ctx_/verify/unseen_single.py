import sys
sys.path.insert(0, "<repo>/src")
from common import *
tr_items = np.unique(np.load(FLAT / "train/cand_item_id.npy"))
f, M, off = feed_meta("val"); it = np.load(FLAT / "val/cand_item_id.npy"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
un = ~np.isin(it, tr_items); fi = np.repeat(np.arange(len(M)), M); u = np.bincount(fi, weights=un, minlength=len(M)) / M
rng = np.random.default_rng(0)
def nd(x, mask):
    fs = np.where(mask)[0]; rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fs]); m = M[fs]; o = np.r_[0, np.cumsum(m)[:-1]]
    s = np.where(np.isnan(x[rows]), -1e9, x[rows]) + rng.random(len(rows)) * 1e-9
    return ndcg(s, y[rows], o, m)
hi = u >= 0.5
print("positive rate unseen vs seen candidates:", y[un].mean().round(4), y[~un].mean().round(4))
for k in ["ctx_hs_t10", "ctx_hn_soft05", "ctx_hs_k128", "ctx_hn_rrw", "ctx_nd_hcent", "maxsim32", "cos_u", "f1", "tte_i_ctr" if (FEATS/"val/tte_i_ctr.npy").exists() else "f2"]:
    x = np.load(FEATS / "val" / f"{k}.npy").astype(np.float64)
    print(f"{k:16s} unseen>=0.5 feeds {nd(x, hi):.4f}  others {nd(x, ~hi):.4f}")
