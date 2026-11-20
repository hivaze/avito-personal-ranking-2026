"""Value test of the same user's EARLIER (train-snapshot) history items that are missing from the val history window.
Allowed by the leak rules (earlier snapshots only). Val, leak-masked; features exist only for train-seen users.

Per val candidate: old_maxsim (max cos to old-only history items), old_meancos (cos to their mean vector),
old_cat_share (share of old-only events in the candidate's category). Reports within-category AUC, and a CV test
z(nn_oof) + w * z(feature) (w chosen on train folds from a 3-value grid) on train-seen feeds and overall.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys, json, time
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
d = pl.read_parquet(f"{OUT}/feed_table.parquet"); keep = ~d["leak"].to_numpy()
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet"); ft = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
n = fv.height
M = fv["M"].to_numpy().astype(np.int64); coff = fv["cand_off"].to_numpy(); vH = fv["H"].to_numpy(); voff = fv["hist_off"].to_numpy()
fid = np.repeat(np.arange(n), M)

def load_emb(split):
    meta = json.loads(open(f"{ROOT}/data/competition/{split}/item_embeddings/data.meta").read())
    return np.memmap(f"{ROOT}/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
E_tr = load_emb("train"); E_va = np.asarray(load_emb("val"))

vu = fv.filter(pl.col("user_id") != 0).select("fidx", "user_id")
tr = ft.join(vu.select("user_id").unique(), on="user_id").select("user_id", "H", "hist_off")
th_item = np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r"); th_cat = np.load(f"{FLAT}/train/hist_category_id.npy", mmap_mode="r")
th_emb = np.load(f"{FLAT}/train/hist_emb_idx.npy", mmap_mode="r")
vh_item = np.load(f"{FLAT}/val/hist_item_id.npy")
by_user = {}
for u, h, o in tr.iter_rows():
    by_user.setdefault(u, []).append((o, h))

# old-only history items per val feed: (item, category, train emb row), deduplicated by item
o_off = np.zeros(n + 1, np.int64); o_cat = []; o_embrow = []
for fi, u in vu.iter_rows():
    if u not in by_user:
        continue
    vi = set(vh_item[voff[fi]:voff[fi] + vH[fi]].tolist())
    seen = set()
    for o, h in by_user[u]:
        it = np.asarray(th_item[o:o + h]); ct = np.asarray(th_cat[o:o + h]); em = np.asarray(th_emb[o:o + h])
        for a, b, c in zip(it.tolist(), ct.tolist(), em.tolist()):
            if a in vi or a in seen or c < 0:
                continue
            seen.add(a); o_cat.append(b); o_embrow.append(c)
    o_off[fi + 1] = len(seen)
o_off = np.cumsum(o_off)
o_cat = np.asarray(o_cat, np.int64); o_embrow = np.asarray(o_embrow, np.int64)
log("old-only items", len(o_cat), "feeds with any", int((np.diff(o_off) > 0).sum()))
uniq, inv = np.unique(o_embrow, return_inverse=True)
O = np.asarray(E_tr[uniq]).astype(np.float32); O /= np.linalg.norm(O, axis=1, keepdims=True) + 1e-6
log("gathered train embeddings", O.shape)
cemb = np.load(f"{FLAT}/val/cand_emb_idx.npy").astype(np.int64)
ccat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)


@nb.njit(parallel=True, fastmath=True)
def feats(o_off, inv, o_cat, O, coff, M, cemb, ccat, E, out):
    for f in nb.prange(len(M)):
        a, b = o_off[f], o_off[f + 1]
        k = b - a
        if k == 0:
            for j in range(M[f]):
                out[coff[f] + j, 0] = np.nan; out[coff[f] + j, 1] = np.nan; out[coff[f] + j, 2] = np.nan
            continue
        mean = np.zeros(O.shape[1], np.float32)
        for t in range(a, b):
            mean += O[inv[t]]
        mean /= np.sqrt((mean * mean).sum()) + 1e-6
        catcnt = np.zeros(64)
        for t in range(a, b):
            catcnt[o_cat[t]] += 1
        for j in range(M[f]):
            r = coff[f] + j
            if cemb[r] < 0:
                out[r, 0] = np.nan; out[r, 1] = np.nan; out[r, 2] = catcnt[ccat[r]] / k; continue
            v = E[cemb[r]].astype(np.float32); v /= np.sqrt((v * v).sum()) + 1e-6
            best = -1.0
            for t in range(a, b):
                s = (O[inv[t]] * v).sum()
                if s > best:
                    best = s
            out[r, 0] = best; out[r, 1] = (mean * v).sum(); out[r, 2] = catcnt[ccat[r]] / k


X = np.zeros((len(y), 3), np.float32)
feats(o_off, inv, o_cat, O, coff, M, cemb, ccat, E_va, X)
log("features computed")
names = ["old_maxsim", "old_meancos", "old_cat_share"]
has = np.diff(o_off) > 0
seg = keep & has
print(f"clean val feeds with old-only history: {seg.sum()} ({seg.sum() / keep.sum():.1%}), mean old-only items {np.diff(o_off)[seg].mean():.1f}")

Z = {k: feed_rank_z(coff, M, np.load(f"{ROOT}/preds/{fn}.npy").astype(np.float32))[:, 1].astype(np.float64)
     for k, fn in [("nn_oof", "nn_final2_val_oof"), ("nn_tr", "nn_final2_trainonly_val")]}
users = fv["user_id"].to_numpy(); users = np.where(users == 0, -np.arange(n) - 1, users)
feeds_idx = np.flatnonzero(keep)
gkf = list(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(feeds_idx, groups=users[feeds_idx]))
sel = np.flatnonzero(keep[fid]); Mk = M[keep]; offk = np.r_[0, np.cumsum(Mk)[:-1]]; yk = y[sel]
hk = has[keep]
def paired(a, b):
    x = a - b; return f"{x.mean():+.4f} (SE {x.std(ddof=1) / np.sqrt(len(x)):.4f})"
for i, nm in enumerate(names):
    x = np.nan_to_num(X[:, i], nan=np.nanmin(X[:, i]) - 1)
    zx = feed_rank_z(coff, M, x.astype(np.float32))[:, 1].astype(np.float64)
    zx[~has[fid]] = 0.0
    single = ndcg_per_feed(x, y, coff, M)[seg].mean()
    for base in ["nn_oof", "nn_tr"]:
        b0 = ndcg_per_feed(Z[base][sel], yk, offk, Mk)
        grid = [0.05, 0.1, 0.2]
        nds = np.stack([ndcg_per_feed(Z[base][sel] + w * zx[sel], yk, offk, Mk) for w in grid])
        out = np.zeros(len(Mk))
        for trf, vaf in gkf:
            out[vaf] = nds[int(np.argmax(nds[:, trf].mean(1))), vaf]
        print(f"  {nm:14s} single NDCG on its feeds {single:.4f} | {base} + w*z CV: all {paired(out, b0)}  train-seen feeds {paired(out[hk], b0[hk])}"
              f"  in-sample per w " + ", ".join(f"{g}:{v.mean() - b0.mean():+.4f}" for g, v in zip(grid, nds)))
log("done")
