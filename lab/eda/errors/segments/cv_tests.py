"""Follow-up tests for the SEGMENTS lens (val, leak-masked, CPU only).

1. M-adjusted NDCG deviations per segment (removes the M x npos difficulty confound).
2. Val -> test mix reweighting by (M bucket x cold-share bucket): predicted test-mix NDCG of each model.
3. Decomposition of the val fine-tune gain (nn_oof - nn_tr): 5-fold GroupKFold(user) logistic regressions on top of
   z(nn_tr) with (a) category one-hot, (b) item-warmness flags/counts, (c) both. How much of +0.0034 do simple priors explain?
4. Segment-routed blends (nested CV): per-bucket weight for z(nn_oof) vs z(stk), chosen on train folds, vs a global weight.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/eda/errors/segments")
from segdefs import seg_defs

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

d_all = pl.read_parquet(f"{OUT}/feed_table.parquet")
te = pl.read_parquet(f"{OUT}/test_segments.parquet")
keep = ~d_all["leak"].to_numpy()
d = d_all.filter(pl.col("leak").not_())
S = seg_defs(d); St = seg_defs(te, test=True)
N = len(d)

# ---------------- 1. M-adjusted deviations ----------------
M = d["M"].to_numpy(); npos = d["npos"].to_numpy()
mq = np.searchsorted(np.quantile(M, np.linspace(0, 1, 21)[1:-1]), M, side="right")
cell = mq * 3 + np.minimum(npos, 3) - 1
print("\n## 1. M x npos adjusted NDCG deviation (nn_oof), per segment bucket")
for k in ["nn_oof", "nn_tr"]:
    x = d[f"ndcg_{k}"].to_numpy()
    cm = np.bincount(cell, x) / np.bincount(cell)
    d = d.with_columns(pl.Series(f"adj_{k}", x - cm[cell]))
rows = []
for sname, lab in S.items():
    for b in sorted(set(lab)):
        m = lab == b
        a = d["adj_nn_oof"].to_numpy()[m]
        rows.append(dict(seg=sname, bucket=b, n=int(m.sum()), adj_oof=a.mean(), se=a.std(ddof=1) / np.sqrt(m.sum()),
                         adj_mass=a.sum() / N))
A = pl.DataFrame(rows)
A.write_csv(f"{OUT}/m_adjusted.csv")
for r in A.filter((pl.col("adj_oof").abs() > 2.5 * pl.col("se")) & (pl.col("n") > 100)).sort("adj_mass").iter_rows(named=True):
    print(f"  {r['seg']:>20s} {r['bucket']:>18s} n={r['n']:5d} adj={r['adj_oof']:+.4f} (SE {r['se']:.4f}) mass={r['adj_mass']:+.4f}")

# ---------------- 2. test-mix reweighting ----------------
print("\n## 2. Val reweighted to the test (M x cold-share) mix")
key_v = np.char.add(S["M"].astype(str), S["cold_share"].astype(str))
key_t = np.char.add(St["M"].astype(str), St["cold_share"].astype(str))
uk = np.unique(np.r_[key_v, key_t])
pv = np.array([(key_v == k).mean() for k in uk]); pt = np.array([(key_t == k).mean() for k in uk])
w = np.zeros(N)
for k, a, b in zip(uk, pv, pt):
    if a > 0:
        w[key_v == k] = b / a
w /= w.mean()
keyM_v = S["M"].astype(str); keyM_t = St["M"].astype(str)
wM = np.zeros(N)
for k in np.unique(keyM_v):
    wM[keyM_v == k] = (keyM_t == k).mean() / (keyM_v == k).mean()
wM /= wM.mean()
for k in ["nn_oof", "nn_tr", "stk", "blend", "cb"]:
    x = d[f"ndcg_{k}"].to_numpy()
    print(f"  {k:7s} val {x.mean():.4f}  M-mix {np.average(x, weights=wM):.4f}  M x cold mix {np.average(x, weights=w):.4f}")
for a, b in [("nn_oof", "nn_tr"), ("stk", "nn_tr"), ("blend", "nn_tr")]:
    dd = d[f"ndcg_{a}"].to_numpy() - d[f"ndcg_{b}"].to_numpy()
    print(f"  {a}-{b}: val {dd.mean():+.4f}  M-mix {np.average(dd, weights=wM):+.4f}  M x cold mix {np.average(dd, weights=w):+.4f}"
          f"  (SE ~{dd.std() / np.sqrt(N):.4f}; eff. n with weights {w.sum() ** 2 / (w ** 2).sum():.0f})")

# ---------------- candidate-level data ----------------
fva = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
Mv = fva["M"].to_numpy().astype(np.int64); off = fva["cand_off"].to_numpy()
fid_all = np.repeat(np.arange(len(Mv)), Mv)
rowkeep = keep[fid_all]
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
cat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
item = np.load(f"{FLAT}/val/cand_item_id.npy")
P = {k: np.load(f"{ROOT}/preds/{fn}.npy").astype(np.float32) for k, fn in
     [("nn_oof", "nn_final2_val_oof"), ("nn_tr", "nn_final2_trainonly_val"), ("stk", "stk2_res2_r3_val"), ("cb", "cb_tr300k_none_val")]}
Z = {k: feed_rank_z(off, Mv, v)[:, 1].astype(np.float64) for k, v in P.items()}
seen_c = np.zeros(111_400_000, bool); seen_c[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
seen_h = np.zeros(111_400_000, bool); seen_h[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
warm_c = seen_c[item].astype(np.float64); warm_h = seen_h[item].astype(np.float64)
del seen_c, seen_h
lc = np.log1p(np.load(f"{ROOT}/data/feats/val/it_cand_cnt_train.npy").astype(np.float64))
lh = np.log1p(np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy").astype(np.float64))
log("candidate data loaded")

# restrict to clean feeds
sel = np.flatnonzero(rowkeep)
Mk = Mv[keep]; offk = np.r_[0, np.cumsum(Mk)[:-1]]
users = fva["user_id"].to_numpy()[keep]
users = np.where(users == 0, -np.arange(len(users)) - 1, users)  # each user-0 feed its own group
fidk = np.repeat(np.arange(len(Mk)), Mk)
yk = y[sel]


def nd(s):
    return ndcg_per_feed(s, yk, offk, Mk)


def paired(a, b):
    x = a - b
    return f"{x.mean():+.4f} (SE {x.std(ddof=1) / np.sqrt(len(x)):.4f})"


base = {k: nd(Z[k][sel]) for k in Z}
print("\n## 3. What does the val fine-tune learn? LR on z(nn_tr) + priors, 5-fold GroupKFold by user (OOF)")
print(f"  nn_tr {base['nn_tr'].mean():.4f}  nn_oof {base['nn_oof'].mean():.4f}  stk {base['stk'].mean():.4f}")
cat_oh = np.zeros((len(sel), 64), np.float32); cat_oh[np.arange(len(sel)), cat[sel]] = 1
cat_oh = cat_oh[:, cat_oh.sum(0) > 0]
warm = np.stack([warm_c[sel], warm_h[sel], lc[sel], lh[sel]], 1)
zt = Z["nn_tr"][sel][:, None]
variants = {
    "z_tr only": zt,
    "+cat": np.hstack([zt, cat_oh]),
    "+warm": np.hstack([zt, warm]),
    "+cat+warm": np.hstack([zt, cat_oh, warm]),
}
gkf = list(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.arange(len(Mk)), groups=users))
oof_store = {}
for vname, X in variants.items():
    oof = np.zeros(len(sel))
    for trf, vaf in gkf:
        trr = np.isin(fidk, trf); var = ~trr
        lr = LogisticRegression(C=1.0, max_iter=300)
        lr.fit(X[trr], yk[trr])
        oof[var] = X[var] @ lr.coef_[0]
    oof_store[vname] = oof
    s = nd(oof)
    print(f"  {vname:12s} {s.mean():.4f}   vs nn_tr {paired(s, base['nn_tr'])}   vs nn_oof {paired(s, base['nn_oof'])}")
    # where does it gain? pos_all_cold vs warm, by positive category in top-2
    pc = S["pos_cold"]
    print("      by pos_cold: " + ", ".join(f"{b}: {paired(s[pc == b], base['nn_tr'][pc == b])}" for b in ["pos_all_cold", "pos_warm"]))
log("LR done")

# ---------------- 4. segment-routed blends (nested CV) ----------------
print("\n## 4. Segment-routed blend weights (choose per-bucket w on train folds, evaluate on held-out folds)")
grid = [0.0, 0.25, 0.5, 0.75, 1.0]
pairs = {"oof|stk": ("nn_oof", "stk"), "oof|cb": ("nn_oof", "cb"), "oof|tr": ("nn_oof", "nn_tr")}
for pname, (a, b) in pairs.items():
    g = grid if pname != "oof|cb" else [0.0, 0.1, 0.2, 0.3]
    nds = np.stack([nd((1 - wt) * Z[a][sel] + wt * Z[b][sel]) for wt in g])  # (len(g), feeds)
    # global
    def route(lab):
        out = np.zeros(len(Mk))
        for trf, vaf in gkf:
            for bkt in np.unique(lab):
                mtr = trf[lab[trf] == bkt]; mva = vaf[lab[vaf] == bkt]
                if len(mva) == 0:
                    continue
                j = int(np.argmax(nds[:, mtr].mean(1))) if len(mtr) else 0
                out[mva] = nds[j, mva]
        return out
    glob = route(np.zeros(len(Mk), int))
    print(f"  [{pname}] global-w CV {glob.mean():.4f} (vs nn_oof {paired(glob, base['nn_oof'])}); in-sample best "
          f"w={g[int(np.argmax(nds.mean(1)))]} {nds.mean(1).max():.4f}")
    for sname in ["M", "H", "user", "cold_share", "n_cat", "hist_top_cat_share", "cand_cat_in_hist", "rho_nn_cb", "hour", "day"]:
        r = route(S[sname])
        print(f"      route by {sname:20s} {r.mean():.4f}  vs global {paired(r, glob)}")
log("done")
