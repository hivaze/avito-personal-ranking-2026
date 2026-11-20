"""Within-(feed, positive category) AUC of models and item-level features (val, leak-masked), split by positive
warm/cold, plus H256 x user-seen M-adjusted cross-segment and the test-mix share of all-cold-positive feeds."""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from segdefs import seg_defs

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
d = pl.read_parquet(f"{OUT}/feed_table.parquet"); keep = ~d["leak"].to_numpy()
f = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
cat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
item = np.load(f"{FLAT}/val/cand_item_id.npy")
seen = np.zeros(111_400_000, bool)
seen[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
seen[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
cold = ~seen[item]; del seen

ck = fid * 64 + cat
order = np.lexsort((ck,)); cks = ck[order]
starts = np.r_[0, np.flatnonzero(np.diff(cks)) + 1]; ends = np.r_[starts[1:], len(cks)]


@nb.njit(parallel=True)
def group_auc(s, yy, cc, order, starts, ends, out_auc, out_w, out_cold):
    for g in nb.prange(len(starts)):
        a, b = starts[g], ends[g]
        idx = order[a:b]
        npos = 0
        for i in idx:
            npos += yy[i] > 0
        nneg = len(idx) - npos
        if npos == 0 or nneg == 0:
            out_w[g] = 0; continue
        num = 0.0
        coldp = 0
        for i in idx:
            if yy[i] > 0:
                coldp += cc[i]
                for j in idx:
                    if yy[j] == 0:
                        if s[i] > s[j]:
                            num += 1.0
                        elif s[i] == s[j]:
                            num += 0.5
        out_auc[g] = num / (npos * nneg); out_w[g] = npos * nneg; out_cold[g] = coldp == npos


gfeed = cks[starts] // 64
gkeep = keep[gfeed]
scores = {
    "nn_oof": np.load(f"{ROOT}/preds/nn_final2_val_oof.npy"), "nn_tr": np.load(f"{ROOT}/preds/nn_final2_trainonly_val.npy"),
    "stk_clean": np.load(f"{ROOT}/preds/stk2_res4_clean_val.npy"), "cb": np.load(f"{ROOT}/preds/cb_tr300k_none_val.npy"),
}
for k in ["f1", "f2", "f4", "f5"]:
    scores[k] = np.load(f"{FLAT}/val/cand_{k}.npy")
for k in ["cos_h10", "maxsim32", "seq_dmax_hl10", "pre_v3_fe", "it_tr_ctr", "it_hist_all_luo", "clu_ic1024_ctr", "tte_it_ctr"]:
    scores[k] = np.load(f"{ROOT}/data/feats/val/{k}.npy")
print("## Within-(feed, category) AUC, groups with >=1 positive and >=1 negative, pair-weighted mean")
print("| score | all groups | groups with warm positives | groups with all-cold positives |")
print("|---|---|---|---|")
for k, s in scores.items():
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=-1e9)
    auc = np.zeros(len(starts)); w = np.zeros(len(starts)); gc = np.zeros(len(starts), np.bool_)
    group_auc(s, y, cold.astype(np.int64), order, starts, ends, auc, w, gc)
    m = gkeep & (w > 0)
    r = [np.average(auc[mm], weights=w[mm]) for mm in [m, m & ~gc, m & gc]]
    print(f"| {k} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} |")
print(f"groups: {int((gkeep & (w > 0)).sum())}, all-cold-positive groups {int((gkeep & (w > 0) & gc).sum())}")

# cross segment: H256 x user seen in train, M x npos adjusted
dk = d.filter(pl.col("leak").not_())
Mk = dk["M"].to_numpy(); npos = dk["npos"].to_numpy()
mq = np.searchsorted(np.quantile(Mk, np.linspace(0, 1, 21)[1:-1]), Mk, side="right"); cell = mq * 3 + np.minimum(npos, 3) - 1
print("\n## M x npos adjusted NDCG, H256 x user group")
for k in ["nn_oof", "nn_tr"]:
    x = dk[f"ndcg_{k}"].to_numpy(); adj = x - (np.bincount(cell, x) / np.bincount(cell))[cell]
    S = seg_defs(dk)
    for h in ["H256", "H64-191", "H192-255"]:
        for u in ["tr0", "tr1", "tr2-4", "uid0"]:
            m = (S["H"] == h) & (S["user"] == u)
            if m.sum() > 50:
                print(f"  {k} {h:9s} {u:5s} n={m.sum():5d} adj {adj[m].mean():+.4f} (SE {adj[m].std() / np.sqrt(m.sum()):.4f})")

# test-mix share of all-cold-positive feeds (reweight val by M x cold-share bucket to the test mix)
te = pl.read_parquet(f"{OUT}/test_segments.parquet"); St = seg_defs(te, test=True); S = seg_defs(dk)
kv = np.char.add(S["M"].astype(str), S["cold_share"].astype(str)); kt = np.char.add(St["M"].astype(str), St["cold_share"].astype(str))
w = np.zeros(len(dk))
for k in np.unique(kv):
    w[kv == k] = (kt == k).mean() / (kv == k).mean()
pac = dk["pos_cold_all"].to_numpy()
print(f"\nall-cold-positive feeds: val {pac.mean():.3f}, val reweighted to test M x cold mix {np.average(pac, weights=w):.3f}")
print(f"mean cold share of candidates: val {dk['cold_share'].mean():.3f}, test {te['cold_share'].mean():.3f}")

# stk2_res2_r3 was replaced by the leak-free stk2_res4_clean during this analysis: its per-segment numbers
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
off = f["cand_off"].to_numpy()
sc = ndcg_per_feed(np.load(f"{ROOT}/preds/stk2_res4_clean_val.npy"), y, off, M)[keep]
for k in ["nn_tr", "nn_oof", "stk"]:
    x = sc - dk[f"ndcg_{k}"].to_numpy()
    print(f"stk_clean - {k}: all {x.mean():+.4f} (SE {x.std()/np.sqrt(len(x)):.4f}) | pos_all_cold {x[pac].mean():+.4f} (SE {x[pac].std()/np.sqrt(pac.sum()):.4f}) | warm-pos {x[~dk['pos_cold_any'].to_numpy()].mean():+.4f}")
print(f"stk_clean val NDCG (clean feeds) {sc.mean():.4f}; pos_all_cold {sc[pac].mean():.4f}")
