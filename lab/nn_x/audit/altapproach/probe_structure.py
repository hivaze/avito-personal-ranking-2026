"""Alt-approach audit, probe 2: structural slices of G3 on val (leak-free), plus test-side density of transductive signals.

  (a) user_id = 0 and H = 0 feeds: G3 / FT-branch / TO-branch NDCG, FT - TO paired delta per slice, and the post-hoc
      routing "TO-only on user-0 feeds" (pre-declared: report only).
  (b) multi-positive feeds: share of same-category positive pairs vs the within-feed chance level, G3 NDCG and
      top-1 precision by number of positives, and whether G3's top-ranked positive's category predicts the others.
  (c) test-side coverage of the transductive other-user history signals (val+test histories), public vs private,
      from the eda/history_leak cache (label-free).
  (d) private-mix weights of the slices.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"; HL = ROOT / "eda/history_leak"
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M); H = f["H"].to_numpy()
fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64); cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
uid0 = f["user_id"].to_numpy() == 0
keep = ~leak_mask(); wpriv = pmix_weights()["private"]
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zft, zto = z(np.load(P / "ens_G3_ft_val.npy")), z(np.load(P / "ens_G3_to_val.npy")); zg = 0.5 * zft + 0.5 * zto
nd = {k: ndcg_per_feed(v, y, off, M) for k, v in [("G3", zg), ("ft", zft), ("to", zto)]}
npos = np.bincount(fid, y, n)
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))

print("=== (a) user-0 / H=0 slices (leak-free val) ===")
for name, m in [("all", keep), ("user0", keep & uid0), ("user>0", keep & ~uid0), ("H=0", keep & (H == 0)), ("H=0 & user>0", keep & (H == 0) & ~uid0),
                ("H<10", keep & (H < 10)), ("H=256", keep & (H == 256))]:
    d = nd["ft"][m] - nd["to"][m]
    print(f"{name:14s} feeds {m.sum():6d} ({m.sum()/keep.sum():.3f}; priv-mix w {wpriv[m].mean():.2f}) G3 {nd['G3'][m].mean():.4f} ft {nd['ft'][m].mean():.4f} "
          f"to {nd['to'][m].mean():.4f}  ft-to {d.mean():+.4f} ± {se(d):.4f}  G3-to {(nd['G3'][m]-nd['to'][m]).mean():+.4f}")
# routing: TO-only on user-0 feeds (post hoc; report only)
route = np.where(uid0[fid], zto, zg)
d = (ndcg_per_feed(route, y, off, M) - nd["G3"])[keep]
pm, pse = wmean_se(d, wpriv[keep])
print(f"routing TO-only on user-0 feeds vs G3: {d.mean():+.5f} ± {se(d):.5f} (private mix {pm:+.5f} ± {pse:.5f})")

print("\n=== (b) multi-positive structure (leak-free val) ===")
# same-category share among positive pairs vs chance (pairs of candidates in the same feed)
ck = fid * 64 + cat
uk, inv = np.unique(ck, return_inverse=True); cnt = np.bincount(inv); kf = uk // 64
pos_per_cat = np.bincount(inv, y)
same_pos_pairs = (pos_per_cat * (pos_per_cat - 1) / 2); all_pos_pairs = npos * (npos - 1) / 2
exp_same = np.bincount(kf, cnt * (cnt - 1) / 2, n) / np.maximum(M * (M - 1) / 2, 1)   # chance that a random pair shares a category
mp = keep & (npos >= 2)
obs = np.bincount(kf, same_pos_pairs, n)[mp].sum() / all_pos_pairs[mp].sum()
expv = (exp_same[mp] * all_pos_pairs[mp]).sum() / all_pos_pairs[mp].sum()
print(f"multi-positive feeds {mp.sum()} ({mp.sum()/keep.sum():.3f}); share of positive pairs in the same category: observed {obs:.3f} vs chance {expv:.3f}")
# NDCG and top-1 precision by npos
order = np.lexsort((y, -zg, fid)); top = order[np.searchsorted(fid[order], np.arange(n))]
top1 = y[top]
for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 100)]:
    m = keep & (npos >= lo) & (npos <= hi)
    print(f"npos {lo}-{hi}: feeds {m.sum():5d} share {m.sum()/keep.sum():.3f} priv-w {wpriv[m].mean():.2f} G3 {nd['G3'][m].mean():.4f} to {nd['to'][m].mean():.4f} ft {nd['ft'][m].mean():.4f} top1-prec {top1[m].mean():.3f} meanM {M[m].mean():.0f}")
# does the top-ranked candidate's category predict the other positives?  P(other positive shares top-1 category) vs chance
topcat = cat[top]
shares = np.bincount(inv, minlength=len(uk)) / M[kf]        # category share within feed
sh_top = np.zeros(n); sh_top[kf] = 0
cat_share_top = np.array([shares[np.searchsorted(uk, g * 64 + topcat[g])] for g in range(n)])
yy = y.copy(); yy[top] = 0  # other positives
oth_same = np.bincount(fid, yy * (cat == topcat[fid]), n); oth = np.bincount(fid, yy, n)
m = mp
print(f"other positives in G3's top-1 category: {oth_same[m].sum()/oth[m].sum():.3f} vs chance (feed share of that category) {np.average(cat_share_top[m], weights=oth[m]):.3f}")

print("\n=== (c) test-side density of transductive other-user history signals (label-free) ===")
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = ft["M"].to_numpy(); tst = np.repeat(ft["timestamp"].to_numpy(), Mt)
pub = tst < 1790497920
tr = pl.read_parquet(HL / "out_item_trans_test.parquet").sort("row")
va = pl.read_parquet(HL / "out_item_trans_val.parquet").sort("row")
for c in ["ho_later_all", "ho_later_fc", "ho_earlier_all", "ho_earlier_fc", "cand_other", "cand_other_later"]:
    xt = tr[c].to_numpy(); xv = va[c].to_numpy()
    print(f"{c:18s} nonzero share: val {np.mean(xv>0):.4f} | test public {np.mean(xt[pub]>0):.4f} private {np.mean(xt[~pub]>0):.4f}")
cvt = np.load(HL / "out_covis_test_train.npy", mmap_mode="r")
print(f"cv_invd20 (train sources only) nonzero: test public {np.mean(cvt[pub,3]>0):.4f} private {np.mean(cvt[~pub,3]>0):.4f} (val train-src 0.113)")
