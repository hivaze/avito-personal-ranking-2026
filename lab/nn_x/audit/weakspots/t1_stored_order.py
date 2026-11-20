"""T1: does the STORED candidate order of val (and test, label-free) carry any label signal?
Dataset page: val/test candidates were uniformly shuffled before display and "the candidate order was additionally
shuffled by one common permutation". If the stored index is a deterministic function of the display slot, display
position bias (random-like train feeds: 2.5% at rank 0 vs 1.15% base) would leak into the stored index.
Checks (val, labels): NDCG of score = -index / +index / random; positive rate by relative index decile; per-M chi-square
of positives over stored index for M<=40 (many feeds per M); rank-correlation of index with item_id / emb_idx / f1 /
category (structure of the stored order) on val and test. Train: is the stored order the production order (pos)?
"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from scipy import stats

rng = np.random.default_rng(0)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

for split in ["val", "test"]:
    f, M, off = feed_meta(split)
    n = len(M); fid = np.repeat(np.arange(n), M)
    idx = np.arange(len(fid)) - np.repeat(off, M)
    rel = idx / np.maximum(M[fid] - 1, 1)
    item = np.load(FLAT / split / "cand_item_id.npy"); emb = np.load(FLAT / split / "cand_emb_idx.npy")
    cat = np.load(FLAT / split / "cand_category_id.npy"); f1 = np.load(FLAT / split / "cand_f1.npy")
    # per-feed Spearman of index with item_id / emb_idx / f1 ; category run structure
    def per_feed_spearman(x):
        rs = []
        sel = rng.choice(n, 3000, replace=False)
        for i in sel:
            if M[i] < 8: continue
            s = slice(off[i], off[i] + M[i])
            rs.append(stats.spearmanr(idx[s], x[s]).statistic)
        rs = np.array(rs); return rs.mean(), np.nanmean(np.abs(rs)), (rs > 0.9).mean(), (rs < -0.9).mean()
    for name, x in [("item_id", item.astype(np.float64)), ("emb_idx", emb.astype(np.float64)), ("f1", f1.astype(np.float64)), ("category", cat.astype(np.float64))]:
        m, am, hi, lo = per_feed_spearman(x)
        say(f"{split}: Spearman(stored index, {name}) over 3000 feeds: mean {m:+.4f}, mean|rho| {am:.4f}, rho>0.9 {hi:.3f}, rho<-0.9 {lo:.3f}")
    # category runs: are same-category candidates adjacent more than chance?
    same_adj = (cat[1:] == cat[:-1]) & (fid[1:] == fid[:-1])
    # chance level: expected adjacency under random order within feed = sum_c p_c^2 per feed, approx via shuffle
    cat_sh = cat.copy()
    for i in rng.choice(n, 2000, replace=False):
        s = slice(off[i], off[i] + M[i]); cat_sh[s] = rng.permutation(cat[s])
    sel_rows = np.isin(fid, rng.choice(n, 2000, replace=False))
    say(f"{split}: adjacent same-category share {same_adj.mean():.4f} (within-feed shuffle of 2000 feeds: {((cat_sh[1:] == cat_sh[:-1]) & (fid[1:] == fid[:-1]))[sel_rows[1:]].mean():.4f} vs real on same feeds {same_adj[sel_rows[1:]].mean():.4f})")
    if split == "val":
        y = np.load(FLAT / split / "cand_target.npy").astype(np.float64)
        for name, sc in [("-index", -idx.astype(np.float64)), ("+index", idx.astype(np.float64)), ("random", rng.random(len(idx)))]:
            nd = ndcg_per_feed(sc, y, off, M)
            say(f"val NDCG score={name}: {nd.mean():.5f} +- {nd.std() / np.sqrt(n):.5f}")
        # positive rate by relative-index decile, overall and by M bucket
        dec = np.minimum((rel * 10).astype(int), 9)
        base = y.mean()
        rates = np.bincount(dec, weights=y, minlength=10) / np.bincount(dec, minlength=10)
        say("val positive rate by relative stored-index decile (base %.4f): " % base + " ".join(f"{r:.4f}" for r in rates))
        for lo_, hi_ in [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
            sel = (M[fid] >= lo_) & (M[fid] < hi_)
            r = np.bincount(dec[sel], weights=y[sel], minlength=10) / np.bincount(dec[sel], minlength=10)
            cnt = np.bincount(dec[sel], weights=y[sel], minlength=10)
            chi = stats.chisquare(cnt, cnt.sum() * np.bincount(dec[sel], minlength=10) / sel.sum())
            say(f"  M in [{lo_},{hi_}): rate by decile " + " ".join(f"{v:.4f}" for v in r) + f"  chi2 p={chi.pvalue:.3g}")
        # absolute index for the first 10 slots and last 3 slots (rank-from-end), feeds M>=20
        sel = M[fid] >= 20
        rk_end = M[fid] - 1 - idx
        say("val M>=20: positive rate by absolute stored index 0..9: " + " ".join(f"{y[sel & (idx == i)].mean():.4f}" for i in range(10)))
        say("val M>=20: positive rate by rank-from-end 0..4: " + " ".join(f"{y[sel & (rk_end == i)].mean():.4f}" for i in range(5)))
        # per-M chi-square for M in 5..40: positives over index
        ps = []
        for m in range(5, 41):
            sel = M[fid] == m
            if sel.sum() < 2000: continue
            cnt = np.bincount(idx[sel], weights=y[sel], minlength=m)
            if cnt.sum() < 20: continue
            chi = stats.chisquare(cnt, np.full(m, cnt.sum() / m)); ps.append(chi.pvalue)
        ps = np.array(ps)
        fis = -2 * np.log(np.clip(ps, 1e-300, 1)).sum()
        say(f"val per-M chi-square (M=5..40, {len(ps)} values): min p {ps.min():.3g}, median p {np.median(ps):.3f}, Fisher combined p {1 - stats.chi2.cdf(fis, 2 * len(ps)):.3g}")
        # does the position of the positive correlate with anything positional in the user's feed?  first/last slot rates for all M
        say(f"val all feeds: P(y | index 0) {y[idx == 0].mean():.4f}, P(y | last index) {y[rk_end == 0].mean():.4f}, base {base:.4f}")

# train: stored order vs production pos
f, M, off = feed_meta("train")
pos = np.load(FLAT / "train/cand_pos.npy")
n = len(M); sel = rng.choice(n, 5000, replace=False)
rs = []
for i in sel:
    if M[i] < 8: continue
    s = slice(off[i], off[i] + M[i]); rs.append(stats.spearmanr(np.arange(M[i]), pos[s]).statistic)
rs = np.array(rs)
say(f"train: Spearman(stored index, pos) over 5000 feeds: mean {rs.mean():.4f}, share == 1: {(rs > 0.9999).mean():.3f}")
open("<repo>/nn_x/audit/weakspots/t1_stored_order.txt", "w").write("\n".join(out) + "\n")
