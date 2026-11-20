# Fourth pass (2026-11-13): the feeds where the pool fails. A tail feed has no positive in the pool's top 8.
# The script compares tail and head feeds, compares each best-ranked positive with the negatives above it on
# every cue the ranker has, and fits a label-free classifier of 'is this a tail feed'.
# Run with no arguments on CPU (12 threads, 21 seconds). No ranker is trained.
# Output: ../results/pass4_tail_anatomy.txt.
# Result: 51.2% of feeds carry 66.0% of the loss; feed size alone predicts a tail feed at AUC 0.766 and
# everything else adds 0.007; in tail feeds the positive loses to its beaters on every cue.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/insights/tail_1113/tail.py
"""TAIL ERROR ANATOMY of G3 on val (2026-11-13). CPU only, 12 threads. Descriptive (label side) + one label-free
predictability test. No ranker is trained; the only fitted models are the nested 'is this a tail feed' classifiers of §3.

TAIL feed = >= 1 positive and NO positive in G3's top-8 (best-positive 0-based rank >= 8); HEAD feed = the rest with positives.
Focal positive = the feed's best-ranked positive under G3 (it defines tail membership). Top negative = the feed's best-ranked
negative. Ranks follow src/metrics.py exactly (score desc, pessimistic ties: negatives first).
SEs: user-clustered (user_id 0 feeds = singleton clusters); M-adjusted differences use M-decile fixed effects.
Outputs (this directory only): tail_log.txt, tail_results.json.
"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[v] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed

R = "<repo>"
OUT = f"{R}/nn_x/insights/tail_1113"
HEADK = 8
T0 = time.time()
LOG = []
RES = {}


def say(s=""):
    print(s, flush=True); LOG.append(s)


# ----------------------------------------------------------------------------------------------- load
f, M, off = feed_meta("val")
M = M.astype(np.int64); off = off.astype(np.int64); n = len(M); N = int(M.sum())
fid = np.repeat(np.arange(n), M)
user = f["user_id"].to_numpy().astype(np.int64)
H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
s = np.load(f"{R}/preds/ens_G3_final_val.npy").astype(np.float64)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
item = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)
cemb = np.load(FLAT / "val/cand_emb_idx.npy").astype(np.int64)
F = {k: np.load(FLAT / f"val/cand_{k}.npy").astype(np.float64) for k in ["f1", "f2", "f3", "f4", "f5"]}
hfid = np.load(FLAT / "val/hist_fidx.npy").astype(np.int64)
hitem = np.load(FLAT / "val/hist_item_id.npy").astype(np.int64)
hcat = np.load(FLAT / "val/hist_category_id.npy").astype(np.int64)
hpos = np.load(FLAT / "val/hist_hpos.npy").astype(np.int64)
hemb = np.load(FLAT / "val/hist_emb_idx.npy").astype(np.int64)
pop = np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64), nan=0.0)
pop1 = np.nan_to_num(np.load(FEATS / "val/it_hist_et1_luo.npy").astype(np.float64), nan=0.0)
cosu = np.load(FEATS / "val/cos_u.npy").astype(np.float64)
assert len(s) == N == len(y) == len(cat) and len(hfid) == H.sum()
assert np.all(np.diff(hfid) >= 0) and np.all(hfid == np.repeat(np.arange(n), H))
lp = np.log1p(pop)

nd = ndcg_per_feed(s, y, off, M)
say(f"TAIL ERROR ANATOMY, G3 final val (preds/ens_G3_final_val.npy). Run {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
say(f"sanity: G3 val NDCG {nd.mean():.5f} (recorded 0.40032); feeds {n}, candidates {N}, positives {int(y.sum())}")
assert abs(nd.mean() - 0.40032) < 1e-5

# clusters
cl = user.copy(); z = cl == 0
cl[z] = -(np.arange(z.sum()) + 1)
_, cl = np.unique(cl, return_inverse=True)
G = cl.max() + 1

# ranks (0-based, metric tie rule)
order = np.lexsort((y, -s, fid))
rank = np.empty(N, np.int64); rank[order] = np.arange(N) - np.repeat(off, M)
disc = lambda r: 1.0 / np.log2(np.asarray(r, np.float64) + 2.0)
npos = np.bincount(fid, y, n)
cumd = np.concatenate([[0.0], np.cumsum(disc(np.arange(int(M.max()))))])
idcg = cumd[npos.astype(np.int64)]
P = np.where(y > 0)[0]; Ng = np.where(y == 0)[0]
po = P[np.lexsort((rank[P], fid[P]))]
first = np.ones(len(po), bool); first[1:] = fid[po][1:] != fid[po][:-1]
foc = np.full(n, -1); foc[fid[po][first]] = po[first]                # focal (best-ranked) positive per feed
no = Ng[np.lexsort((rank[Ng], fid[Ng]))]
firstn = np.ones(len(no), bool); firstn[1:] = fid[no][1:] != fid[no][:-1]
tneg = np.full(n, -1); tneg[fid[no][firstn]] = no[firstn]            # top-ranked negative per feed
haspos = npos > 0
bpr = np.where(haspos, rank[np.maximum(foc, 0)], 10 ** 9)
tail = haspos & (bpr >= HEADK); head = haspos & ~tail
loss = 1.0 - nd
say(f"feeds with >=1 positive {haspos.sum()} ({haspos.mean():.4f}); TAIL {tail.sum()} ({tail.mean():.4f}) / HEAD {head.sum()}")
say(f"loss (1-NDCG): total {loss.sum():.1f} = {loss.mean():.5f}/feed; tail share {loss[tail].sum()/loss.sum():.4f}; "
    f"mean NDCG tail {nd[tail].mean():.4f}, head {nd[head].mean():.4f}")
RES["n_tail"] = int(tail.sum()); RES["n_head"] = int(head.sum()); RES["tail_loss_share"] = float(loss[tail].sum() / loss.sum())

# M deciles (all feeds) for adjustment
Mq = np.quantile(M, np.linspace(0, 1, 11)[1:-1])
Mbin = np.digitize(M, Mq)


def ols_cr(v, X, c):
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ (X.T @ v)
    e = v - X @ b
    g = c.max() + 1
    Sg = np.stack([np.bincount(c, X[:, j] * e, g) for j in range(X.shape[1])], 1)
    k = X.shape[1]; nn = len(v)
    V = (g / max(g - 1, 1)) * ((nn - 1) / max(nn - k, 1)) * XtX @ (Sg.T @ Sg) @ XtX
    return b, np.sqrt(np.maximum(np.diag(V), 0))


def tdiff(v, name, fmt="{:.4f}", sel=None):
    """head mean | tail mean | tail-head (cluster SE) | M-adjusted tail-head (M-decile FE, cluster SE)."""
    v = np.asarray(v, np.float64)
    m = (head | tail) & np.isfinite(v)
    if sel is not None:
        m &= sel
    vv, t, c, mb = v[m], tail[m].astype(np.float64), cl[m], Mbin[m]
    _, c = np.unique(c, return_inverse=True)
    b, se = ols_cr(vv, np.stack([np.ones_like(t), t], 1), c)
    X = np.concatenate([t[:, None], (mb[:, None] == np.arange(10)[None, :]).astype(np.float64)], 1)
    X = X[:, np.r_[True, X[:, 1:].sum(0) > 0]]
    ba, sea = ols_cr(vv, X, c)
    hm, tm = vv[t == 0].mean(), vv[t == 1].mean()
    say(f"  {name:44s} head {fmt.format(hm):>9s} | tail {fmt.format(tm):>9s} | diff {b[1]:+.4f} +- {se[1]:.4f} | "
        f"M-adj {ba[0]:+.4f} +- {sea[0]:.4f}   (n {int((t==0).sum())}/{int(t.sum())})")
    RES.setdefault("compare", {})[name] = dict(head=float(hm), tail=float(tm), diff=float(b[1]), se=float(se[1]),
                                               diff_madj=float(ba[0]), se_madj=float(sea[0]))
    return hm, tm


def pct_in_feed(x):
    """average-rank percentile of x within its feed in [0,1] (NaN -> lowest)."""
    x = np.where(np.isfinite(x), x, -np.inf)
    o = np.lexsort((x, fid)); xs = x[o]; fs = fid[o]
    newg = np.ones(N, bool); newg[1:] = (fs[1:] != fs[:-1]) | (xs[1:] != xs[:-1])
    gid = np.cumsum(newg) - 1
    gmean = np.bincount(gid, np.arange(N, dtype=np.float64)) / np.bincount(gid)
    r = gmean[gid] - off[fs]
    out = np.empty(N); out[o] = r / np.maximum(M[fs] - 1, 1)
    return out


def ratio_cl(num, den, c=None):
    c = cl if c is None else c
    r = num.sum() / den.sum(); g = c.max() + 1
    u = np.bincount(c, num - r * den, g)
    return r, np.sqrt(g / (g - 1) * (u ** 2).sum()) / den.sum()


# ----------------------------------------------------------------------------------------------- per-candidate cues
t1 = time.time()
emb_path = f"{R}/data/competition/val/item_embeddings/data.memmap"
table = np.fromfile(emb_path, dtype=np.int8).reshape(2516711, 128)
cmax_all = np.full(N, np.nan); cmax_n5 = np.full(N, np.nan); cmean_n5 = np.full(N, np.nan)
for i in range(n):
    h = H[i]
    if h == 0:
        continue
    ho = hoff[i]
    he = hemb[ho:ho + h]; ok = he >= 0
    if not ok.any():
        continue
    Hv = table[he[ok]].astype(np.float32)
    Hv /= np.linalg.norm(Hv, axis=1, keepdims=True) + 1e-6
    newest = hpos[ho:ho + h][ok] < 5
    co, m = off[i], M[i]
    ce = cemb[co:co + m]; okc = ce >= 0
    if not okc.any():
        continue
    Cv = table[ce[okc]].astype(np.float32)
    Cv /= np.linalg.norm(Cv, axis=1, keepdims=True) + 1e-6
    S = Cv @ Hv.T
    idx = co + np.where(okc)[0]
    cmax_all[idx] = S.max(1)
    if newest.any():
        cmax_n5[idx] = S[:, newest].max(1)
        cmean_n5[idx] = S[:, newest].mean(1)
del table
say(f"[cosines computed in {time.time()-t1:.0f}s; candidates without embedding {np.mean(cemb < 0):.4f}, "
    f"without any cos to history {np.mean(~np.isfinite(cmax_all)):.4f}]")

# categories / items in history
KC = 256
hk = np.unique(hfid * KC + hcat + 128)
hk5 = np.unique((hfid * KC + hcat + 128)[hpos < 5])
hk1 = np.unique((hfid * KC + hcat + 128)[hpos < 1])
ck = fid * KC + cat + 128
cat_in_hist = np.isin(ck, hk); cat_in_n5 = np.isin(ck, hk5); cat_in_n1 = np.isin(ck, hk1)
ik = np.unique((hfid << 32) | hitem)
item_in_hist = np.isin((fid << 32) | item, ik)
# category share of history for the candidate's category
uh, ch = np.unique(hfid * KC + hcat + 128, return_counts=True)
j = np.searchsorted(uh, ck); j = np.minimum(j, len(uh) - 1)
cat_hist_share = np.where(uh[j] == ck, ch[j], 0) / np.maximum(H[fid], 1)
# candidate-set category counts
uc, cc = np.unique(ck, return_counts=True)
jc = np.searchsorted(uc, ck)
cat_cnt_feed = cc[jc]
# within-feed percentiles
pct = {k: pct_in_feed(v) for k, v in [("pop", pop), ("cos_all", cmax_all), ("cos_n5", cmax_n5), ("cosu", cosu),
                                      ("f1", F["f1"]), ("f2", F["f2"]), ("f3", F["f3"]), ("f4", F["f4"]), ("f5", F["f5"])]}
# per-feed z of G3 (G3 final is already a per-feed z-mix; z again for comparability)
mu = np.bincount(fid, s, n) / M; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
sz = (s - mu[fid]) / sd[fid]

fp = np.maximum(foc, 0); tn = np.maximum(tneg, 0)
tn_ok = tneg >= 0
say(f'feeds without any negative: {(~tn_ok).sum()} (their top-neg statistics are NaN)')


def TN(x):
    return np.where(tn_ok, np.asarray(x, np.float64)[tn], np.nan)


feed_median_pop = np.zeros(n)
for_med = np.lexsort((pop, fid))
# median popularity per feed
ps = pop[for_med]
lo = off + (M - 1) // 2; hi = off + M // 2
feed_median_pop = 0.5 * (ps[lo] + ps[hi])

# above-positive counts
above = (rank < bpr[fid]) & (y == 0)
same = cat == cat[fp][fid]
n_same_above = np.bincount(fid, above & same, n); n_other_above = np.bincount(fid, above & ~same, n)
within_cat_rank = n_same_above  # number of same-category candidates above the focal positive (all are negatives)
pcat_share = cat_cnt_feed[fp] / M
topcat_is_pcat = np.where(tn_ok, (cat[tn] == cat[fp]).astype(float), np.nan)

# ----------------------------------------------------------------------------------------------- §2 comparisons
say("\n=== §2 TAIL vs HEAD (feed level; focal positive = best-ranked positive; top neg = best-ranked negative) ===")
say("  columns: head mean | tail mean | tail-head +- user-clustered SE | same with M-decile fixed effects")
say(" -- feed descriptors --")
tdiff(M, "M", "{:.1f}")
tdiff(np.log(M), "log M")
say(f"  median M: head {np.median(M[head]):.0f}, tail {np.median(M[tail]):.0f}; share M>=50: head {np.mean(M[head]>=50):.3f}, tail {np.mean(M[tail]>=50):.3f}")
tdiff(H, "history length H", "{:.1f}")
tdiff((H >= 256).astype(float), "H at the 256 cap")
tdiff((H == 0).astype(float), "H = 0 (empty history)")
tdiff((user == 0).astype(float), "user_id 0 share")
tdiff(npos, "n positives (label side)")
tdiff(npos / M, "positive rate npos/M (label side)")
say(" -- the positive's category --")
tdiff(pcat_share, "share of candidates in positive's category")
tdiff(cat_cnt_feed[fp].astype(float), "n candidates in positive's category", "{:.1f}")
tdiff(bpr.astype(float), "rank of focal positive (0-based)", "{:.1f}")
tdiff(n_same_above.astype(float), "n same-category candidates above positive", "{:.1f}")
tdiff(n_other_above.astype(float), "n other-category candidates above positive", "{:.1f}")
sh_same = np.where(bpr > 0, n_same_above / np.maximum(bpr, 1), np.nan)
tdiff(sh_same, "share of beaters in the positive's category")
wcr_rel = n_same_above / np.maximum(cat_cnt_feed[fp] - 1, 1)
tdiff(np.where(cat_cnt_feed[fp] > 1, wcr_rel, np.nan), "pos within-category rank / (n same-cat - 1)")
tdiff((n_same_above == 0).astype(float), "pos is its category's top item")
tdiff(topcat_is_pcat.astype(float), "top neg in positive's category")
tdiff(cat_in_hist[fp].astype(float), "pos category in history (any)")
tdiff(cat_in_n5[fp].astype(float), "pos category in newest-5 events")
tdiff(cat_in_n1[fp].astype(float), "pos category = newest event's category")
tdiff(cat_in_hist[tn].astype(float), "top-neg category in history (any)", sel=tn_ok)
tdiff(cat_in_n5[tn].astype(float), "top-neg category in newest-5 events", sel=tn_ok)
tdiff(cat_hist_share[fp], "pos category share of history")
tdiff(cat_hist_share[tn], "top-neg category share of history", sel=tn_ok)
say(" -- popularity (it_hist_all_luo) --")
tdiff(lp[fp], "log1p pop, positive")
tdiff(lp[tn], "log1p pop, top neg", sel=tn_ok)
tdiff(lp[fp] - lp[tn], "log1p pop, positive - top neg", sel=tn_ok)
tdiff(lp[fp] - np.log1p(feed_median_pop), "log1p pop, positive - feed median")
tdiff((pop[fp] > feed_median_pop).astype(float), "P(pos pop > feed median)")
tdiff((pop[fp] > pop[tn]).astype(float) + 0.5 * (pop[fp] == pop[tn]), "P(pos pop > top-neg pop) (ties .5)", sel=tn_ok)
tdiff(pct["pop"][fp], "pos pop within-feed percentile")
tdiff((pop[fp] == 0).astype(float), "pos is cold (pop 0)")
tdiff((pop[tn] == 0).astype(float), "top neg is cold (pop 0)", sel=tn_ok)
say(" -- embedding similarity to the history --")
tdiff(cmax_all[fp], "cos max to history, positive")
tdiff(cmax_all[tn], "cos max to history, top neg", sel=tn_ok)
tdiff(cmax_all[fp] - cmax_all[tn], "cos max, positive - top neg", sel=tn_ok)
tdiff(cmax_n5[fp], "cos max to newest 5, positive")
tdiff(cmax_n5[tn], "cos max to newest 5, top neg", sel=tn_ok)
tdiff(cmax_n5[fp] - cmax_n5[tn], "cos max newest 5, positive - top neg", sel=tn_ok)
tdiff(pct["cos_all"][fp], "pos cos-max within-feed percentile")
tdiff(pct["cos_n5"][fp], "pos cos-newest5 within-feed percentile")
tdiff(cosu[fp] - cosu[tn], "cos to user_emb, positive - top neg", sel=tn_ok)
tdiff((cmax_all[fp] > 0.98).astype(float), "pos near-duplicate of a history item (cos>.98)")
tdiff((cmax_all[tn] > 0.98).astype(float), "top neg near-duplicate of a history item", sel=tn_ok)
tdiff(item_in_hist[fp].astype(float), "pos item_id itself in history")
tdiff(item_in_hist[tn].astype(float), "top neg item_id itself in history", sel=tn_ok)
say(" -- f1..f5 (positive vs top neg) --")
for k in ["f1", "f2", "f3", "f4", "f5"]:
    x = F[k]
    tdiff(x[fp] - x[tn], f"{k}: positive - top neg", sel=tn_ok)
    tdiff((x[fp] > x[tn]).astype(float) + 0.5 * (x[fp] == x[tn]), f"{k}: P(pos > top neg) (ties .5)", sel=tn_ok)
    tdiff(pct[k][fp], f"{k}: pos within-feed percentile")
tdiff(F["f4"][fp], "f4 positive (level)")
tdiff(F["f4"][tn], "f4 top neg (level)", sel=tn_ok)
say(" -- G3 score --")
tdiff(sz[fp], "G3 within-feed z of the positive")
tdiff(sz[tn], "G3 within-feed z of the top neg", sel=tn_ok)

# near-duplicate shares as plain proportions
nd_t = np.nanmean(cmax_all[fp][tail] > 0.98); nd_h = np.nanmean(cmax_all[fp][head] > 0.98)
anyd = np.bincount(fid, (y > 0) & (cmax_all > 0.98), n) > 0
say(f"\nnear-duplicate (cos>0.98 to any history item): focal positive tail {nd_t:.4f} / head {nd_h:.4f}; "
    f"ANY positive near-dup: tail {anyd[tail].mean():.4f} / head {anyd[head].mean():.4f}; "
    f"all negatives {np.nanmean(cmax_all[y==0] > 0.98):.4f}, all positives {np.nanmean(cmax_all[y>0] > 0.98):.4f}")
# where are tail near-dup positives ranked, and how many near-dup negatives sit above them
tl = tail & (cmax_all[fp] > 0.98)
ndneg_above = np.bincount(fid, above & (cmax_all > 0.98), n)
say(f"  tail feeds with near-dup focal positive: {tl.sum()}; their median rank {np.median(bpr[tl]):.0f}; "
    f"near-dup NEGATIVES above it: median {np.median(ndneg_above[tl]):.0f}, mean {ndneg_above[tl].mean():.1f}")
RES["neardup"] = dict(tail=float(nd_t), head=float(nd_h), any_tail=float(anyd[tail].mean()), any_head=float(anyd[head].mean()))

# ----------------------------------------------------------------------------------------------- beaters: cue AUCs
say("\n=== §2b CUE AUCs: focal positive vs the negatives ranked ABOVE it (tail beaters) and vs all negatives ===")
say("  P(cue_pos > cue_neg), ties 0.5, pairs with a NaN cue dropped; pooled over pairs, user-clustered SE")
negm = y == 0
cues = [("pop it_hist_all_luo", pop), ("pop it_hist_et1_luo", pop1), ("cos max to history", cmax_all),
        ("cos max newest-5", cmax_n5), ("cos mean newest-5", cmean_n5), ("cos to user_emb", cosu),
        ("cat in history", cat_in_hist.astype(float)), ("cat in newest-5", cat_in_n5.astype(float)),
        ("cat = newest event cat", cat_in_n1.astype(float)), ("cat share of history", cat_hist_share),
        ("item in history", item_in_hist.astype(float)),
        ("f1", F["f1"]), ("f2", F["f2"]), ("f3", F["f3"]), ("f4", F["f4"]), ("f5", F["f5"])]
RES["cue_auc"] = {}
for nm, x in cues:
    xp = x[fp][fid]
    ok = negm & np.isfinite(x) & np.isfinite(xp)
    win = (xp > x).astype(float) + 0.5 * (xp == x)
    row = []
    for lab, msk in [("tail vs beaters", ok & above & tail[fid]), ("tail vs all neg", ok & tail[fid]),
                     ("head vs all neg", ok & head[fid]), ("head vs negs below pos", ok & head[fid] & ~above)]:
        num = np.bincount(fid, win * msk, n); den = np.bincount(fid, msk, n).astype(float)
        r, se = ratio_cl(num, den)
        row.append((lab, r, se))
    say(f"  {nm:24s} " + " | ".join(f"{lab} {r:.3f}+-{se:.3f}" for lab, r, se in row))
    RES["cue_auc"][nm] = {lab: [float(r), float(se)] for lab, r, se in row}

# ----------------------------------------------------------------------------------------------- Pareto dominance
say("\n=== §2d PARETO DOMINANCE: negatives that are >= the focal positive on EVERY monotone cue at once ===")
say("  any scoring that is non-decreasing in each cue ranks such a negative at or above the positive (ties pessimistic);")
say("  cues used: those with head-feed cue AUC >= 0.6 vs all negatives; NaN cue -> the pair is not counted as dominated")
say(f"  item_id of a candidate present in its own feed's history: {item_in_hist.mean():.6f} of all candidates")
cuesets = {"8 cues (pop, cos max, cos newest-5, cos user_emb, cat in newest-5, f1, f3, f5)":
           [pop, cmax_all, cmax_n5, cosu, cat_in_n5.astype(float), F["f1"], F["f3"], F["f5"]],
           "5 cues (cos max, cos newest-5, cat in newest-5, f1, f3)": [cmax_all, cmax_n5, cat_in_n5.astype(float), F["f1"], F["f3"]],
           "3 cues (cos max, f1, pop)": [cmax_all, F["f1"], pop]}
RES["pareto"] = {}
for nm, cs in cuesets.items():
    dom = negm.copy()
    for x in cs:
        xp = x[fp][fid]
        dom &= np.isfinite(x) & np.isfinite(xp) & (x >= xp)
    nd_ = np.bincount(fid, dom, n)
    row = {}
    for lab, msk in [("tail", tail), ("head", head)]:
        row[lab] = dict(ge1=float(np.mean(nd_[msk] >= 1)), ge8=float(np.mean(nd_[msk] >= 8)), mean=float(nd_[msk].mean()),
                        median=float(np.median(nd_[msk])))
    say(f"  {nm}")
    for lab in ["tail", "head"]:
        r_ = row[lab]
        say(f"     {lab}: feeds with >=1 dominating negative {r_['ge1']:.3f}, with >=8 {r_['ge8']:.3f}; dominating negatives per feed mean {r_['mean']:.1f}, median {r_['median']:.0f}")
    # tail loss in feeds where >= 8 negatives dominate (the positive cannot reach the top-8 under any monotone scoring of these cues)
    m8 = tail & (nd_ >= 8)
    say(f"     tail feeds with >=8 dominating negatives hold {loss[m8].sum()/loss.sum():.3f} of ALL val loss; the minimal rank any monotone scoring can give the focal positive in tail feeds: median {np.median(nd_[tail]):.0f} vs its G3 rank median {np.median(bpr[tail]):.0f}")
    RES["pareto"][nm] = row

# ----------------------------------------------------------------------------------------------- within vs cross category
say("\n=== §2c WITHIN- vs CROSS-CATEGORY decomposition of the lost NDCG (exact, Shapley over the two oracles) ===")
say("  W = within-category oracle: each category keeps its rank slots, positives move to their category's earliest slots")
say("  C = category oracle: all candidates of positive-containing categories move above the rest (relative order kept)")
say("  WC = both; residual 1 - f(WC) = interleaving of >= 2 positive categories")


def apply_W(rk):
    g = fid * KC + cat + 128
    o_slots = np.lexsort((rk, g)); o_items = np.lexsort((rk, -y, g))
    new = np.empty(N, np.int64); new[o_items] = rk[o_slots]
    return new


poscat = np.bincount(fid * KC + cat + 128, y, n * KC)[fid * KC + cat + 128] > 0


def apply_C(rk):
    key = np.where(poscat, 0, 1)
    o = np.lexsort((rk, key, fid))
    new = np.empty(N, np.int64); new[o] = np.arange(N) - np.repeat(off, M)
    return new


f0 = nd
rW = apply_W(rank); fW = ndcg_per_feed(-rW.astype(float), y, off, M)
rC = apply_C(rank); fC = ndcg_per_feed(-rC.astype(float), y, off, M)
rWC = apply_W(rC); fWC = ndcg_per_feed(-rWC.astype(float), y, off, M)
rCW = apply_C(rW); fCW = ndcg_per_feed(-rCW.astype(float), y, off, M)
say(f"  check: f(WC) vs f(CW) max |diff| {np.abs(fWC - fCW).max():.2e}; mean f0 {f0.mean():.5f} fW {fW.mean():.5f} fC {fC.mean():.5f} fWC {fWC.mean():.5f}")
Wp = 0.5 * ((fW - f0) + (fWC - fC)); Cp = 0.5 * ((fC - f0) + (fWC - fW)); Mx = 1 - fWC
Wseq = fW - f0; Cseq = 1 - fW
RES["decomp"] = {}
for lab, msk in [("all", haspos), ("tail", tail), ("head", head), ("tail M<50", tail & (M < 50)), ("tail M>=50", tail & (M >= 50))]:
    den = loss * msk
    out = []
    for nm, part in [("within (Shapley)", Wp), ("cross (Shapley)", Cp), ("interleave", Mx), ("within (W first)", Wseq), ("cross (W first)", Cseq)]:
        r, se = ratio_cl(part * msk, den)
        out.append((nm, r, se))
    say(f"  {lab:11s} loss/feed {loss[msk].mean():.4f} (n {msk.sum()}): " + "; ".join(f"{nm} {r:.3f}+-{se:.3f}" for nm, r, se in out))
    RES["decomp"][lab] = {nm: [float(r), float(se)] for nm, r, se in out}
say(f"  oracle gains on ALL val (per feed): W {np.mean(fW-f0):+.4f}, C {np.mean(fC-f0):+.4f}, WC {np.mean(fWC-f0):+.4f}")

# ----------------------------------------------------------------------------------------------- §4 loss attribution
say("\n=== §4 WHERE THE TAIL LOSS IS: exact per-positive split of 1-NDCG ===")
say("  positive j (0-based order among the feed's positives, rank r_j) carries (disc(j) - disc(r_j)) / IDCG; sums to 1-NDCG")
ps_ = po  # positives sorted by (feed, rank)
jj = np.arange(len(ps_)) - np.repeat(np.r_[0, np.cumsum(npos[haspos].astype(np.int64))[:-1]], npos[haspos].astype(np.int64))
contrib = (disc(jj) - disc(rank[ps_])) / idcg[fid[ps_]]
chk = np.bincount(fid[ps_], contrib, n)
assert np.allclose(chk, loss), "per-positive split does not add up"
pf = fid[ps_]; pt = tail[pf]
qpos = np.quantile(pop[y > 0], [1 / 3, 2 / 3])
say(f"  popularity terciles of ALL val positives (it_hist_all_luo): cuts {qpos[0]:.0f} / {qpos[1]:.0f}; "
    f"share of positives with pop 0: {np.mean(pop[y>0]==0):.3f}")
pp = pop[ps_]
if qpos[0] == 0:
    q2 = np.quantile(pop[y > 0][pop[y > 0] > 0], 0.5)
    tb = np.select([pp == 0, pp <= q2], [0, 1], 2)
    tnames = ["cold (pop 0)", f"low (1..{q2:.0f})", f"high (>{q2:.0f})"]
    say(f"  the lower tercile cut is 0, so the groups are: pop 0 | 1..{q2:.0f} (median of non-zero) | > {q2:.0f}")
else:
    tb = np.select([pp <= qpos[0], pp <= qpos[1]], [0, 1], 2)
    tnames = [f"T1 (<= {qpos[0]:.0f})", f"T2 (<= {qpos[1]:.0f})", f"T3 (> {qpos[1]:.0f})"]
rel = pct["pop"][ps_]
rb = np.select([rel < 1 / 3, rel < 2 / 3], [0, 1], 2)
rnames = ["feed-bottom third", "feed-middle third", "feed-top third"]
chs = np.select([cat_in_n5[ps_], cat_in_hist[ps_]], [0, 1], 2)
cnames = ["cat in newest-5", "cat in hist, not newest-5", "cat NOT in history"]
tot_loss = loss.sum(); tail_loss = loss[tail].sum()
RES["loss_split"] = {}


def table(lab, grp, names):
    say(f"  -- by {lab}: share of TAIL loss +- SE | share of ALL val loss | share of tail positives | mean loss per tail positive | NDCG-pts per val feed")
    for b, nm in enumerate(names):
        m = pt & (grp == b)
        num = np.bincount(pf, contrib * m, n)
        r, se = ratio_cl(num, loss * tail)
        npt = m.sum() / pt.sum()
        say(f"     {nm:28s} {r:.3f} +- {se:.3f} | {num.sum()/tot_loss:.3f} | {npt:.3f} | {contrib[m].mean() if m.any() else 0:.4f} | {num.sum()/n:.4f}")
        RES["loss_split"][f"{lab}:{nm}"] = dict(share_tail=float(r), se=float(se), share_all=float(num.sum() / tot_loss),
                                                pos_share=float(npt), per_feed=float(num.sum() / n))


table("popularity group (absolute)", tb, tnames)
table("within-feed popularity third", rb, rnames)
table("category vs history", chs, cnames)
say("  -- 3x3: category vs history (rows) x absolute popularity group (cols): share of TAIL loss (share of tail positives)")
for a, an in enumerate(cnames):
    cells = []
    for b, bn in enumerate(tnames):
        m = pt & (chs == a) & (tb == b)
        cells.append(f"{contrib[m].sum()/tail_loss:.3f} ({m.sum()/pt.sum():.3f})")
    say(f"     {an:28s} " + " | ".join(f"{bn}: {c}" for bn, c in zip(tnames, cells)))
say("  -- same 3x3 for HEAD feeds (share of head loss), for contrast")
ph = head[pf]; head_loss = loss[head].sum()
for a, an in enumerate(cnames):
    cells = []
    for b, bn in enumerate(tnames):
        m = ph & (chs == a) & (tb == b)
        cells.append(f"{contrib[m].sum()/head_loss:.3f} ({m.sum()/ph.sum():.3f})")
    say(f"     {an:28s} " + " | ".join(f"{bn}: {c}" for bn, c in zip(tnames, cells)))
# within/cross for tail per positive-attribute cell (feed-level focal attributes)
say("  -- tail feeds by focal-positive cell: n feeds | mean NDCG | Shapley within share | cross share")
fch = np.select([cat_in_n5[fp], cat_in_hist[fp]], [0, 1], 2)
ftb = np.select([pop[fp] == 0, pop[fp] <= (q2 if qpos[0] == 0 else qpos[1])], [0, 1], 2) if qpos[0] == 0 else \
    np.select([pop[fp] <= qpos[0], pop[fp] <= qpos[1]], [0, 1], 2)
for a, an in enumerate(cnames):
    for b, bn in enumerate(tnames):
        m = tail & (fch == a) & (ftb == b)
        if m.sum() < 30:
            continue
        rw, _ = ratio_cl(Wp * m, loss * m); rc, _ = ratio_cl(Cp * m, loss * m)
        say(f"     {an:28s} x {bn:16s} n {m.sum():5d} | NDCG {nd[m].mean():.3f} | within {rw:.3f} | cross {rc:.3f} | median rank {np.median(bpr[m]):.0f}")

# what a partial tail lift would buy
say("\n  -- what moving tail positives up would buy (focal positive only, other items unchanged; val NDCG points) --")
foc_c = np.zeros(n)
for lab, newr in [("to rank 8 (just into top-8; r=7)", np.full(n, 7)), ("rank halved", bpr // 2),
                  ("rank x0.9", (bpr * 0.9).astype(np.int64)), ("up by 1", bpr - 1)]:
    g_ = np.where(tail, (disc(newr) - disc(bpr)) / np.maximum(idcg, 1e-9), 0.0)
    say(f"     {lab:36s} +{g_.mean():.5f}")
    RES.setdefault("lift", {})[lab] = float(g_.mean())

# ----------------------------------------------------------------------------------------------- §3 predictability
say("\n=== §3 IS FAILURE PREDICTABLE FROM THE FEED ALONE? nested user-grouped 5-fold logistic, label-free feed stats ===")
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier


def ent_per(keys, owner, size, nfeed):
    u, c = np.unique(keys, return_counts=True)
    ow = owner[np.searchsorted(keys, u)] if False else (u // KC)
    p = c / size[ow]
    return np.bincount(ow, -p * np.log(p), nfeed)


cent = ent_per(ck, fid, M, n)
hent = ent_per(hfid * KC + hcat + 128, hfid, np.maximum(H, 1), n)
ncat = np.bincount(np.unique(ck) // KC, minlength=n)
fmean = lambda x: np.bincount(fid, np.nan_to_num(x), n) / np.maximum(np.bincount(fid, np.isfinite(x), n), 1)


def fsd(x):
    okx = np.isfinite(x); c = np.maximum(np.bincount(fid, okx, n), 1)
    m1 = np.bincount(fid, np.where(okx, x, 0), n) / c; m2 = np.bincount(fid, np.where(okx, x * x, 0), n) / c
    return np.sqrt(np.maximum(m2 - m1 ** 2, 0))


def fmax(x):
    o = np.full(n, -np.inf); np.maximum.at(o, fid, np.where(np.isfinite(x), x, -np.inf)); return np.where(np.isfinite(o), o, 0)


def fmed(x):
    xx = np.where(np.isfinite(x), x, -1.0)
    o = np.lexsort((xx, fid)); xs = xx[o]
    return 0.5 * (xs[off + (M - 1) // 2] + xs[off + M // 2])


cos_sd = fsd(cmax_all); cos_gap = fmax(cmax_all) - fmed(cmax_all); cos_max = fmax(cmax_all)
pop_sd = fsd(lp); pop_gap = fmax(lp) - fmed(lp); cold_sh = np.bincount(fid, pop == 0, n) / M
nd_sh = np.bincount(fid, cmax_all > 0.98, n) / M
cih_sh = np.bincount(fid, cat_in_hist, n) / M; ci5_sh = np.bincount(fid, cat_in_n5, n) / M
miss_sh = np.bincount(fid, cemb < 0, n) / M
f_sd = np.stack([fsd(F[k]) for k in ["f1", "f2", "f3", "f4", "f5"]], 1)
# G3 score shape (label-free at test time)
o = np.lexsort((-sz, fid)); szs = sz[o]
top1 = szs[off]; top8 = szs[off + np.minimum(7, M - 1)]
gap18 = top1 - top8
ex = np.exp(sz - top1[fid]); pz = ex / np.bincount(fid, ex, n)[fid]
sent = np.bincount(fid, -pz * np.log(pz + 1e-300), n) / np.log(np.maximum(M, 2))
logM = np.log(M); logH = np.log1p(H)
FA = dict(logM=logM, logM2=logM ** 2, logH=logH, Hcap=(H >= 256).astype(float), H0=(H == 0).astype(float),
          user0=(user == 0).astype(float), cand_cat_entropy=cent, hist_cat_entropy=hent, ncat_over_M=ncat / M,
          pop_sd=pop_sd, pop_gap=pop_gap, cold_share=cold_sh, cos_sd=cos_sd, cos_gap=cos_gap, cos_max=cos_max)
FB = dict(FA, cat_in_hist_share=cih_sh, cat_in_n5_share=ci5_sh, neardup_share=nd_sh, emb_missing_share=miss_sh,
          **{f"{k}_sd": f_sd[:, i] for i, k in enumerate(["f1", "f2", "f3", "f4", "f5"])})
FC_ = dict(FB, g3_gap_top1_top8=gap18, g3_top1_z=top1, g3_softmax_entropy=sent)
yb = tail[haspos].astype(int)
grp = cl[haspos]
Cs = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def nested_lr(X):
    X = X[haspos]
    oof = np.zeros(len(yb)); chosen = []
    for tr, te in GroupKFold(5).split(X, yb, grp):
        best = None
        for C in Cs:
            sc = []
            for itr, ite in GroupKFold(4).split(X[tr], yb[tr], grp[tr]):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
                mdl.fit(X[tr][itr], yb[tr][itr])
                sc.append(roc_auc_score(yb[tr][ite], mdl.predict_proba(X[tr][ite])[:, 1]))
            if best is None or np.mean(sc) > best[0]:
                best = (np.mean(sc), C)
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=5000)).fit(X[tr], yb[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]; chosen.append(best[1])
    return oof, chosen


def oof_hgb(X):
    X = X[haspos]; oof = np.zeros(len(yb))
    for tr, te in GroupKFold(5).split(X, yb, grp):
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                                             l2_regularization=1.0, random_state=0).fit(X[tr], yb[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return oof


mats = {"M only (logM, logM^2)": np.stack([logM, logM ** 2], 1),
        "A: M, H, entropy, user0, pop spread, cos spread": np.stack(list(FA.values()), 1),
        "B: A + category/near-dup/f-spread shares": np.stack(list(FB.values()), 1),
        "C: B + G3 score shape (gap, top-1 z, entropy)": np.stack(list(FC_.values()), 1)}
preds = {}
for k, X in mats.items():
    oof, ch = nested_lr(X)
    preds["LR " + k] = oof
    say(f"  [LR] {k:52s} chosen C per fold {ch}")
preds["HGB on C features (fixed params, 5-fold)"] = oof_hgb(mats["C: B + G3 score shape (gap, top-1 z, entropy)"])
gcl = grp.copy(); _, gcl = np.unique(gcl, return_inverse=True); GG = gcl.max() + 1
rng = np.random.default_rng(1113)
names = list(preds)
base_auc = {k: roc_auc_score(yb, preds[k]) for k in names}
B = 300
boot = np.zeros((B, len(names)))
for b in range(B):
    w = np.bincount(rng.integers(0, GG, GG), minlength=GG)[gcl]
    boot[b] = [roc_auc_score(yb, preds[k], sample_weight=w) for k in names]
Mh = M[haspos]; Mb = Mbin[haspos]


def strat_auc(p):
    num = 0.0; den = 0.0
    for b in np.unique(Mb):
        m = Mb == b
        if yb[m].min() == yb[m].max():
            continue
        npair = yb[m].sum() * (1 - yb[m]).sum()
        num += roc_auc_score(yb[m], p[m]) * npair; den += npair
    return num / den


say(f"  base rate P(tail) {yb.mean():.4f}; tail rate by M decile: " +
    " ".join(f"{int(Mq[b-1]) if b>0 else 0}+:{yb[Mb==b].mean():.2f}" for b in range(10)))
say("  model | OOF AUC +- cluster-bootstrap SE | AUC within M deciles (pair-weighted) | AUC - M-only")
RES["predict"] = {}
for i, k in enumerate(names):
    dd = boot[:, i] - boot[:, 0]
    sa = strat_auc(preds[k])
    say(f"  {k:60s} {base_auc[k]:.4f} +- {boot[:, i].std(ddof=1):.4f} | strat {sa:.4f} | {base_auc[k]-base_auc[names[0]]:+.4f} +- {dd.std(ddof=1):.4f}")
    RES["predict"][k] = dict(auc=float(base_auc[k]), se=float(boot[:, i].std(ddof=1)), strat=float(sa),
                             d_vs_M=float(base_auc[k] - base_auc[names[0]]), d_se=float(dd.std(ddof=1)))
# a usable reading: what fraction of the tail loss sits in the top-scored decile of predicted tail risk (beyond M)
pC = preds["LR C: B + G3 score shape (gap, top-1 z, entropy)"]
lh = loss[haspos]
for q in [0.5, 0.8]:
    thr = np.quantile(pC, q)
    m = pC >= thr
    say(f"  LR-C top {round((1-q)*100)}% risk feeds: tail rate {yb[m].mean():.3f} (rest {yb[~m].mean():.3f}), share of all loss {lh[m].sum()/lh.sum():.3f}")
# per-M-bucket AUC
for lab, m in [("M<50", Mh < 50), ("M>=50", Mh >= 50)]:
    say(f"  within {lab}: tail rate {yb[m].mean():.3f}; AUC M-only {roc_auc_score(yb[m], preds[names[0]][m]):.4f}, "
        f"A {roc_auc_score(yb[m], preds[names[1]][m]):.4f}, B {roc_auc_score(yb[m], preds[names[2]][m]):.4f}, "
        f"C {roc_auc_score(yb[m], preds[names[3]][m]):.4f}, HGB {roc_auc_score(yb[m], preds[names[4]][m]):.4f}")
# standardized LR coefficients on the full data (descriptive)
Xf = mats["B: A + category/near-dup/f-spread shares"][haspos]
mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xf, yb)
co_ = mdl[-1].coef_[0]
say("  descriptive standardized LR-B coefficients (full fit, C=1): " +
    ", ".join(f"{k} {c:+.2f}" for k, c in sorted(zip(FB.keys(), co_), key=lambda t: -abs(t[1]))[:12]))

json.dump(RES, open(f"{OUT}/tail_results.json", "w"), indent=1)
say(f"\n[done in {time.time()-T0:.0f}s]")
open(f"{OUT}/tail_log.txt", "w").write("\n".join(LOG) + "\n")
