"""M2 (CPU, stored predictions only): is there an exploration-propensity x new-category bias left in the current best file?

Table: lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet (sibling stream, read-only): one row per val candidate with
score = 0.5 ft + 0.5 to of FD16P20@m0.4_0.2, label, cat_newest_idx (-1 = category absent from history), and user-history
columns built from the user's OWN history snapshot only: h_nov32r (share of the newest 32 events whose category is absent
from the K older events), h_ent (category entropy), h_ncat, h_topsh; past_npos / past_nnew (earlier labelled TRAIN feeds of
the same user: positives / positives in a category new to that feed's history; train is earlier than every val feed).
Classes of candidates: NEW = category absent from the history; FAR = newest event of the category older than index 31;
NEAR = in the newest 32.
1. O/E of a within-feed softmax(score / T) by class x tercile of each propensity column (T fitted on all clicks).
2. Cross-fitted (user halves A/B of the sibling's half.npy) offset s' = s + sum_c 1[class c] (a_c + b_c * zprop) fitted by
   softmax-CE on one half, NDCG read on the other; paired per feed, val / pmix-public / pmix-private / M<50 / M>=50.
   This is the most a TRAINED input carrying the propensity could do if it acted only through this interaction
   (it can also act through other channels, see the report's caveat: residual gates see 10-26% of trained-input effects).
"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from scipy.optimize import minimize
from common import feed_meta
from metrics import ndcg_per_feed
import pmix

T = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet",
                    columns=["fidx", "row", "score", "label", "cat_newest_idx", "h_nov32r", "h_ent", "h_ncat", "h_topsh",
                             "past_npos", "past_nnew", "M", "user_id"])
fv, Mv, ov = feed_meta("val")
assert len(T) == Mv.sum()
T = T.sort("row") if "row" in T.columns else T
half = np.load("<repo>/lab/nn_x/w12/fixmiss/systematic/data/half.npy", allow_pickle=True)
print("half.npy", half.dtype, half.shape, np.unique(half)[:5])
W = pmix.weights(); big = Mv >= 50
s = T["score"].to_numpy().astype(np.float64); y = T["label"].to_numpy().astype(np.float64)
fid = np.repeat(np.arange(len(Mv)), Mv)
assert (T["fidx"].to_numpy() == fid).all(), "row order differs from stored order"
cni = T["cat_newest_idx"].to_numpy()
cls = np.where(cni < 0, 0, np.where(cni > 31, 1, 2))  # 0 NEW, 1 FAR, 2 NEAR
names = ["NEW", "FAR", "NEAR"]

def softmax_feed(x):
    mx = np.maximum.reduceat(x, ov); e = np.exp(x - np.repeat(mx, Mv)); return e / np.repeat(np.add.reduceat(e, ov), Mv)
K = np.add.reduceat(y, ov); yK = y / np.repeat(np.maximum(K, 1), Mv)
def nll(t):
    p = softmax_feed(s / t); return -(yK * np.log(p + 1e-30)).sum()
Tfit = minimize(lambda v: nll(v[0]), [1.0], method="Nelder-Mead").x[0]
p = softmax_feed(s / Tfit); E = p * np.repeat(K, Mv)
print(f"T* {Tfit:.4f}; candidate shares NEW/FAR/NEAR " + " ".join(f"{(cls == c).mean():.3f}" for c in range(3))
      + "; positive shares " + " ".join(f"{y[cls == c].sum() / y.sum():.3f}" for c in range(3)))

print("\n1. O/E by class x tercile of the propensity column (feed-level column repeated on rows)")
for col in ["h_nov32r", "h_ent", "h_ncat", "h_topsh", "past_nnew"]:
    v = T[col].to_numpy().astype(np.float64)
    if col == "past_nnew":
        npos = T["past_npos"].to_numpy().astype(np.float64)
        v = np.where(npos >= 3, v / np.maximum(npos, 1), np.nan)
    ok = np.isfinite(v)
    q = np.nanquantile(v[ok], [1 / 3, 2 / 3]); tb = np.where(~ok, -1, np.digitize(v, q))
    out = []
    for c in range(3):
        cells = []
        for t in range(3):
            m = (cls == c) & (tb == t)
            o, e = y[m].sum(), E[m].sum()
            cells.append(f"{o / e:.3f}(n+{int(o)})")
        out.append(f"{names[c]} " + " ".join(cells))
    print(f"  {col:9s} terciles low/mid/high (rows with the column: {ok.mean():.2f}):  " + " | ".join(out))

print("\n2. cross-fitted class offsets with propensity slope (fit on one user half, NDCG on the other)")
hv = np.asarray(half).astype(np.int64)
if hv.shape[0] == len(Mv):
    hrow = np.repeat(hv, Mv)
else:
    hrow = hv
base = ndcg_per_feed(s.astype(np.float32), y.astype(np.float32), ov, Mv)
def zcol(v):
    v = np.where(np.isfinite(v), v, np.nanmean(v)); return (v - v.mean()) / (v.std() + 1e-12)
for col in ["h_nov32r", "h_ent", "h_topsh"]:
    zp = zcol(T[col].to_numpy().astype(np.float64))
    X = np.stack([(cls == 0), (cls == 1), (cls == 0) * zp, (cls == 1) * zp, (cls == 2) * zp], 1).astype(np.float64)
    newsc = np.empty_like(s)
    coefs = []
    for h in (0, 1):
        tr = hrow == h
        ftr = np.unique(fid[tr])
        # fit on feeds of half h: restrict arrays to those feeds
        sel = np.isin(fid, ftr)
        Ms = Mv[ftr]; os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]])
        ss, ys, Xs = s[sel], y[sel], X[sel]
        Ks = np.add.reduceat(ys, os_); yKs = ys / np.repeat(np.maximum(Ks, 1), Ms)
        def f(b):
            x = (ss + Xs @ b) / Tfit
            mx = np.maximum.reduceat(x, os_); e = np.exp(x - np.repeat(mx, Ms))
            lse = np.log(np.add.reduceat(e, os_)) + mx
            return -(yKs * (x - np.repeat(lse, Ms))).sum()
        b = minimize(f, np.zeros(X.shape[1]), method="L-BFGS-B").x
        coefs.append(b)
        te = ~sel
        newsc[te] = s[te] + X[te] @ b
    d = ndcg_per_feed(newsc.astype(np.float32), y.astype(np.float32), ov, Mv) - base
    f_ = lambda d, w=None: ("%+.5f+-%.5f" % (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))) if w is None else ("%+.5f+-%.5f" % pmix.wmean_se(d, w))
    print(f"  {col:8s} coefs half0 {np.round(coefs[0], 3)} half1 {np.round(coefs[1], 3)}")
    print(f"           cross-fitted NDCG delta: val {f_(d)} | pub {f_(d, W['public'])} | priv {f_(d, W['private'])} | M<50 {f_(d[~big])} | M>=50 {f_(d[big])}")
