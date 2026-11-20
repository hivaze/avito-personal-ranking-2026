"""Step 4: the untested cell. Positives in a category new to the user (category absent from the newest 8 history
events, the pass-3 definition: 32% of positives, 37% of the loss).

(a) is the model under-predicting new-category candidates? actual/expected by novelty class, both halves
(b) where in the list does any bias sit (rank bucket x novelty)?
(c) feed-level: does a label-free "this user explores" signal predict the feed's new-category click mass beyond
    the model's own predicted mass? logistic of 1[a positive is new-category] (and the count share) on
    logit(model mass) + explorer signals, fitted on one half, scored on the other.
(d) cross-fitted conditional-logit offsets nov8 x explorer signal (continuous) and NDCG deltas.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table()
y = D["label"].to_numpy().astype(np.float64)
s = D["score"].to_numpy().astype(np.float64)
S = np.load(HERE / "data/setup_final.npz"); E = S["E"]; p = S["p"]
St = np.load(HERE / "data/setup_final_tf.npz"); Et = St["E"]
rows_r, loss_r = S["rows"].astype(np.int64), S["loss"]
R = rows_r.shape[0]
hr = half[fid]
npos = np.bincount(fid, y, len(M))


def col(c):
    return D[c].to_numpy().astype(np.float64)


cni = col("cat_newest_idx")
nov8 = (cni < 0) | (cni >= 8)
first = off


def ae_line(mask, Ex, label):
    out = [f"{label:38s} rows {int(mask.sum()):8d}"]
    for h in (0, 1):
        mh = mask & (hr == h)
        a, e = y[mh].sum(), Ex[mh].sum()
        null = (mh[rows_r]).sum(1)
        out.append(f"{'AB'[h]}: act {int(a):5d} A/E {a / e:.3f} z {(a - e) / max(null.std(), 1e-9):+5.2f}")
    print(" | ".join(out))


print("=== (a) actual / expected by novelty class (global T | per-feed T) ===")
cls = np.select([cni < 0, cni == 0, cni < 8, cni < 32], ["never", "newest", "h1-7", "h8-31"], "h32+")
for c in ["newest", "h1-7", "h8-31", "h32+", "never"]:
    ae_line(cls == c, E, f"global  {c}"); ae_line(cls == c, Et, f"perfeed {c}")
ae_line(nov8, E, "global  nov8 (new-to-user, pass-3 def)"); ae_line(nov8, Et, "perfeed nov8")
L = S["L"]
mr = nov8[rows_r]
print(f"loss share of nov8 positives: actual {L[nov8].sum() / L.sum():.4f}  re-draw {((loss_r * mr).sum(1) / loss_r.sum(1)).mean():.4f} "
      f"± {((loss_r * mr).sum(1) / loss_r.sum(1)).std():.4f}; positive share actual {y[nov8].sum() / y.sum():.4f} "
      f"expected {E[nov8].sum() / E.sum():.4f}")

print("\n=== (b) rank bucket x novelty, global T ===")
rk = col("rank")
rb = np.select([rk == 1, rk <= 3, rk <= 10, rk <= 30, rk <= 100], ["1", "2-3", "4-10", "11-30", "31-100"], "101+")
for b in ["1", "2-3", "4-10", "11-30", "31-100", "101+"]:
    ae_line((rb == b) & nov8, E, f"rank {b:6s} nov8=1"); ae_line((rb == b) & ~nov8, E, f"rank {b:6s} nov8=0")
hq = col("hmean5") >= np.nanquantile(col("hmean5"), 0.9)
for b in ["1", "2-3", "4-10", "11-30", "31-100", "101+"]:
    ae_line((rb == b) & hq, E, f"rank {b:6s} hmean5 top decile")

print("\n=== (c) feed-level exploration: does an explorer signal predict new-category clicks beyond the model? ===")
mass = np.bincount(fid, p * nov8, len(M))                       # model's predicted share of clicks on nov8 items
has_nov = np.bincount(fid, nov8, len(M)) > 0
yf = (np.bincount(fid, y * nov8, len(M)) > 0).astype(float)     # feed has a new-category positive
share = np.bincount(fid, y * nov8, len(M)) / np.maximum(npos, 1)
past = np.where(col("past_npos")[first] > 0, col("past_nnew")[first] / np.maximum(col("past_npos")[first], 1), np.nan)
sig = {k: np.nan_to_num(col(k)[first], nan=np.nanmean(col(k)[first])) for k in
       ["h_nov32", "h_nov32r", "h_ent", "h_brk10", "n_hist_cats32", "h_topsh", "H", "n_feed_cats"]}
sig["past_novel_rate"] = np.nan_to_num(past, nan=np.nanmean(past)); sig["past_missing"] = np.isnan(past).astype(float)
sig["logM"] = np.log(M); sig["guest"] = col("guest")[first]
keep = has_nov & (npos > 0)
lm = np.log(np.clip(mass, 1e-6, 1 - 1e-6) / (1 - np.clip(mass, 1e-6, 1 - 1e-6)))
# expected P(any of npos positives is nov8) under the model, approx 1 - (1 - mass)^npos
pexp = 1 - (1 - mass) ** npos
print(f"feeds with >=1 new-category candidate: {int(keep.sum())} ({keep.mean():.3f}); with a new-category positive "
      f"{yf[keep].mean():.4f}; model-expected {pexp[keep].mean():.4f}")
for h in (0, 1):
    k = keep & (half == h)
    print(f"  half {'AB'[h]}: actual {yf[k].mean():.4f} expected {pexp[k].mean():.4f}  AUC(model mass) {roc_auc_score(yf[k], mass[k]):.4f}")


def logit_fit(X, t, w=None):
    def fg(b):
        z = X @ b
        pr = 1 / (1 + np.exp(-z))
        f = -(t * np.log(pr + 1e-12) + (1 - t) * np.log(1 - pr + 1e-12)).sum() + 1e-3 * (b[1:] @ b[1:]) * len(t)
        g = X.T @ (pr - t) + np.r_[0, 2e-3 * b[1:] * len(t)]
        return f, g
    return minimize(fg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def design(names):
    cols_ = [np.ones(len(M)), np.log(pexp / (1 - pexp + 1e-9) + 1e-9)]
    for n_ in names:
        v = sig[n_]
        cols_.append((v - v[keep].mean()) / (v[keep].std() + 1e-12))
    return np.column_stack(cols_)


for names in [[], ["h_nov32"], ["h_nov32r"], ["h_ent"], ["h_brk10"], ["n_hist_cats32"], ["past_novel_rate", "past_missing"],
              ["h_nov32", "h_nov32r", "h_ent", "h_brk10", "n_hist_cats32", "h_topsh", "H", "past_novel_rate", "past_missing",
               "logM", "guest", "n_feed_cats"]]:
    X = design(names)
    ll, aucs, coefs = [], [], []
    for h in (0, 1):
        tr, te = keep & (half == h), keep & (half == 1 - h)
        b = logit_fit(X[tr], yf[tr])
        pr = 1 / (1 + np.exp(-(X[te] @ b)))
        b0 = logit_fit(X[tr][:, :2], yf[tr]); pr0 = 1 / (1 + np.exp(-(X[te][:, :2] @ b0)))
        ll.append(-(yf[te] * np.log(pr) + (1 - yf[te]) * np.log(1 - pr)).mean() + (yf[te] * np.log(pr0) + (1 - yf[te]) * np.log(1 - pr0)).mean())
        aucs.append(roc_auc_score(yf[te], pr) - roc_auc_score(yf[te], pr0))
        coefs.append(b)
    print(f"  + {','.join(names) or '(model mass only)':60s} out-of-half dLogLoss {np.mean(ll):+.5f} dAUC {np.mean(aucs):+.4f}  "
          f"slope on model {coefs[0][1]:.3f}/{coefs[1][1]:.3f}  coefs A {np.round(coefs[0][2:], 3).tolist()}")

print("\n=== (d) cross-fitted offsets nov8 x explorer signal (global T base), NDCG ===")
nd0 = ndcg_per_feed(s, y, off, M)
TA = fit_T(s, y, off, M, half == 0); TB = fit_T(s, y, off, M, half == 1)
rows_B = hr == 1
for names in [["h_nov32"], ["h_nov32r"], ["h_ent"], ["h_brk10"], ["n_hist_cats32"], ["past_novel_rate", "past_missing"],
              ["h_nov32", "h_ent", "h_brk10", "n_hist_cats32", "past_novel_rate", "past_missing"]]:
    cols_ = [nov8.astype(float)]
    for n_ in names:
        v = sig[n_]; v = (v - v.mean()) / (v.std() + 1e-12)
        cols_.append(nov8 * v[fid])
    X = np.column_stack(cols_)
    bA = clogit_fit(s / TA, X, y, off, M, half == 0, 1e-4); bB = clogit_fit(s / TB, X, y, off, M, half == 1, 1e-4)
    new = np.where(rows_B, s / TA + X @ bA, s / TB + X @ bB)
    report_delta(ndcg_per_feed(new, y, off, M), nd0, M, half, f"nov8 x [{','.join(names)}]"[:40])
    print(f"      b(A) {np.round(bA, 3).tolist()}  b(B) {np.round(bB, 3).tolist()}")
