"""winner_model recheck, check 4: does candidate SUB-SAMPLING turn a production feed (top-M of the ranked list) into a
val/test-like candidate set (random M-subset of the pool)?  Label-free, position-free feed-level set-composition
aggregates (mean / sd / min / max of f1..f5, cos_u, cos_hall, cos_h10, cos_het1 + categories per candidate), then a
cross-validated LightGBM AUC of [val + test feeds of a size band] versus each constructed population of the same size.
AUC 0.5 = indistinguishable set composition. Populations (sizes matched to the band's val/test M distribution):
  E_nat   train exploration feeds (v2 posterior > 0.9), natural size in the band          (floor: time drift only)
  P_nat   train production feeds (v2 posterior < 0.1), natural size in the band            (what the model trains on)
  E_long  train exploration feeds with M >= 200, random subset                             (control: heavy-scroller users)
  P_long  train production feeds with M >= 200, random subset (label-free)
  P_long+ same, every positive kept (the train.py --aug_cand rule)
  P_2x    train production feeds with M = 2 x target, random half (what --aug_cand 0.5 does at the low end)
Only columns that do not depend on the candidate set are used. No model input is produced; test rows are used for
their unlabelled covariates only (the same use as the exploration detector v2).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = "<repo>/data/"
COLS = [("flat", "cand_f1"), ("flat", "cand_f2"), ("flat", "cand_f3"), ("flat", "cand_f4"), ("flat", "cand_f5"),
        ("feats", "cos_u"), ("feats", "cos_hall"), ("feats", "cos_h10"), ("feats", "cos_het1")]
rng = np.random.default_rng(20261114)
NMAX = 6000


def meta(sp):
    f = pl.read_parquet(f"{R}flat/{sp}/feeds.parquet")
    return f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def feats(sp, groups):
    """groups: list of row-index arrays (one per constructed feed) -> [n, F] aggregates."""
    rows = np.concatenate(groups)
    gid = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    order = np.argsort(rows, kind="stable")
    n = len(groups)
    m = np.bincount(gid, minlength=n).astype(np.float64)
    F = []
    for d, c in COLS:
        x = np.empty(len(rows), np.float64)
        x[order] = np.nan_to_num(np.asarray(np.load(f"{R}{d}/{sp}/{c}.npy", mmap_mode="r")[rows[order]], np.float64))
        mu = np.bincount(gid, x, n) / m
        sd = np.sqrt(np.maximum(np.bincount(gid, x * x, n) / m - mu * mu, 0))
        mn = np.full(n, np.inf); np.minimum.at(mn, gid, x)
        mxv = np.full(n, -np.inf); np.maximum.at(mxv, gid, x)
        F += [mu, sd, mn, mxv]
    cat = np.empty(len(rows), np.int64)
    cat[order] = np.asarray(np.load(f"{R}flat/{sp}/cand_category_id.npy", mmap_mode="r")[rows[order]], np.int64)
    ncat = np.bincount(np.unique(gid * 64 + cat) // 64, minlength=n)
    F += [ncat / m, ncat.astype(np.float64), m]
    return np.stack(F, 1).astype(np.float32)


def auc(Xa, Xb, drop_m=True):
    X = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    if drop_m:
        X = X[:, :-1]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, num_threads=int(os.environ.get("NT", "12")), verbose=-1)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    a = roc_auc_score(y, p)
    # bootstrap-free SE (Hanley-McNeil)
    n1, n0 = len(Xa), len(Xb)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    return a, se


Mtr, otr = meta("train")
ytr = np.load(f"{R}flat/train/cand_target.npy", mmap_mode="r")
p2 = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom_v2.npz")["p_random"].astype(np.float32), nan=0.5)
prod, expl = p2 < 0.1, p2 > 0.9
print(f"train feeds: production (p2<0.1) {int(prod.sum())}, exploration (p2>0.9) {int(expl.sum())}; "
      f"M>=200: production {int((prod & (Mtr >= 200)).sum())}, exploration {int((expl & (Mtr >= 200)).sum())}", flush=True)

for lo, hi in ((9, 15), (24, 36), (60, 90)):
    tgt = []
    for sp in ("val", "test"):
        M, off = meta(sp)
        ids = np.where((M >= lo) & (M <= hi))[0]
        tgt.append((sp, ids, M, off))
    XV = np.vstack([feats(sp, [np.arange(off[f], off[f] + M[f]) for f in ids]) for sp, ids, M, off in tgt])
    sizes = XV[:, -1].astype(np.int64)
    n = min(NMAX, len(XV))
    print(f"== size band [{lo},{hi}]: val+test feeds {len(XV)} (mean M {sizes.mean():.1f})", flush=True)

    def nat(mask):
        ids = np.where(mask & (Mtr >= lo) & (Mtr <= hi))[0]
        ids = rng.choice(ids, min(n, len(ids)), replace=False)
        return [np.arange(otr[f], otr[f] + Mtr[f]) for f in ids]

    def sub(mask, mmin, keep_pos=False, ratio=None):
        tg = rng.choice(sizes, n)
        out = []
        if ratio is None:
            pool = np.where(mask & (Mtr >= mmin))[0]
            src = rng.choice(pool, n, replace=len(pool) < n)
        else:  # source feed of size ratio x target (+-1)
            src = []
            byM = {}
            for t in np.unique(tg):
                c = np.where(mask & (np.abs(Mtr - ratio * t) <= 1))[0]
                byM[t] = c
            src = np.array([rng.choice(byM[t]) for t in tg])
        for f, t in zip(src, tg):
            o, m = otr[f], Mtr[f]
            if keep_pos:
                y = np.asarray(ytr[o:o + m]) > 0
                pos, neg = np.where(y)[0], np.where(~y)[0]
                k = max(t - len(pos), 0)
                sel = np.r_[pos, rng.choice(neg, min(k, len(neg)), replace=False)]
            else:
                sel = rng.choice(m, t, replace=False)
            out.append(o + np.sort(sel))
        return out

    pops = [("E_nat  exploration, natural size", nat(expl)),
            ("P_nat  production, natural size", nat(prod)),
            ("E_long exploration M>=200, random subset", sub(expl, 200)),
            ("P_long production M>=200, random subset", sub(prod, 200)),
            ("P_long+ production M>=200, positives kept", sub(prod, 200, keep_pos=True)),
            ("P_2x   production M=2x target, random half", sub(prod, 0, ratio=2)),
            ("P_2x+  production M=2x target, positives kept", sub(prod, 0, keep_pos=True, ratio=2))]
    for name, g in pops:
        X = feats("train", g)
        a, se = auc(XV, X)
        print(f"   val+test vs {name:48s} n {len(X):5d}  AUC {a:.4f} +- {se:.4f}", flush=True)
