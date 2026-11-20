# First pass (2026-11-11, morning), on the train-only network ensemble of that hour (nn_final3, 0.38662 val on
# leak-free feeds). The cheap gate of the pass: for each candidate column it reports the within-feed pairwise AUC
# (all pairs, covered pairs, same-category pairs), the t-statistic of the column in a pairwise logit next to the
# base score (feed-clustered SE), and a nested blend z(base) + w * z(column) with w chosen on 4/5 of the feeds by user.
# Run as (argparse; the exact calls are not on record): python 02_screen.py <col,col,...> [--base nn_final3_trainonly]
# It wrote lab/eda/errors/synthesis/logs/02_screen_*.log; two of them are ../results/pass1_feature_screen.txt.
# Result: the best of 11 columns, tp_cosC_h, read +0.00093 +- 0.00038 on the train-only base and -0.00003 on the
# fine-tuned base. It imports helpers from lab/eda/errors/synthesis/qt.py and lab/src/ (not copied).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/eda/errors/synthesis/02_screen.py
"""Cheap gates for candidate features on leak-free val feeds, against z(nn_final3_trainonly) (no val labels in the base).

For each feature x (NaN = unknown -> within-feed z set to 0):
  auc_all  : within-feed pairwise AUC over all (positive, negative) pairs (ties 0.5)
  auc_cov  : same, restricted to pairs where both candidates have x defined
  auc_samecat : pairs inside one category (both defined)
  t_resid  : t-stat of beta_x in a no-intercept pairwise logit  1[pos first] ~ b0*dz_base + b1*dz_x (feed-clustered sandwich SE)
  nested   : z_base + w*z_x, w in {0, .05, .1, .2, .3, .5} chosen on 4/5 of the feeds, applied to the other 1/5
             (5-fold GroupKFold by user, 2 seeds) -> paired per-feed delta vs w = 0 with SE.
"""
import sys, argparse
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from common import *
from stack_val import feed_rank_z
from qt import leak_mask, path_of
from sklearn.model_selection import GroupKFold

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn


ap = argparse.ArgumentParser()
ap.add_argument("feats")
ap.add_argument("--base", default="nn_final3_trainonly")
ap.add_argument("--sign", default="", help="comma list of +1/-1 per feature (default +1)")
args = ap.parse_args()
names = args.feats.split(",")
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.int8)
cat = np.load(FLAT / "val/cand_category_id.npy")
fidx = np.load(FLAT / "val/cand_fidx.npy")
kf = ~leak_mask()
P = Path("<repo>/preds")
zb = feed_rank_z(off, M, np.load(pp(f"{args.base}.npy" if args.base.endswith("oof") else f"{args.base}_val.npy")).astype(np.float32))[:, 1].astype(np.float64)
users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)

# all within-feed (pos, neg) pairs of leak-free feeds
pi, ni, pf = [], [], []
for g in np.nonzero(kf)[0]:
    s, m = off[g], M[g]
    yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) == 0 or len(q) == 0:
        continue
    pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p))); pf.append(np.full(len(p) * len(q), g))
pi = np.concatenate(pi); ni = np.concatenate(ni); pf = np.concatenate(pf)
same = cat[pi] == cat[ni]
print(f"pairs {len(pi):,}  same-category {same.mean():.3f}")


def auc(d):
    return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")


def cluster_logit(X, cl):
    """no-intercept logit on symmetric pairs: all labels 1 with X = (pos - neg); fit via Newton on y=1 & y=0 mirrored."""
    Xs = np.vstack([X, -X]); ys = np.r_[np.ones(len(X)), np.zeros(len(X))]; cls = np.r_[cl, cl]
    b = np.zeros(X.shape[1])
    for _ in range(25):
        pr = 1 / (1 + np.exp(-Xs @ b)); W = pr * (1 - pr)
        H = (Xs * W[:, None]).T @ Xs; g = Xs.T @ (ys - pr)
        b += np.linalg.solve(H, g)
    pr = 1 / (1 + np.exp(-Xs @ b)); G = Xs * (ys - pr)[:, None]
    o = np.argsort(cls, kind="stable"); cs = cls[o]; bnd = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
    Gc = np.add.reduceat(G[o], bnd, axis=0)
    Hinv = np.linalg.inv((Xs * (pr * (1 - pr))[:, None]).T @ Xs)
    V = Hinv @ (Gc.T @ Gc) @ Hinv
    return b, np.sqrt(np.diag(V))


signs = [float(s) for s in args.sign.split(",")] if args.sign else [1.0] * len(names)
base_per = ndcg_per_feed(zb, y.astype(np.float32), off, M)
for n, sg in zip(names, signs):
    x = np.load(path_of(n, "val")).astype(np.float64) * sg
    dfn = np.isfinite(x)
    xf = np.where(dfn, x, np.nan)
    # within-feed z with NaN -> 0
    zx = np.zeros(len(x))
    for g in range(len(M)):
        s, m = off[g], M[g]
        v = xf[s:s + m]; ok = np.isfinite(v)
        if ok.sum() >= 2:
            sd = v[ok].std()
            if sd > 0:
                zx[s:s + m][ok] = (v[ok] - v[ok].mean()) / sd
    d = zx[pi] - zx[ni]
    both = dfn[pi] & dfn[ni]
    a_all, a_cov, a_sc = auc(d), auc(d[both]), auc(d[both & same])
    b, se = cluster_logit(np.c_[zb[pi] - zb[ni], d], pf)
    # nested linear blend
    W = [0, .05, .1, .2, .3, .5]
    pers = {w: ndcg_per_feed(zb + w * zx, y.astype(np.float32), off, M) for w in W}
    deltas = []
    for seed in range(2):
        out = np.zeros(len(M))
        for tr, va in GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
            trk = tr[kf[tr]]
            wbest = max(W, key=lambda w: pers[w][trk].mean())
            out[va] = pers[wbest][va] - base_per[va]
        deltas.append(out)
    dn = np.mean(deltas, 0)[kf]
    wfull = max(W, key=lambda w: pers[w][kf].mean())
    print(f"{n:22s} cov {dfn.mean():.3f} | AUC all {a_all:.4f} cov {a_cov:.4f} samecat {a_sc:.4f} | resid b {b[1]:+.4f} t {b[1] / se[1]:+.1f} | "
          f"nested {dn.mean():+.5f} ± {dn.std() / np.sqrt(len(dn)):.5f} (full-val best w {wfull}: {pers[wfull][kf].mean() - base_per[kf].mean():+.5f})",
          flush=True)
