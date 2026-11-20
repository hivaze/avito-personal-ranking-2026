"""web_shift probe s6: nested output-offset read (PREDECLARED.txt addendum 2). score' = z(pool) + c * delta, c fitted on 4 user-grouped folds."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feat_new")
from metrics import ndcg_per_feed
import hz_common as H
ROOT = Path("<repo>"); HERE = ROOT / "reports/TRICKS_1115/web_shift_probe"
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); n = len(M); fid = np.repeat(np.arange(n), M); off = f["cand_off"].to_numpy().astype(np.int64)
uid = f["user_id"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy().astype(np.int64)
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
cnt = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy"); cold = (cnt <= 0).astype(np.float64)
dv = np.load(HERE / "cf/delta_val.npy").astype(np.float64)
npos = np.bincount(fid, y, n)
grp = np.where(uid == 0, -(np.arange(n) + 1), uid)
ug = np.unique(grp); perm = np.random.default_rng(0).permutation(len(ug)); fold_of = dict(zip(ug.tolist(), (perm % 5).tolist()))
fold = np.array([fold_of[g] for g in grp.tolist()])
day = H.day_of(ts); big = [d for d in np.unique(day) if (day == d).sum() > 1000]
def z(s):
    s = s.astype(np.float64); mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
def fit(X, fmask):
    rows = fmask[fid] & (npos[fid] > 0)
    w = np.where(rows, y / np.maximum(npos[fid], 1), 0.0)
    b = np.zeros(X.shape[1]); b[0] = 1.0
    st = np.r_[0, np.cumsum(M)[:-1]]
    for it in range(60):
        l = X @ b; l -= np.maximum.reduceat(l, st)[fid]
        e = np.exp(l); p = e / np.bincount(fid, e, n)[fid]; p = np.where(rows, p, 0.0)
        g = X.T @ (w - p); Xp = X * p[:, None]
        mu = np.stack([np.bincount(fid, Xp[:, j], n) for j in range(X.shape[1])], 1)
        Hm = X.T @ Xp - mu.T @ mu
        step = np.linalg.solve(Hm, g); b += step
        if np.abs(step).max() < 1e-8:
            break
    R = X * (w - p)[:, None]; G = np.stack([np.bincount(fid, R[:, j], n) for j in range(X.shape[1])], 1)
    Hi = np.linalg.inv(Hm); V = Hi @ (G.T @ G) @ Hi
    return b, np.sqrt(np.diag(V))
def rd(x):
    se = x.std(ddof=1) / np.sqrt(len(x)); return f"{x.mean():+.5f} +- {se:.5f} (z {x.mean()/se:+.2f}, n {len(x)})"
allm = np.ones(n, bool)
for tag in sys.argv[1:]:
    zt = z(np.load(ROOT / "preds" / f"ens_{tag}_to_val.npy")); zf = z(np.load(ROOT / "preds" / f"ens_{tag}_ft_val.npy"))
    for name, s in (("final", z(0.5 * zf + 0.5 * zt)), ("train-only", zt), ("fine-tuned", zf)):
        base = ndcg_per_feed(s, y, off, M); out = s.copy(); cs = []
        for k in range(5):
            b, _ = fit(np.stack([s, cold, dv], 1), fold != k)
            c = b[2] / b[0]; cs.append(c)
            te = (fold == k)[fid]; out[te] = s[te] + c * dv[te]
        d = ndcg_per_feed(out, y, off, M) - base
        print(f"\n{tag} {name}: base {base.mean():.5f} | nested offset delta {rd(d)} | c per fold {np.round(cs, 3).tolist()}")
        print("   M<50  ", rd(d[M < 50]), "| M>=50 ", rd(d[M >= 50]))
        for dd in big:
            print(f"   day {dd}", rd(d[day == dd]))
        if name == "final":
            b, se = fit(np.stack([s, cold, dv], 1), allm)
            print(f"   in-sample g {b[2]:+.4f} +- {se[2]:.4f} (z {b[2]/se[2]:+.2f}); c = g/b1 = {b[2]/b[0]:.4f}")
            for dd in big:
                b, se = fit(np.stack([s, cold, dv], 1), day == dd)
                print(f"   day {dd}: g {b[2]:+.4f} +- {se[2]:.4f}")
            X = np.stack([s, cold, ((dv > 0.05) & (dv <= 0.3)).astype(float), ((dv > 0.3) & (dv <= 0.8)).astype(float), (dv > 0.8).astype(float)], 1)
            b, se = fit(X, allm)
            for j, lab in ((2, "delta 0.05-0.3"), (3, "delta 0.3-0.8"), (4, "delta > 0.8")):
                m_ = X[:, j] > 0
                print(f"   dose bin {lab}: coefficient {b[j]:+.4f} +- {se[j]:.4f} | rows {m_.mean():.3f} | mean delta {dv[m_].mean():.2f} | linear model predicts {0:+.0f}".replace("predicts +0", f"predicts {dv[m_].mean():.2f} x g"))
