"""syn ceiling_read: amendment A4 of DECLARATION.txt (information only). Best LINEAR use of the neighbour-imputed columns inside
the cold rows of a feed: conditional logit over cold rows, in-sample (upper bound) and nested by user (honest).
CPU only, read-only on the project. usage: python ceiling_read.py NPZ TAG"""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>")
HERE = Path(__file__).resolve().parent
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def feed_z(x, fid, nF):
    cnt = np.maximum(np.bincount(fid, minlength=nF), 1)
    mu = np.bincount(fid, weights=x, minlength=nF) / cnt
    d = x - mu[fid]
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
    return d / np.maximum(sd[fid], 1e-9)


def main():
    npz, tag = sys.argv[1], sys.argv[2]
    Z = np.load(npz)
    selF, qrows, nv, s1, s2, dg = Z["selF"], Z["qrows"], Z["nv"], Z["s1"], Z["s2"], Z["dg"]
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    coff = f["cand_off"].to_numpy().astype(np.int64)
    user = f["user_id"].to_numpy().astype(np.int64)
    nF = len(M)
    fid = np.repeat(np.arange(nF), M)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    wgt = np.where(nv >= 0.90, nv, 0.0)
    ws = np.maximum(wgt.sum(1), 1e-9)
    imp = [
        (wgt * s1).sum(1) / ws, (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws, np.where(wgt > 0, s1, 0).max(1),
        (wgt * s2 / np.maximum(dg, 1)).sum(1) / ws, ((wgt > 0) & (s1 > 0)).sum(1).astype(np.float64),
        (wgt * np.log1p(dg)).sum(1) / ws, nv[:, 0],
    ]
    logt = [True, True, True, True, True, False, False]
    zf = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)  # noqa: E731
    pools = {
        "F": zf(0.5 * zf(np.load(ROOT / "preds/ens_FD16P20kd_ft_val.npy")) + 0.5 * zf(np.load(ROOT / "preds/ens_FD16P20kd_to_val.npy"))),
        "W": zf(0.5 * zf(np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * zf(np.load(ROOT / "preds/ens_W16kd_to_val.npy"))),
    }
    Ms = M[selF]
    offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
    rsel = np.concatenate([np.arange(coff[x], coff[x] + M[x]) for x in selF])
    pos_in_sel = np.full(len(y), -1, np.int64)
    pos_in_sel[rsel] = np.arange(len(rsel))
    q = pos_in_sel[qrows]
    nS = len(selF)
    fS = np.repeat(np.arange(nS), Ms)
    qf = fS[q]
    ys = y[rsel]
    yq = ys[q]
    ncold = np.bincount(qf, minlength=nS)
    cpos = np.bincount(qf, weights=yq, minlength=nS)
    has_pair = (cpos > 0) & (ncold - cpos > 0)
    qstart = np.concatenate([[0], np.cumsum(ncold)[:-1]]).astype(np.int64)
    zc = lambda v: feed_z(np.asarray(v, np.float64), qf, nS)  # noqa: E731

    def crank(v):
        r = np.zeros(len(v))
        for ff in range(nS):
            k = ncold[ff]
            if k:
                a = qstart[ff]
                r[a:a + k] = (rankdata(v[a:a + k]) - 0.5) / k - 0.5
        return r

    def auc_cold(v):
        out = np.full(nS, np.nan)
        for ff in np.nonzero(has_pair)[0]:
            a, k = qstart[ff], ncold[ff]
            r = rankdata(v[a:a + k])
            yy = yq[a:a + k] > 0
            n1 = yy.sum()
            out[ff] = (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * (k - n1))
        return out

    cols = []
    for v, lt in zip(imp, logt):
        v = np.asarray(v, np.float64)
        if lt:
            med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
            cols.append(zc(np.log1p(v / med)))
        else:
            cols.append(zc(v))
        cols.append(crank(v) * np.sqrt(12.0))
    ind = (imp[0] > 0).astype(np.float64)
    cols.append(ind - (np.bincount(qf, weights=ind, minlength=nS) / np.maximum(ncold, 1))[qf])
    X = np.stack(cols, 1)                      # 15 extra features, all centred within the feed's cold rows
    P = X.shape[1]
    log(f"[{tag}] feeds {nS}, cold rows {len(q)}, pair feeds {int(has_pair.sum())}, extra features {P}")

    def fit(pz, feeds_mask, lam=1.0):
        rows = feeds_mask[qf]
        Xr, pr, yr, gr = X[rows], pz[rows], yq[rows], qf[rows]
        # compact group ids
        ug, gi = np.unique(gr, return_inverse=True)
        ng = len(ug)
        npos = np.bincount(gi, weights=yr, minlength=ng)

        def obj(th):
            s = th[0] * pr + Xr @ th[1:]
            mx = np.full(ng, -np.inf)
            np.maximum.at(mx, gi, s)
            e = np.exp(s - mx[gi])
            den = np.bincount(gi, weights=e, minlength=ng)
            loss = -(yr * s).sum() + (npos * (mx + np.log(den))).sum() + 0.5 * lam * (th[1:] ** 2).sum()
            pgi = e / den[gi]
            r = -yr + npos[gi] * pgi
            g = np.concatenate([[r @ pr], Xr.T @ r + lam * th[1:]])
            return loss / ng, g / ng

        th0 = np.zeros(P + 1)
        th0[0] = 1.0
        o = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        return o.x

    def reorder_scores(ps, key):
        pc = ps[q]
        o1 = np.lexsort((-key, qf))
        o2 = np.lexsort((-pc, qf))
        out = ps.copy()
        out[q[o1]] = pc[o2]
        return out

    us = user[selF]
    grp = np.where(us == 0, -1 - np.arange(nS), us)
    folds = list(GroupKFold(n_splits=5).split(np.arange(nS), groups=grp))
    res = {}
    for pn, p in pools.items():
        ps = p[rsel].copy()
        base = ndcg_per_feed(ps, ys, offs, Ms)
        pz = zc(ps[q])
        a_pool = auc_cold(pz)
        # (a) in-sample
        th = fit(pz, has_pair)
        sc = th[0] * pz + X @ th[1:]
        a_in = auc_cold(sc)
        d_in = ndcg_per_feed(reorder_scores(ps, sc), ys, offs, Ms) - base
        # control of the fit path: pool alone must give delta 0
        d0 = ndcg_per_feed(reorder_scores(ps, pz), ys, offs, Ms) - base
        assert np.abs(d0).max() < 1e-12
        # (b) nested
        sc_n = np.zeros(len(q))
        for trf, vaf in folds:
            mtr = np.zeros(nS, bool)
            mtr[trf] = True
            t_ = fit(pz, mtr & has_pair)
            mva = np.zeros(nS, bool)
            mva[vaf] = True
            rows = mva[qf]
            sc_n[rows] = t_[0] * pz[rows] + X[rows] @ t_[1:]
        a_n = auc_cold(sc_n)
        d_n = ndcg_per_feed(reorder_scores(ps, sc_n), ys, offs, Ms) - base
        se = lambda d: d.std() / np.sqrt(nS)  # noqa: E731
        dA_in = a_in - a_pool
        dA_n = a_n - a_pool
        ok = ~np.isnan(a_pool)
        log(f"[A4 pool {pn}] AUC among cold rows: pool {np.nanmean(a_pool):.4f} | in-sample fit {np.nanmean(a_in):.4f} (delta {np.nanmean(dA_in):+.4f} ± {np.nanstd(dA_in) / np.sqrt(ok.sum()):.4f}) | nested {np.nanmean(a_n):.4f} (delta {np.nanmean(dA_n):+.4f} ± {np.nanstd(dA_n) / np.sqrt(ok.sum()):.4f})")
        log(f"[A4 pool {pn}] slot-preserving NDCG delta, all {nS} feeds: in-sample fit {d_in.mean():+.5f} ± {se(d_in):.5f} | nested {d_n.mean():+.5f} ± {se(d_n):.5f} (z {d_n.mean() / se(d_n):+.2f}) | feeds with a cold positive: in-sample {d_in[cpos > 0].mean():+.5f}, nested {d_n[cpos > 0].mean():+.5f}")
        log(f"[A4 pool {pn}] in-sample coefficients: pool {th[0]:+.3f}; extras " + " ".join(f"{t:+.3f}" for t in th[1:]))
        res[pn] = dict(auc_pool=float(np.nanmean(a_pool)), auc_in=float(np.nanmean(a_in)), auc_nested=float(np.nanmean(a_n)),
                       ndcg_in=float(d_in.mean()), ndcg_in_se=float(se(d_in)), ndcg_nested=float(d_n.mean()), ndcg_nested_se=float(se(d_n)))
    (HERE / f"ceiling_{tag}.json").write_text(json.dumps(res, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
