"""syn gate_read: pre-declared reads A1-A3 of nn_x/w12/lit/syn/DECLARATION.txt on a neighbour file of cold val rows.
CPU only, read-only on the project, writes only next to this script.
usage: python gate_read.py NPZ TAG      (NPZ has selF, qrows, nv, s1, s2, dg as written by int_gnn s5_coldsignal.py)"""
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
    luo = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64)
    cemb = np.load(FLAT / "val/cand_emb_idx.npy").astype(np.int64)
    inF = np.zeros(nF, bool)
    inF[selF] = True
    chk = np.nonzero(inF[fid] & (luo == 0) & (cemb >= 0))[0]
    assert len(chk) == len(qrows) and (chk == qrows).all(), "qrows are not the cold rows of selF"

    # columns exactly as in int_gnn s5_coldsignal.py
    wgt = np.where(nv >= 0.90, nv, 0.0)
    ws = np.maximum(wgt.sum(1), 1e-9)
    imp = {
        "x1 imputed s1 (cos-weighted mean)": (wgt * s1).sum(1) / ws,
        "x2 imputed s2/deg^.5": (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws,
        "imputed s1 (max)": np.where(wgt > 0, s1, 0).max(1),
        "imputed s2/deg": (wgt * s2 / np.maximum(dg, 1)).sum(1) / ws,
        "number of linked neighbours": ((wgt > 0) & (s1 > 0)).sum(1).astype(np.float64),
        "imputed log-degree": (wgt * np.log1p(dg)).sum(1) / ws,
        "density (top-1 cos)": nv[:, 0],
    }
    log(f"[{tag}] feeds {len(selF)}, cold rows {len(qrows)}, cold positives {int(y[qrows].sum())}; coverage x1>0 {np.mean(imp['x1 imputed s1 (cos-weighted mean)'] > 0):.4f}")

    zf = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)  # noqa: E731
    pools = {
        "F": zf(0.5 * zf(np.load(ROOT / "preds/ens_FD16P20kd_ft_val.npy")) + 0.5 * zf(np.load(ROOT / "preds/ens_FD16P20kd_to_val.npy"))),
        "W": zf(0.5 * zf(np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * zf(np.load(ROOT / "preds/ens_W16kd_to_val.npy"))),
    }

    # sample-local indexing: rows of the sampled feeds, in feed order
    Ms = M[selF]
    offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
    rsel = np.concatenate([np.arange(coff[x], coff[x] + M[x]) for x in selF])
    pos_in_sel = np.full(len(y), -1, np.int64)
    pos_in_sel[rsel] = np.arange(len(rsel))
    q = pos_in_sel[qrows]                      # cold rows, sample-local row index
    nS = len(selF)
    fS = np.repeat(np.arange(nS), Ms)
    qf = fS[q]                                 # sample-local feed of every cold row (non-decreasing)
    assert (np.diff(qf) >= 0).all()
    ys = y[rsel]
    yq = ys[q]
    ncold = np.bincount(qf, minlength=nS)
    cpos = np.bincount(qf, weights=yq, minlength=nS)
    has_pair = (cpos > 0) & (ncold - cpos > 0)
    log(f"feeds with a cold positive {np.mean(cpos > 0):.4f}; with a cold positive and a cold negative {int(has_pair.sum())}")
    for pn, p in pools.items():
        b = ndcg_per_feed(p[rsel], ys, offs, Ms)
        log(f"pool {pn}: NDCG on the sampled feeds {b.mean():.5f} (all val {ndcg_per_feed(p, y, coff, M).mean():.5f}); feeds with a cold positive {b[cpos > 0].mean():.5f}")

    # ---------- helpers on cold rows ----------
    qstart = np.concatenate([[0], np.cumsum(ncold)[:-1]]).astype(np.int64)

    def crank(v):                              # centred percentile rank among the cold rows of the feed
        r = np.zeros(len(v))
        for ff in range(nS):
            k = ncold[ff]
            if k:
                a = qstart[ff]
                r[a:a + k] = (rankdata(v[a:a + k]) - 0.5) / k - 0.5
        return r

    def zc(v):                                 # z among the cold rows of the feed
        return feed_z(np.asarray(v, np.float64), qf, nS)

    def auc_cold(v):                           # per-feed AUC among cold rows, ties 0.5
        out = np.full(nS, np.nan)
        for ff in np.nonzero(has_pair)[0]:
            a, k = qstart[ff], ncold[ff]
            r = rankdata(v[a:a + k])
            yy = yq[a:a + k] > 0
            n1 = yy.sum()
            n0 = k - n1
            out[ff] = (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return out

    def ms(au):
        ok = ~np.isnan(au)
        return f"{np.nanmean(au):.4f}±{np.nanstd(au) / np.sqrt(ok.sum()):.4f}"

    def tr(n, v):                              # transform for the NDCG reads (as int_gnn)
        v = np.asarray(v, np.float64)
        if n.startswith("x") or n.startswith("imputed s") or n.startswith("number"):
            med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
            v = np.log1p(v / med)
        return zc(v)

    ctrl_feats = [np.nan_to_num(np.asarray(np.load(FEATS / "val/seq_max_new8.npy"), np.float64), nan=-9.0)[qrows],
                  np.asarray(np.load(FEATS / "val/pre2_fe_noid.npy"), np.float64)[qrows]]
    rc = [crank(c) for c in ctrl_feats]
    res = {"tag": tag, "n_feeds": int(nS), "n_cold_rows": int(len(q)), "n_pair_feeds": int(has_pair.sum()), "A1": {}, "A2": {}, "A3": {}}

    # ---------- A1 ----------
    for pn, p in pools.items():
        pq = p[rsel][q]
        rp = crank(pq)
        log(f"[A1 pool {pn}] raw AUC among cold rows: pool {ms(auc_cold(pq))}")
        for n, v in imp.items():
            rx = crank(v)
            raw = auc_cold(v)
            A = rp[:, None]
            b1, *_ = np.linalg.lstsq(A, rx, rcond=None)
            a1 = auc_cold(rx - A @ b1)
            A3_ = np.stack([rp] + rc, 1)
            b3, *_ = np.linalg.lstsq(A3_, rx, rcond=None)
            a3 = auc_cold(rx - A3_ @ b3)
            res["A1"][f"{pn}|{n}"] = dict(raw=float(np.nanmean(raw)), cond_pool=float(np.nanmean(a1)),
                                          cond_pool_se=float(np.nanstd(a1) / np.sqrt((~np.isnan(a1)).sum())),
                                          cond_pool_2content=float(np.nanmean(a3)), beta_pool=float(b1[0]))
            log(f"[A1 pool {pn}] {n}: raw {ms(raw)} | (i) after rank(pool) {ms(a1)} (beta {b1[0]:+.2f}) | (ii) after rank(pool)+2 content {ms(a3)}")

    # ---------- A2 / A3 ----------
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    us = user[selF]
    grp = np.where(us == 0, -1 - np.arange(nS), us)
    folds = list(GroupKFold(n_splits=5).split(np.arange(nS), groups=grp))
    try:
        sys.path.insert(0, str(ROOT / "eda/errors/segments"))
        import pmix
        W = pmix.weights()
        wpub, wpri = W["public"][selF], W["private"][selF]
    except Exception as e:  # noqa: BLE001
        log("pmix weights unavailable:", repr(e))
        wpub = wpri = None

    def wm(d, w):
        m = np.average(d, weights=w)
        neff = w.sum() ** 2 / (w ** 2).sum()
        return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)

    def reorder_scores(ps, pzc, x, w):
        """slot-preserving: cold rows re-ordered by pzc + w x, receiving the sorted pool scores of the feed's cold rows."""
        key = pzc + w * x
        pc = ps[q]
        o1 = np.lexsort((-key, qf))            # by feed, key descending
        o2 = np.lexsort((-pc, qf))             # by feed, pool score descending
        out = ps.copy()
        out[q[o1]] = pc[o2]
        return out

    def nested(ps, base, pzc, x):
        per = {w: ndcg_per_feed(reorder_scores(ps, pzc, x, w), ys, offs, Ms) for w in grid}
        delta = np.zeros(nS)
        wsel = []
        for trf, vaf in folds:
            best = max(grid, key=lambda w: per[w][trf].mean())
            wsel.append(best)
            delta[vaf] = per[best][vaf] - base[vaf]
        return delta, wsel, {w: float((per[w] - base).mean()) for w in grid[1:]}

    for pn, p in pools.items():
        ps = p[rsel].copy()
        base = ndcg_per_feed(ps, ys, offs, Ms)
        pzc = zc(ps[q])
        chk0 = ndcg_per_feed(reorder_scores(ps, pzc, np.zeros(len(q)), 0.0), ys, offs, Ms)
        assert np.abs(chk0 - base).max() < 1e-12, "w = 0 must reproduce the pool"
        for n, v in imp.items():
            x = tr(n, v)
            delta, wsel, ins = nested(ps, base, pzc, x)
            se = delta.std() / np.sqrt(nS)
            assert np.abs(delta[cpos == 0]).max() == 0.0, "feeds without a cold positive must not move"
            extra = ""
            if wpub is not None:
                mp, sp = wm(delta, wpub)
                mr, sr = wm(delta, wpri)
                extra = f" | pmix public {mp:+.5f}±{sp:.5f} private {mr:+.5f}±{sr:.5f}"
            res["A2"][f"{pn}|{n}"] = dict(gain=float(delta.mean()), se=float(se), z=float(delta.mean() / max(se, 1e-12)), w=wsel,
                                          insample=ins, coldpos=float(delta[cpos > 0].mean()))
            log(f"[A2 pool {pn}] {n}: nested {delta.mean():+.5f} ± {se:.5f} (z {delta.mean() / max(se, 1e-12):+.2f}) w {wsel} | feeds with a cold positive {delta[cpos > 0].mean():+.5f}{extra} | in-sample "
                + " ".join(f"{w}:{g:+.5f}" for w, g in ins.items()))
            if n.startswith("x1"):
                np.save(HERE / f"delta_{tag}_{pn}_x1.npy", delta)
        # A3 placebo on x1 and x2
        for n in list(imp)[:2]:
            x = tr(n, imp[n])
            gains = []
            for sd in range(20):
                rng = np.random.default_rng(sd)
                # permute x among the cold rows within each feed: random key sorted inside feed blocks
                o = np.lexsort((rng.random(len(q)), qf))
                xp = np.empty_like(x)
                xp[np.arange(len(q))] = x[o]
                d_, _, _ = nested(ps, base, pzc, xp)
                gains.append(float(d_.mean()))
            res["A3"][f"{pn}|{n}"] = gains
            log(f"[A3 pool {pn}] placebo {n}: mean {np.mean(gains):+.5f} sd {np.std(gains):.5f} min {np.min(gains):+.5f} max {np.max(gains):+.5f} (20 draws)")
    (HERE / f"gate_{tag}.json").write_text(json.dumps(res, indent=1))
    log("DONE")


if __name__ == "__main__":
    main()
