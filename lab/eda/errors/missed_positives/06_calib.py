"""Calibration screen: for every feature, decile bins (global quantiles over leak-free candidates; floor values get their own
bin automatically through unique edges) and the ratio actual / expected positives, where expected = the v5 blend's
within-feed softmax probability (scale fitted by conditional logit) times the feed's number of positives.
Cluster-robust z per bin: z_k = sum_f (A_fk - E_fk) / sqrt(sum_f (A_fk - E_fk)^2).
A bin with |z| >= 4 is a region where the best model is systematically wrong -> an exploitable correction.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *

D = Path("<repo>/eda/errors/missed_positives")


def softmax_p(z, y, fid, off, M, nf):
    a = 1.0
    for _ in range(30):
        e = np.exp(a * z - np.repeat(np.maximum.reduceat(a * z, off), M))
        S0 = np.bincount(fid, weights=e, minlength=nf); S1 = np.bincount(fid, weights=e * z, minlength=nf)
        S2 = np.bincount(fid, weights=e * z * z, minlength=nf); npos = np.bincount(fid, weights=y, minlength=nf)
        m1 = S1 / S0; m2 = S2 / S0
        g = (y * z).sum() - (npos * m1).sum(); h = -(npos * (m2 - m1 ** 2)).sum()
        st = g / h; a -= st
        if abs(st) < 1e-8:
            break
    e = np.exp(a * z - np.repeat(np.maximum.reduceat(a * z, off), M))
    return e / np.repeat(np.bincount(fid, weights=e, minlength=nf), M) * np.repeat(np.bincount(fid, weights=y, minlength=nf), M)


def calib(x, y, p, fid, keepc, nf, nb=10):
    xs = x[keepc & np.isfinite(x)]
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, nb + 1)[1:-1]))
    b = np.where(np.isfinite(x), np.searchsorted(edges, x, side="right"), len(edges) + 1)
    K = len(edges) + 2
    w = keepc.astype(float)
    key = fid * K + b
    A = np.bincount(key, weights=y * w, minlength=nf * K).reshape(nf, K)
    E = np.bincount(key, weights=p * w, minlength=nf * K).reshape(nf, K)
    d = A - E
    num = d.sum(0); den = np.sqrt((d ** 2).sum(0)) + 1e-12
    share = np.bincount(b, weights=w, minlength=K) / w.sum()
    ratio = A.sum(0) / np.maximum(E.sum(0), 1e-9)
    lo = np.r_[-np.inf, edges, np.nan]
    return num / den, ratio, share, lo, A.sum(0)


def main():
    fr = np.load(D / "frame.npz")
    y, best, zc, M, off, leakf, fid = (fr[k] for k in ["y", "best", "z_clean", "M", "off", "leakf", "fid"])
    nf = len(M); keepc = ~leakf[fid]
    pb = softmax_p(best, y, fid, off, M, nf)
    pc = softmax_p(zc, y, fid, off, M, nf)
    names = all_feature_names("val")
    dnames = sorted(p.stem for p in (D / "derived/val").glob("*.npy"))
    rows = []
    for n in names + dnames:
        x = np.load((FEATS / "val" / f"{n}.npy") if n in names else (D / "derived/val" / f"{n}.npy")).astype(np.float64)
        if np.nanstd(x) == 0:
            continue
        zb, rb, sh, lo, A = calib(x, y, pb, fid, keepc, nf)
        zcl, rc, _, _, _ = calib(x, y, pc, fid, keepc, nf)
        k = int(np.nanargmax(np.abs(zb) * (sh > 0.005)))
        rows.append(dict(feat=n, maxz_best=float(zb[k]), bin_lo=float(lo[k]) if k < len(lo) else np.nan, bin_share=float(sh[k]),
                         ratio_best=float(rb[k]), ratio_clean=float(rc[k]), z_clean_samebin=float(zcl[k]), n_pos_bin=float(A[k]),
                         chi2_best=float(np.sum((zb ** 2)[sh > 0.005])), chi2_clean=float(np.sum((zcl ** 2)[sh > 0.005]))))
    df = pl.DataFrame(rows).with_columns((pl.col("chi2_best") / 9).alias("chi2_per_df"))
    df.write_parquet(D / "calib.parquet")
    pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(3)
    print(df.sort(pl.col("maxz_best").abs(), descending=True).head(60))


if __name__ == "__main__":
    main()
