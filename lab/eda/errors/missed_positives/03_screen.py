"""Screen every available feature for the missed-positives lens (leak-free val feeds only).

For each feature x:
 (1) group means / medians: MP (positives in bottom half by v5 blend), TP3 (positives ranked top-3), FA3 (negatives ranked top-3,
     in the same feeds); unpaired AUC MP-vs-FA3.
 (2) within-feed win rate WR(MP>FA3): for every feed with >=1 MP, mean over (MP, FA3) pairs of 1[x_MP > x_FA] + 0.5*1[=],
     averaged over feeds; SE over feeds. Same for TP3 vs FA3 (what the model already uses).
 (3) pairwise residual test on ALL within-feed (pos, neg) pairs: conditional logit P(pos above neg) = sigmoid(a*dz + b*dx)
     where dz = model z-score difference, dx = difference of the rank-gauss transformed feature; pairs weighted so every feed
     has total weight 1; cluster-robust (feed) t-stat of b.  Done against v5 blend ("best") and against nn_final2_trainonly
     ("clean").  |t_best| large = signal the best model does not exploit.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb
from scipy.stats import rankdata, norm

D = Path("<repo>/eda/errors/missed_positives")


@nb.njit(cache=True)
def build_pairs(off, M, y, keep):
    n = 0
    for f in range(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        p = 0
        for i in range(m):
            p += y[o + i] > 0
        n += p * (m - p)
    ip = np.empty(n, np.int64); ineg = np.empty(n, np.int64); pf = np.empty(n, np.int64); w = np.empty(n, np.float64)
    k = 0
    for f in range(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        p = 0
        for i in range(m):
            p += y[o + i] > 0
        if p == 0 or p == m:
            continue
        ww = 1.0 / (p * (m - p))
        for i in range(m):
            if y[o + i] > 0:
                for j in range(m):
                    if y[o + j] == 0:
                        ip[k] = o + i; ineg[k] = o + j; pf[k] = f; w[k] = ww; k += 1
    return ip[:k], ineg[:k], pf[:k], w[:k]


@nb.njit(parallel=True, cache=True)
def _grad_hess(a, b, dz, dx, w):
    # returns gradient (2), hessian (3: aa, ab, bb), loglik
    n = len(dz)
    nt = 16
    acc = np.zeros((nt, 6))
    chunk = (n + nt - 1) // nt
    for t in nb.prange(nt):
        s = t * chunk; e = min(n, s + chunk)
        g0 = 0.0; g1 = 0.0; h00 = 0.0; h01 = 0.0; h11 = 0.0; ll = 0.0
        for i in range(s, e):
            u = a * dz[i] + b * dx[i]
            if u > 0:
                sg = 1.0 / (1.0 + np.exp(-u)); ll += w[i] * (-np.log1p(np.exp(-u)))
            else:
                ex = np.exp(u); sg = ex / (1.0 + ex); ll += w[i] * (u - np.log1p(ex))
            r = w[i] * (1.0 - sg)
            g0 += r * dz[i]; g1 += r * dx[i]
            q = w[i] * sg * (1.0 - sg)
            h00 += q * dz[i] * dz[i]; h01 += q * dz[i] * dx[i]; h11 += q * dx[i] * dx[i]
        acc[t, 0] = g0; acc[t, 1] = g1; acc[t, 2] = h00; acc[t, 3] = h01; acc[t, 4] = h11; acc[t, 5] = ll
    return acc.sum(axis=0)


@nb.njit(cache=True)
def _cluster_scores(a, b, dz, dx, w, pf, nf):
    s0 = np.zeros(nf); s1 = np.zeros(nf)
    for i in range(len(dz)):
        u = a * dz[i] + b * dx[i]
        sg = 1.0 / (1.0 + np.exp(-u))
        r = w[i] * (1.0 - sg)
        s0[pf[i]] += r * dz[i]; s1[pf[i]] += r * dx[i]
    return s0, s1


def fit(dz, dx, w, pf, nf, a0=1.0):
    a, b = a0, 0.0
    for _ in range(25):
        g0, g1, h00, h01, h11, ll = _grad_hess(a, b, dz, dx, w)
        Hm = np.array([[h00, h01], [h01, h11]]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(Hm, np.array([g0, g1]))
        a += step[0]; b += step[1]
        if np.abs(step).max() < 1e-7:
            break
    g0, g1, h00, h01, h11, ll = _grad_hess(a, b, dz, dx, w)
    Hm = np.array([[h00, h01], [h01, h11]])
    s0, s1 = _cluster_scores(a, b, dz, dx, w, pf, nf)
    S = np.stack([s0, s1], 1)
    meat = S.T @ S
    Hi = np.linalg.inv(Hm + 1e-9 * np.eye(2))
    V = Hi @ meat @ Hi
    return a, b, b / np.sqrt(V[1, 1] + 1e-30), ll


def rank_gauss(x):
    x = x.astype(np.float64)
    nanm = ~np.isfinite(x)
    if nanm.any():
        x = x.copy(); x[nanm] = np.nanmedian(x[~nanm]) if (~nanm).any() else 0.0
    r = rankdata(x, method="average")
    return norm.ppf((r - 0.5) / len(r))


@nb.njit(parallel=True, cache=True)
def winrate(x, off, M, mp_mask, fa_mask, keep):
    nf = len(M)
    out = np.full(nf, np.nan)
    for f in nb.prange(nf):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        s = 0.0; c = 0
        for i in range(m):
            if not mp_mask[o + i]:
                continue
            for j in range(m):
                if not fa_mask[o + j]:
                    continue
                xi = x[o + i]; xj = x[o + j]
                if xi > xj:
                    s += 1.0
                elif xi == xj:
                    s += 0.5
                c += 1
        if c > 0:
            out[f] = s / c
    return out


def auc(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    r = rankdata(np.concatenate([a, b]))
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    fr = np.load(D / "frame.npz")
    y, best, zc, M, off, leakf, MP, TP3, FA3 = (fr[k] for k in ["y", "best", "z_clean", "M", "off", "leakf", "MP", "TP3", "FA3"])
    keep = ~leakf
    ip, ineg, pf, w = build_pairs(off.astype(np.int64), M.astype(np.int64), y, keep)
    print("pairs:", len(ip), flush=True)
    nf = len(M)
    dz_b = best[ip] - best[ineg]; dz_c = zc[ip] - zc[ineg]
    a_b, _, _, ll_b = fit(dz_b, np.zeros_like(dz_b), w, pf, nf)
    a_c, _, _, ll_c = fit(dz_c, np.zeros_like(dz_c), w, pf, nf)
    print(f"base fits: best a={a_b:.3f} ll={ll_b:.2f}; clean a={a_c:.3f} ll={ll_c:.2f}", flush=True)
    names = [n for n in all_feature_names("val")]
    dnames = sorted(p.stem for p in (D / "derived/val").glob("*.npy"))
    rows = []
    fidc = fr["fid"]
    for n in names + dnames:
        x = np.load((FEATS / "val" / f"{n}.npy") if n in names else (D / "derived/val" / f"{n}.npy")).astype(np.float64)
        if np.nanstd(x) == 0:
            continue
        xn = np.where(np.isfinite(x), x, np.nan)
        u = rank_gauss(x)
        dx = u[ip] - u[ineg]
        if np.abs(dx).max() == 0:
            # feed-constant feature: no within-feed information
            wb = (np.nan, np.nan, np.nan, 0.0); wc = wb
        else:
            wb = fit(dz_b, dx, w, pf, nf, a_b)
            wc = fit(dz_c, dx, w, pf, nf, a_c)
        xw = np.where(np.isfinite(x), x, -1e30)
        wr_mp = winrate(xw, off.astype(np.int64), M.astype(np.int64), MP, FA3, keep)
        wr_tp = winrate(xw, off.astype(np.int64), M.astype(np.int64), TP3, FA3, keep)
        m1 = np.isfinite(wr_mp); m2 = np.isfinite(wr_tp)
        rows.append(dict(
            feat=n,
            mean_MP=np.nanmean(xn[MP]), mean_TP3=np.nanmean(xn[TP3]), mean_FA3=np.nanmean(xn[FA3]),
            med_MP=np.nanmedian(xn[MP]), med_TP3=np.nanmedian(xn[TP3]), med_FA3=np.nanmedian(xn[FA3]),
            nan_MP=float(np.isnan(xn[MP]).mean()), nan_FA3=float(np.isnan(xn[FA3]).mean()),
            auc_MP_FA3=auc(xn[MP], xn[FA3]),
            wr_MP_FA3=float(np.nanmean(wr_mp)), wr_MP_se=float(np.nanstd(wr_mp) / np.sqrt(m1.sum())),
            wr_TP3_FA3=float(np.nanmean(wr_tp)), wr_TP3_se=float(np.nanstd(wr_tp) / np.sqrt(m2.sum())),
            b_best=wb[1], t_best=wb[2], dll_best=wb[3] - ll_b if np.isfinite(wb[1]) else 0.0,
            b_clean=wc[1], t_clean=wc[2], dll_clean=wc[3] - ll_c if np.isfinite(wc[1]) else 0.0,
        ))
        if len(rows) % 50 == 0:
            print(len(rows), n, flush=True)
    df = pl.DataFrame(rows)
    df.write_parquet(D / "screen.parquet")
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    print("\nTOP |t_best| (signal not exploited by v5 blend):")
    print(df.sort(pl.col("t_best").abs(), descending=True, nulls_last=True).head(60).select(
        "feat", "t_best", "b_best", "t_clean", "wr_MP_FA3", "wr_MP_se", "wr_TP3_FA3", "mean_MP", "mean_TP3", "mean_FA3"))
    print("\nTOP wr_MP_FA3 (missed positives beat their top-3 negatives on this feature):")
    print(df.sort("wr_MP_FA3", descending=True).head(40).select(
        "feat", "wr_MP_FA3", "wr_MP_se", "wr_TP3_FA3", "t_best", "t_clean", "mean_MP", "mean_FA3"))
    print("\nBOTTOM wr_MP_FA3 (why MPs were missed: the model followed these):")
    print(df.sort("wr_MP_FA3").head(40).select(
        "feat", "wr_MP_FA3", "wr_MP_se", "wr_TP3_FA3", "t_best", "t_clean", "mean_MP", "mean_FA3"))


if __name__ == "__main__":
    main()
