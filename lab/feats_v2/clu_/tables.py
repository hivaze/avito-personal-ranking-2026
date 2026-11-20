"""Train-label tables for the clu_ family (one pass over train; all tables are per train calendar day so that
train rows can use strictly earlier days (expanding window), and val/test use the sum over all train days).

Expected clicks (position debiasing): ex = CTR of the (pos bucket x rank-from-end bucket x M bucket) cell over all train
(17 x 8 x 6 = 816 cells; a nuisance model with 247M rows -> its own-day leakage is negligible).
Outputs (data/clu_work/):
  train_ex.npy                         float32 per train candidate row
  te_ic{K}.npz                         imp/pos/ex [D, K]               item-cluster
  te_u{Ku}i{Ki}.npz                    imp/pos/ex [D, Ku*Ki]           user-cluster x item-cluster
  te_dp{K}.npz                         imp/pos/ex [D, 4*K]             daypart (MSK 0-6,6-12,12-18,18-24) x item-cluster
  te_wk{K}.npz                         imp/pos/ex [D, 2*K]             weekday/weekend x item-cluster
  cf_h{K}.npz                          pos/ex [D, K*K]                 last-16-history-cluster x candidate-cluster
"""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *

L_CF = 16
HALF = 5.0


def dayparts(ts):
    h = ((ts + 3 * 3600) // 3600) % 24
    return (h // 6).astype(np.int64)


def is_weekend(day):
    # day 0 = Sunday; Saturday = day % 7 == 6
    return ((day % 7 == 0) | (day % 7 == 6)).astype(np.int64)


def bc3(key, nbin, y, ex):
    imp = np.bincount(key, minlength=nbin).astype(np.float32)
    ps = np.bincount(key, weights=y, minlength=nbin).astype(np.float32)
    e = np.bincount(key, weights=ex, minlength=nbin).astype(np.float32)
    return imp, ps, e


@nb.njit(parallel=True, cache=True)
def cf_accum(f0, f1, co, M, ho, H, ccl, hcl, hpos, y, ex, K, L, wtab, nth):
    P = np.zeros((nth, K * K), np.float64)
    E = np.zeros((nth, K * K), np.float64)
    nf = f1 - f0
    for t in nb.prange(nth):
        hs = np.empty(L, np.int64)
        ws = np.empty(L, np.float64)
        for f in range(f0 + t * nf // nth, f0 + (t + 1) * nf // nth):
            n = 0
            o = ho[f]
            for j in range(H[f]):
                c = hcl[o + j]
                if c >= 0:
                    hs[n] = c
                    ws[n] = wtab[hpos[o + j]]
                    n += 1
                    if n == L:
                        break
            if n == 0:
                continue
            sw = 0.0
            for k in range(n):
                sw += ws[k]
            for i in range(M[f]):
                r = co[f] + i
                c = ccl[r]
                if c < 0:
                    continue
                yy = y[r]
                ee = ex[r]
                for k in range(n):
                    w = ws[k] / sw
                    P[t, hs[k] * K + c] += w * yy
                    E[t, hs[k] * K + c] += w * ee
    Ps = np.zeros(K * K, np.float64)
    Es = np.zeros(K * K, np.float64)
    for t in range(nth):
        Ps += P[t]
        Es += E[t]
    return Ps, Es


def main(stages):
    nb.set_num_threads(24)
    fe = feeds("train")
    nf = len(fe["M"])
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float32)
    N = len(y)
    exp_path = WORK / "train_ex.npy"
    if not exp_path.exists():
        pos = np.load(FLAT / "train/cand_pos.npy")
        rk = np.empty(N, np.int16)
        rank_in_feed(fe["co"], fe["M"], pos, rk)
        Mr = fe["M"][fidx]
        rfe = Mr - 1 - rk
        pb = np.searchsorted(POS_EDGES, pos, side="right") - 1
        rb = np.searchsorted(RFE_EDGES, rfe, side="right") - 1
        mb = np.searchsorted(M_EDGES, Mr, side="right") - 1
        cell = (pb * NRB + rb) * NMB + mb
        del rk, rfe, pb, rb, mb, Mr
        nc = NPB * NRB * NMB
        ctr = np.bincount(cell, weights=y, minlength=nc) / np.maximum(np.bincount(cell, minlength=nc), 1)
        ex = ctr[cell].astype(np.float32)
        np.save(exp_path, ex)
        np.save(WORK / "pos_cell_ctr.npy", ctr)
        log("expected clicks: cells", nc, "ex mean", ex.mean(), "y mean", y.mean(),
            "corr", np.corrcoef(ex[:5_000_000], y[:5_000_000])[0, 1])
        del cell, pos
    ex = np.load(exp_path)
    dayf = day_index(fe["ts"])
    day = dayf[fidx].astype(np.int64)
    if "te" in stages:
        icl = {K: np.asarray(cand_cl("train", K)).astype(np.int64) for K in IKS}
        for K in IKS:
            c = icl[K]
            ok = c >= 0
            imp, ps, e = bc3(day[ok] * K + c[ok], D * K, y[ok], ex[ok])
            np.savez(WORK / f"te_ic{K}.npz", imp=imp.reshape(D, K), pos=ps.reshape(D, K), ex=e.reshape(D, K))
            log("te_ic", K)
        dp = dayparts(fe["ts"])[fidx]
        wk = is_weekend(day)
        for K in (256, 1024):
            c = icl[K]
            ok = c >= 0
            imp, ps, e = bc3((day[ok] * 4 + dp[ok]) * K + c[ok], D * 4 * K, y[ok], ex[ok])
            np.savez(WORK / f"te_dp{K}.npz", imp=imp.reshape(D, 4 * K), pos=ps.reshape(D, 4 * K), ex=e.reshape(D, 4 * K))
            imp, ps, e = bc3((day[ok] * 2 + wk[ok]) * K + c[ok], D * 2 * K, y[ok], ex[ok])
            np.savez(WORK / f"te_wk{K}.npz", imp=imp.reshape(D, 2 * K), pos=ps.reshape(D, 2 * K), ex=e.reshape(D, 2 * K))
            log("te_dp/wk", K)
        del dp, wk
        for Ku in UKS:
            ucl = user_cl("train", Ku)[fidx]
            for Ki in (256, 1024, 4096):
                if Ku * Ki > 1024 * 4096 or (Ku, Ki) in ((64, 4096),):
                    continue
                c = icl[Ki]
                ok = (c >= 0) & (ucl >= 0)
                nbin = D * Ku * Ki
                imp, ps, e = bc3((day[ok] * Ku + ucl[ok]) * Ki + c[ok], nbin, y[ok], ex[ok])
                np.savez(WORK / f"te_u{Ku}i{Ki}.npz", imp=imp.reshape(D, Ku * Ki), pos=ps.reshape(D, Ku * Ki),
                         ex=e.reshape(D, Ku * Ki))
                log("te_u", Ku, Ki)
            del ucl
        del icl
    if "cf" in stages:
        hpos = np.load(FLAT / "train/hist_hpos.npy")
        wtab = 0.5 ** (np.arange(256) / HALF)
        dstart = np.searchsorted(dayf, np.arange(D + 1))
        for K in [int(k) for k in stages["cf"]]:
            ccl = np.asarray(cand_cl("train", K))
            hcl = np.asarray(hist_cl("train", K))
            P = np.zeros((D, K * K), np.float32)
            E = np.zeros((D, K * K), np.float32)
            for d in range(D):
                f0, f1 = dstart[d], dstart[d + 1]
                if f1 <= f0:
                    continue
                p, e = cf_accum(f0, f1, fe["co"], fe["M"], fe["ho"], fe["H"], ccl, hcl, hpos, y, ex, K, L_CF, wtab, 24)
                P[d] = p; E[d] = e
                log("cf", K, "day", d, f1 - f0)
            np.savez(WORK / f"cf_h{K}.npz", pos=P, ex=E)
            del ccl, hcl


if __name__ == "__main__":
    st = {}
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        st[k] = v.split(",") if v else True
    main(st)
