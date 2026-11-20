"""Cheap marginal-utility screen: NDCG of (within-feed pct-rank of base OOF) + a * (within-feed pct-rank of feature),
NaN -> lowest rank, for a in +-{0.03, 0.1}. Reports the best delta vs base OOF with its paired SE.
Optimistic (a is picked on val) -> only for pre-filtering. Usage: python blend_screen.py <prefix> [oof.npy]"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

SCR = Path("/tmp/scratchpad")


@nb.njit(parallel=True, cache=True)
def feed_pct(x, off, M, out):
    for f in nb.prange(len(M)):
        a = off[f]; m = M[f]
        v = x[a:a + m]
        o = np.argsort(v, kind="mergesort")
        r = np.empty(m)
        # average ranks for ties
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        for k in range(m):
            out[a + k] = r[k] / max(m - 1, 1)


def main():
    pref = sys.argv[1]
    oofp = sys.argv[2] if len(sys.argv) > 2 else str(SCR / "xgb_base_oof_s0.npy")
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    ob = np.load(oofp)
    rb = np.empty(len(ob)); feed_pct(ob.astype(np.float64), off, M.astype(np.int64), rb)
    noise = np.random.default_rng(0).random(len(y)) * 1e-7
    pb = ndcg_per_feed(rb + noise, y, off, M)
    print("base", pb.mean().round(5))
    for n in sorted(p.stem for p in (FEATS / "val").glob(f"{pref}*.npy")):
        x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)
        res = []
        for sgn in (1, -1):
            v = np.where(np.isfinite(x), sgn * x, -1e30)
            rx = np.empty(len(v)); feed_pct(v, off, M.astype(np.int64), rx)
            for a in (0.03, 0.1):
                d = ndcg_per_feed(rb + a * rx + noise, y, off, M) - pb
                res.append((d.mean(), d.std() / np.sqrt(len(d)), sgn * a))
        best = max(res, key=lambda z: z[0])
        print(f"{n:28s} best_a={best[2]:+.2f} delta={best[0]:+.5f} se={best[1]:.5f}  t={best[0]/best[1]:+.1f}", flush=True)


if __name__ == "__main__":
    main()
