"""Quick screen for trd_ features on val: single-feature NDCG (both directions, random tie-break),
NaN share, and a rank-blend delta on top of a fixed strong reference score
ref = pct(maxsim32) + pct(cos_u) + pct(f1)  (within-feed percentile ranks).
Usage: python screen.py name1,name2,...   or   python screen.py --prefix trd_cl_
"""
import sys
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

nb.set_num_threads(16)


@nb.njit(parallel=True, cache=True)
def pct_rank(off, M, x, noise, out):
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        v = x[o:o + m] + noise[o:o + m]
        order = np.argsort(v)
        for k in range(m):
            out[o + order[k]] = k / (m - 1) if m > 1 else 0.5


def main():
    args = sys.argv[1:]
    if args[0] == "--prefix":
        names = sorted(p.stem for p in (FEATS / "val").glob(f"{args[1]}*.npy"))
    else:
        names = args[0].split(",")
    f = feeds_df("val"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    rng = np.random.default_rng(0)
    noise = rng.random(len(y)) * 1e-6
    def pr(x):
        x = np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]) - 1 if np.isfinite(x).any() else 0)
        o = np.empty(len(x)); pct_rank(off, M, x.astype(np.float64) * 1.0, noise * (np.abs(x).mean() + 1e-9), o); return o
    ref = sum(pr(np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)) for n in ["maxsim32", "cos_u", "f1"])
    pb = ndcg_per_feed(ref + noise, y, off, M)
    print(f"ref ndcg {pb.mean():.4f}")
    print(f"{'feature':32s} {'nan%':>6s} {'asc':>7s} {'desc':>7s} {'d(+.3)':>8s} {'d(-.3)':>8s} {'se':>7s}")
    for n in names:
        x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)
        a, d = single_ndcg(x, y, off, M)
        r = pr(x)
        dp = ndcg_per_feed(ref + 0.3 * r + noise, y, off, M) - pb
        dm = ndcg_per_feed(ref - 0.3 * r + noise, y, off, M) - pb
        print(f"{n:32s} {100*np.mean(~np.isfinite(x)):6.2f} {a:7.4f} {d:7.4f} {dp.mean():+8.4f} {dm.mean():+8.4f} {dp.std()/np.sqrt(len(dp)):7.4f}", flush=True)


if __name__ == "__main__":
    main()
