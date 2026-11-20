"""Marginal-information screen: per-feed conditional logit (softmax over the feed's candidates) of the target on
  a * base_oof_score  (+ b * q(x) + c * isnan(x))
where q(x) is the global quantile transform of the feature (NaN -> 0 with an indicator). Likelihood-ratio statistic
vs the base-only model. Measures whether a feature carries ranking information the base model's OOF score lacks.
Usage: python clogit_screen.py <prefix> [base_oof.npy]"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from scipy.optimize import minimize
from scipy.stats import chi2

BASE_OOF = "<repo>/data/eval_cache/base_oof_s0.npy"


def fit(Z, y, starts, fid, posf):
    """Z: (n, k) covariates; maximize sum over positives of log softmax within feed."""
    def nll(w):
        u = Z @ w
        m = np.maximum.reduceat(u, starts)
        e = np.exp(u - np.repeat(m, np.diff(np.r_[starts, len(u)])))
        se = np.add.reduceat(e, starts)
        lse = np.log(se) + m
        ll = (u * y).sum() - (lse * posf).sum()
        p = e / np.repeat(se, np.diff(np.r_[starts, len(u)]))
        g = Z.T @ y - Z.T @ (p * np.repeat(posf, np.diff(np.r_[starts, len(u)])))
        return -ll, -g
    r = minimize(nll, np.zeros(Z.shape[1]) + 0.0, jac=True, method="L-BFGS-B")
    return -r.fun, r.x


def qtrans(x):
    out = np.zeros(len(x)); fin = np.isfinite(x)
    if fin.sum() > 0:
        r = np.unique(x[fin], return_inverse=True)[1].astype(np.float64)
        out[fin] = r / max(r.max(), 1) - 0.5
    return out, (~fin).astype(np.float64)


def main():
    pref = sys.argv[1]
    oof = np.load(sys.argv[2] if len(sys.argv) > 2 else BASE_OOF)
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    npos = np.bincount(fidx, weights=y, minlength=len(M))
    keepf = npos > 0
    rows = keepf[fidx]
    y2 = y[rows]; fid2 = fidx[rows]
    starts = np.r_[0, np.flatnonzero(np.diff(fid2)) + 1]
    posf = npos[keepf]
    s = (oof[rows] - oof[rows].mean()) / oof[rows].std()
    ll0, w0 = fit(s[:, None], y2, starts, fid2, posf)
    print(f"base-only LL {ll0:.1f} a={w0[0]:.3f}  feeds with positives: {keepf.sum()}")
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(p.stem for p in (FEATS / "val").glob(f"{pref}*.npy"))
    for n in names:
        x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)[rows]
        q, nan = qtrans(x)
        cols = [s, q] + ([nan] if 0 < nan.mean() < 1 else [])
        if np.std(q) == 0:
            print(f"{n:28s} constant"); continue
        ll1, w1 = fit(np.stack(cols, 1), y2, starts, fid2, posf)
        lr = 2 * (ll1 - ll0)
        df = len(cols) - 1
        print(f"{n:28s} LR={lr:7.2f} df={df} p={chi2.sf(lr, df):.2e}  b_q={w1[1]:+.3f}" + (f" c_nan={w1[2]:+.3f}" if df == 2 else ""), flush=True)


if __name__ == "__main__":
    main()
