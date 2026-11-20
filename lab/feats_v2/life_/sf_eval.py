"""Single-feature NDCG on val (both directions, random tie-break, NaN -> bottom) + coverage + drift (PSI val vs
test-public / test-private) for every feature file with a given prefix.
Usage: python sf_eval.py life_ts_ [--nodrift]"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *

PUB_END = 1790497920  # public/private boundary timestamp (test)


def sf_ndcg(x, y, off, M, n_rep=3):
    x = np.asarray(x, np.float64)
    out = []
    for sgn in (1.0, -1.0):
        v = np.where(np.isfinite(x), sgn * x, -np.inf)
        fin = np.isfinite(v)
        # rank-transform so that the noise is a pure tie-breaker
        r = np.full(len(v), -1.0)
        if fin.any():
            r[fin] = np.unique(v[fin], return_inverse=True)[1].astype(np.float64)
        vals = []
        for s in range(n_rep):
            noise = np.random.default_rng(s).random(len(v)) * 0.5
            vals.append(ndcg(r + noise, y, off, M))
        out.append(float(np.mean(vals)))
    return out


def psi(a, b, nb=10):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    qs = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)[1:-1]))
    pa = np.bincount(np.searchsorted(qs, a, side="right"), minlength=len(qs) + 1) / len(a)
    pb = np.bincount(np.searchsorted(qs, b, side="right"), minlength=len(qs) + 1) / len(b)
    pa = np.clip(pa, 1e-4, None); pb = np.clip(pb, 1e-4, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def main():
    pref = sys.argv[1]
    drift = "--nodrift" not in sys.argv
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    names = [n for n in all_feature_names("val") if n.startswith(pref)]
    if drift:
        ft = pl.read_parquet(FLAT / "test/feeds.parquet")
        tfi = np.load(FLAT / "test/cand_fidx.npy")
        pub = (ft["timestamp"].to_numpy() < PUB_END)[tfi]
    print(f"{'feature':28s} {'nd+':>7s} {'nd-':>7s} {'cov_v':>6s} {'cov_pub':>7s} {'cov_prv':>7s} {'psi_v_pub':>9s} {'psi_v_prv':>9s}  quantiles val | pub | prv (25/50/75)")
    for n in names:
        x = np.load(FEATS / "val" / f"{n}.npy")
        a, b = sf_ndcg(x, y, off, M)
        line = f"{n:28s} {a:7.4f} {b:7.4f} {np.isfinite(x).mean():6.3f}"
        if drift and (FEATS / "test" / f"{n}.npy").exists():
            xt = np.load(FEATS / "test" / f"{n}.npy")
            xp, xq = xt[pub], xt[~pub]
            q = lambda z: np.round(np.nanquantile(z, [0.25, 0.5, 0.75]), 2) if np.isfinite(z).any() else None
            line += (f" {np.isfinite(xp).mean():7.3f} {np.isfinite(xq).mean():7.3f} {psi(x, xp):9.3f} {psi(x, xq):9.3f}"
                     f"  {q(x)} | {q(xp)} | {q(xq)}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
