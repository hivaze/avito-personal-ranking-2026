"""prew_{fe,new,hmax32}_wrk: within-feed mid-rank percentile of the UNCLIPPED prew_* score among history-warm
candidates (non-NaN prew), NaN for cold candidates. Same function on every split (parity by construction).

Why (reports/featrev_prep.md, PREP-1): nn/data.py attach_xt clips every "other" xt column to +-10 before the model
computes the within-feed rank/z of XT_REL columns. prew_fe / prew_new have median ~15 (85-94% of warm rows >= 10 on
every split), so the in-model rank/z of the clipped value are mostly ties (val per-feed AUC 0.562 vs 0.608 unclipped).
These files carry the rank of the unclipped value; names end in _rk-like '_wrk' -> use them via a name ending in '_rk'
or add a pass-through rule. Output: data/feats_new/{split}/prew_*_wrk.npy (float32, aligned with data/flat/<split>/cand_*).
Usage: python feat_new/prew_wrk.py [split ...]"""
import sys, time
import numpy as np, polars as pl, numba as nb
ROOT = "<repo>"
nb.set_num_threads(16)


@nb.njit(parallel=True, cache=False)
def warm_rank(x, off, M, out):
    for f in nb.prange(len(off)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        idx = np.where(~np.isnan(v))[0]
        k = len(idx)
        for i in range(m):
            out[o + i] = np.nan
        if k == 0:
            continue
        if k == 1:
            out[o + idx[0]] = 0.5
            continue
        w = v[idx]
        order = np.argsort(w, kind="mergesort")
        s = w[order]
        i = 0
        while i < k:
            j = i
            while j + 1 < k and s[j + 1] == s[i]:
                j += 1
            r = 0.5 * (i + j) / (k - 1)
            for t in range(i, j + 1):
                out[o + idx[order[t]]] = r
            i = j + 1


def main(splits):
    for split in splits:
        t0 = time.time()
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet")
        off = f["cand_off"].to_numpy().astype(np.int64); M = f["M"].to_numpy().astype(np.int64)
        for n in ["prew_fe", "prew_new", "prew_hmax32"]:
            x = np.load(f"{ROOT}/data/feats/{split}/{n}.npy").astype(np.float32)
            out = np.empty_like(x)
            warm_rank(x, off, M, out)
            assert np.array_equal(np.isnan(out), np.isnan(x))
            np.save(f"{ROOT}/data/feats_new/{split}/{n}_wrk.npy", out.astype(np.float32))
            print(split, n, f"rows {len(x)} warm {np.mean(~np.isnan(x)):.3f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["val", "test", "train"])
