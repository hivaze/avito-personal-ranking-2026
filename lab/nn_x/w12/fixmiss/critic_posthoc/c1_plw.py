"""Critic check 1 (CPU): independent rebuild of FD16P20@m<ws>_<wl> from the 20 member files, own NDCG (pessimistic ties),
paired deltas, nested-by-user choice of the long-feed weight, the public additivity identity, and the stale-witness claim.
Writes nothing outside this directory."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT
import pmix

P = "<repo>/preds"
S8 = range(951, 959); S4 = range(951, 955)
FD16 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]
FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]

fv, M, off = feed_meta("val")
n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
assert len(y) == M.sum()


def z(x):
    x = np.asarray(x, np.float64)
    mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


# own NDCG: per positive, pessimistic rank = #(score > s) + #(neg with score == s) + index among tied positives
def ndcg(s):
    s = np.asarray(s, np.float64)
    out = np.ones(n)
    for f in range(n):
        o, m = off[f], M[f]
        v = s[o:o + m]; t = y[o:o + m]
        K = int(t.sum())
        if K == 0:
            continue
        # sort by -score, then label ascending (negatives first inside ties)
        order = np.lexsort((t, -v))
        ts = t[order]
        disc = 1.0 / np.log2(np.arange(m) + 2.0)
        out[f] = (ts * disc).sum() / disc[:K].sum()
    return out


def pool(names, suf):
    return np.mean([z(np.load(f"{P}/nn_{m}_{suf}.npy").astype(np.float32)) for m in names], 0)


ftb, ftp = pool(FD16, "n2cpv3s3_val_oof"), pool(FDP, "n2cpv3s3_val_oof")
tob, top = pool(FD16, "trainonly_val"), pool(FDP, "trainonly_val")
Mr = np.repeat(M, M); big = M >= 50


def fin(ws, wl):
    w = np.where(Mr >= 50, wl, ws)
    return 0.5 * z((1 - w) * z(ftb) + w * z(ftp)) + 0.5 * z((1 - w) * z(tob) + w * z(top))


ref_s = fin(0.4, 0.2)
st = 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_ft_val.npy")) + 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_to_val.npy"))
print("rebuild vs stored final: max|d|", np.abs(ref_s - st).max())
ref = ndcg(ref_s)
print(f"own NDCG of current file: {ref.mean():.5f} (stored-score version {ndcg(st).mean():.5f})")
W = pmix.weights()


def rep(lbl, e, b=ref):
    d = e - b
    def f(x, w=None):
        m, se = pmix.wmean_se(x, w); return f"{m:+.5f}+-{se:.5f}"
    print(f"{lbl:34s} val {e.mean():.5f} | d {f(d)} | pub {f(d, W['public'])} | priv {f(d, W['private'])} | M<50 {f(d[~big])} | M>=50 {f(d[big])}", flush=True)
    return d


grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
E = {}
for wl in grid:
    E[wl] = ndcg(fin(0.4, wl))
    rep(f"m0.4_{wl}", E[wl])
E02_02 = ndcg(fin(0.2, 0.2)); E02_04 = ndcg(fin(0.2, 0.4))
dA = E[0.4] - ref; dB = E02_04 - E02_02
print("additivity check: max |(m0.4_0.4 - m0.4_0.2) - (m0.2_0.4 - m0.2_0.2)| per feed =", np.abs(dA - dB).max())

# nested-by-user choice of wl (two independent user splits + 5-fold), M>=50 feeds only change
u = fv["user_id"].to_numpy().astype(np.int64)
rng = np.random.default_rng(7)
for split_name, nf in (("2-fold", 2), ("5-fold", 5)):
    for rep_i in range(3):
        uu = np.unique(u); perm = rng.permutation(len(uu)); fold_of = dict(zip(uu[perm], np.arange(len(uu)) % nf))
        fold = np.array([fold_of[x] for x in u])
        g = np.where(u == 0)[0]; fold[g] = rng.integers(0, nf, len(g))
        out = np.zeros(n); picks = []
        for k in range(nf):
            tr = fold != k
            best = max(grid, key=lambda w: E[w][tr].mean())
            picks.append(best); out[fold == k] = E[best][fold == k]
        d = out - ref
        m, se = d.mean(), d.std() / np.sqrt(n)
        mp, sep = pmix.wmean_se(d, W['private'])
        print(f"nested {split_name} rep{rep_i}: picks {picks} | vs current m0.4_0.2 {m:+.5f}+-{se:.5f} | priv {mp:+.5f}+-{sep:.5f}")

# public identity: public(ws, wl) = A(ws) + B(wl) exactly (rules act on disjoint feeds)
pub_FD16P20, pub_m02_04, pub_m04_02 = 0.394892, 0.3947465, 0.3952092
Bpub04 = pub_m02_04 - pub_FD16P20
print(f"public B(0.4)-B(0.2) = {Bpub04:+.7f}  => implied public of m0.4_0.4 = {pub_m04_02 + Bpub04:.7f}")
d03 = (E[0.3] - ref)[big].mean() * big.mean(); d04 = (E[0.4] - ref)[big].mean() * big.mean()
c = (d04 - 2 * d03) / 0.02; b = (d03 - c * 0.01) / 0.1
bp = (Bpub04 - c * 0.04) / 0.2
print(f"val curve B(0.3)={d03:+.5f} B(0.4)={d04:+.5f}; quadratic with val curvature, public slope -> public B(0.3) = {bp * 0.1 + c * 0.01:+.6f}; chord bound >= {Bpub04 / 2:+.6f}")
np.save("<repo>/lab/nn_x/w12/fixmiss/critic_posthoc/E_wl.npy", np.stack([E[w] for w in grid]))
np.save("<repo>/lab/nn_x/w12/fixmiss/critic_posthoc/E_ref.npy", ref)
