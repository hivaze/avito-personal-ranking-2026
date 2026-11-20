"""Reviewer's own helpers (nn_x/w12/seg/review). Independent of nn_x/w12/seg/diag/lib.py: own z, own NDCG, own cluster SE,
own AUC, own E_random. CPU only; read-only outside this directory."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
P = ROOT / "preds"
FC = ROOT / "nn/feat_cache/val"
HERE = ROOT / "nn_x/w12/seg/review"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

fv = pl.read_parquet(FLAT / "val/feeds.parquet")
M = fv["M"].to_numpy().astype(np.int64)
OFF = fv["cand_off"].to_numpy().astype(np.int64)
H = fv["H"].to_numpy().astype(np.int64)
HOFF = fv["hist_off"].to_numpy().astype(np.int64)
UID = fv["user_id"].to_numpy().astype(np.int64)
NF = len(M)
N = int(M.sum())
assert (OFF == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
FID = np.repeat(np.arange(NF), M)
Y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
assert len(Y) == N
NPOS = np.bincount(FID, weights=Y, minlength=NF).astype(np.int64)
# clusters: one per user; every feed of the pooled anonymous bucket (user_id 0) is its own cluster
_key = np.where(UID == 0, -(np.arange(NF) + 1), UID)
CLU = np.unique(_key, return_inverse=True)[1]
NCL = int(CLU.max()) + 1

CAP = H >= 256
NEAR = (H >= 100) & (H <= 255)
REST = H < 256
BUCK = [("H=0", H == 0), ("1-4", (H >= 1) & (H <= 4)), ("5-9", (H >= 5) & (H <= 9)), ("10-19", (H >= 10) & (H <= 19)),
        ("20-49", (H >= 20) & (H <= 49)), ("50-99", (H >= 50) & (H <= 99)), ("100-255", NEAR), ("H=256", CAP)]


def z(x):
    """per-feed standardisation (x - mean) / (sd + 1e-9), population sd; own numpy code."""
    x = np.asarray(x, np.float64)
    s1 = np.bincount(FID, weights=x, minlength=NF) / M
    s2 = np.bincount(FID, weights=x * x, minlength=NF) / M
    sd = np.sqrt(np.maximum(s2 - s1 * s1, 0.0)) + 1e-9
    return (x - s1[FID]) / sd[FID]


_DISC = 1.0 / np.log2(np.arange(2, 3002))
_CD = np.concatenate([[0.0], np.cumsum(_DISC)])


def ndcg(s):
    """own per-feed NDCG: binary gain, log2 discount, pessimistic ties (negatives first inside a tie), no-positive feed = 1."""
    s = np.asarray(s, np.float64)
    order = np.lexsort((Y, -s, FID))
    pos = np.arange(N) - OFF[FID]            # FID is sorted, so position within the feed after the sort
    dcg = np.bincount(FID, weights=Y[order] * _DISC[pos], minlength=NF)
    out = np.ones(NF)
    ok = NPOS > 0
    out[ok] = dcg[ok] / _CD[NPOS[ok]]
    return out


def auc(s):
    """per-feed AUC (share of negatives below a positive, ties 1/2), NaN without positives or negatives. Pairwise count
    through ranks: own implementation with average ranks."""
    s = np.asarray(s, np.float64)
    order = np.lexsort((s, FID))
    ss = s[order]
    yy = Y[order]
    r = (np.arange(N) - OFF[FID]).astype(np.float64)
    # average ranks over ties within a feed
    brk = np.ones(N, bool)
    brk[1:] = (ss[1:] != ss[:-1]) | (FID[1:] != FID[:-1])
    g = np.cumsum(brk) - 1
    r = (np.bincount(g, weights=r) / np.bincount(g))[g]
    npos = NPOS.astype(np.float64)
    nneg = M - npos
    rs = np.bincount(FID, weights=r * yy, minlength=NF)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (rs - npos * (npos - 1) / 2) / (npos * nneg)
    a[(npos == 0) | (nneg == 0)] = np.nan
    return a


def e_rand():
    """expected NDCG of a uniformly random order: every position equally likely for each positive."""
    e = np.ones(NF)
    ok = NPOS > 0
    e[ok] = NPOS[ok] * (_CD[M[ok]] / M[ok]) / _CD[NPOS[ok]]
    return e


def cmean(x, m=None):
    """mean, cluster-robust SE (by user), n over feeds in mask m (NaN dropped)."""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    if m is not None:
        ok &= m
    n = int(ok.sum())
    if n == 0:
        return np.nan, np.nan, 0
    mu = x[ok].mean()
    r = np.bincount(CLU[ok], weights=x[ok] - mu, minlength=NCL)
    g = int((np.bincount(CLU[ok], minlength=NCL) > 0).sum())
    return mu, float(np.sqrt((r ** 2).sum() * g / max(g - 1, 1)) / n), n


def cdiff(x, a, b, w=None):
    """mean(x | a) - mean(x | b) with a cluster-robust SE through the influence function; optional feed weights w."""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    w = np.ones(NF) if w is None else np.asarray(w, np.float64)
    wa, wb = w * (a & ok), w * (b & ok)
    xa = np.nan_to_num(x)
    ma, mb = (wa * xa).sum() / wa.sum(), (wb * xa).sum() / wb.sum()
    infl = wa * (xa - ma) / wa.sum() - wb * (xa - mb) / wb.sum()
    r = np.bincount(CLU, weights=infl, minlength=NCL)
    return ma - mb, float(np.sqrt((r ** 2).sum()))


def pool(names, suf):
    return np.mean([z(np.load(P / f"nn_{m}_{suf}.npy")) for m in names], 0)


def final_from(base, pop, w=0.2):
    """as build_r4.py: per branch (1 - w) z(base pool) + w z(pop pool); final 0.5 z(FT) + 0.5 z(TO)."""
    br = []
    for suf in ("n2cpv3s3_val_oof", "trainonly_val"):
        b = pool(base, suf)
        if pop:
            b = (1 - w) * z(b) + w * z(pool(pop, suf))
        br.append(b)
    return 0.5 * z(br[0]) + 0.5 * z(br[1])


S8 = list(range(951, 959))
S4 = list(range(951, 955))
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
FDP = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]
W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
P1W = [f"abl_e10_pop1kd05w192_s{s}" for s in S4]
T11 = [f"r6_cd_m02_r{r}" for r in range(101, 108)] + [f"w5_e10_base_s{s}" for s in S4]


def feat(n):
    return np.nan_to_num(np.load(FC / f"{n}.npy").astype(np.float64), nan=0.0)


def boot_weights(B=2000, seed=11):
    rng = np.random.default_rng(seed)
    return rng.multinomial(NCL, np.ones(NCL) / NCL, size=B).astype(np.float64)


def tab(rows, hdr):
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = lambda r: "| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |"
    return "\n".join([line(hdr), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])
