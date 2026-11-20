"""Skeptic check of the P2 mechanism: PSI of the trainer tied rank/z vs warm-only rank/z measured from the TRAINING
marginal (trn subset = 20K random train feeds) to val / public / private, all rows and warm rows only.
Read-only on the proposer's extracted arrays. CPU only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, numba as nb, torch
P = Path("/tmp/scratchpad/dpa/pipeline/out")
CK = torch.load("<repo>/nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
XT = CK["cfg"]["xt_names"]; xi = {n: j for j, n in enumerate(XT)}

@nb.njit(parallel=True, cache=True)
def rz(x, off, M, warm):
    r = np.zeros(len(x)); z = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; n = 0; s = 0.0
        for k in range(m):
            if (not warm) or x[o + k] > 0: n += 1; s += x[o + k]
        if n == 0: continue
        mu = s / n; v = 0.0
        for k in range(m):
            if (not warm) or x[o + k] > 0: v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / n)
        for k in range(m):
            u = x[o + k]
            if warm and u <= 0: continue
            less = 0; eq = 0
            for j in range(m):
                w = x[o + j]
                if warm and w <= 0: continue
                if w < u: less += 1
                elif w == u: eq += 1
            r[o + k] = (less + 0.5 * (eq - 1)) / max(n - 1, 1)
            z[o + k] = (u - mu) / (sd + 1e-4)
    return r, z

def psi(a, b, ref):
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    pa = np.bincount(np.searchsorted(qs, a, side="right"), minlength=len(qs) + 1) / len(a)
    pb = np.bincount(np.searchsorted(qs, b, side="right"), minlength=len(qs) + 1) / len(b)
    pa = np.clip(pa, 1e-4, None); pb = np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())

D = {}
for sub in ["trn", "val", "test"]:
    R = np.load(P / f"{sub}_raw.npy", mmap_mode="r"); mt = dict(np.load(P / f"{sub}_meta.npz"))
    D[sub] = (R, mt)
BOUND = 1790497920
for col in ["it_hist_all_luo", "seq_cv_invd20"]:
    S = {}
    for sub, (R, mt) in D.items():
        x = np.nan_to_num(np.asarray(R[:, xi[col]], np.float64)); M, off = mt["M"], mt["off"]
        tr, tz = rz(x, off, M, False); wr, wz = rz(x, off, M, True); w = x > 0
        rowsel = {"all": np.ones(len(x), bool)}
        if sub == "test":
            pub = np.repeat(mt["ts"] < BOUND, M); S["pub"] = (tr[pub], tz[pub], wr[pub], wz[pub], w[pub]); S["priv"] = (tr[~pub], tz[~pub], wr[~pub], wz[~pub], w[~pub])
        else:
            S[sub] = (tr, tz, wr, wz, w)
    ref = S["trn"]
    print(f"== {col}  cold share: " + " ".join(f"{k} {1 - v[4].mean():.3f}" for k, v in S.items()))
    for tgt in ["val", "pub", "priv"]:
        t = S[tgt]
        line = f"  trn->{tgt:4s} PSI all rows: tied_rk {psi(ref[0], t[0], ref[0]):.3f} tied_z {psi(ref[1], t[1], ref[1]):.3f} | warm_rk {psi(ref[2], t[2], ref[2]):.3f} warm_z {psi(ref[3], t[3], ref[3]):.3f}"
        line += f" || warm rows only: tied_rk {psi(ref[0][ref[4]], t[0][t[4]], ref[0][ref[4]]):.3f} tied_z {psi(ref[1][ref[4]], t[1][t[4]], ref[1][ref[4]]):.3f} | warm_rk {psi(ref[2][ref[4]], t[2][t[4]], ref[2][ref[4]]):.3f} warm_z {psi(ref[3][ref[4]], t[3][t[4]], ref[3][ref[4]]):.3f}"
        print(line)
    v = S["val"]; p = S["priv"]
    print(f"  val->priv PSI all rows: tied_rk {psi(v[0], p[0], v[0]):.3f} tied_z {psi(v[1], p[1], v[1]):.3f} | warm_rk {psi(v[2], p[2], v[2]):.3f} warm_z {psi(v[3], p[3], v[3]):.3f}")
