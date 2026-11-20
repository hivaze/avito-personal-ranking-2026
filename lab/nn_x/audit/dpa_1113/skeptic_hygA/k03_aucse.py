"""Bootstrap (feed-level) SE of the within-capped-group AUC of the uncapped value and of the G3 score, and of their
difference, plus AUC of the uncapped value conditional on the G3 order (pairs where G3 is wrong): what share of G3's
mis-ordered contested pairs would the uncapped value fix. Val + hold. CPU only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from pathlib import Path
ROOT = Path("<repo>")
IN = Path("/tmp/scratchpad/dpa/pipeline/out")
XT = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)["cfg"]["xt_names"]
xi = {n: j for j, n in enumerate(XT)}
def zf(a, off, M):
    out = np.empty(len(a))
    for o, m in zip(off, M):
        s = a[o:o + m]; sd = s.std(); out[o:o + m] = (s - s.mean()) / (sd if sd > 0 else 1)
    return out
rng = np.random.default_rng(0)
for sub in ["val", "hold"]:
    R = np.load(IN / f"{sub}_raw.npy", mmap_mode="r"); mt = dict(np.load(IN / f"{sub}_meta.npz"))
    M, off, y = mt["M"], mt["off"], mt["y"].astype(np.float64)
    if sub == "val":
        b = zf(0.5 * zf(np.load(ROOT / "preds/ens_G3_ft_val.npy"), off, M) + 0.5 * zf(np.load(ROOT / "preds/ens_G3_to_val.npy"), off, M), off, M)
    else:
        MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
        b = zf(np.mean([zf(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), off, M) for m in MEM], 0), off, M)
    for n in ["ctx_hn_soft05", "ctx_hs_e1"]:
        x = np.nan_to_num(np.asarray(R[:, xi[n]], np.float64), nan=0.0)
        recs = []  # per feed: (wins_x, wins_b, pairs, x_fixes_b_wrong, b_wrong)
        for o, m in zip(off, M):
            xx = x[o:o + m]; c = xx >= 10
            if c.sum() < 2: continue
            yy = y[o:o + m][c]; p = yy > 0
            if p.sum() == 0 or (~p).sum() == 0: continue
            xs = xx[c]; bs = b[o:o + m][c]
            dx = xs[p][:, None] - xs[~p][None, :]; db = bs[p][:, None] - bs[~p][None, :]
            wx = (dx > 0).sum() + 0.5 * (dx == 0).sum(); wb = (db > 0).sum() + 0.5 * (db == 0).sum()
            bw = db < 0
            recs.append((wx, wb, dx.size, (dx[bw] > 0).sum(), bw.sum()))
        A = np.array(recs, float)
        def stat(A):
            return A[:, 0].sum() / A[:, 2].sum(), A[:, 1].sum() / A[:, 2].sum(), A[:, 3].sum() / max(A[:, 4].sum(), 1)
        s0 = stat(A); bs_ = np.array([stat(A[rng.integers(0, len(A), len(A))]) for _ in range(2000)])
        se = bs_.std(0); dd = bs_[:, 0] - bs_[:, 1]
        print(f"{sub:4s} {n:14s} feeds {len(A):5d} pairs {int(A[:,2].sum()):6d} | AUC uncapped x {s0[0]:.4f} (SE {se[0]:.4f})  G3 {s0[1]:.4f} (SE {se[1]:.4f})  "
              f"diff x-G3 {s0[0]-s0[1]:+.4f} (SE {dd.std():.4f}) | among G3-misordered pairs ({int(A[:,4].sum())}) x orders correctly {s0[2]:.4f} (SE {se[2]:.4f})", flush=True)
