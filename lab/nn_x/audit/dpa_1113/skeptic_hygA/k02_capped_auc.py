"""Skeptic check: does the uncapped value of the 4 attention-mass inputs order candidates WITHIN the capped (>=10) group?
For each split (val, hold) and each column: feeds with >=1 positive and >=1 negative among rows with x>=10 ->
pairwise within-feed AUC among capped rows of (a) raw uncapped x, (b) the G3 score (val) / G3-TO member mean (hold).
Also: share of all val positives that sit in such contested capped groups. CPU only, reads dpa/pipeline/out read-only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from pathlib import Path
ROOT = Path("<repo>")
IN = Path("/tmp/scratchpad/dpa/pipeline/out")
XT = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)["cfg"]["xt_names"]
xi = {n: j for j, n in enumerate(XT)}
MASS = ["ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]

def zf(a, off, M):
    out = np.empty(len(a))
    for o, m in zip(off, M):
        s = a[o:o + m]; sd = s.std(); out[o:o + m] = (s - s.mean()) / (sd if sd > 0 else 1)
    return out

def grp_auc(score, y, x, off, M):
    num = den = 0.0; nf = 0; npos_c = 0
    for o, m in zip(off, M):
        xx = x[o:o + m]; c = xx >= 10
        if c.sum() < 2: continue
        yy = y[o:o + m][c]; ss = score[o:o + m][c]
        p = yy > 0; n = ~p
        if p.sum() == 0 or n.sum() == 0: continue
        nf += 1; npos_c += p.sum()
        sp = ss[p][:, None]; sn = ss[n][None, :]
        num += (sp > sn).sum() + 0.5 * (sp == sn).sum(); den += p.sum() * n.sum()
    return num / den, nf, npos_c, den

for sub in ["val", "hold"]:
    R = np.load(IN / f"{sub}_raw.npy", mmap_mode="r"); mt = dict(np.load(IN / f"{sub}_meta.npz"))
    M, off, y = mt["M"], mt["off"], mt["y"].astype(np.float64)
    if sub == "val":
        b = zf(0.5 * zf(np.load(ROOT / "preds/ens_G3_ft_val.npy"), off, M) + 0.5 * zf(np.load(ROOT / "preds/ens_G3_to_val.npy"), off, M), off, M)
    else:
        MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
        b = zf(np.mean([zf(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), off, M) for m in MEM], 0), off, M)
    totpos = y.sum()
    for n in MASS:
        x = np.nan_to_num(np.asarray(R[:, xi[n]], np.float64), nan=0.0)
        a_x, nf, npc, npairs = grp_auc(x, y, x, off, M)
        a_b, _, _, _ = grp_auc(b, y, x, off, M)
        # control: same AUC of raw x within the 5..10 band (uncapped region) for comparison
        print(f"{sub:4s} {n:14s} contested capped feeds {nf:5d} | positives in them {int(npc):5d} = {npc/totpos:.4f} of all positives | pairs {int(npairs):7d} | "
              f"AUC within capped group: uncapped x {a_x:.4f}  G3 score {a_b:.4f}", flush=True)
