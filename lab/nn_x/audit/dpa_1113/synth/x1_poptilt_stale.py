"""Synthesiser check: does the POP-TILT test-time tilt (s = pool - w * z_feed(log1p seq_cv_pop)) survive the
stale-val lens (the project rule for cold/warm trades)? TO half only (stale preds exist for TO members only).
Live:  G3 TO pool from the 20 *_base stale-run files (== stored train-only val preds) and live xt_seq_cv_pop.
Stale: G3 TO pool from the 20 *_both files and the 'both' counterfactual seq_cv_pop (nn_x/r6/stale_cf).
CPU only, read-only on project files, writes only stdout."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402
import pmix  # noqa: E402

D = "/tmp/scratchpad/dpa"
ST = f"{D}/validation/out/stale"
z = np.load(f"{D}/nonlinear/val.npz")
M = z["M"].astype(np.int64); off = z["off"].astype(np.int64); y = z["y"].astype(np.float64)
nf = len(M); fid = np.repeat(np.arange(nf), M)
cold = z["cold"].astype(np.float64)


def fz(x):
    x = np.asarray(x, np.float64)
    m = np.bincount(fid, x, minlength=nf) / M
    v = np.maximum(np.bincount(fid, x * x, minlength=nf) / M - m ** 2, 0)
    s = np.sqrt(v)
    return (x - m[fid]) / np.where(s[fid] > 1e-9, s[fid], 1.0)


F10 = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)]
F7x = [f"r6_cd_m02_r{s}" for s in range(101, 108)]
F7x70 = [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]
G3 = F10 + F7x + F7x70
pool = {}
for sc in ("base", "both"):
    pool[sc] = np.mean([fz(np.load(f"{ST}/nn_{m}_{sc}.npy")) for m in G3], 0)
g_rec = np.load(f"{D}/nonlinear/val.npz")["gto"]
print(f"corr(base TO pool, nonlinear gto) {np.corrcoef(pool['base'], g_rec)[0, 1]:.6f}")
x_live = fz(z["xt_seq_cv_pop"])
x_stale = fz(np.log1p(np.maximum(np.load("<repo>/nn_x/r6/stale_cf/both_seq_cv_pop.npy").astype(np.float64), 0)))
print(f"corr(live z, stale z) {np.corrcoef(x_live, x_stale)[0, 1]:.4f}")
wpriv = pmix.weights()["private"]

y_pos = np.bincount(fid, y, minlength=nf); ycold = np.bincount(fid, y * cold, minlength=nf)
has = y_pos > 0; acp = has & (ycold == y_pos); awp = has & (ycold == 0)


def rd(name, s, base):
    d = ndcg_per_feed(s, y, off, M) - ndcg_per_feed(base, y, off, M)
    se = d.std(ddof=1) / np.sqrt(nf)
    pm = (wpriv * d).sum() / wpriv.sum()
    pse = np.sqrt(((wpriv * (d - pm)) ** 2).sum()) / wpriv.sum()
    print(f"{name:34s} {d.mean():+.5f} +- {se:.5f} (z {d.mean() / se:+.2f}) | pmix {pm:+.5f} +- {pse:.5f} | "
          f"all-cold-pos {d[acp].mean():+.5f} all-warm-pos {d[awp].mean():+.5f}")


b = ndcg_per_feed(pool["base"], y, off, M).mean(); s = ndcg_per_feed(pool["both"], y, off, M).mean()
print(f"TO pool live {b:.5f} stale {s:.5f} (drop {s - b:+.5f})")
for w in (-0.01, -0.02, -0.03):
    rd(f"live  w {w:+.2f}", pool["base"] + w * x_live, pool["base"])
    rd(f"stale w {w:+.2f} (stale x)", pool["both"] + w * x_stale, pool["both"])
    rd(f"stale w {w:+.2f} (live x)", pool["both"] + w * x_live, pool["both"])
