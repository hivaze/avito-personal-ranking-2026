"""Review R8: re-check 'fine-tuning 3-5 members reproduces the FT branch of all 20' (ensemble a4_ft_fewer) on G3, for
the standard 0.5/0.5 blend and for the M-split blend; FT branch from k random members, TO branch from all 20.
Reports mean delta over 12 random subsets and the WORST subset (risk of an unlucky pick)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import sys; sys.path.insert(0, "<repo>/src")
from common import *
P = Path("<repo>/preds")
fv, Mv, offv = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy"); fid = np.repeat(np.arange(len(Mv)), Mv)
def z(s):
    s = s.astype(np.float64); n = len(Mv); mu = np.bincount(fid, s, n) / Mv
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / Mv - mu ** 2, 0)) + 1e-9; return (s - mu[fid]) / sd[fid]
G3 = [f"v23_gru_pre2s_s{i}" for i in range(501, 507)] + [f"r6_cd_m02_r{i}" for i in range(101, 108)] + [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]
FT = np.stack([z(np.load(P / f"nn_{m}_n2cpv3_val_oof.npy")).astype(np.float32) for m in G3])
TO = z(np.mean([z(np.load(P / f"nn_{m}_trainonly_val.npy")) for m in G3], 0))
small = (Mv < 50)[fid]
def score(ftb, msplit):
    fin = 0.5 * z(ftb) + 0.5 * TO
    if msplit: fin = np.where(small, TO, fin)
    return ndcg_per_feed(fin.astype(np.float32), yv, offv, Mv)
rng = np.random.default_rng(0)
for msplit in (False, True):
    full = score(FT.mean(0), msplit)
    print(f"{'M-split' if msplit else '0.5/0.5'}: full-FT pool {full.mean():.5f}")
    for k in (1, 2, 3, 4, 5, 8):
        ds = []
        for r in range(12):
            idx = rng.choice(20, k, replace=False)
            ds.append((score(FT[idx].mean(0), msplit) - full).mean())
        ds = np.array(ds)
        print(f"   k={k}: mean {ds.mean():+.5f}  sd over subsets {ds.std():.5f}  worst {ds.min():+.5f}")
