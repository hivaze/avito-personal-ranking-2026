"""Popularity reliance by holdout day: paired per-feed NDCG delta base (w5_e10_base s951-954) minus pop-free
(abl_e10_xdrop_pop s951-954) and minus pop1 (abl_e10_pop1 s951-954), member-paired by seed, on the holdout split by
train day (Sep 6-8 = pre2-missing regime) vs val (Tue / Thu) and the val private-mix.  Also the LUO cold share and the
LUO count level per holdout day.  CPU only."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
import pmix
ROOT = Path("<repo>"); OUT = Path(__file__).resolve().parent / "out"
mh = np.load(OUT / "hold_meta.npz"); M, off, y = mh["M"], mh["off"], mh["y"].astype(np.float64)
day = (mh["ts"].astype(np.int64) + 10800) // 86400; day = day - day.min()
R = np.load(OUT / "hold_raw.npy", mmap_mode="r"); luo = np.asarray(R[:, 3]); fid = np.repeat(np.arange(len(M)), M)
cold_f = np.bincount(fid, weights=(luo == 0).astype(float), minlength=len(M)) / M
lvl_f = np.bincount(fid, weights=np.log1p(luo), minlength=len(M)) / M
mv = np.load(OUT / "val_meta.npz"); Mv, offv, yv = mv["M"], mv["off"], mv["y"].astype(np.float64)
dv = (mv["ts"].astype(np.int64) + 10800) // 86400; dv = dv - dv.min()
Rv = np.load(OUT / "val_raw.npy", mmap_mode="r"); luov = np.asarray(Rv[:, 3]); fidv = np.repeat(np.arange(len(Mv)), Mv)
cold_v = np.bincount(fidv, weights=(luov == 0).astype(float), minlength=len(Mv)) / Mv
W = pmix.weights()["private"]


def per(fam, split):
    out = []
    for s in (951, 952, 953, 954):
        if split == "hold":
            out.append(ndcg_per_feed(np.load(ROOT / f"preds/holdout/nn_{fam}_s{s}_holdout.npy"), y, off, M))
        else:
            out.append(ndcg_per_feed(np.load(ROOT / f"preds/nn_{fam}_s{s}_trainonly_val.npy"), yv, offv, Mv))
    return np.mean(out, 0)


for alt in ["abl_e10_xdrop_pop", "abl_e10_pop1"]:
    dh = per("w5_e10_base", "hold") - per(alt, "hold"); dvv = per("w5_e10_base", "val") - per(alt, "val")
    print(f"\n== base minus {alt} (4 seed-paired members, per-feed NDCG) ==")
    print(f"holdout all {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f}); Sep6-8 {dh[day <= 2].mean():+.5f} ({dh[day<=2].std()/np.sqrt((day<=2).sum()):.5f}); "
          f"Sep9-19 {dh[day > 2].mean():+.5f} ({dh[day>2].std()/np.sqrt((day>2).sum()):.5f})")
    print("holdout by day: " + " ".join(f"d{d}:{dh[day == d].mean():+.4f}(n{(day == d).sum()},cold{cold_f[day == d].mean():.3f},lvl{lvl_f[day == d].mean():.2f})" for d in np.unique(day)))
    pm, pse = pmix.wmean_se(dvv, W)
    print(f"val all {dvv.mean():+.5f} ({dvv.std()/np.sqrt(len(dvv)):.5f}); Tue/Wed/Thu " + " ".join(f"{dvv[dv == d].mean():+.5f}" for d in np.unique(dv))
          + f"; val M>=50 {dvv[Mv >= 50].mean():+.5f}; pmix {pm:+.5f} ({pse:.5f}); val cold share by day " + " ".join(f"{cold_v[dv == d].mean():.3f}" for d in np.unique(dv)))
    # holdout restricted to val-like cold share (feed cold share >= 0.3)
    for lo in (0.3, 0.45):
        m = cold_f >= lo
        print(f"holdout feeds with cold share >= {lo}: n {m.sum()} delta {dh[m].mean():+.5f} ({dh[m].std()/np.sqrt(m.sum()):.5f}); val same cut {dvv[cold_v >= lo].mean():+.5f}")
