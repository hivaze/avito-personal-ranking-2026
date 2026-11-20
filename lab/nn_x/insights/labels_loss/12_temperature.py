"""Calibration of the listwise softmax on val/holdout per member: best global temperature T* (grid) and the CE gap
CE(T=1) - CE(T*). T* > 1 = over-confident logits (a memorisation signature), T* < 1 = under-confident.
Grouped by recipe family (c0 = mix 0.1 re-cool, m02, m0, ...). Writes out/12_temperature.txt"""
import sys, os, re, glob
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
PR = "<repo>/preds"; OUT = "<repo>/nn_x/insights/labels_loss/out"
f, Mv, ov = feed_meta("val"); yv = np.asarray(np.load(FLAT / "val/cand_target.npy")).astype(np.float64)
hm = np.load(f"{PR}/holdout/holdout_meta.npy"); Mh, oh = hm[0].astype(np.int64), hm[1].astype(np.int64)
yh = np.load(f"{PR}/holdout/holdout_y.npy").astype(np.float64)
def ce(s, y, o, M, T):
    s = s.astype(np.float64) / T; fid = np.repeat(np.arange(len(M)), M)
    mx = np.maximum.reduceat(s, o); e = np.exp(s - np.repeat(mx, M)); Z = np.add.reduceat(e, o)
    lp = s - np.repeat(mx + np.log(Z), M); K = np.add.reduceat(y, o)
    return (-np.bincount(fid, weights=lp * y, minlength=len(M)) / K).mean()
TS = np.arange(0.80, 1.41, 0.04)
pat = ["r6_cd_c0_r10", "r6_cd_m02_r10", "r6_cd_m02_p70k_r10", "r6_cd_m0_r10", "r6_cd_d0_r10", "r6_cd_d20_r10",
       "r6_cd_w10_r10", "r6_cd_pp02_r10", "r6_cd_aa5_r10", "r6_cd_aug_r10", "r6_cd_cf4_r10", "r6_cd_e3_r10",
       "v17_pre2s_s10", "v28_lr1_m02_s41", "v22_pre2s_lr1_s40", "v23_gru_pre2s_s50", "v27_gru_m02_s51", "v29_d192_m02_s71",
       "v26_d192_pre2s_s70", "v30_f7_m02_s90"]
out = []
for pfx in pat:
    files = sorted(glob.glob(f"{PR}/holdout/nn_{pfx}*_holdout.npy"))
    files = [x for x in files if "@" not in x and "cs813" not in x]
    rows = []
    for fh in files:
        n = os.path.basename(fh)[3:-12]
        fvp = f"{PR}/nn_{n}_trainonly_val.npy"
        if not os.path.exists(fvp): continue
        sv, sh = np.load(fvp), np.load(fh)
        cv = np.array([ce(sv, yv, ov, Mv, T) for T in TS]); ch = np.array([ce(sh, yh, oh, Mh, T) for T in TS])
        rows.append((TS[cv.argmin()], cv[np.abs(TS - 1).argmin()] - cv.min(), TS[ch.argmin()], ch[np.abs(TS - 1).argmin()] - ch.min()))
    if rows:
        r = np.array(rows)
        s = (f"{pfx:22s} n {len(rows)} | val T* {r[:,0].mean():.2f} gap {r[:,1].mean():.4f} | hold T* {r[:,2].mean():.2f} "
             f"gap {r[:,3].mean():.4f}")
        print(s, flush=True); out.append(s)
open(f"{OUT}/12_temperature.txt", "w").write("\n".join(out) + "\n")
