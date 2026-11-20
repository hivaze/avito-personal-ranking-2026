"""Review R6: the LUO-future explanation of the TO age miscalibration predicts that the TO halves are CALIBRATED on
train-period rows (the 30K holdout has the same input semantics as training rows) and miscalibrated only on val.
Recompute act/exp by listing age (f4 -> age, feats_v2/life_/post_time.age_hours) for the G3 TO pool on val (own code)
and on the holdout (same 20 members, z-mean), temperature fitted per split."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import sys; sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feats_v2/life_")
from common import *
import post_time as ptm
P = Path("<repo>/preds"); knots = np.load(FEATS / "life_f4_age_knots.npy")
bins = [0, 6, 24, 72, 168, 720, 2160, 1e9]; lab = ['<6h', '6-24h', '1-3d', '3-7d', '7-30d', '30-90d', '>90d']
G3 = [f"v23_gru_pre2s_s{i}" for i in range(501, 507)] + [f"r6_cd_m02_r{i}" for i in range(101, 108)] + [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]
def zf(s, fid, M):
    s = s.astype(np.float64); n = len(M); mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9; return (s - mu[fid]) / sd[fid]
def table(score, y, fid, M, age, name, luo=None):
    n = len(M); npos = np.bincount(fid, y, n); keep = (npos > 0)[fid]
    best = None
    for T in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        e = np.exp(score * T); p = e / np.bincount(fid, e, n)[fid] * npos[fid]
        ll = np.sum((y * np.log(p / np.maximum(npos[fid], 1) + 1e-12))[keep])
        if best is None or ll > best[0]: best = (ll, T, p)
    p = best[2]; b = np.digitize(np.where(np.isnan(age), 1e8, age), bins) - 1
    row = [f"{y[(b == i) & keep].sum() / p[(b == i) & keep].sum():.3f}±{np.sqrt(p[(b == i) & keep].sum()) / p[(b == i) & keep].sum():.3f}" for i in range(7)]
    print(f"{name:22s} T={best[1]:.1f} " + " ".join(f"{r:>12s}" for r in row))
    if luo is not None:
        for nm, mk in [("LUO=0", luo == 0), ("LUO 1-3", (luo >= 1) & (luo <= 3)), ("LUO>3", luo > 3)]:
            print(f"{'   ' + nm:22s}       " + " ".join(f"{y[(b == i) & keep & mk].sum() / max(p[(b == i) & keep & mk].sum(), 1e-9):12.3f}" for i in range(7)))
print(f"{'':28s}" + " ".join(f"{l:>12s}" for l in lab))
# val
fv, Mv, offv = feed_meta("val"); fidv = np.repeat(np.arange(len(Mv)), Mv); yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
agev = ptm.age_hours(np.load(FLAT / "val/cand_f4.npy"), knots)
luov = np.load(FEATS / "val/it_hist_all_luo.npy")
zv = np.mean([zf(np.load(P / f"nn_{m}_trainonly_val.npy"), fidv, Mv) for m in G3], 0)
table(zv, yv, fidv, Mv, agev, "VAL G3 TO (own z-mean)", luov)
# holdout: reconstruct the 30K feed ids exactly as nn_x/r6/train.py does
ft, Mt, offt = feed_meta("train")
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz"); prn = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); prn[Mt < 50] = 0
cand = np.where((prn > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
hm = np.load(P / "holdout/holdout_meta.npy"); Mh = hm[0].astype(np.int64)
assert len(hold) == len(Mh) and (Mt[hold] == Mh).all(), "holdout reconstruction mismatch"
rows = np.concatenate([np.arange(offt[f], offt[f] + Mt[f]) for f in hold])
yh = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
assert np.allclose(yh, np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows])
fidh = np.repeat(np.arange(len(Mh)), Mh)
ageh = ptm.age_hours(np.load(FLAT / "train/cand_f4.npy", mmap_mode="r")[rows], knots)
luoh = np.load(FEATS / "train/it_hist_all_luo.npy", mmap_mode="r")[rows]
zh = np.mean([zf(np.load(P / f"holdout/nn_{m}_holdout.npy"), fidh, Mh) for m in G3], 0)
table(zh, yh, fidh, Mh, ageh, "HOLDOUT G3 TO", luoh)
print("\ncandidate age mix: " + " ".join(f"{lab[i]}: val {np.mean(np.digitize(np.nan_to_num(agev, nan=1e8), bins)-1 == i):.3f} hold {np.mean(np.digitize(np.nan_to_num(ageh, nan=1e8), bins)-1 == i):.3f}" for i in range(7)))
print("LUO zero-rate by age (val | hold): " + " ".join(f"{lab[i]} {np.mean(luov[np.digitize(np.nan_to_num(agev, nan=1e8), bins)-1 == i] == 0):.2f}|{np.mean(luoh[np.digitize(np.nan_to_num(ageh, nan=1e8), bins)-1 == i] == 0):.2f}" for i in range(7)))
