"""Skeptic 'duplicates', new check for lb_stats#5 / pool_composition#12 (HOLD-SHORT) and for a NATURAL short-feed witness.
Question 1 (premise of HOLD-SHORT): is a random m-subset of a long held-out exploration feed as hard as a NATURAL short exploration feed?
Question 2: do out-of-sample predictions for natural short exploration-like train feeds already exist?  Yes: teacher_oof.npy holds, for every
  train feed, the mean logit of the 3 hard-label fold models that never saw its user (E_kd_predeclared.txt 12:20Z), and the XPOOL detector
  (nn_x/w12/xdet/out/prand_v2.npz) marks exploration-like feeds of every size.
Scorer everywhere = a 3-member hard-label fold pool (same strength):
  NAT  = train feeds with p_random > 0.9 and M < 50 (not in the cooldown pool, which is M >= 50), scored by teacher_oof (out-of-fold by user);
  SYN  = the 30,000 holdout feeds (M >= 50), sub-sampled to a size m drawn from NAT's M distribution (uniform subset, redrawn until >= 1 positive),
         scored by the stored full-feed teacher_oof logits restricted to the subset;
  VAL  = val feeds with M < 50 scored by the xf0 fold pool (3 members), later period.
CPU only, read-only."""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/recheck/forensics_skeptic")
from hx import *
f, hold = hold_feeds(); Mtr = f["M"].to_numpy().astype(np.int64); co = f["cand_off"].to_numpy().astype(np.int64); uid = f["user_id"].to_numpy()
pr = np.load(R / "nn_x/w12/xdet/out/prand_v2.npz"); assert (pr["fidx"] == np.arange(f.height)).all(); p = np.nan_to_num(pr["p_random"], nan=0.0)
T = np.load(R / "nn_x/w12/kd/out/teacher_oof.npy", mmap_mode="r"); Y = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
isH = np.zeros(f.height, bool); isH[hold] = True
nat = np.where((p > 0.9) & (Mtr < 50) & ~isH)[0]
def gather(idx):
    Ms = Mtr[idx]; rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co[idx], Ms)]); off = np.r_[0, np.cumsum(Ms)[:-1]]
    return np.asarray(T[rows], np.float32), np.asarray(Y[rows], np.float32), Ms, off
sN, yN, MN, oN = gather(nat); eN = ndcg_per_feed(sN, yN, oN, MN)
sH, yH, MH, oH = gather(hold); eHfull = ndcg_per_feed(sH, yH, oH, MH)
rng = np.random.default_rng(20261114); S = []; Yy = []; Ms = []
for i in range(len(MH)):
    s = sH[oH[i]:oH[i] + MH[i]]; y = yH[oH[i]:oH[i] + MH[i]]
    for _ in range(200):
        m = int(MN[rng.integers(len(MN))]); k = rng.choice(MH[i], m, replace=False)
        if y[k].sum() > 0: break
    else: continue
    S.append(s[k]); Yy.append(y[k]); Ms.append(m)
Ms = np.array(Ms); oS = np.r_[0, np.cumsum(Ms)[:-1]]; sS = np.concatenate(S); yS = np.concatenate(Yy); eS = ndcg_per_feed(sS, yS, oS, Ms)
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
xv = np.mean([zfeed(np.load(W / f"nn_w5_e10_xf0_s{s}_trainonly_val.npy"), ov, Mv) for s in (951, 952, 953)], 0); eV = ndcg_per_feed(xv, yv, ov, Mv)
npos = lambda y, off, M: np.add.reduceat(y, off)
print(f"NAT feeds {len(MN)} (users {len(np.unique(uid[nat]))}, guest feeds {(uid[nat] == 0).sum()}), SYN feeds {len(Ms)} of {len(MH)}, VAL M<50 feeds {(Mv < 50).sum()}")
print(f"holdout full feeds (M>=50) by teacher_oof: {eHfull.mean():.5f}  (E_kd_predeclared: OOF teacher on the holdout 0.32100)")
print(f"{'M bucket':10s} | {'NAT n':>7s} {'pos/feed':>8s} {'NDCG':>18s} | {'SYN n':>7s} {'pos/feed':>8s} {'NDCG':>18s} | {'SYN-NAT':>18s} | {'VAL n':>6s} {'pos/feed':>8s} {'NDCG':>18s}")
pN, pS, pV = npos(yN, oN, MN), npos(yS, oS, Ms), npos(yv, ov, Mv)
for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50), (5, 50)):
    a = (MN >= lo) & (MN < hi); b = (Ms >= lo) & (Ms < hi); c = (Mv >= lo) & (Mv < hi)
    dse = np.sqrt(se(eN[a]) ** 2 + se(eS[b]) ** 2)
    print(f"{lo:2d}-{hi - 1:<7d} | {a.sum():7d} {pN[a].mean():8.3f} {eN[a].mean():.5f}+-{se(eN[a]):.5f} | {b.sum():7d} {pS[b].mean():8.3f} {eS[b].mean():.5f}+-{se(eS[b]):.5f} | "
          f"{eS[b].mean() - eN[a].mean():+.5f}+-{dse:.5f} | {c.sum():6d} {pV[c].mean():8.3f} {eV[c].mean():.5f}+-{se(eV[c]):.5f}")
# M-standardised comparison (weights = NAT's M histogram) so the size mix cannot explain a gap
w = np.bincount(MN, minlength=50)[:50].astype(float); w /= w.sum()
mN = np.array([eN[MN == m].mean() if (MN == m).any() else np.nan for m in range(50)]); mS = np.array([eS[Ms == m].mean() if (Ms == m).any() else np.nan for m in range(50)])
mV = np.array([eV[Mv == m].mean() if (Mv == m).any() else np.nan for m in range(50)]); ok = ~np.isnan(mN) & ~np.isnan(mS) & ~np.isnan(mV)
print(f"M-standardised (NAT's M mix): NAT {np.nansum(w[ok] * mN[ok]) / w[ok].sum():.5f} | SYN {np.nansum(w[ok] * mS[ok]) / w[ok].sum():.5f} | VAL M<50 {np.nansum(w[ok] * mV[ok]) / w[ok].sum():.5f}")
# how many NAT feeds belong to fold-0 users is not computed here (fold hash lives in train.py); by construction about 20%.
print(f"p_random > 0.5 instead of 0.9 would give {int(((p > 0.5) & (Mtr < 50) & ~isH).sum())} NAT feeds")
