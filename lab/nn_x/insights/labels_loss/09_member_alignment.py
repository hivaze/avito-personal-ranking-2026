"""(2c) Is the training loss a good proxy for the metric AT THE MARGIN where our members differ?
For every train-only member with val + holdout predictions: mean listwise CE (raw logits, T=1, the training loss),
temperature-optimised CE (best global T, removes calibration/scale), and NDCG. Then: across members, Spearman of
CE vs NDCG; within seed pairs, per-feed correlation of dCE vs dNDCG. Writes out/09_member_alignment.txt"""
import sys, os, re, glob
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

OUT = "<repo>/nn_x/insights/labels_loss/out"
PR = "<repo>/preds"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

f, Mv, ov = feed_meta("val"); yv = np.asarray(np.load(FLAT / "val/cand_target.npy")).astype(np.float64)
hm = np.load(f"{PR}/holdout/holdout_meta.npy"); Mh, oh = hm[0].astype(np.int64), hm[1].astype(np.int64)
yh = np.load(f"{PR}/holdout/holdout_y.npy").astype(np.float64)

def ce_feed(s, y, o, M, T=1.0):
    s = s.astype(np.float64) / T; fid = np.repeat(np.arange(len(M)), M)
    mx = np.maximum.reduceat(s, o); e = np.exp(s - np.repeat(mx, M)); Z = np.add.reduceat(e, o)
    lp = s - np.repeat(mx + np.log(Z), M); K = np.add.reduceat(y, o)
    return -np.bincount(fid, weights=lp * y, minlength=len(M)) / K

names = sorted(set(re.sub(r"_holdout\.npy$", "", os.path.basename(x))[3:] for x in glob.glob(f"{PR}/holdout/nn_*_holdout.npy")))
rows = []
for n in names:
    fv = f"{PR}/nn_{n}_trainonly_val.npy"
    if not os.path.exists(fv): continue
    sv = np.load(fv); sh = np.load(f"{PR}/holdout/nn_{n}_holdout.npy")
    ndv = ndcg_per_feed(sv, yv, ov, Mv); ndh = ndcg_per_feed(sh, yh, oh, Mh)
    cv = ce_feed(sv, yv, ov, Mv); ch = ce_feed(sh, yh, oh, Mh)
    Ts = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    cvT = min(ce_feed(sv, yv, ov, Mv, T).mean() for T in Ts); chT = min(ce_feed(sh, yh, oh, Mh, T).mean() for T in Ts)
    rows.append((n, ndv.mean(), cv.mean(), cvT, ndh.mean(), ch.mean(), chT, ndv, cv, ndh, ch))
P(f"{len(rows)} train-only members with val + holdout predictions")
P("member | val ndcg | val CE | val CE(best T) | hold ndcg | hold CE | hold CE(best T)")
for r in sorted(rows, key=lambda r: -r[1]):
    P(f"{r[0]} | {r[1]:.5f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.5f} | {r[5]:.4f} | {r[6]:.4f}")
nd = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows]); cT = np.array([r[3] for r in rows])
ndh = np.array([r[4] for r in rows]); chh = np.array([r[5] for r in rows]); chT = np.array([r[6] for r in rows])
P(f"\nacross members: Spearman(val NDCG, -val CE) {spearmanr(nd, -c)[0]:+.3f}; with best-T CE {spearmanr(nd, -cT)[0]:+.3f}")
P(f"across members: Spearman(hold NDCG, -hold CE) {spearmanr(ndh, -chh)[0]:+.3f}; best-T {spearmanr(ndh, -chT)[0]:+.3f}")
b = np.polyfit(cT, nd, 1)[0]; bh = np.polyfit(chT, ndh, 1)[0]
P(f"slope dNDCG/dCE(best T): val {b:+.4f}, holdout {bh:+.4f} (NDCG per nat of CE)")
P(f"Spearman(val NDCG, hold NDCG) {spearmanr(nd, ndh)[0]:+.3f}; Spearman(val CE bestT, hold CE bestT) {spearmanr(cT, chT)[0]:+.3f}")
# per-feed alignment within seed pairs of the same family
fam = {}
for r in rows:
    k = re.sub(r"(_s|_r)\d+$", "", r[0]); fam.setdefault(k, []).append(r)
cs, csh = [], []
for k, rs in fam.items():
    for i in range(len(rs) - 1):
        a, bb = rs[i], rs[i + 1]
        dn = a[7] - bb[7]; dc = a[8] - bb[8]; m = dn != 0
        cs.append(pearsonr(dn, -dc)[0]); csh.append(pearsonr(a[9] - bb[9], -(a[10] - bb[10]))[0])
P(f"within-family seed pairs ({len(cs)}): per-feed corr(dNDCG, -dCE) val mean {np.mean(cs):+.3f}, holdout {np.mean(csh):+.3f}")
open(f"{OUT}/09_member_alignment.txt", "w").write("\n".join(lines) + "\n")
