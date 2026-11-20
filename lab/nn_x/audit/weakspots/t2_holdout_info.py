"""T2: is the M>=50-only train holdout informative about val M<50 (41% of val, 39% of test)?
For every member with holdout + train-only val preds: val NDCG all / M<50 / M>=50, holdout NDCG. Correlations across
members, overall and within family (seed level, after removing the family mean)."""
import os, re, sys
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
P = "<repo>/preds"
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
hm = np.load(f"{P}/holdout/holdout_meta.npy"); hy = np.load(f"{P}/holdout/holdout_y.npy")
HM, HO = hm[0].astype(np.int64), hm[1].astype(np.int64)
rows = []
for fn in sorted(os.listdir(f"{P}/holdout")):
    m = re.match(r"nn_(.+)_holdout\.npy$", fn)
    if not m: continue
    v = m.group(1)
    vp = f"{P}/nn_{v}_trainonly_val.npy"
    if not os.path.exists(vp): continue
    s = np.load(vp).astype(np.float32)
    if s.shape[0] != len(y): continue
    nd = ndcg_per_feed(s, y, off, M)
    h = np.load(f"{P}/holdout/{fn}").astype(np.float32)
    ndh = ndcg_per_feed(h, hy, HO, HM)
    rows.append((v, nd.mean(), nd[M < 50].mean(), nd[M >= 50].mean(), nd[M < 20].mean(), ndh.mean()))
    print(f"{v:32s} val {nd.mean():.5f} M<50 {nd[M < 50].mean():.5f} M>=50 {nd[M >= 50].mean():.5f} M<20 {nd[M < 20].mean():.5f} hold {ndh.mean():.5f}", flush=True)
names = np.array([r[0] for r in rows]); A = np.array([r[1:] for r in rows])
out = []
def say(s): print(s, flush=True); out.append(s)
say(f"members with holdout + train-only val preds: {len(rows)}")
from scipy.stats import spearmanr, pearsonr
cols = ["val", "val_M<50", "val_M>=50", "val_M<20", "holdout"]
say("ACROSS ALL MEMBERS (Spearman):")
for i in range(4):
    say(f"  rho(holdout, {cols[i]}) = {spearmanr(A[:, 4], A[:, i]).statistic:+.3f}   rho(val_M>=50, {cols[i]}) = {spearmanr(A[:, 2], A[:, i]).statistic:+.3f}")
# family = name with the trailing seed/root token stripped
def fam(v):
    v = re.sub(r"_(s|r)\d+(_cs\d+)?$", "", v); v = re.sub(r"@\d+$", "", v); return v
fams = np.array([fam(v) for v in names])
say("WITHIN-FAMILY (seed level, family mean removed; families with >= 4 members):")
dev = np.zeros_like(A); keep = np.zeros(len(A), bool)
for g in np.unique(fams):
    s = fams == g
    if s.sum() >= 4:
        dev[s] = A[s] - A[s].mean(0); keep[s] = True
        say(f"  {g:28s} n={s.sum()} seed-sd val {A[s, 0].std():.5f} M<50 {A[s, 1].std():.5f} M>=50 {A[s, 2].std():.5f} hold {A[s, 4].std():.5f} | rho(hold,val) {spearmanr(A[s, 4], A[s, 0]).statistic:+.2f} rho(hold,M<50) {spearmanr(A[s, 4], A[s, 1]).statistic:+.2f} rho(hold,M>=50) {spearmanr(A[s, 4], A[s, 2]).statistic:+.2f} rho(M<50,M>=50) {spearmanr(A[s, 1], A[s, 2]).statistic:+.2f}")
D = dev[keep]
say(f"pooled within-family (n={keep.sum()}): rho(hold, val) {pearsonr(D[:, 4], D[:, 0]).statistic:+.3f}, rho(hold, val M<50) {pearsonr(D[:, 4], D[:, 1]).statistic:+.3f}, rho(hold, val M>=50) {pearsonr(D[:, 4], D[:, 2]).statistic:+.3f}, rho(val M<50, val M>=50) {pearsonr(D[:, 1], D[:, 2]).statistic:+.3f}, rho(hold, val M<20) {pearsonr(D[:, 4], D[:, 3]).statistic:+.3f}")
np.save("<repo>/nn_x/audit/weakspots/t2_member_table.npy", np.array(rows, dtype=object), allow_pickle=True)
open("<repo>/nn_x/audit/weakspots/t2_holdout_info.txt", "w").write("\n".join(out) + "\n")
