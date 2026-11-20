"""Correlations of f1..f5 with label-free and label-based reference features (val, all rows):
global Spearman and pooled within-feed rank correlation (feed-centred percentile ranks).
Plus: f5 vs item CTR / impressions / history counts (rate or count?), f5 for fresh listings, f5 floor composition,
f3/f1 explained by visible-history category stats (GBM R^2), f2 vs the user's history share in the candidate's region (K2 component)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
from scipy.stats import spearmanr, rankdata

REF = ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et4_luo", "tte_it_imp", "tte_it_ctr", "tte_it_coec", "it_cand_cnt_split",
       "cat_share_all", "cat_cnt_all", "cat_cnt_20", "cat_min_hpos", "feed_cat_share", "cos_hall", "cos_h10", "cos_u", "maxsim32", "item_norm", "centrality", "H"]
d = load("val", None, extra_feats=REF)
feed = d["feed"]

def wpct(v):
    o = np.lexsort((v, feed))
    r = np.empty(len(v)); r[o] = np.arange(len(v))
    # average ties
    rr = rankdata(np.c_[feed, v].view([("a", feed.dtype), ("b", v.dtype)]).ravel() if False else 0) if False else None
    fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
    start = np.flatnonzero(fs)[np.cumsum(fs) - 1]
    rk = np.arange(len(o)) - start
    # tie-average within feed
    vs = v[o]; key_break = np.r_[True, (feed[o][1:] != feed[o][:-1]) | (vs[1:] != vs[:-1])]
    gid = np.cumsum(key_break) - 1
    avg = (np.bincount(gid, rk) / np.bincount(gid))[gid]
    M = np.bincount(feed)[feed[o]]
    out = np.empty(len(v)); out[o] = (avg + 0.5) / M - 0.5
    return out

W = {k: wpct(d[k].astype(np.float64)) for k in FS}
print("Spearman: global | pooled within-feed (val, all rows; NaN rows dropped)")
print("ref".ljust(20) + "".join(f"{k:>18}" for k in FS))
for r in ["f1", "f2", "f3", "f4", "f5"] + REF:
    x = d[r].astype(np.float64)
    ok = ~np.isnan(x)
    wx = wpct(np.where(ok, x, -1e9))
    cells = []
    for k in FS:
        g = spearmanr(x[ok], d[k][ok]).correlation
        w = np.corrcoef(wx[ok], W[k][ok])[0, 1] if r != "H" else np.nan
        cells.append(f"{g:+.3f} | {w:+.3f}".rjust(18))
    print(r.ljust(20) + "".join(cells))

# f5: rate or count? items with train coverage
imp = d["tte_it_imp"]; ctr = d["tte_it_ctr"]; hi = d["it_hist_all_luo"]
cov = imp > 0
print(f"\nf5 vs item stats on val rows with train impressions (n={cov.sum()}, {cov.mean():.3f} of rows):")
for nm, x in [("tte_it_ctr (smoothed train CTR)", ctr), ("tte_it_imp (train impressions)", imp), ("it_hist_all_luo (history appearances)", hi), ("f4 (age)", d["f4"])]:
    print(f"   Spearman(f5, {nm}) = {spearmanr(d['f5'][cov], x[cov]).correlation:+.3f}")
# raw empirical CTR on random-like-free basis: positives / impressions with >= 50 impressions
pos_ = np.load(FEATS / "val" / "tte_it_pos.npy")[d["rows"]]
m = imp >= 50
print(f"   items with >= 50 train impressions (rows {m.sum()}): Spearman(f5, pos/imp) {spearmanr(d['f5'][m], (pos_/imp)[m]).correlation:+.3f}, Spearman(f5, imp) {spearmanr(d['f5'][m], imp[m]).correlation:+.3f}")
# partial: f5 vs ctr controlling for imp (rank-residual)
def resid(a, b):
    ra = rankdata(a); rb = rankdata(b); beta = np.polyfit(rb, ra, 1); return ra - np.polyval(beta, rb)
print(f"   partial Spearman(f5, CTR | imp) {np.corrcoef(resid(d['f5'][m], imp[m]), resid((pos_/imp)[m], imp[m]))[0,1]:+.3f}; partial (f5, imp | CTR) {np.corrcoef(resid(d['f5'][m], (pos_/imp)[m]), resid(imp[m], (pos_/imp)[m]))[0,1]:+.3f}")
# f5 by age (f4 -> hours via knots)
kn = np.load(FEATS / "life_f4_age_knots.npy")
print(f"   age knots file shape {kn.shape}")
f4 = d["f4"].astype(np.float64)
for lo, hi_ in [(0, 2/64), (2/64, 9/64), (9/64, 20/64), (20/64, 40/64), (40/64, 1.01)]:
    mm = (f4 >= lo) & (f4 < hi_)
    f5m = d["f5"][mm]
    print(f"   f4 in [{lo:.3f},{hi_:.3f}) rows {mm.sum()}: f5 mean {f5m.mean():.3f}, f5 floor(1/62) share {np.mean(np.abs(f5m-1/62)<1e-6):.3f}, f5==0 share {np.mean(f5m==0):.3f}, f5 sd {f5m.std():.3f}; hist count 0 share {np.mean(hi[mm]==0):.3f}")
fl = np.abs(d["f5"] - 1/62) < 1e-6; z0 = d["f5"] == 0
print(f"   f5 floor rows: hist count 0 share {np.mean(hi[fl]==0):.3f} (all rows {np.mean(hi==0):.3f}); train imp 0 share {np.mean(imp[fl]==0):.3f} (all {np.mean(imp==0):.3f}); target rate {d['target'][fl].mean()*100:.2f}% vs {d['target'].mean()*100:.2f}%")
print(f"   f5 == 0 rows: hist count 0 share {np.mean(hi[z0]==0):.3f}; train imp 0 share {np.mean(imp[z0]==0):.3f}; target rate {d['target'][z0].mean()*100:.2f}%; mean f4 {f4[z0].mean():.3f}")

# f3 / f1 explained by visible-history category stats
import lightgbm as lgb
hist_cols = ["cat_share_all", "cat_cnt_all", "cat_cnt_20", "cat_min_hpos", "H"]
et = [f"cat_cnt_et{i}" for i in range(1, 11)]
dd = load("val", None, extra_feats=et)
X = np.c_[np.stack([d[c] for c in hist_cols], 1), np.stack([dd[c] for c in et], 1), d["cat"]]
# one row per (feed, cat) for f3
key = feed.astype(np.int64) * 64 + d["cat"]
_, first = np.unique(key, return_index=True)
users = d["user_id"][first]
rng = np.random.default_rng(0)
ug = np.unique(users); tr_u = set(rng.choice(ug, len(ug) // 2, replace=False))
trm = np.array([u in tr_u for u in users])
for tgt in ["f3", "f1"]:
    rows = first if tgt == "f3" else np.arange(len(feed))
    trm_r = trm if tgt == "f3" else np.isin(d["user_id"], list(tr_u))
    Xr = X[rows]; yr = d[tgt][rows]
    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, n_jobs=12)
    mdl.fit(Xr[trm_r], yr[trm_r], categorical_feature=[X.shape[1] - 1])
    p = mdl.predict(Xr[~trm_r])
    r2 = 1 - np.mean((yr[~trm_r] - p) ** 2) / np.var(yr[~trm_r])
    print(f"\n{tgt} ~ visible-history category stats (+category id), held-out users: R^2 {r2:.3f}, Spearman {spearmanr(p, yr[~trm_r]).correlation:.3f}")
    # by H bucket
    Hh = d["H"][rows][~trm_r]
    for lo, hi_ in [(0, 1), (1, 50), (50, 256), (256, 257)]:
        mm = (Hh >= lo) & (Hh < hi_)
        if mm.sum() > 100:
            print(f"   H in [{lo},{hi_}): R^2 {1 - np.mean((yr[~trm_r][mm] - p[mm])**2) / np.var(yr[~trm_r][mm]):.3f} (n={mm.sum()})")
    # residual vs user0
    u0 = d["user_id"][rows][~trm_r] == 0
    print(f"   user0 rows: R^2 {1 - np.mean((yr[~trm_r][u0] - p[u0])**2) / np.var(yr[~trm_r][u0]):.3f} (n={u0.sum()})")
# category absent from history but f3 above floor
absent = d["cat_cnt_all"] == 0
print(f"\nrows whose category is absent from the visible history: {absent.mean():.3f}; of them f3 above floor {np.mean(np.abs(d['f3'][absent]-1/62)>1e-6):.3f}, f1 above floor {np.mean(np.abs(d['f1'][absent]-1/56)>1e-6):.3f}; target rate f3>floor {d['target'][absent & (np.abs(d['f3']-1/62)>1e-6)].mean()*100:.2f}% vs floor {d['target'][absent & (np.abs(d['f3']-1/62)<1e-6)].mean()*100:.2f}%")
present = ~absent
print(f"rows whose category IS in the history: f3 at floor {np.mean(np.abs(d['f3'][present]-1/62)<1e-6):.3f}, f1 at floor {np.mean(np.abs(d['f1'][present]-1/56)<1e-6):.3f}")
