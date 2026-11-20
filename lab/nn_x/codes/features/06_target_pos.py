"""Target relation and production-ranker test for f1..f5.
(1) per-feed AUC of each +f alone: val; train random-like (P>0.9, M>=50, 60K feeds); train production-like (P<0.05, M>=50, 60K feeds).
(2) target rate by GLOBAL decile of f and by WITHIN-FEED percentile decile (monotone? U-shaped?), val and train random-like.
(3) relationship with the original position pos: mean per-feed Spearman(f, pos) on production-like vs random-like feeds,
    and the share of production-like feeds where |Spearman| > 0.9 (a production ranker score would be ~ -1)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
from scipy.stats import rankdata

def within_pct(v, feed):
    o = np.lexsort((np.random.default_rng(0).random(len(v)), v, feed))
    fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
    rk = np.arange(len(o)) - np.flatnonzero(fs)[np.cumsum(fs) - 1]
    M = np.bincount(feed)
    p = np.empty(len(v)); p[o] = rk / np.maximum(M[feed[o]] - 1, 1)
    return p

def per_feed_spearman(a, b, feed):
    # average ranks within feed via rankdata on composite key
    ra = np.empty(len(a)); rb = np.empty(len(b))
    o = np.argsort(feed, kind="stable")
    fs = feed[o]; starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]]); ends = np.r_[starts[1:], len(fs)]
    out = np.full(len(starts), np.nan)
    A = a[o]; B = b[o]
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e - s < 10: continue
        x = rankdata(A[s:e]); y = rankdata(B[s:e])
        if x.std() == 0 or y.std() == 0: continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out

sets = [("val", None), ("train_rl", "rl"), ("train_prod", "prod")]
D = {}
for name, kind in sets:
    fidx = None if kind is None else train_sample(kind, 60_000, 3)
    D[name] = load("val" if kind is None else "train", fidx)

print("(1) per-feed AUC of +f (ties 0.5); se in brackets")
for name, d in D.items():
    print(f"  {name}: " + "  ".join(f"{k} {feed_auc(d[k].astype(np.float64), d['target'], d['feed'])[0]:.4f}({feed_auc(d[k].astype(np.float64), d['target'], d['feed'])[1]:.4f})" for k in FS))

print("\n(2) target rate / mean rate by decile (low -> high)")
for name in ["val", "train_rl", "train_prod"]:
    d = D[name]; y = d["target"]; base = y.mean()
    for k in FS:
        v = d[k].astype(np.float64)
        qs = np.quantile(v, np.linspace(0, 1, 11))
        g = np.clip(np.searchsorted(qs[1:-1], v, side="right"), 0, 9)
        rg = np.bincount(g, y, minlength=10) / np.maximum(np.bincount(g, minlength=10), 1) / base
        p = within_pct(v, d["feed"])
        w = np.clip((p * 10).astype(int), 0, 9)
        rw = np.bincount(w, y, minlength=10) / np.maximum(np.bincount(w, minlength=10), 1) / base
        print(f"  {name} {k} global  : " + " ".join(f"{x:.2f}" for x in rg))
        print(f"  {name} {k} in-feed : " + " ".join(f"{x:.2f}" for x in rw))

print("\n(3) per-feed Spearman(f, pos) [pos 0 = top; a production score would be strongly NEGATIVE]")
for name in ["train_prod", "train_rl"]:
    d = D[name]
    for k in FS + ["target"]:
        s = per_feed_spearman(d[k].astype(np.float64), d["pos"].astype(np.float64), d["feed"])
        s = s[~np.isnan(s)]
        print(f"  {name} {k}: mean {s.mean():+.4f} sd {s.std():.3f} | share feeds rho < -0.5 {np.mean(s<-0.5):.4f}, < -0.9 {np.mean(s<-0.9):.4f} | feeds {len(s)}")
# NDCG-like check: AUC of -pos for target on production feeds vs AUC of f
d = D["train_prod"]
print(f"\n  train_prod AUC of -pos for target: {feed_auc(-d['pos'].astype(np.float64), d['target'], d['feed'])[0]:.4f};  train_rl: {feed_auc(-D['train_rl']['pos'].astype(np.float64), D['train_rl']['target'], D['train_rl']['feed'])[0]:.4f}")
# within production feeds, is f's AUC explained by position? AUC of f within pos strata
for k in FS:
    v = d[k].astype(np.float64); pos = d["pos"]
    out = []
    for lo, hi in [(0, 10), (10, 50), (50, 400)]:
        m = (pos >= lo) & (pos < hi)
        out.append(f"pos[{lo},{hi}) {feed_auc(v[m], d['target'][m], d['feed'][m])[0]:.4f}")
    print(f"  train_prod {k} AUC within position strata: " + ", ".join(out))
