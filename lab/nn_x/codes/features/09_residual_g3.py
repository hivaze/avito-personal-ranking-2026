"""Does the current pick (G3 ensemble, val predictions = OOF fine-tune half + train-only half) already use the decoded structure?
Candidate decoded features (label-free, val):
  f2_mode    1 if the row's f2 equals the feed's modal non-zero f2 level (= the user's dominant region)
  f2_zero    1 if f2 == 0 (region the user never touched)
  f2_lvlrank rank of the row's f2 among the feed's distinct non-zero levels (0 = lowest, 1 = highest), 0.5 for f2 == 0
  f1_in_cat  within-(feed, category) percentile of f1 (finer preference inside the category)
  f1_m_f3    f1 - f3
  f5_nodata  f5 == 0 ; f5_floor f5 == 1/62
Readouts: (a) act/exp of each flag within G3 within-feed deciles; (b) nested 2-fold-by-user weight choice for z(G3) + w * x,
delta NDCG with paired SE. Pre-declared: a feature is 'information G3 lacks' only if the nested delta is > 2 SE."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

d = load("val")
f, M, off = feeds("val"), feeds("val")["M"].to_numpy(), feeds("val")["cand_off"].to_numpy()
y = d["target"].astype(np.float64); feed = d["feed"]
g = np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float64)
nf = d["n_feeds"]
print(f"val feeds {nf}; feeds with no positive {np.mean(np.bincount(feed, y, minlength=nf)==0):.4f}; positives per feed mean {y.sum()/nf:.2f}")
def fz(x):
    m = np.bincount(feed, x, minlength=nf) / M
    s = np.sqrt(np.maximum(np.bincount(feed, x * x, minlength=nf) / M - m ** 2, 0))
    return (x - m[feed]) / np.where(s[feed] > 0, s[feed], 1)
gz = fz(g)
base = ndcg_per_feed(gz, y, off, M)
print(f"G3 val NDCG {base.mean():.5f}")

# decoded features
v2 = d["f2"]
df = pl.DataFrame({"feed": feed, "f2": v2})
dom = (df.filter(pl.col("f2") > 0).group_by(["feed", "f2"]).agg(pl.len().alias("n"))
       .sort(["feed", "n", "f2"], descending=[False, True, True]).group_by("feed", maintain_order=True).first())
domv = np.full(nf, np.nan); domv[dom["feed"].to_numpy()] = dom["f2"].to_numpy()
f2_mode = (v2 == domv[feed]).astype(np.float64)
f2_zero = (v2 == 0).astype(np.float64)
lv = df.filter(pl.col("f2") > 0).unique(["feed", "f2"]).with_columns(
    ((pl.col("f2").rank("dense").over("feed") - 1) / (pl.col("f2").n_unique().over("feed") - 1).clip(1, None)).alias("r"))
r = df.with_row_index("i").join(lv.select(["feed", "f2", "r"]), on=["feed", "f2"], how="left").sort("i")["r"].fill_null(0.5).to_numpy()
cat = d["cat"].astype(np.int64)
key = feed.astype(np.int64) * 64 + cat
o = np.lexsort((d["f1"], key)); ks = key[o]
st = np.r_[True, ks[1:] != ks[:-1]]; start = np.flatnonzero(st)[np.cumsum(st) - 1]
rk = np.arange(len(o)) - start; n_k = np.bincount(np.cumsum(st) - 1)[np.cumsum(st) - 1]
f1_in_cat = np.empty(len(o)); f1_in_cat[o] = np.where(n_k > 1, rk / np.maximum(n_k - 1, 1), 0.5)
X = {"f2_mode": f2_mode, "f2_zero": f2_zero, "f2_lvlrank": r, "f1_in_cat": f1_in_cat, "f1_m_f3": d["f1"] - d["f3"],
     "f5_nodata": (d["f5"] == 0).astype(np.float64), "f5_floor": (np.abs(d["f5"] - 1/62) < 1e-6).astype(np.float64)}

# (a) act/exp within G3 deciles: expected = per-feed share via within-feed G3 percentile decile rates
o = np.lexsort((gz, feed)); fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
rk = np.arange(len(o)) - np.flatnonzero(fs)[np.cumsum(fs) - 1]
pct = np.empty(len(o)); pct[o] = rk / np.maximum(M[feed[o]] - 1, 1)
dec = np.clip((pct * 10).astype(int), 0, 9)
rate_dec = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10)
exp_ = rate_dec[dec]
# per-feed normalisation: expected positives in each feed = actual positives (removes feed-size / base-rate confounding)
exp_ = exp_ * (np.bincount(feed, y, minlength=nf) / np.bincount(feed, exp_, minlength=nf))[feed]
print("\n(a) actual / expected target (expected = G3 within-feed decile rate, renormalised per feed to its actual positives)")
for k in ["f2_mode", "f2_zero", "f5_nodata", "f5_floor"]:
    m = X[k] == 1
    print(f"   {k}=1: rows {m.mean():.3f}, act/exp {y[m].sum()/exp_[m].sum():.3f} ; {k}=0: act/exp {y[~m].sum()/exp_[~m].sum():.3f}")
for k in ["f2_lvlrank", "f1_in_cat", "f1_m_f3"]:
    edges = [-1e9, 1e-4, 0.25, 0.4999, 0.5001, 0.75, 0.9999, 1e9] if k != "f1_m_f3" else list(np.quantile(X[k], [0, 0.2, 0.4, 0.6, 0.8])) + [1e9]
    b = np.searchsorted(np.array(edges[1:-1]), X[k], side="right")
    print(f"   {k} bins {np.round(edges[1:-1],3).tolist()} row share / act/exp: " + " ".join(f"{np.mean(b==i):.3f}/{y[b==i].sum()/max(exp_[b==i].sum(),1e-9):.3f}" for i in range(len(edges)-1)))

# (b) nested 2-fold by user
users = d["user_id"][np.r_[0, np.cumsum(M)[:-1]]]
rng = np.random.default_rng(0)
uu = np.unique(users[users > 0]); half = set(rng.choice(uu, len(uu) // 2, replace=False).tolist())
fold = np.array([(u in half) if u > 0 else (i % 2 == 0) for i, u in enumerate(users)])
W = [-1, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
print("\n(b) nested 2-fold-by-user: z(G3) + w * z_feed(x)")
for k, x in X.items():
    xz = fz(x.astype(np.float64))
    per = {w: ndcg_per_feed(gz + w * xz, y, off, M) for w in W}
    out = np.empty(nf)
    ws = []
    for a in [True, False]:
        sel = fold == a
        wb = max(W, key=lambda w: per[w][sel].mean())
        ws.append(wb)
        out[~sel] = per[wb][~sel]
    dlt = out - base
    wbest = max(W, key=lambda w: per[w].mean())
    print(f"   {k:11s}: nested delta {dlt.mean():+.5f} (se {dlt.std()/np.sqrt(nf):.5f}, z {dlt.mean()/(dlt.std()/np.sqrt(nf)):+.2f}); fold weights {ws}; in-sample best w {wbest} delta {per[wbest].mean()-base.mean():+.5f}")
