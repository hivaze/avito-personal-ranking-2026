"""Follow-up of 14: f5 is an unsmoothed engagement rate (top decile = tiny denominators). Does G3 already discount noisy high f5?
Label-free candidates on val: f5_shrunk = (f5*n + 0.44*k)/(n + k) with n = train impressions + history appearances (k = 10, 30);
f5_noisy_hi = 1 if f5 in the top val decile and n < 5. Same nested 2-fold-by-user readout as 09 (pre-declared: > 2 SE)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

d = load("val", None, extra_feats=["tte_it_imp", "it_hist_all_luo"])
f = feeds("val"); M = f["M"].to_numpy(); off = f["cand_off"].to_numpy()
y = d["target"].astype(np.float64); feed = d["feed"]; nf = d["n_feeds"]
g = np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float64)
def fz(x):
    m = np.bincount(feed, x, minlength=nf) / M
    s = np.sqrt(np.maximum(np.bincount(feed, x * x, minlength=nf) / M - m ** 2, 0))
    return (x - m[feed]) / np.where(s[feed] > 0, s[feed], 1)
gz = fz(g); base = ndcg_per_feed(gz, y, off, M)
n = d["tte_it_imp"] + d["it_hist_all_luo"]
f5 = d["f5"].astype(np.float64)
X = {"f5_shrunk10": (f5 * n + 0.44 * 10) / (n + 10), "f5_shrunk30": (f5 * n + 0.44 * 30) / (n + 30),
     "f5_noisy_hi": ((f5 >= np.quantile(f5, 0.9)) & (n < 5)).astype(np.float64)}
o = np.lexsort((gz, feed)); fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
rk = np.arange(len(o)) - np.flatnonzero(fs)[np.cumsum(fs) - 1]
pct = np.empty(len(o)); pct[o] = rk / np.maximum(M[feed[o]] - 1, 1)
dec = np.clip((pct * 10).astype(int), 0, 9)
rate_dec = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10)
exp_ = rate_dec[dec]; exp_ = exp_ * (np.bincount(feed, y, minlength=nf) / np.bincount(feed, exp_, minlength=nf))[feed]
m = X["f5_noisy_hi"] == 1
print(f"f5_noisy_hi rows {m.mean():.4f}: act/exp {y[m].sum()/exp_[m].sum():.3f}; top-decile f5 with n >= 5: act/exp {y[(f5>=np.quantile(f5,0.9))&(n>=5)].sum()/exp_[(f5>=np.quantile(f5,0.9))&(n>=5)].sum():.3f}")
users = d["user_id"][np.r_[0, np.cumsum(M)[:-1]]]
rng = np.random.default_rng(0)
uu = np.unique(users[users > 0]); half = set(rng.choice(uu, len(uu) // 2, replace=False).tolist())
fold = np.array([(u in half) if u > 0 else (i % 2 == 0) for i, u in enumerate(users)])
W = [-1, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
for k, x in X.items():
    xz = fz(x)
    per = {w: ndcg_per_feed(gz + w * xz, y, off, M) for w in W}
    out = np.empty(nf); ws = []
    for a in [True, False]:
        sel = fold == a
        wb = max(W, key=lambda w: per[w][sel].mean()); ws.append(wb); out[~sel] = per[wb][~sel]
    dl = out - base; se = dl.std() / np.sqrt(nf)
    print(f"   {k}: nested delta {dl.mean():+.5f} (se {se:.5f}); fold weights {ws}")
