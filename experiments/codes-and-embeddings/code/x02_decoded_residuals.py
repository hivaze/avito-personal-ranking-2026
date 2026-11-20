# Residual screen of six features that exist only because of the decoding (vertical shares, delivery
# class). Each feature x is added to the reference pool G3 as z(G3) + w * z_feed(x); w comes from a
# 13-value grid chosen on one half of the users and applied to the other half. The rule was written
# before the run: a feature passes if the nested gain is above +0.0003 and above 2 SE.
# Run without arguments on CPU with 12 threads; the command line is not on record.
# Produced logs/x02.log: G3 val NDCG 0.40032, 0 of 6 features pass.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/codes/synth/x02_decoded_residuals.py
#
"""Pre-declared decoding-enabled residual screen on G3 (val, own-feed history only; label-free features).
Rule (declared before running): a feature counts as 'information G3 lacks' only if nested 2-fold-by-user delta > +2 SE
AND > +0.0003. Harness identical to nn_x/codes/features/09_residual_g3.py (z(G3) + w * z_feed(x), 13-value grid).
Features (all need the decoded category -> vertical map or the delivery reading):
  v_share        share of the feed's history events in the candidate's vertical
  v_share_xleaf  share of history in the candidate's vertical but in a DIFFERENT category (what per-leaf f3 cannot see)
  v_last         candidate vertical == vertical of the newest history event
  v_rec          -log1p(min hpos of a history event in the candidate's vertical), -log(257) if none
  deliv_x        log1p(# e5/e6/e8 events in history) * [candidate category has delivery share >= 2%] (C1 table)
  deliv_mask     [user has >= 1 delivery event] * [candidate category is hard-zero delivery]  (structural-zero cross)"""
import os, sys, json
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS']:
    os.environ[k] = '12'
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/features"); sys.path.insert(0, "<repo>/src")
from fcommon import load, feeds, ROOT
from metrics import ndcg_per_feed

VERT = json.load(open(ROOT / 'nn_x/codes/categories/out/code_vertical.json'))
vnames = sorted(set(VERT.values())); vid = np.full(64, len(vnames), np.int64)
for c, v in VERT.items(): vid[int(c)] = vnames.index(v)
NV = len(vnames) + 1
c1 = pl.read_csv(ROOT / 'nn_x/codes/synth/logs/c1_delivery_share.csv')
dshare = np.zeros(64); dshare[c1['cat'].to_numpy()] = c1['share'].to_numpy()

d = load("val"); f = feeds("val"); M = f["M"].to_numpy(); off = f["cand_off"].to_numpy(); nf = len(M)
y = d["target"].astype(np.float64); feed = d["feed"]; cat = d["cat"].astype(np.int64)
g = np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float64)
# history
FL = ROOT / 'data/flat/val'
hf = np.load(FL / 'hist_fidx.npy').astype(np.int64); hc = np.load(FL / 'hist_category_id.npy').astype(np.int64)
ht = np.load(FL / 'hist_event_type.npy').astype(np.int64); hp = np.load(FL / 'hist_hpos.npy').astype(np.int64)
assert (f['fidx'].to_numpy() == np.arange(nf)).all()
hv = vid[hc]
H = np.bincount(hf, minlength=nf).astype(float)
cnt_v = np.zeros((nf, NV)); np.add.at(cnt_v, (hf, hv), 1)
cnt_c = np.zeros((nf, 64)); np.add.at(cnt_c, (hf, hc), 1)
minh_v = np.full((nf, NV), 256.0); np.minimum.at(minh_v, (hf, hv), hp)
newest = np.full(nf, -1); o = np.lexsort((hp, hf)); first = np.r_[True, hf[o][1:] != hf[o][:-1]]
newest[hf[o][first]] = hv[o][first]
ndel = np.bincount(hf, weights=np.isin(ht, [5, 6, 8]), minlength=nf)
cv = vid[cat]
Hs = np.maximum(H[feed], 1)
X = {
    'v_share': cnt_v[feed, cv] / Hs,
    'v_share_xleaf': (cnt_v[feed, cv] - cnt_c[feed, cat]) / Hs,
    'v_last': (cv == newest[feed]).astype(float),
    'v_rec': -np.log1p(minh_v[feed, cv]),
    'deliv_x': np.log1p(ndel[feed]) * (dshare[cat] >= 0.02),
    'deliv_mask': ((ndel[feed] > 0) & (dshare[cat] == 0)).astype(float),
}


def fz(x):
    m = np.bincount(feed, x, minlength=nf) / M
    s = np.sqrt(np.maximum(np.bincount(feed, x * x, minlength=nf) / M - m ** 2, 0))
    return (x - m[feed]) / np.where(s[feed] > 0, s[feed], 1)


gz = fz(g); base = ndcg_per_feed(gz, y, off, M)
print(f"G3 val NDCG {base.mean():.5f}")
users = f['user_id'].to_numpy()
rng = np.random.default_rng(0)
uu = np.unique(users[users > 0]); half = set(rng.choice(uu, len(uu) // 2, replace=False).tolist())
fold = np.array([(u in half) if u > 0 else (i % 2 == 0) for i, u in enumerate(users)])
W = [-1, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
# act/exp of G3 by feature quintile (calibration readout)
o = np.lexsort((gz, feed)); fs_ = np.r_[True, feed[o][1:] != feed[o][:-1]]
rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
pct = np.empty(len(o)); pct[o] = rk / np.maximum(M[feed[o]] - 1, 1)
dec = np.clip((pct * 10).astype(int), 0, 9)
rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
ex = ex * (np.bincount(feed, y, minlength=nf) / np.bincount(feed, ex, minlength=nf))[feed]
for k, x in X.items():
    xz = fz(x.astype(np.float64))
    per = {w: ndcg_per_feed(gz + w * xz, y, off, M) for w in W}
    out = np.empty(nf); ws = []
    for a in [True, False]:
        sel = fold == a
        wb = max(W, key=lambda w: per[w][sel].mean()); ws.append(wb); out[~sel] = per[wb][~sel]
    dl = out - base; se = dl.std() / np.sqrt(nf)
    qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(np.unique(qs), x, side='right')
    ae = ' '.join(f"{np.mean(b == i):.2f}/{y[b == i].sum() / max(ex[b == i].sum(), 1e-9):.3f}" for i in range(b.max() + 1))
    verdict = 'PASS' if (dl.mean() > 2 * se and dl.mean() > 0.0003) else 'FAIL'
    print(f"{k:14s}: nested {dl.mean():+.5f} (se {se:.5f}, z {dl.mean() / max(se, 1e-12):+.2f}) folds w {ws} -> {verdict} | G3 act/exp by bin (share/ae): {ae}")
