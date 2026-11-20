"""V4 (bonus): where is the remaining headroom of the best model? Decompose within-feed pairwise errors of nn_final into
same-category vs cross-category (positive, negative) pairs, plus category-oracle NDCG on top of the NN, by segment."""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

FLAT = Path("<repo>/data/flat")
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy(); H = f["H"].to_numpy(); U = f["user_id"].to_numpy()
y = np.load(FLAT / "val/cand_target.npy").astype(np.int64)
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
P = {n: np.load(f"<repo>/preds/{n}.npy").astype(np.float64) for n in
     ["nn_final_val_oof", "nn_ens_stage1_trainonly_val", "cb_tr300k_none_val"]}
P["f1+1e-3f5"] = np.load(FLAT / "val/cand_f1.npy") + 1e-3 * np.load(FLAT / "val/cand_f5.npy").astype(np.float64)


def feed_z(s):
    mu = np.add.reduceat(s, off) / M; d = s - np.repeat(mu, M)
    sd = np.sqrt(np.add.reduceat(d * d, off) / M); return d / np.repeat(np.where(sd > 0, sd, 1), M)


fid = np.repeat(np.arange(len(M)), M)
poscat = np.zeros(len(y), bool)
# candidate shares a category with some positive of its feed
key = fid * 100 + cat
pk = np.unique(key[y == 1]); poscat = np.isin(key, pk)
print(f"share of candidates in a positive's category: {poscat.mean():.3f}; share of negatives: {poscat[y==0].mean():.3f}")

for name, s in P.items():
    same_err = same_tot = cross_err = cross_tot = 0.0
    for i in range(len(M)):
        a, m = off[i], M[i]
        yy = y[a:a + m]; ss = s[a:a + m]; cc = cat[a:a + m]
        pi = np.where(yy == 1)[0]; ni = np.where(yy == 0)[0]
        if len(pi) == 0 or len(ni) == 0: continue
        for p in pi:
            sm = cc[ni] == cc[p]
            err = (ss[ni] >= ss[p]).astype(np.float64)  # pessimistic ties count as errors
            w = 1.0 / (len(pi) * len(ni))
            same_err += err[sm].sum() * w; same_tot += sm.sum() * w
            cross_err += err[~sm].sum() * w; cross_tot += (~sm).sum() * w
    base = ndcg_per_feed(s, y, off, M)
    orc = ndcg_per_feed(10 * poscat + feed_z(s), y, off, M)
    print(f"{name:28s} NDCG {base.mean():.4f}  cat-oracle+model {orc.mean():.4f} (+{(orc-base).mean():.4f})  "
          f"pairwise err same-cat {same_err/same_tot:.3f} (mass {same_tot/len(M):.3f})  cross-cat {cross_err/cross_tot:.3f} "
          f"(mass {cross_tot/len(M):.3f})  -> share of errors cross-cat {cross_err/(cross_err+same_err):.2f}", flush=True)

nn = ndcg_per_feed(P["nn_final_val_oof"], y, off, M)
rnd_exp = None
npos = np.add.reduceat(y.astype(float), off)
disc = 1 / np.log2(np.arange(2, 302)); cd = np.concatenate([[0], np.cumsum(disc)])
er = (npos / M) * cd[M] / cd[npos.astype(int)]
f1 = ndcg_per_feed(P["f1+1e-3f5"], y, off, M)
print("\nSegment lift over expected-random: nn_final vs f1+1e-3f5")
for nm, m in [("H=0", H == 0), ("H 1-31", (H > 0) & (H < 32)), ("H>=32", H >= 32), ("user0", U == 0),
              ("M<30", M < 30), ("M 30-149", (M >= 30) & (M < 150)), ("M>=150", M >= 150)]:
    print(f"  {nm:9s} n={m.sum():5d}  nn {nn[m].mean():.4f} (+{(nn-er)[m].mean():.4f})  f1 {f1[m].mean():.4f} (+{(f1-er)[m].mean():.4f})")
