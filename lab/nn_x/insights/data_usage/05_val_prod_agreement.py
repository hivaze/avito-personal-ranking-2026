"""Val: does the model's ranking agree with the production ranker's item-level opinion (it_tr_pos = item mean
normalised production position over train feeds; lower = shown higher) more than labels justify? CPU, read-only."""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); P = ROOT / "preds"
fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet").sort("fidx")
M = fv["M"].to_numpy().astype(np.int64); off = fv["cand_off"].to_numpy().astype(np.int64); n = len(M)
fid = np.repeat(np.arange(n), M)
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
tp = np.load(ROOT / "data/feats/val/it_tr_pos.npy").astype(np.float64)
luo = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy")
print("it_tr_pos finite share", np.isfinite(tp).mean(), "min/max", np.nanmin(tp), np.nanmax(tp), "cold(luo==0) share", (luo == 0).mean())
ok = np.isfinite(tp)
def wrank(s, mask):
    # within-feed average rank among masked candidates (ties averaged), NaN elsewhere
    s2 = np.where(mask, s, np.inf); o = np.lexsort((s2, fid)); r = np.empty(len(s)); r[o] = np.arange(len(s)) - off[fid[o]]
    cnt = np.bincount(fid, weights=mask, minlength=n)
    return np.where(mask, r, np.nan), cnt
def spearman_pf(a, b, mask):
    ra, c = wrank(a, mask); rb, _ = wrank(b, mask)
    ma = np.bincount(fid[mask], weights=ra[mask], minlength=n) / np.maximum(c, 1); mb = np.bincount(fid[mask], weights=rb[mask], minlength=n) / np.maximum(c, 1)
    da = np.where(mask, ra - ma[fid], 0); db = np.where(mask, rb - mb[fid], 0)
    num = np.bincount(fid, weights=da * db, minlength=n); den = np.sqrt(np.bincount(fid, weights=da * da, minlength=n) * np.bincount(fid, weights=db * db, minlength=n))
    r = np.where((c >= 5) & (den > 0), num / np.maximum(den, 1e-12), np.nan)
    return r
def auc_pf(s, mask):
    # within-feed AUC of s for labels, restricted to mask; mean over feeds with >=1 pos and >=1 neg in mask
    o = np.lexsort((s, fid)); rr = np.empty(len(s)); rr[o] = np.arange(len(s))
    # rank within feed among masked only
    s2 = np.where(mask, s, -np.inf); o = np.lexsort((s2, fid)); r = np.empty(len(s)); r[o] = np.arange(len(s)) - off[fid[o]]
    nm = np.bincount(fid, weights=~mask, minlength=n); r = r - nm[fid]  # masked-out sit at the bottom (-inf)
    pos = mask & (y > 0); neg = mask & (y == 0)
    npos = np.bincount(fid, weights=pos, minlength=n); nneg = np.bincount(fid, weights=neg, minlength=n)
    sr = np.bincount(fid, weights=np.where(pos, r, 0), minlength=n)
    auc = (sr - npos * (npos - 1) / 2) / np.maximum(npos * nneg, 1)
    m = (npos > 0) & (nneg > 0)
    return np.where(m, auc, np.nan)
mods = {"ref (pool-only)": ROOT / "nn_x/r6/preds/nn_r6_rhoref_pool_s1_trainonly_val.npy"}
for r in (101, 102):
    mods[f"snap@77500 r{r}"] = P / f"nn_v17_pre2s_s{r}@77500_trainonly_val.npy"
    mods[f"F7 r{r}"] = P / f"nn_v17_pre2s_s{r}_trainonly_val.npy"
    mods[f"F7x r{r}"] = P / f"nn_r6_cd_m02_r{r}_trainonly_val.npy"
a_lab = auc_pf(-tp, ok)
print(f"label within-feed AUC of -it_tr_pos (warm candidates only): {np.nanmean(a_lab):.4f} (feeds {np.isfinite(a_lab).sum()})")
rows = []
for k, f in mods.items():
    s = np.load(f).astype(np.float64)
    rho = spearman_pf(s, -tp, ok)
    a = auc_pf(s, ok)
    a_all = auc_pf(s, np.ones(len(s), bool))
    rows.append(dict(model=k, spearman_score_vs_prodpos=np.nanmean(rho), auc_label_warm=np.nanmean(a), auc_label_all=np.nanmean(a_all)))
df = pl.DataFrame(rows); pl.Config.set_tbl_width_chars(200); print(df)
df.write_csv(ROOT / "nn_x/insights/data_usage/out/05_val_prod_agreement.csv")
