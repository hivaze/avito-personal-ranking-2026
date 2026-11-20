"""Label-feature relation by day type and hour bucket on train: single-feature NDCG lift over random (exact expected ties)
for key base features, weekend minus weekday with SE (feeds are independent units). All feeds and random-like feeds."""
from calcommon import *
F = feeds("train"); M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
tp = time_parts(F["timestamp"].to_numpy())
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"]
valid = np.isin(tp["day"], np.unique(tp["day"])[np.bincount(tp["day"] - tp["day"].min())[np.unique(tp["day"]) - tp["day"].min()] > 1000])
y = cand("train", "target", mmap=False).astype(np.float64)
nd0 = ndcg_pf(np.zeros(len(y)), y, off, M, mode=2)
feats = ["f1", "f2", "f3", "f4", "f5", "maxsim32", "cos_u", "attn32", "cos_h10", "cat_cnt_all", "it_hist_all", "tte_it_coec"]
groups = {"all": valid, "random_like": valid & (pr > 0.5) & (M >= 20)}
rows = []
for fn in feats:
    x = (cand("train", fn, mmap=False) if fn.startswith("f") and len(fn) == 2 else np.load(f"{FEATS}/train/{fn}.npy")).astype(np.float64)
    x = np.where(np.isnan(x), -1e30, x)
    lift = ndcg_pf(x, y, off, M, mode=2) - nd0
    for g, gm in groups.items():
        a = lift[gm & (tp["weekend"] == 1)]; b = lift[gm & (tp["weekend"] == 0)]
        hb = [float(lift[gm & (tp["hb"] == h)].mean()) for h in range(N_HB)]
        rows.append(dict(feature=fn, group=g, lift_weekday=float(b.mean()), lift_weekend=float(a.mean()),
                         we_minus_wd=float(a.mean() - b.mean()), se=float(np.sqrt(a.var() / len(a) + b.var() / len(b))),
                         hb_min=min(hb), hb_max=max(hb), hb_profile=str([round(v, 4) for v in hb])))
    log(fn)
df = pl.DataFrame(rows)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
print(df)
df.write_csv(f"{OUT}/e07_feature_relation.csv")
