"""Single-feature NDCG on val (both directions, exact random tie-break) + NaN rates; train weekend/weekday on random-like feeds."""
from calcommon import *
import glob

F = feeds("val"); M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
y = cand("val", "target", mmap=False).astype(np.float64)
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{FEATS}/val/cal_*.npy"))
base_rand = float(ndcg_pf(np.zeros(len(y)), y, off, M, mode=2).mean())
print("val random (expected) NDCG", round(base_rand, 4))
rows = []
for n in names:
    x = np.load(f"{FEATS}/val/{n}.npy").astype(np.float64)
    s = single_ndcg(x, y, off, M)
    # within-feed variability: share of feeds where feature is non-constant
    rows.append(dict(name=n, nan=float(np.isnan(x).mean()), mean=float(np.nanmean(x)), std=float(np.nanstd(x)),
                     ndcg_pos=s["pos"], ndcg_neg=s["neg"], best=max(s["pos"], s["neg"]) - base_rand))
df = pl.DataFrame(rows)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(df)
df.write_csv(f"{OUT}/e02_single_val.csv")

# train: same on random-like feeds split weekday / weekend (labels position-debiased by design: random order)
Ft = feeds("train"); Mt = Ft["M"].to_numpy().astype(np.int64); offt = Ft["cand_off"].to_numpy().astype(np.int64)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"]
tp = time_parts(Ft["timestamp"].to_numpy())
from b01_build import N_HB
late = tp["day"] >= np.unique(tp["day"])[5]   # days with >=5 earlier days of label stats
rows = []
for wk in [0, 1]:
    sel = np.where((pr > 0.5) & (Mt >= 20) & (tp["weekend"] == wk) & late)[0]
    sel = sel[np.random.default_rng(0).permutation(len(sel))[:60000]]; sel.sort()
    idx = np.concatenate([np.arange(offt[i], offt[i] + Mt[i]) for i in sel])
    M2 = Mt[sel]; o2 = np.r_[0, np.cumsum(M2)[:-1]]
    yt = np.asarray(cand("train", "target")[idx], np.float64)
    br = float(ndcg_pf(np.zeros(len(yt)), yt, o2, M2, mode=2).mean())
    for n in names:
        x = np.asarray(np.load(f"{FEATS}/train/{n}.npy", mmap_mode="r")[idx], np.float64)
        s = single_ndcg(x, yt, o2, M2)
        rows.append(dict(weekend=wk, name=n, n_feeds=len(sel), lift_pos=s["pos"] - br, lift_neg=s["neg"] - br))
dt = pl.DataFrame(rows).pivot(on="weekend", index="name", values=["lift_pos", "lift_neg"])
print(dt)
dt.write_csv(f"{OUT}/e02_single_train_randomlike.csv")
