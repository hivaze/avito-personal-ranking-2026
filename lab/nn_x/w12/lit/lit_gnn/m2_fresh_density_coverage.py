"""M2 (label-free): are the on-disk fresh neighbour-pooled demand densities (trd_*; feats_v2/trd) available on COLD rows,
and are they lag-invariant (val -> public -> private; train sample)?
trd_n95_sday = # pool events (hpos<10, other users, same MSK day) with cos >= 0.95 to the candidate, per 1K pool feeds;
trd_ex_sday  = exact item matches; trd_kde_sday = sum exp(20(cos-1)); trd_sn95_sday = supply (candidates of other feeds).
Reads data/feats/{train,val,test}/{trd_*,it_hist_all_luo,seq_cv_pop}.npy and flat/test feeds. No labels. Stdout only."""
import numpy as np, polars as pl
R = "<repo>/"; BOUND = 1790497920
cols = ["trd_ex_sday", "trd_n95_sday", "trd_n85_sday", "trd_kde_sday", "trd_sn95_sday", "trd_skde_sday", "trd_n95_s6h"]
def L(split, n): return np.load(R + f"data/feats/{split}/{n}.npy", mmap_mode="r")
def rep(tag, split, idx):
    luo = np.nan_to_num(np.asarray(L(split, "it_hist_all_luo")[idx])); pop = np.nan_to_num(np.asarray(L(split, "seq_cv_pop")[idx]))
    cold = (luo == 0) & (pop == 0)
    print(f"\n== {tag}: rows {len(luo):,}; cold share {cold.mean():.4f}")
    print("  column            | >0 share: all / cold / warm | median of positives: cold / warm | NaN")
    for c in cols:
        x = np.asarray(L(split, c)[idx]).astype(np.float64); nan = np.isnan(x).mean(); x = np.nan_to_num(x)
        pos = x > 0
        mc = np.median(x[pos & cold]) if (pos & cold).any() else float("nan"); mw = np.median(x[pos & ~cold]) if (pos & ~cold).any() else float("nan")
        print(f"  {c:17s} | {pos.mean():.3f} / {pos[cold].mean():.3f} / {pos[~cold].mean():.3f} | {mc:.4f} / {mw:.4f} | {nan:.4f}")
n = L("val", "seq_cv_pop").shape[0]; rep("val", "val", slice(None))
fidx = np.asarray(np.load(R + "data/flat/test/cand_fidx.npy", mmap_mode="r"))
ts = pl.read_parquet(R + "data/flat/test/feeds.parquet")["timestamp"].to_numpy()
pub = (ts < BOUND)[fidx]
rep("test public", "test", np.flatnonzero(pub)); rep("test private", "test", np.flatnonzero(~pub))
ntr = L("train", "seq_cv_pop").shape[0]
rng = np.random.default_rng(0); idx = np.sort(rng.choice(ntr, 4_000_000, replace=False))
rep("train (4M random rows, stored as-built counters)", "train", idx)
