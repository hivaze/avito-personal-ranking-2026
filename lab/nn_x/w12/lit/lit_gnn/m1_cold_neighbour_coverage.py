"""M1 (label-free): do COLD candidates have close WARM content neighbours?
Reads existing per-row files only (no labels):
  data/feats/{val,test}/it_hist_all_luo.npy, seq_cv_pop.npy, seq_cv_invd20.npy
  nn_x/w12/cb/famB/feat/{c,d}_{val,test}.npy  (kn_cos1 = cosine to the nearest item of the famB index:
      5.86M much-shown train items, search inside the candidate's own 4096-cluster, own item skipped)
  data/flat/test/feeds.parquet (timestamp) + cand_fidx for the public / private split (boundary ts 1790497920).
CPU only. Writes nothing but stdout.
"""
import json, numpy as np, polars as pl
R = "<repo>/"
B = R + "nn_x/w12/cb/famB/feat/"
cn = json.load(open(B + "c_names.json")); dn = json.load(open(B + "d_names.json"))
BOUND = 1790497920

def load(split):
    f = lambda n: np.load(R + f"data/feats/{split}/{n}.npy", mmap_mode="r")
    luo = np.asarray(f("it_hist_all_luo")); pop = np.asarray(f("seq_cv_pop")); cv = np.asarray(f("seq_cv_invd20"))
    c = np.load(B + f"c_{split}.npy", mmap_mode="r"); d = np.load(B + f"d_{split}.npy", mmap_mode="r")
    fidx = np.load(R + f"data/flat/{split}/cand_fidx.npy", mmap_mode="r")
    return luo, pop, cv, c, d, np.asarray(fidx)

def rep(tag, m, luo, pop, cv, c, d):
    luo, pop, cv = luo[m], pop[m], cv[m]
    k1 = np.asarray(c[:, cn.index("kn_cos1")])[m]; k20 = np.asarray(c[:, cn.index("kn_cos20")])[m]
    ex20 = np.asarray(c[:, cn.index("kn_ex20")])[m]
    n95 = np.asarray(d[:, dn.index("dup_n95")])[m]; n90 = np.asarray(d[:, dn.index("dup_n90")])[m]
    n = len(luo)
    cold_luo = np.nan_to_num(luo) == 0; cold_cv = np.nan_to_num(pop) == 0; pair = np.nan_to_num(cv) > 0
    cold = cold_luo & cold_cv
    print(f"\n== {tag}: rows {n:,}")
    print(f"  it_hist_all_luo == 0: {cold_luo.mean():.4f} | seq_cv_pop == 0: {cold_cv.mean():.4f} | both (cold): {cold.mean():.4f}")
    print(f"  pairwise co-vis score > 0: {pair.mean():.4f} (of rows with seq_cv_pop > 0: {pair[~cold_cv].mean():.4f})")
    for name, sel in (("cold (both counters 0)", cold), ("warm (either counter > 0)", ~cold)):
        a = k1[sel]; b = k20[sel]
        fin = np.isfinite(a)
        q = np.nanquantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  {name}: n {sel.sum():,}; kn_cos1 NaN {1-fin.mean():.4f}; p10/p25/p50/p75/p90 "
              + " / ".join(f"{x:.3f}" for x in q))
        print("     share kn_cos1 >= 0.90 / 0.95 / 0.97 / 0.99: "
              + " / ".join(f"{(a >= t).mean():.3f}" for t in (0.90, 0.95, 0.97, 0.99)))
        print(f"     kn_cos20 (mean cos of 20 nearest) p50 {np.nanmedian(b):.3f}; share >= 0.90: {(b >= 0.90).mean():.3f}")
        print(f"     index items with cos > 0.95: share rows with >= 1 / >= 5: {(n95[sel] >= 1).mean():.3f} / {(n95[sel] >= 5).mean():.3f};"
              f" cos > 0.90: >= 1 / >= 5: {(n90[sel] >= 1).mean():.3f} / {(n90[sel] >= 5).mean():.3f}")
        print(f"     kn_ex20 (expected clicks of the 20 neighbours = support) p10/p50/p90: "
              + " / ".join(f"{x:.2f}" for x in np.nanquantile(ex20[sel], [0.1, 0.5, 0.9])))
    # rows without ANY pairwise co-vis signal that have a near-duplicate index neighbour
    nop = ~pair
    print(f"  rows with NO pairwise co-vis ({nop.mean():.4f} of rows): share with kn_cos1 >= 0.95: {(k1[nop] >= 0.95).mean():.3f}; >= 0.97: {(k1[nop] >= 0.97).mean():.3f}")

luo, pop, cv, c, d, fidx = load("val")
rep("val", np.ones(len(luo), bool), luo, pop, cv, c, d)
luo, pop, cv, c, d, fidx = load("test")
feeds = pl.read_parquet(R + "data/flat/test/feeds.parquet")
tcol = [x for x in feeds.columns if "time" in x.lower() or x.lower() in ("ts", "t")]
print("\nfeeds columns:", feeds.columns, "-> time column", tcol)
ts = feeds[tcol[0]].to_numpy()
if ts.dtype.kind == "M":
    ts = ts.astype("datetime64[s]").astype(np.int64)
pub = (ts < BOUND)[fidx]
print(f"public feeds {int((ts < BOUND).sum()):,} of {len(ts):,}")
rep("test public", pub, luo, pop, cv, c, d)
rep("test private", ~pub, luo, pop, cv, c, d)
