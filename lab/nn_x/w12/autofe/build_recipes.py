"""Step 4: NN-input recipes. Writes float16 arrays in flat candidate order to
nn_x/w12/autofe/feats/<split>/afe_*.npy for train / val / test. All transforms are label-free, per-feed or
frozen-constant (cache/stats.json, computed on val inputs only), so train / val / test are computed identically.
Names are chosen so that attach_xt applies the right fixed transform: *_rk -> raw, others -> clip +-10.
usage: build_recipes.py <split> [block ...]"""
import sys, time, json
import numpy as np, polars as pl
from lib import *
from lib import _rk, _z, _catgroup

split = sys.argv[1]; blocks = sys.argv[2:] or ["S", "P", "C"]
load_stats()
f = pl.read_parquet(FLAT / split / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
OD = OUT / "feats" / split
OD.mkdir(parents=True, exist_ok=True)


def get(n):
    if n in ("f1", "f2", "f3", "f4", "f5"):
        x = np.load(FLAT / split / f"cand_{n}.npy")
    else:
        x = np.load(featfile(split, n))
    return np.ascontiguousarray(nn_tf(n, x.astype(np.float32)).astype(np.float32))


def save(name, x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=10, neginf=-10)
    assert np.abs(x).max() <= 10.0 + 1e-3, (name, float(np.abs(x).max()))
    np.save(OD / f"{name}.npy", x.astype(np.float16))
    print(f"  {split}/{name}: mean {x.mean():+.4f} sd {x.std():.4f} min {x.min():.3f} max {x.max():.3f} {time.time() - t0:.0f}s", flush=True)


def qn(n, x):
    q = np.asarray(STATS[n]["q"], dtype=np.float32)
    return (np.searchsorted(q, x) / len(q)).astype(np.float32)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


t0 = time.time()
RECIPE = {"S": [], "P": [], "C": []}
if "S" in blocks:  # sparse / heavy-tailed inputs: non-zero flag + frozen quantile-normalised value
    for n in ("ctx_hn_top1w", "ctx_hs_mutual", "seq_cv_invd20", "seq_cv_norm"):
        x = get(n); save(f"afe_nz_{n}", (x > 0).astype(np.float32)); save(f"afe_qn_{n}", qn(n, x))
    for n in ("ctx_hn_rrw", "ctx_hn_e1_rrw", "ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"):
        save(f"afe_qn_{n}", qn(n, get(n)))
if "P" in blocks:  # best pair constructions of the screen (all marginal)
    R = lambda n: _rk(off, M, get(n))
    Z = lambda n: _z(off, M, get(n))
    save("afe_rmax_cosu_knnh_rk", np.maximum(R("cos_u"), R("ctx_nd_knn_h")))
    save("afe_rmin_luo_knnf1_rk", np.minimum(R("it_hist_all_luo"), R("ctx_nd_knn_f1")))
    save("afe_rmax_ndmax_hse1_rk", np.maximum(R("ctx_nd_max"), R("ctx_hs_e1")))
    # rmax(cos_u, pre2_new_s) (val z +2.3, holdout +1.4) is NOT built: pre2 is blank on 28% of train feeds, which
    # would make the column regime-dependent (rank ties at 0.5)
    save("afe_rmin_ndmax_hsk128_rk", np.minimum(R("ctx_nd_max"), R("ctx_hs_k128")))
    save("afe_zmul_catshare_inorm", slog(Z("cat_share_all") * Z("item_norm")))
    save("afe_zdiff_smax_coshall", Z("seq_max_all") - Z("cos_hall"))
    xa = (get("seq_top3") - STATS["seq_top3"]["min"]) / (STATS["seq_top3"]["std"] + 1e-6)
    xb = (get("cos_hall") - STATS["cos_hall"]["min"]) / (STATS["cos_hall"]["std"] + 1e-6)
    save("afe_lratio_top3_coshall", np.clip(np.log((xa + 0.1) / (xb + 0.1)), -10, 10))
if "C" in blocks:  # within-(feed, category) relative versions
    cat = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
    for n, keep in (("cos_hall", ("dgm",)), ("f4", ("dgm", "grk")), ("f5", ("grk",)), ("seq_max_all", ("grk", "gmf")), ("f2", ("gmf",))):
        dgm, grk, gmf, dgx = _catgroup(off, M, cat, get(n))
        if "dgm" in keep: save(f"afe_dgm_{n}", dgm)
        if "grk" in keep: save(f"afe_grk_{n}_rk", grk)
        if "gmf" in keep: save(f"afe_gmf_{n}", gmf)
if "Q" in blocks:  # formal passes of waves 2-3 that need no embedding kernel (all marginal)
    R = lambda n: _rk(off, M, get(n))
    Z = lambda n: _z(off, M, get(n))
    save("afe_rmin_het1_hsk128_rk", np.minimum(R("cos_het1"), R("ctx_hs_k128")))
    save("afe_rmin_luo4_top5h_rk", np.minimum(R("it_hist_et4_luo"), R("ctx_nd_top5h")))
    save("afe_rmax_smax_catet3_rk", np.maximum(R("seq_max_all"), R("cat_cnt_et3")))
    save("afe_zmul_luo4_knnh", slog(Z("it_hist_et4_luo") * Z("ctx_nd_knn_h")))
    save("afe_zmax_fcs_soft05", np.maximum(Z("feed_cat_share"), Z("ctx_hn_soft05")))
print("done", split, f"{time.time() - t0:.0f}s", flush=True)
