"""pop1 arm input (nn_x/codes/followup/RESULT.txt section (4) "POP1 DESIGN"; ledger nn_x/abl/ledger/ledger.md section 7).

  xp_imp_c50 = log1p(min(tte_it_imp, 50)) as float32, for train / val / test.

tte_it_imp (src/features_tte.py) = the item's TRAIN impressions: train rows on day d count train days < d only (expanding
window; day-0 rows are 0 by construction), val / test rows count all train days. No labels, no own-day rows -> leak-free.

Writes NEW files only (exclusive create, never overwrites):
  data/feats/<split>/xp_imp_c50.npy
  nn/feat_cache/<split>/xp_imp_c50.npy  = a HARD LINK of the data/feats file (the way every other xt column, e.g.
                                          it_hist_all_luo, is exposed to the nn_x/abl trainer; existing entries untouched)
and prints the parity statistics (mean, share at the cap, share zero, PSI train-vs-val / train-vs-test, day-0 rows).

  cd <repo> && OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/pop1/build_xp_imp_c50.py [--stats-only]
"""
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = "<repo>"
FEATS = f"{ROOT}/data/feats"
CACHE = f"{ROOT}/nn/feat_cache"
FLAT = f"{ROOT}/data/flat"
NAME = "xp_imp_c50"
CAP = 50.0
SPLITS = ("train", "val", "test")
OUT = f"{ROOT}/nn_x/abl/pop1/out"


def build(split):
    imp = np.load(f"{FEATS}/{split}/tte_it_imp.npy")
    assert imp.dtype == np.float32 and not np.isnan(imp).any() and (imp >= 0).all(), split
    return np.log1p(np.minimum(imp, CAP)).astype(np.float32), imp


def save_new(path, x):
    with open(path, "xb") as f:  # exclusive create: refuses to overwrite
        np.save(f, x)


def psi(a, b, edges):
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def main():
    stats_only = "--stats-only" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    X, RAW = {}, {}
    for sp in SPLITS:
        X[sp], RAW[sp] = build(sp)
        if not stats_only:
            dst, lnk = f"{FEATS}/{sp}/{NAME}.npy", f"{CACHE}/{sp}/{NAME}.npy"
            assert not os.path.exists(dst) and not os.path.exists(lnk), f"{dst} / {lnk} exists: refusing to overwrite"
            save_new(dst, X[sp])
            os.link(dst, lnk)
            st_d, st_l = os.stat(dst), os.stat(lnk)
            assert st_d.st_ino == st_l.st_ino and st_d.st_nlink == 2
            back = np.load(lnk)
            assert back.dtype == np.float32 and np.array_equal(back, X[sp])
            print(f"{sp}: wrote {dst} ({X[sp].shape[0]:,} rows), hard link {lnk} (inode {st_d.st_ino}, nlink 2)", flush=True)
    # parity statistics
    ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
    fidx = np.load(f"{FLAT}/train/cand_fidx.npy")
    day_f = (ftr["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400
    day = day_f[fidx]
    d0 = day == day.min()
    res = {"name": NAME, "def": "log1p(min(tte_it_imp, 50)) float32"}
    for sp in SPLITS:
        x, r = X[sp], RAW[sp]
        res[sp] = dict(rows=int(len(x)), mean=float(x.mean()), sd=float(x.std()), min=float(x.min()), max=float(x.max()),
                       share_cap=float((r >= CAP).mean()), share_zero=float((x == 0).mean()), median_raw=float(np.median(r)),
                       mean_raw_capped=float(np.minimum(r, CAP).mean()))
    res["train_day0"] = dict(day=int(day.min()), rows=int(d0.sum()), share=float(d0.mean()),
                             all_zero=bool((X["train"][d0] == 0).all()))
    xt_nd0 = X["train"][~d0]
    res["train_excl_day0"] = dict(mean=float(xt_nd0.mean()), share_zero=float((xt_nd0 == 0).mean()),
                                  share_cap=float((RAW["train"][~d0] >= CAP).mean()))
    # PSI: (a) natural count bins of the capped count; (b) 10 train-quantile bins (unique edges)
    lo = np.array([1, 2, 3, 5, 10, 20, 50], np.float64)  # lower ends of the bins after {0}
    edges_a = np.r_[-1.0, (np.log1p(lo - 1) + np.log1p(lo)) / 2, 10.0]  # bins {0},{1},{2},{3-4},{5-9},{10-19},{20-49},{50}
    q = np.unique(np.quantile(X["train"], np.linspace(0, 1, 11)[1:-1]))
    edges_b = np.r_[-1.0, q + 1e-6, 10.0]
    for other in ("val", "test"):
        res[f"psi_train_{other}"] = dict(count_bins=psi(X["train"], X[other], edges_a), train_quantile_bins=psi(X["train"], X[other], edges_b))
        res[f"psi_train_excl_day0_{other}"] = dict(count_bins=psi(xt_nd0, X[other], edges_a))
    res["psi_val_test"] = dict(count_bins=psi(X["val"], X["test"], edges_a))
    res["count_bin_shares"] = {sp: (np.histogram(X[sp], edges_a)[0] / len(X[sp])).round(4).tolist() for sp in SPLITS}
    res["count_bin_labels"] = ["0", "1", "2", "3-4", "5-9", "10-19", "20-49", "50(cap)"]
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/xp_imp_c50_parity.json", "w"), indent=1)
    assert res["psi_train_val"]["count_bins"] < 0.5, "PSI train-vs-val >= 0.5: parity rule broken"


if __name__ == "__main__":
    main()
