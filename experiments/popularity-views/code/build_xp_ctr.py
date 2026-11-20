# Builder of the input of the second popularity-light view: xp_ctr_c = clip(log(rate / prior), -2, 2),
# the item's smoothed train click rate relative to the prior 0.0159. Train rows use earlier train days
# only; val and test rows use all train days.
# Run as: cd <repo> && OMP_NUM_THREADS=12 .venv/bin/python
# nn_x/abl/pop2/build_xp_ctr.py [--stats-only]
# Produced data/feats/<split>/xp_ctr_c.npy and a hard link in nn/feat_cache/. Four 10-epoch members
# with this input correlated 0.9863 with the exposure-count view: the same view.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/abl/pop2/build_xp_ctr.py
#
"""pop2 arm input (POPULARITY-VIEW FAMILIES follow-up, Task B; ledger nn_x/abl/ledger/ledger.md section 8). Modelled on
nn_x/abl/pop1/build_xp_imp_c50.py.

  xp_ctr_c = clip(log(tte_it_ctr / PRIOR), -2, +2) as float32, PRIOR = 0.0159 (src/features_tte.py), for train / val / test.

tte_it_ctr (src/features_tte.py) = the item's smoothed TRAIN click rate (it_pos + PRIOR * 20) / (it_imp + 20): train rows on day d
use train days < d only (expanding window; day-0 rows and items without earlier impressions sit at exactly PRIOR -> 0 here),
val / test rows use all train days. No labels of the own row / day, no later snapshots -> leak-free. The log ratio to the prior
is 0 for 'no information', negative for under-clicked items, positive for over-clicked ones; +-2 = a click rate 7.4x above /
below the prior (ctr 0.117 / 0.00215), reached only with >= 3-4 all-clicked impressions or >= 128 never-clicked impressions,
so the clip removes the extreme popularity ordering at both ends. The transform was chosen by parity (scratchpad explore_ctr.py,
14:25 UTC): every clip of the log ratio has PSI train-vs-val 0.014 and train-vs-test 0.096-0.105 (the no-train-impression
atom: train 31.7% / val 34.6% / test 46.2%, the same shift as xp_imp_c50's 0.102); +-2 keeps 99.6% of the rows unclipped.

Writes NEW files only (exclusive create, never overwrites):
  data/feats/<split>/xp_ctr_c.npy
  nn/feat_cache/<split>/xp_ctr_c.npy  = a HARD LINK of the data/feats file (as xp_imp_c50 / it_hist_all_luo are exposed to the
                                        nn_x/abl trainer; existing entries untouched)
and prints the parity statistics (mean, sd, share at the clips, share at the prior, PSI train-vs-val / train-vs-test on natural
log-ratio bins and on 10 train-quantile bins, day-0 rows).

  cd <repo> && OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/pop2/build_xp_ctr.py [--stats-only]
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
NAME = "xp_ctr_c"
PRIOR = 0.0159  # src/features_tte.py PRIOR
LO, HI = -2.0, 2.0
SPLITS = ("train", "val", "test")
OUT = f"{ROOT}/nn_x/abl/pop2/out"


def build(split):
    ctr = np.load(f"{FEATS}/{split}/tte_it_ctr.npy")
    assert ctr.dtype == np.float32 and not np.isnan(ctr).any() and (ctr > 0).all(), split
    x = np.clip(np.log(ctr / np.float32(PRIOR)), LO, HI).astype(np.float32)
    assert np.isfinite(x).all() and x.min() >= LO and x.max() <= HI
    return x, ctr


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
    ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
    fidx = np.load(f"{FLAT}/train/cand_fidx.npy")
    day_f = (ftr["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400
    day = day_f[fidx]
    d0 = day == day.min()
    res = {"name": NAME, "def": f"clip(log(tte_it_ctr / {PRIOR}), {LO}, {HI}) float32"}
    for sp in SPLITS:
        x, r = X[sp], RAW[sp]
        res[sp] = dict(rows=int(len(x)), mean=float(x.mean()), sd=float(x.std()), min=float(x.min()), max=float(x.max()),
                       share_at_lo=float((x <= LO).mean()), share_at_hi=float((x >= HI).mean()),
                       share_prior=float((np.abs(r - PRIOR) < 1e-7).mean()), median_raw=float(np.median(r)), mean_raw=float(r.mean()),
                       q=dict(zip(["p01", "p10", "p25", "p50", "p75", "p90", "p99"], np.quantile(x, [.01, .1, .25, .5, .75, .9, .99]).round(4).tolist())))
    res["train_day0"] = dict(day=int(day.min()), rows=int(d0.sum()), share=float(d0.mean()), all_zero=bool((X["train"][d0] == 0).all()))
    xt_nd0 = X["train"][~d0]
    res["train_excl_day0"] = dict(mean=float(xt_nd0.mean()), share_prior=float((xt_nd0 == 0).mean()))
    edges_a = np.array([-np.inf, -2 + 1e-6, -1.5, -1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1, 1.5, 2 - 1e-6, np.inf])  # natural bins incl. the clip atoms + the prior atom
    q = np.unique(np.quantile(X["train"], np.linspace(0, 1, 11)[1:-1]))
    edges_b = np.r_[-np.inf, q + 1e-6, np.inf]
    for other in ("val", "test"):
        res[f"psi_train_{other}"] = dict(natural_bins=psi(X["train"], X[other], edges_a), train_quantile_bins=psi(X["train"], X[other], edges_b))
        res[f"psi_train_excl_day0_{other}"] = dict(natural_bins=psi(xt_nd0, X[other], edges_a))
    res["psi_val_test"] = dict(natural_bins=psi(X["val"], X["test"], edges_a))
    res["natural_bin_shares"] = {sp: (np.histogram(X[sp], edges_a)[0] / len(X[sp])).round(4).tolist() for sp in SPLITS}
    res["natural_bin_labels"] = ["-2(clip)", "(-2,-1.5]", "(-1.5,-1]", "(-1,-.5]", "(-.5,-.2]", "(-.2,-.05]", "(-.05,.05] incl prior", "(.05,.2]", "(.2,.5]", "(.5,1]", "(1,1.5]", "(1.5,2)", "2(clip)"]
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/{NAME}_parity.json", "w"), indent=1)
    assert res["psi_train_val"]["natural_bins"] < 0.5 and res["psi_train_test"]["natural_bins"] < 0.5, "PSI >= 0.5: parity rule broken"


if __name__ == "__main__":
    main()
