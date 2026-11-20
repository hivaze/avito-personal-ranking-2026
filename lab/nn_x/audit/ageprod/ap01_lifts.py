"""ap01: listing-age lift table L[M bucket, age bin] = log(prod lift / exploration lift) from the FULL train set
(holdout excluded), following nn_x/audit/avito/a08_age_train.py exactly for the age proxy (f4 -> hours via
life_f4_age_knots.npy) and the 8 age bins. Production = cls 0 (NN p_random <= 0.5, all M), exploration-like = cls 1
(p_random > 0.5, M >= 50) of data_usage out/train_feeds.parquet. Also: the a08 definition (p > 0.9 vs p < 0.05, M >= 50),
the same lifts on the U2 proxy sample, and the val / HO lifts for reference. Writes out/age_lifts.json + out/ap01.log.
CPU only, read-only on every input."""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
DU = ROOT / "nn_x/insights/data_usage/out"
G = DU / "gb"
RV = ROOT / "nn_x/insights/review/out/gb"
OUT = ROOT / "nn_x/audit/ageprod/out"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
logf = open(OUT / "ap01.log", "w")


def log(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


# ---- age proxy exactly as a06/a07/a08 ----
KNOTS = np.load(ROOT / "data/feats/life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6
EDGES = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]
LABELS = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]
NB = len(LABELS)


def age_hours(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = KNOTS[k] * (1 - w) + KNOTS[k + 1] * w
    return np.where(x < CAP, a, np.nan)


def age_bin(f4):
    return np.digitize(np.nan_to_num(age_hours(f4), nan=1e8), EDGES) - 1


def table(b, y, name):
    """per-bin rows, positives, rate, lift vs base; SE of log rate ~ sqrt((1-r)/(r n))."""
    n = np.bincount(b, minlength=NB).astype(np.float64)
    p = np.bincount(b, weights=y, minlength=NB)
    base = y.mean()
    rate = p / np.maximum(n, 1)
    lift = rate / base
    se = np.sqrt((1 - rate) / np.maximum(p, 1))
    log(f"{name:34s} rows {int(n.sum()):>11d} base {base:.5f} | share {np.round(n / n.sum(), 3)} | rate% {np.round(rate * 100, 3)} | lift {np.round(lift, 3)} | pos {p.astype(int)}")
    return dict(rows=n.astype(int).tolist(), pos=p.astype(int).tolist(), rate=rate.tolist(), base=float(base), lift=lift.tolist(), se_log=se.tolist())


def lifts(prod, expl):
    L = np.log(np.array(prod["lift"]) / np.array(expl["lift"]))
    se = np.sqrt(np.array(prod["se_log"]) ** 2 + np.array(expl["se_log"]) ** 2)
    return L, se


res = dict(age_proxy="age_hours(f4) via data/feats/life_f4_age_knots.npy as a06/a07/a08; f4 >= 63/64-1e-6 -> NaN -> 1e8 h (last bin)",
           edges_hours=EDGES[:-1] + ["inf"], labels=LABELS,
           definition="L[b] = log((rate_prod[b]/base_prod) / (rate_expl[b]/base_expl)); unweighted positive rates over candidate rows; "
                      "production = cls 0 (NN p_random <= 0.5; all M), exploration-like = cls 1 (p_random > 0.5, M>=50); holdout (cls 2) excluded",
           groups={})

# ---- full train ----
t = pl.read_parquet(DU / "train_feeds.parquet", columns=["M", "cls", "prand_nn"])
M = t["M"].to_numpy().astype(np.int64)
cls = t["cls"].to_numpy()
prand = t["prand_nn"].to_numpy()
n_feeds = len(M)
log(f"train feeds {n_feeds}: cls counts {np.bincount(cls)}")
f4 = np.load(ROOT / "data/feats/train/f4.npy", mmap_mode="r")
y_all = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")
N = int(M.sum())
assert f4.shape[0] == N == y_all.shape[0], (f4.shape, N, y_all.shape)
log("computing age bins on", N, "train rows")
b_all = np.empty(N, np.int8)
CH = 20_000_000
for s in range(0, N, CH):
    b_all[s:s + CH] = age_bin(np.asarray(f4[s:s + CH]))
y = np.asarray(y_all).astype(np.float32)
cls_row = np.repeat(cls, M)
M_row = np.repeat(M, M)
prand_row = np.repeat(prand, M)
log("rows by cls", np.bincount(cls_row))

grp = {}
grp["expl_cls1_M>=50"] = table(b_all[cls_row == 1], y[cls_row == 1], "exploration cls1 (M>=50)")
grp["prod_all"] = table(b_all[cls_row == 0], y[cls_row == 0], "production cls0 (all M)")
MB = [("M<50", 1, 49), ("M>=50", 50, 10 ** 6), ("M1-16", 1, 16), ("M17-32", 17, 32), ("M33-49", 33, 49), ("M50-99", 50, 99), ("M100-199", 100, 199), ("M>=200", 200, 10 ** 6)]
for name, lo, hi in MB:
    m = (cls_row == 0) & (M_row >= lo) & (M_row <= hi)
    grp[f"prod_{name}"] = table(b_all[m], y[m], f"production cls0 {name}")
grp["holdout_cls2"] = table(b_all[cls_row == 2], y[cls_row == 2], "holdout cls2 (reference only)")
# a08 definition check (p>0.9 random-like vs p<0.05 production-like, M>=50, holdout excluded)
m_rl = (prand_row > 0.9) & (M_row >= 50) & (cls_row != 2)
m_pr = (prand_row < 0.05) & (M_row >= 50) & (cls_row != 2)
grp["a08_randomlike_p>0.9_M>=50"] = table(b_all[m_rl], y[m_rl], "a08 random-like p>0.9 M>=50")
grp["a08_productionlike_p<0.05_M>=50"] = table(b_all[m_pr], y[m_pr], "a08 production-like p<0.05 M>=50")
res["groups"] = grp

log("== log(prod lift / expl lift) by age bin " + str(LABELS) + " ==")
LT = {}
for name in ["prod_all", "prod_M<50", "prod_M>=50", "prod_M1-16", "prod_M17-32", "prod_M33-49", "prod_M50-99", "prod_M100-199", "prod_M>=200"]:
    L, se = lifts(grp[name], grp["expl_cls1_M>=50"])
    LT[name] = dict(L=L.tolist(), se=se.tolist())
    log(f"  {name:14s} vs expl cls1: L {np.round(L, 3)}  se {np.round(se, 3)}")
L08, se08 = lifts(grp["a08_productionlike_p<0.05_M>=50"], grp["a08_randomlike_p>0.9_M>=50"])
LT["a08_check"] = dict(L=L08.tolist(), se=se08.tolist())
log(f"  a08 check (p<0.05 vs p>0.9, M>=50): L {np.round(L08, 3)} (a08 log: 0.891 0.763 0.545 0.199 -0.021 -0.134 -0.119 -0.116)")
Lh, seh = lifts(grp["prod_M>=50"], grp["holdout_cls2"])
log(f"  prod M>=50 vs HOLDOUT (cls2, not used): L {np.round(Lh, 3)}")
res["lifts"] = LT

# ---- the U2 proxy sample itself (what the arm's training rows look like) ----
m2 = dict(np.load(RV / "U2_meta.npz"))
X2 = np.load(RV / "U2_X.npy", mmap_mode="r")
feats = json.load(open(G / "feats.json"))
j4 = feats.index("f4")
b2 = age_bin(np.asarray(X2[:, j4]))
fid2 = np.repeat(np.arange(len(m2["M"])), m2["M"])
c2 = m2["cls"][fid2]; M2 = m2["M"][fid2]; y2 = m2["y"]
u2 = {}
u2["expl"] = table(b2[c2 == 1], y2[c2 == 1], "U2 exploration cls1")
u2["prod_M<50"] = table(b2[(c2 == 0) & (M2 < 50)], y2[(c2 == 0) & (M2 < 50)], "U2 production M<50")
u2["prod_M>=50"] = table(b2[(c2 == 0) & (M2 >= 50)], y2[(c2 == 0) & (M2 >= 50)], "U2 production M>=50")
for k in ["prod_M<50", "prod_M>=50"]:
    L, se = lifts(u2[k], u2["expl"])
    u2[k]["L_vs_U2_expl"] = L.tolist()
    log(f"  U2-sample {k} vs U2 expl: L {np.round(L, 3)} se {np.round(se, 3)}")
res["U2_sample"] = u2
# row shares of the age bins among U2 production rows (where the offset acts)
log(f"U2 production rows by age bin: {np.bincount(b2[c2 == 0], minlength=NB)} ({np.round(np.bincount(b2[c2 == 0], minlength=NB) / (c2 == 0).sum(), 3)})")

# ---- val and HO reference lifts ----
mv = dict(np.load(G / "val_meta.npz"))
Xv = np.load(G / "val_X.npy", mmap_mode="r")
bv = age_bin(np.asarray(Xv[:, j4]))
res["val"] = table(bv, mv["y"].astype(np.float32), "VAL (reference)")
mh = dict(np.load(G / "HO_meta.npz"))
Xh = np.load(G / "HO_X.npy", mmap_mode="r")
bh = age_bin(np.asarray(Xh[:, j4]))
res["HO"] = table(bh, mh["y"].astype(np.float32), "HO (reference)")
np.save(OUT / "val_agebin.npy", bv.astype(np.int8))
np.save(OUT / "HO_agebin.npy", bh.astype(np.int8))
np.save(OUT / "val_agehours.npy", age_hours(np.asarray(Xv[:, j4])).astype(np.float32))
np.save(OUT / "HO_agehours.npy", age_hours(np.asarray(Xh[:, j4])).astype(np.float32))
json.dump(res, open(OUT / "age_lifts.json", "w"), indent=1)
log("wrote out/age_lifts.json; done")
