"""ap03: paired per-feed readout of the AGE-PROD proxy arms (seed-mean over 11-14), slices, gate verdict (plan.txt),
calibration by age bin, and the NN spec JSON (out/age_lifts_mbucket.json). Reads out/gb/res_*.npz; writes out/results.txt."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
DU = ROOT / "nn_x/insights/data_usage/out"
G = DU / "gb"
RV = ROOT / "nn_x/insights/review/out/gb"
AP = ROOT / "nn_x/audit/ageprod/out"
OUT = AP / "gb"
SEEDS = [11, 12, 13, 14]
ROUNDS = [100, 200, 300, 400]
PRIMARY = 200
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def load(arm):
    fs = [OUT / f"res_{arm.replace('+', '_')}_s{s}.npz" for s in SEEDS]
    if not all(f.exists() for f in fs):
        return None
    d = [np.load(f) for f in fs]
    keys = [f"{p}{r}" for p in ("val", "stale", "ho") for r in ROUNDS]
    out = {k: np.mean([x[k] for x in d], 0).astype(np.float64) for k in keys}
    out["pred_val200"] = np.mean([x["pred_val200"] for x in d], 0).astype(np.float64)
    out["per_seed"] = {k: [x[k] for x in d] for k in ("val200", "ho200", "val300", "ho300")}
    return out


def pm(d):
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(d):
    m, se = pm(d)
    return f"{m:+.5f}±{se:.5f} (z{m / se if se > 0 else 0:+.1f})"


# ---- val / HO structure ----
mv = dict(np.load(G / "val_meta.npz")); Mv = mv["M"].astype(np.int64); yv = mv["y"].astype(np.float64)
offv = np.concatenate([[0], np.cumsum(Mv)[:-1]]); nv = len(Mv); fidv = np.repeat(np.arange(nv), Mv)
mh = dict(np.load(G / "HO_meta.npz")); Mh = mh["M"].astype(np.int64); yh = mh["y"].astype(np.float64)
nh = len(Mh); fidh = np.repeat(np.arange(nh), Mh)
fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet").sort("fidx")
assert (fv["M"].to_numpy() == Mv).all()
tsv = fv["timestamp"].to_numpy().astype(np.int64) + 3 * 3600
dayv = tsv // 86400
TUE, THU = 20718, 20720
ageh_v = np.load(AP / "val_agehours.npy").astype(np.float64); ageh_v = np.where(np.isnan(ageh_v), 1e8, ageh_v)
ageh_h = np.load(AP / "HO_agehours.npy").astype(np.float64); ageh_h = np.where(np.isnan(ageh_h), 1e8, ageh_h)
binv = np.load(AP / "val_agebin.npy").astype(np.int64)
LAB8 = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]
EDGES8 = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]


def feed_age_slices(M, y, ageh, fid, n):
    npos = np.bincount(fid, weights=y, minlength=n)
    mean_age = np.bincount(fid, weights=y * ageh, minlength=n) / np.maximum(npos, 1)
    young = np.full(n, 1e8); np.minimum.at(young, fid[y > 0], ageh[y > 0])
    any3h = np.bincount(fid, weights=y * (ageh < 3), minlength=n) > 0
    c04 = np.digitize(mean_age, [24, 72, 168, 576, 1824])  # <1d, 1-3d, 3-7d, 7-24d, 24-76d, >76d
    yb = np.digitize(young, EDGES8) - 1
    return dict(npos=npos, mean_age=mean_age, c04=c04, youngest8=yb, any3h=any3h)


SV = feed_age_slices(Mv, yv, ageh_v, fidv, nv)
SH = feed_age_slices(Mh, yh, ageh_h, fidh, nh)
C04 = ["<1d", "1-3d", "3-7d", "7-24d", "24-76d", ">76d"]

ARMS = ["base", "age1", "tprod1", "tprod1+age1", "ageh1", "age0.5"]
R = {a: load(a) for a in ARMS}
missing = [a for a in ARMS if R[a] is None]
log(f"AGE-PROD proxy gate readout ({time.strftime('%FT%TZ', time.gmtime())}); seeds {SEEDS}; missing arms: {missing}")

# ---- reproduction check vs r2 ----
log("\n== reproduction check of base / tprod1 vs the stored r2 fresh-sample results (same seeds, same params) ==")
for a in ["base", "tprod1"]:
    if R[a] is None:
        continue
    for s in SEEDS:
        mine = np.load(OUT / f"res_{a}_s{s}.npz"); ref = np.load(RV / f"res_{a}_s{s}.npz")
        log(f"  {a:7s} s{s}: val200 mine {mine['val200'].mean():.5f} r2 {ref['val200'].mean():.5f} corr {np.corrcoef(mine['val200'], ref['val200'])[0, 1]:.4f} | ho200 mine {mine['ho200'].mean():.5f} r2 {ref['ho200'].mean():.5f}")

# ---- levels ----
log("\n== seed-mean levels (val / stale-val / holdout) by rounds ==")
for a in ARMS:
    if R[a] is None:
        continue
    log(f"  {a:12s} " + " | ".join(f"r{r}: val {R[a][f'val{r}'].mean():.5f} stale {R[a][f'stale{r}'].mean():.5f} ho {R[a][f'ho{r}'].mean():.5f}" for r in ROUNDS))

# ---- paired contrasts ----
CON = [("age1", "base"), ("tprod1", "base"), ("tprod1+age1", "tprod1"), ("tprod1+age1", "base"), ("ageh1", "base"), ("age0.5", "base"), ("tprod1+age1", "age1")]
log("\n== paired per-feed deltas (seed-mean), all rounds: val | stale-val | holdout ==")
D = {}
for a, b in CON:
    if R[a] is None or R[b] is None:
        continue
    log(f"  {a} - {b}:")
    for r in ROUNDS:
        dv = R[a][f"val{r}"] - R[b][f"val{r}"]; ds = R[a][f"stale{r}"] - R[b][f"stale{r}"]; dh = R[a][f"ho{r}"] - R[b][f"ho{r}"]
        D[(a, b, r)] = (dv, ds, dh)
        log(f"    r{r}: val {fmt(dv)} | stale {fmt(ds)} | ho {fmt(dh)}")
    # per-seed at 200 (seed spread)
    ps = [float((R[a]["per_seed"]["val200"][i] - R[b]["per_seed"]["val200"][i]).mean()) for i in range(len(SEEDS))]
    ph = [float((R[a]["per_seed"]["ho200"][i] - R[b]["per_seed"]["ho200"][i]).mean()) for i in range(len(SEEDS))]
    log(f"    per-seed val200 deltas {np.round(ps, 5).tolist()}; ho200 {np.round(ph, 5).tolist()}")


def slice_table(dv, dh, r, title):
    log(f"  -- {title} (r{r}) --")
    rows = []
    def one(name, mask_v, mask_h=None):
        m, se = pm(dv[mask_v]); share = mask_v.mean(); contrib = m * share
        s = f"    {name:22s} val n {mask_v.sum():5d} ({share:.3f}) {m:+.5f}±{se:.5f} contrib {contrib:+.5f}"
        if mask_h is not None and mask_h.sum() > 0:
            mh_, seh = pm(dh[mask_h]); s += f" | ho n {mask_h.sum():5d} {mh_:+.5f}±{seh:.5f}"
        log(s); rows.append((name, m, se, share, contrib))
    one("M<50", Mv < 50)
    one("M>=50", Mv >= 50, np.ones(nh, bool))
    one("Tue", dayv == TUE)
    one("Thu", dayv == THU)
    for i, lab in enumerate(C04):
        one(f"posmeanage {lab}", SV["c04"] == i, SH["c04"] == i)
    for i, lab in enumerate(LAB8):
        one(f"youngestpos {lab}", SV["youngest8"] == i, SH["youngest8"] == i)
    one("any pos <3h", SV["any3h"], SH["any3h"])
    one("no pos <3h", ~SV["any3h"], ~SH["any3h"])
    return rows


SL = {}
for (a, b) in [("age1", "base"), ("tprod1+age1", "tprod1"), ("tprod1", "base"), ("ageh1", "base")]:
    for r in (PRIMARY, 300):
        if (a, b, r) in D:
            dv, ds, dh = D[(a, b, r)]
            SL[(a, b, r)] = slice_table(dv, dh, r, f"slices of {a} - {b}")

# ---- gate ----
log("\n== GATE (plan.txt, primary round 200, seed-mean paired deltas) ==")
verdict = None
if all(k in D for k in [("age1", "base", PRIMARY), ("tprod1+age1", "tprod1", PRIMARY)]):
    dv, ds, dh = D[("age1", "base", PRIMARY)]
    m1, se1 = pm(dv); mh1, seh1 = pm(dh); ms1, ses1 = pm(ds)
    no3 = ~SV["any3h"]; m3, se3 = pm(dv[no3]); contrib3 = m3 * no3.mean()
    dv2, ds2, dh2 = D[("tprod1+age1", "tprod1", PRIMARY)]
    m2, se2 = pm(dv2); mh2, seh2 = pm(dh2)
    c1 = m1 > se1; c2 = mh1 >= 0; c3 = (m3 > 0) and (contrib3 >= 0.5 * m1); c4 = m2 > se2
    log(f"  (i)   age1-base val      {m1:+.5f} ± {se1:.5f}  > +1 SE ? {c1}")
    log(f"  (ii)  age1-base holdout  {mh1:+.5f} ± {seh1:.5f}  >= 0 ? {c2}")
    log(f"  (iii) age1-base val on feeds with no positive <3h: {m3:+.5f} ± {se3:.5f} (share {no3.mean():.3f}, contrib {contrib3:+.5f} of {m1:+.5f}) > 0 and >= 50% ? {c3}")
    log(f"  (iv)  tprod1+age1 - tprod1 val {m2:+.5f} ± {se2:.5f}  > +1 SE ? {c4}")
    log(f"  advisory: stale-val age1-base {ms1:+.5f} ± {ses1:.5f} (>= -1 SE ? {ms1 >= -ses1}); holdout tprod1+age1-tprod1 {mh2:+.5f} ± {seh2:.5f} (>= -1 SE ? {mh2 >= -seh2}); "
        f"300-round sign: age1-base val {pm(D[('age1', 'base', 300)][0])[0]:+.5f}, tprod1+age1-tprod1 val {pm(D[('tprod1+age1', 'tprod1', 300)][0])[0]:+.5f}")
    verdict = "PASS" if (c1 and c2 and c3 and c4) else "FAIL"
    log(f"  VERDICT: {verdict}  (clauses i {c1}, ii {c2}, iii {c3}, iv {c4})")

# ---- calibration by age bin on val (softmax within feed x npos, as a06) ----
log("\n== val act/exp by age bin (r200 seed-mean preds; softmax within feed x npos) ==")
npos_v = SV["npos"]


def act_exp(s):
    z = s - np.repeat(np.maximum.reduceat(s, offv), Mv)
    e = np.exp(z); p = e / np.repeat(np.add.reduceat(e, offv), Mv)
    ex = p * npos_v[fidv]
    return np.bincount(binv, weights=yv, minlength=8) / np.maximum(np.bincount(binv, weights=ex, minlength=8), 1e-9)


log(f"  bins {LAB8}; val row share {np.round(np.bincount(binv, minlength=8) / len(binv), 3).tolist()}")
for a in ARMS:
    if R[a] is not None:
        log(f"  {a:12s} act/exp {np.round(act_exp(R[a]['pred_val200']), 3).tolist()}")

# ---- NN spec JSON ----
LJ = json.load(open(AP / "age_lifts.json"))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


ref = LJ["groups"]["expl_cls1_M>=50"]
spec = dict(
    description="per-M-bucket LISTING-AGE log lifts for a training-only additive logit offset (init_score / logit margin) on PRODUCTION "
                "train rows (NN p_random <= 0.5, i.e. cls 0 incl. every M<50 feed); pool / random-like (p_random > 0.5) and holdout feeds get no "
                "offset; never applied at inference. Applied value: buckets[i].L[age bin of the candidate]. Age bin = a08 bins of "
                "age_hours(f4) (data/feats/life_f4_age_knots.npy, f4 >= 63/64-1e-6 -> last bin). Gate verdict in nn_x/audit/ageprod/out/results.txt.",
    definition="L[b, k] = log( (rate_prod[b, k] / base_prod[b]) / (rate_expl[k] / base_expl) ); rate = positive rate over candidate rows "
               "(unweighted) of bucket-b production feeds (cls 0) / of exploration-like feeds (cls 1: p_random > 0.5, M >= 50), holdout excluded; "
               "full train (3,151,208 feeds). Single M>=50 exploration reference for every M bucket.",
    gate_verdict=verdict, proxy_arm="age1 = 1.0 x L on all production rows (nn_x/audit/ageprod/ap02_arms.py)",
    age_proxy=LJ["age_proxy"], age_edges_hours=EDGES8[:-1] + ["inf"], age_labels=LAB8,
    reference=dict(name="exploration-like cls1 M>=50", n_feeds=134118, rows=ref["rows"], pos=ref["pos"], rate=ref["rate"], base=ref["base"], lift=ref["lift"]),
    buckets=[], created=time.strftime("%FT%TZ", time.gmtime()), script="nn_x/audit/ageprod/ap01_lifts.py",
    inputs=dict(train_feeds=md5(DU / "train_feeds.parquet"), f4=md5(ROOT / "data/feats/train/f4.npy"), knots=md5(ROOT / "data/feats/life_f4_age_knots.npy"), target=md5(ROOT / "data/flat/train/cand_target.npy")))
tf = pl.read_parquet(DU / "train_feeds.parquet", columns=["M", "cls"])
Mt = tf["M"].to_numpy(); ct = tf["cls"].to_numpy()
for name, lo, hi in [("M<50", 1, 49), ("M>=50", 50, None)]:
    g = LJ["groups"][f"prod_{name}"]; L = LJ["lifts"][f"prod_{name}"]
    nfe = int(((ct == 0) & (Mt >= lo) & ((Mt <= hi) if hi else True)).sum())
    spec["buckets"].append(dict(name=name, lo=lo, hi=hi, n_feeds=nfe, rows=g["rows"], pos=g["pos"], rate=g["rate"], base=g["base"], lift=g["lift"], L=L["L"], se=L["se"]))
spec["finer_M_buckets_for_reference"] = {k: LJ["lifts"][k]["L"] for k in ["prod_M1-16", "prod_M17-32", "prod_M33-49", "prod_M50-99", "prod_M100-199", "prod_M>=200"]}
json.dump(spec, open(AP / "age_lifts_mbucket.json", "w"), indent=1)
log(f"\nwrote out/age_lifts_mbucket.json (verdict {verdict})")
open(AP / "results.txt", "w").write("\n".join(lines) + "\n")
