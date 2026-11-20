"""hzcd pre-flight (nn_x/w5/hzcd/preflight.txt holds the pre-declared kill rule; this script appends the numbers).
CPU only. Compares, per listing-age bucket, the train-row distribution of the horizon-limited popularity it_hist_hz_*
(data/feats_new/train) and of the live input it_hist_*_luo (nn/feat_cache/train) with val / public / private luo.
  OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/hzcd/preflight.py
"""
import json
import os
import time

import numpy as np
import polars as pl

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
FC = f"{ROOT}/nn/feat_cache"
FN = f"{ROOT}/data/feats_new"
HERE = os.path.dirname(os.path.abspath(__file__))
BOUND = 1790497920  # calib.py / pmix.py: public = test feeds with timestamp < BOUND
KN = np.load(f"{ROOT}/data/feats/life_f4_age_knots.npy")
EDGES = [0, 1, 3, 7, 24, 76, 1e9]
LAB = ["<1d", "1-3d", "3-7d", "7-24d", "24-76d", ">76d"]
EDGES2 = [0, 1, 3, 7, 10, 24, 76, 1e9]  # diagnostic split of 7-24d
LAB2 = ["<1d", "1-3d", "3-7d", "7-10d", "10-24d", "24-76d", ">76d"]
COLS = [("all", "it_hist_all_luo", "it_hist_hz_all"), ("et1", "it_hist_et1_luo", "it_hist_hz_et1"),
        ("et2", "it_hist_et2_luo", "it_hist_hz_et2"), ("et3", "it_hist_et3_luo", "it_hist_hz_et3"),
        ("et4", "it_hist_et4_luo", "it_hist_hz_et4")]
CAP = 100000
t0 = time.time()
out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def age_days(x):  # exactly nn_x/audit/features/s06_joint_age_cov.py
    x = x.astype(np.float64)
    u = x * 64
    k = np.clip(np.floor(u).astype(int), 1, 62)
    w = u - k
    a = KN[k] * (1 - w) + KN[k + 1] * w
    return np.where(x < 63 / 64, a, 5 * 365 * 24.0) / 24.0


def stats(cnt):
    """zero rate, mean log1p, q50/q75/q90/q99 of the integer count, P(>=4) via bincount (exact, O(n))."""
    n = len(cnt)
    if n == 0:
        return dict(n=0)
    c = np.minimum(cnt, CAP).astype(np.int64)
    bc = np.bincount(c, minlength=CAP + 1)
    cdf = np.cumsum(bc) / n
    q = {f"q{p}": int(np.searchsorted(cdf, p / 100, side="left")) for p in (50, 75, 90, 99)}
    return dict(n=int(n), zero=float(bc[0] / n), mean_log1p=float(np.log1p(np.arange(CAP + 1)) @ bc / n),
                ge4=float(1 - cdf[3]), **q)


def js(p, q):
    m = 0.5 * (p + q)
    e = 1e-9
    return 0.5 * (p * np.log((p + e) / (m + e))).sum() + 0.5 * (q * np.log((q + e) / (m + e))).sum()


# ---------------------------------------------------------------- train feed meta
f = pl.read_parquet(f"{FLAT}/train/feeds.parquet", columns=["M", "timestamp"])
M = f["M"].to_numpy().astype(np.int64)
n = len(M)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(n)).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hmask = np.zeros(n, bool)
hmask[hold] = True
pop_feed = {"trAll": ~hmask, "trPool": (p > 0.5) & (M >= 50) & ~hmask, "trHold": hmask}
fid = np.repeat(np.arange(n, dtype=np.int32), M)
f4 = np.load(f"{FLAT}/train/cand_f4.npy")
assert len(f4) == len(fid) == int(M.sum())
a_tr = age_days(f4)
del f4
b_tr = np.digitize(a_tr, EDGES[1:-1]).astype(np.int8)
b2_tr = np.digitize(a_tr, EDGES2[1:-1]).astype(np.int8)
del a_tr
P(f"train: {n} feeds, {len(fid)} rows; trAll {int(pop_feed['trAll'].sum())} feeds, trPool {int(pop_feed['trPool'].sum())}, "
  f"trHold {int(hmask.sum())}; {time.time() - t0:.0f}s")
# val / test
fv = np.load(f"{FLAT}/val/cand_f4.npy")
b_val = np.digitize(age_days(fv), EDGES[1:-1]).astype(np.int8)
b2_val = np.digitize(age_days(fv), EDGES2[1:-1]).astype(np.int8)
ft = pl.read_parquet(f"{FLAT}/test/feeds.parquet", columns=["M", "timestamp"])
Mt = ft["M"].to_numpy().astype(np.int64)
pub = (ft["timestamp"].to_numpy() < BOUND)[np.repeat(np.arange(len(Mt)), Mt)]
f4t = np.load(f"{FLAT}/test/cand_f4.npy")
b_te = np.digitize(age_days(f4t), EDGES[1:-1]).astype(np.int8)
b2_te = np.digitize(age_days(f4t), EDGES2[1:-1]).astype(np.int8)
del fv, f4t

RES = {}
for tag, luo_n, hz_n in COLS:
    luo = np.load(f"{FC}/train/{luo_n}.npy")
    hz = np.load(f"{FN}/train/{hz_n}.npy")
    assert luo.shape == hz.shape == fid.shape and not np.isnan(hz).any()
    d = luo - hz
    sub_ok = bool((d >= 0).all())
    changed = float((d != 0).mean())
    P(f"\n### column {tag}: train {luo_n} vs {hz_n}: hz <= luo everywhere {sub_ok}; rows changed {changed:.4f}; "
      f"{time.time() - t0:.0f}s")
    del d
    vl = np.load(f"{FC}/val/{luo_n}.npy")
    tl = np.load(f"{FC}/test/{luo_n}.npy")
    R = {}
    for pop, fm in pop_feed.items():
        rm = fm[fid]
        for nm, arr in (("luo", luo), ("hz", hz)):
            for j, lab in enumerate(LAB):
                R[(pop, nm, lab)] = stats(arr[rm & (b_tr == j)])
            for j, lab in enumerate(LAB2):
                if lab in ("7-10d", "10-24d"):
                    R[(pop, nm, lab)] = stats(arr[rm & (b2_tr == j)])
        del rm
    for j, lab in enumerate(LAB):
        R[("val", "luo", lab)] = stats(vl[b_val == j])
        R[("public", "luo", lab)] = stats(tl[pub & (b_te == j)])
        R[("private", "luo", lab)] = stats(tl[~pub & (b_te == j)])
    for j, lab in enumerate(LAB2):
        if lab in ("7-10d", "10-24d"):
            R[("val", "luo", lab)] = stats(vl[b2_val == j])
            R[("public", "luo", lab)] = stats(tl[pub & (b2_te == j)])
            R[("private", "luo", lab)] = stats(tl[~pub & (b2_te == j)])
    # table
    hdr = f"{'bucket':7s} {'pop':8s} {'col':4s} {'rows':>10s} {'zero':>6s} {'mlog1p':>7s} {'P>=4':>6s} {'q50':>5s} {'q75':>5s} {'q90':>5s} {'q99':>6s} | {'|dzero-val|':>11s} {'|dmean-val|':>11s}"
    P(hdr)
    for lab in LAB2:
        for pop in ("trAll", "trPool", "trHold", "val", "public", "private"):
            for nm in (("luo", "hz") if pop.startswith("tr") else ("luo",)):
                s = R[(pop, nm, lab)]
                v = R[("val", "luo", lab)]
                dz = abs(s["zero"] - v["zero"])
                dm = abs(s["mean_log1p"] - v["mean_log1p"])
                P(f"{lab:7s} {pop:8s} {nm:4s} {s['n']:>10d} {s['zero']:6.3f} {s['mean_log1p']:7.3f} {s['ge4']:6.3f} "
                  f"{s['q50']:5d} {s['q75']:5d} {s['q90']:5d} {s['q99']:6d} | {dz:11.3f} {dm:11.3f}")
    # JS of the 12-cell (age x cold) distribution
    dist = {}
    for pop, fm in pop_feed.items():
        rm = fm[fid]
        for nm, arr in (("luo", luo), ("hz", hz)):
            cell = b_tr[rm].astype(np.int64) * 2 + (arr[rm] == 0)
            dist[(pop, nm)] = np.bincount(cell, minlength=12) / len(cell)
        del rm
    dist[("val", "luo")] = np.bincount(b_val.astype(np.int64) * 2 + (vl == 0), minlength=12) / len(vl)
    dist[("public", "luo")] = np.bincount(b_te[pub].astype(np.int64) * 2 + (tl[pub] == 0), minlength=12) / int(pub.sum())
    dist[("private", "luo")] = np.bincount(b_te[~pub].astype(np.int64) * 2 + (tl[~pub] == 0), minlength=12) / int((~pub).sum())
    P("JS divergence of the 12-cell (age x cold) distribution:")
    JS = {}
    for pop in ("trAll", "trPool", "trHold"):
        for nm in ("luo", "hz"):
            for ref in ("val", "private"):
                JS[f"{pop}/{nm}->{ref}"] = float(js(dist[(pop, nm)], dist[(ref, "luo")]))
    for k, v in JS.items():
        P(f"  JS({k}) = {v:.4f}")
    P(f"  JS(val->public) = {js(dist[('val', 'luo')], dist[('public', 'luo')]):.4f}; JS(val->private) = "
      f"{js(dist[('val', 'luo')], dist[('private', 'luo')]):.4f}")
    RES[tag] = dict(sub_ok=sub_ok, changed=changed, table={"|".join(k): v for k, v in R.items()}, js=JS)
    del luo, hz, vl, tl

# ---------------------------------------------------------------- verdict (pre-declared rule, column 'all')
P("\n### KILL-RULE EVALUATION (column it_hist_all_luo vs it_hist_hz_all; buckets 3-7d, 7-24d; populations trAll, trPool)")
T = RES["all"]["table"]
cells = []
for lab in ("3-7d", "7-24d"):
    v = T[f"val|luo|{lab}"]
    for pop in ("trAll", "trPool"):
        l_, h_ = T[f"{pop}|luo|{lab}"], T[f"{pop}|hz|{lab}"]
        dz_l, dz_h = abs(l_["zero"] - v["zero"]), abs(h_["zero"] - v["zero"])
        dm_l, dm_h = abs(l_["mean_log1p"] - v["mean_log1p"]), abs(h_["mean_log1p"] - v["mean_log1p"])
        ok = (dz_h < dz_l) and (dm_h <= dm_l)
        cells.append(ok)
        P(f"  {lab:6s} {pop:7s}: zero gap to val luo {dz_l:.4f} -> hz {dz_h:.4f} ({'closer' if dz_h < dz_l else 'NOT closer'}); "
          f"mean log1p gap luo {dm_l:.4f} -> hz {dm_h:.4f} ({'ok' if dm_h <= dm_l else 'NOT ok'}) => {'PASS' if ok else 'FAIL'}")
verdict = "PASS" if all(cells) else "FAIL"
P(f"VERDICT (pre-declared rule): {verdict} ({sum(cells)}/4 cells); {time.time() - t0:.0f}s")
# context: 1-3d
for lab in ("1-3d",):
    v = T[f"val|luo|{lab}"]
    for pop in ("trAll", "trPool"):
        l_, h_ = T[f"{pop}|luo|{lab}"], T[f"{pop}|hz|{lab}"]
        P(f"  context {lab} {pop}: zero luo {l_['zero']:.3f} hz {h_['zero']:.3f} val {v['zero']:.3f}; mean log1p luo "
          f"{l_['mean_log1p']:.3f} hz {h_['mean_log1p']:.3f} val {v['mean_log1p']:.3f}")
json.dump(dict(verdict=verdict, cells=cells, res=RES), open(f"{HERE}/preflight.json", "w"), indent=1, default=str)
with open(f"{HERE}/preflight.txt", "a") as fo:
    fo.write(f"\n==== RESULTS appended {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} (preflight.py; json: preflight.json)\n")
    fo.write("\n".join(out_lines) + "\n")
