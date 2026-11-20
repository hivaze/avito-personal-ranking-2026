# Builds the 'fresh exposure' counts: for each candidate of a val or test feed, the number of
# other users' val and test feeds that show the same item, in six time windows (strictly earlier,
# earlier within 24 hours, same day, within one day either side, whole period, strictly later).
# Feeds of the same user are never counted. No label is read.
# How it was run: cd <project root> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
# NUMBA_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s1_build.py (from the docstring).
# What it produced: out/cnt_main.npy and coverage tables (results/fresh_s1_build.log). The
# strictly-earlier count is 1 or more for 10.8% of val, 20.1% of public-like and 28.0% of
# private-like candidates.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/fresh/pregate/s1_build.py
"""Step 1: build the fresh-exposure counts for val and test rows and report coverage (label-free).

Writes (own directory only): out/cnt_main.npy [N_val + N_test, 6] int32 (windows a, a24, b, c, d, e; uid-0 feeds are sources),
out/cnt_nzsrc.npy (uid-0 feeds never sources), out/rowmeta.npz (split, part, cold, day, gid), out/s1_coverage.json.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s1_build.py
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import BOUND, OUT, WIN, count, load_union, psi, PSI_LABELS, PSI_EDGES

t0 = time.time()
F, R = load_union()
N = len(R["item"]); ng = len(F["ts"])
print(f"union: {ng} feeds, {N} rows; uid-0 feeds {(F['uid'] == 0).sum()}", flush=True)
cm = count(R["item"], R["gid"], F, R["item"], R["gid"])
print(f"main counts {time.time() - t0:.0f}s", flush=True)
cz = count(R["item"], R["gid"], F, R["item"], R["gid"], src_ok_feed=F["uid"] != 0)
print(f"nzsrc counts {time.time() - t0:.0f}s", flush=True)
# identities of the windows (sanity): a + e + same-tick = d ; a24 <= a ; b <= d ; c <= d
assert (cm[:, 0] + cm[:, 5] <= cm[:, 4]).all() and (cm[:, 1] <= cm[:, 0]).all() and (cm[:, 2] <= cm[:, 4]).all()
assert (cz <= cm).all()
# brute-force check of 300 random rows against a direct python count
rng = np.random.default_rng(0)
gid = R["gid"]; ts_r = F["ts"][gid]; us_r = F["eff_user"][gid]; day_r = F["day"][gid]
order = np.argsort(R["item"], kind="stable"); si = R["item"][order]
for q in rng.choice(N, 300, replace=False):
    lo, hi = np.searchsorted(si, R["item"][q]), np.searchsorted(si, R["item"][q], side="right")
    rr = order[lo:hi]; rr = rr[us_r[rr] != us_r[q]]
    t = ts_r[q]
    ref = [(ts_r[rr] < t).sum(), ((ts_r[rr] < t) & (ts_r[rr] >= t - 86400)).sum(), (day_r[rr] == day_r[q]).sum(),
           (np.abs(ts_r[rr] - t) <= 86400).sum(), len(rr), (ts_r[rr] > t).sum()]
    assert list(cm[q]) == [int(x) for x in ref], (q, cm[q], ref)
print("brute-force check of 300 rows: OK", flush=True)
np.save(f"{OUT}/cnt_main.npy", cm); np.save(f"{OUT}/cnt_nzsrc.npy", cz)
split_r = F["split"][gid]
part = np.where(split_r == 0, 0, np.where(ts_r < BOUND, 1, 2)).astype(np.int8)  # 0 val, 1 public-like, 2 private-like
np.savez(f"{OUT}/rowmeta.npz", split=split_r, part=part, cold=R["cold"], day=day_r, gid=gid, ts=ts_r, uid0=F["uid"][gid] == 0)
PART = ["val", "public-like", "private-like"]
res = {"windows": WIN, "parts": PART, "n_rows": {p: int((part == i).sum()) for i, p in enumerate(PART)},
       "cold_share": {p: float(R["cold"][part == i].mean()) for i, p in enumerate(PART)}}
print("rows", res["n_rows"], "cold share", res["cold_share"])
for tag, c in (("main", cm), ("nzsrc", cz)):
    res[tag] = {}
    print(f"\n=== coverage [{tag}] share of candidates with count >= 1 / >= 2 / >= 5 (percent); mean count")
    print(f"{'window':6s} {'part':13s} {'subset':5s} {'>=1':>7s} {'>=2':>7s} {'>=5':>7s} {'mean':>7s}")
    for w, wn in enumerate(WIN):
        for i, p in enumerate(PART):
            for sub, m in (("all", part == i), ("cold", (part == i) & R["cold"]), ("warm", (part == i) & ~R["cold"])):
                x = c[m, w]
                r = dict(ge1=float((x >= 1).mean()), ge2=float((x >= 2).mean()), ge5=float((x >= 5).mean()), mean=float(x.mean()),
                         n=int(m.sum()))
                res[tag][f"{wn}|{p}|{sub}"] = r
                print(f"{wn:6s} {p:13s} {sub:5s} {100 * r['ge1']:7.2f} {100 * r['ge2']:7.2f} {100 * r['ge5']:7.2f} {r['mean']:7.3f}")
# by day (main): coverage of each window, cold / warm
res["by_day"] = {}
print("\n=== coverage by Moscow day [main], share with count >= 1 (percent): all / cold / warm")
for d in np.unique(day_r):
    m = day_r == d
    if m.sum() < 5000:
        continue
    line = f"day {d} rows {m.sum():8d} cold {R['cold'][m].mean():.3f} |"
    res["by_day"][int(d)] = {}
    for w, wn in enumerate(WIN):
        v = [float((cm[m, w] >= 1).mean()), float((cm[m & R["cold"], w] >= 1).mean()), float((cm[m & ~R["cold"], w] >= 1).mean())]
        res["by_day"][int(d)][wn] = v
        line += f" {wn} {100 * v[0]:5.1f}/{100 * v[1]:5.1f}/{100 * v[2]:5.1f} |"
    print(line)
# feeds with at least one covered candidate, and covered cold candidates per feed
print("\n=== feed level [main]: share of feeds with >= 1 covered candidate; mean number of covered candidates per feed")
for w, wn in enumerate(WIN):
    line = f"{wn:4s}"
    for i, p in enumerate(PART):
        fm = np.unique(gid[part == i])
        nc = np.bincount(gid, weights=(cm[:, w] >= 1), minlength=ng)[fm]
        ncc = np.bincount(gid, weights=(cm[:, w] >= 1) & R["cold"], minlength=ng)[fm]
        res.setdefault("feed_level", {})[f"{wn}|{p}"] = dict(feeds_ge1=float((nc >= 1).mean()), mean_cov=float(nc.mean()),
                                                              feeds_coldcov_ge1=float((ncc >= 1).mean()))
        line += f" | {p}: feeds>=1 {100 * (nc >= 1).mean():5.1f}% mean {nc.mean():5.2f} cold-covered feeds {100 * (ncc >= 1).mean():5.1f}%"
    print(line)
# the indicator "cold in train but shown elsewhere": share of ALL rows and of cold rows
print("\n=== indicator cold & shown elsewhere (cs_w) [main]: share of all rows / of cold rows (percent)")
for w, wn in enumerate(WIN):
    line = f"{wn:4s}"
    for i, p in enumerate(PART):
        m = part == i
        cs = R["cold"][m] & (cm[m, w] >= 1)
        res.setdefault("cs", {})[f"{wn}|{p}"] = [float(cs.mean()), float(cs.sum() / R["cold"][m].sum())]
        line += f" | {p} {100 * cs.mean():5.2f} / {100 * cs.sum() / R['cold'][m].sum():5.2f}"
    print(line)
json.dump(res, open(f"{OUT}/s1_coverage.json", "w"), indent=1)
print(f"\nDONE {time.time() - t0:.0f}s")
