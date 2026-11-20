"""WAVE 10 R2 (XOFF) PREMISE CHECK (CPU only, no model). Is there a measurable examination (position) gradient in the LABELS of the
random-order (exploration) train feeds, and on val?

Rows:
  POOL    = the m02 pool feeds = NN p_random > 0.5 (eda/labels_position/out/train_prandom.npz, NaN -> 0, p := 0 for M < 50),
            M >= 50, minus the 30,000 holdout feeds (train.py rule: rng 12345 choice from p > 0.9 & M >= 50)  -> 134,118 feeds
  P>0.9 / 0.5<P<=0.9 splits of POOL; HOLDOUT (30K, p > 0.9) as information; PROD (p <= 0.05, M >= 50, 200K sample) as the
  confounded reference curve.
  VAL     = all 14,423 val feeds (random-order by construction; the organisers SHUFFLED the candidate order of val/test by one
            permutation and do not publish pos, readme), so the only available index is the candidate index.
Position = the displayed slot `pos` (train only; gaps = items already viewed by the user, removed from the dataset) and the
within-feed rank of pos (0 = first kept item); rkend = rank from the end (0 = last shown kept item, the truncation spike).
Tests (all exclude rkend 0, so the truncation spike cannot create a gradient):
  (T1) chi-square homogeneity of the positive rate across pos bins;
  (T2) feed-level paired test: mean normalised rank of the positives (rank/(M-1)) minus 0.5, SE over feeds (0 under no position effect);
  (T3) weighted LS slope of log(rate) on log1p(pos) over the bins;
  (T4) the same (T1)-(T3) on VAL by candidate index (expected flat: a permutation).
Output: premise_check.txt (table + tests) and g_noend_pos.json (log lift by raw pos on POOL rows, rkend 0 excluded; the offset table
candidate, smoothed), both in this directory.
"""
import json, os, sys, time
import numpy as np
from scipy import stats

os.environ.setdefault("OMP_NUM_THREADS", "12")
ROOT = "<repo>"
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
import pyarrow.parquet as pq

def feeds(split):
    t = pq.read_table(f"{ROOT}/data/flat/{split}/feeds.parquet", columns=["M", "cand_off", "timestamp"])
    return t.column("M").to_numpy().astype(np.int64), t.column("cand_off").to_numpy().astype(np.int64), t.column("timestamp").to_numpy()

M, off, ts = feeds("train")
n = len(M)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(n)).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
is_hold = np.zeros(n, bool); is_hold[hold] = True
pool = np.where((p > 0.5) & (M >= 50) & ~is_hold)[0]
print(f"train feeds {n}; p>0.5&M>=50 {(p > 0.5).sum()}; holdout {len(hold)}; POOL {len(pool)}", flush=True)
assert len(pool) == 134118, len(pool)

cpos = np.load(f"{ROOT}/data/flat/train/cand_pos.npy", mmap_mode="r")
ctgt = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")

def gather(sel):
    """rows of the feeds sel: pos (raw), rank, rkend, target, fid (local), M"""
    o, m = off[sel], M[sel]
    rows = np.concatenate([np.arange(a, a + b) for a, b in zip(o, m)])
    pos = np.asarray(cpos[rows]).astype(np.int64)
    y = np.asarray(ctgt[rows]).astype(np.int64)
    fid = np.repeat(np.arange(len(sel)), m)
    Mr = np.repeat(m, m)
    # within-feed rank of pos (ordinal; the builder keeps feed order, pos may be unsorted -> argsort per feed)
    order = np.lexsort((pos, fid))
    rank = np.empty_like(pos)
    start = np.concatenate([[0], np.cumsum(m)[:-1]])
    rank[order] = np.arange(len(pos)) - np.repeat(start, m)
    rkend = Mr - 1 - rank
    return dict(pos=pos, rank=rank, rkend=rkend, y=y, fid=fid, M=Mr, m=m)

BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 9), (10, 14), (15, 19), (20, 29), (30, 49), (50, 99), (100, 199), (200, 10**6)]

def table(d, key, name, lines, exclude_end=True):
    x = d[key]; y = d["y"]
    keep = d["rkend"] != 0 if exclude_end else np.ones_like(y, bool)
    base = y[keep].mean()
    lines.append(f"\n[{name}] by {key} (rkend 0 excluded: {exclude_end}); rows {keep.sum()}, feeds {len(d['m'])}, base rate {base:.5f}")
    lines.append(f"{'bin':>9} {'n':>9} {'rate':>8} {'SE':>8} {'lift':>6} {'z_vs_base':>9}")
    cnts, poss, rates, mids = [], [], [], []
    for lo, hi in BINS:
        mk = keep & (x >= lo) & (x <= hi)
        nn_ = int(mk.sum())
        if nn_ < 200:
            continue
        r = y[mk].mean(); se = np.sqrt(max(r * (1 - r), 1e-12) / nn_)
        lines.append(f"{(f'{lo}' if lo == hi else f'{lo}-{min(hi, 299)}'):>9} {nn_:>9} {r:>8.5f} {se:>8.5f} {r / base:>6.3f} {(r - base) / se:>9.2f}")
        cnts.append(nn_); poss.append(int(y[mk].sum())); rates.append(r); mids.append(np.sqrt((lo + 1) * (min(hi, 299) + 1)) - 1)
    cnts, poss, rates, mids = map(np.asarray, (cnts, poss, rates, mids))
    # T1 chi-square homogeneity
    obs = np.stack([poss, cnts - poss], 1)
    chi2, pv, dof, _ = stats.chi2_contingency(obs)
    # T3 weighted LS of log rate on log1p(mid)
    w = cnts * rates * (1 - rates)  # ~ 1/var(log rate) * rate^2 ... use delta: var(log r) = (1-r)/(n r)
    vlog = (1 - rates) / (cnts * rates)
    X = np.stack([np.ones_like(mids), np.log1p(mids)], 1)
    Wd = 1 / vlog
    beta = np.linalg.solve(X.T @ (Wd[:, None] * X), X.T @ (Wd * np.log(rates)))
    cov = np.linalg.inv(X.T @ (Wd[:, None] * X))
    lines.append(f"  T1 chi-square homogeneity over {len(cnts)} bins: chi2 {chi2:.1f}, df {dof}, p {pv:.3g}")
    lines.append(f"  T3 WLS slope of log(rate) on log1p({key}): {beta[1]:+.4f} +- {np.sqrt(cov[1, 1]):.4f} (z {beta[1] / np.sqrt(cov[1, 1]):+.1f})")
    # T2 feed-level paired: mean normalised rank of positives (excluding end) minus 0.5
    pos_ok = keep & (y == 1)
    nr = x[pos_ok] / np.maximum(d["M"][pos_ok] - 1, 1) if key == "rank" else None
    if nr is not None:
        f = d["fid"][pos_ok]
        s = np.bincount(f, nr, minlength=len(d["m"])); c = np.bincount(f, minlength=len(d["m"]))
        has = c > 0
        pf = s[has] / c[has]
        se = pf.std(ddof=1) / np.sqrt(len(pf))
        lines.append(f"  T2 feed-level mean normalised rank of positives - 0.5: {pf.mean() - 0.5:+.5f} +- {se:.5f} (z {(pf.mean() - 0.5) / se:+.1f}; feeds with a non-end positive {len(pf)})")
    return dict(base=float(base), chi2=float(chi2), chi2_p=float(pv), slope=float(beta[1]), slope_se=float(np.sqrt(cov[1, 1])))

lines = [f"R2 XOFF premise check, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"]
res = {}
d_pool = gather(pool)
lines.append(f"POOL rows {len(d_pool['y'])}, positives {int(d_pool['y'].sum())}, rate incl. end {d_pool['y'].mean():.5f}; last-shown-item (rkend 0) rate "
             f"{d_pool['y'][d_pool['rkend'] == 0].mean():.4f} (truncation spike, excluded below); gaps: mean pos/rank ratio "
             f"{(d_pool['pos'].mean() / d_pool['rank'].mean()):.3f}")
res["pool_rank"] = table(d_pool, "rank", "POOL 134,118 (p>0.5, M>=50, holdout excluded)", lines)
res["pool_pos"] = table(d_pool, "pos", "POOL 134,118", lines)
hi9 = p[pool] > 0.9
d9 = gather(pool[hi9]); d5 = gather(pool[~hi9])
res["pool_p9_rank"] = table(d9, "rank", f"POOL p>0.9 ({hi9.sum()} feeds)", lines)
res["pool_p59_rank"] = table(d5, "rank", f"POOL 0.5<p<=0.9 ({(~hi9).sum()} feeds)", lines)
d_h = gather(hold)
res["holdout_rank"] = table(d_h, "rank", "HOLDOUT 30,000 (p>0.9, information)", lines)
prod = np.where((p <= 0.05) & (M >= 50) & ~is_hold)[0]
prod = np.sort(np.random.default_rng(0).choice(prod, 200000, replace=False))
d_p = gather(prod)
res["prod_rank"] = table(d_p, "rank", "PROD-LIKE 200K sample (p<=0.05, M>=50; confounded reference)", lines)
# by M bin inside the pool (examination gradient should be present in every M bin)
for lo, hi in [(50, 99), (100, 199), (200, 10**6)]:
    mk = (M[pool] >= lo) & (M[pool] <= hi)
    res[f"pool_M{lo}_rank"] = table(gather(pool[mk]), "rank", f"POOL M in [{lo},{min(hi, 300)}] ({mk.sum()} feeds)", lines)

# VAL by candidate index (expected flat: permutation)
Mv, offv, tsv = feeds("val")
ctv = np.load(f"{ROOT}/data/flat/val/cand_target.npy")
yv = ctv.astype(np.int64)
fidv = np.repeat(np.arange(len(Mv)), Mv)
idx = np.arange(len(yv)) - np.repeat(offv, Mv)
d_v = dict(pos=idx, rank=idx, rkend=np.repeat(Mv, Mv) - 1 - idx, y=yv, fid=fidv, M=np.repeat(Mv, Mv), m=Mv)
res["val_index_all"] = table(d_v, "rank", "VAL 14,423 feeds by CANDIDATE INDEX (all M; incl. last index)", lines, exclude_end=False)
mk = Mv >= 50
sel = np.where(mk)[0]
rows = np.concatenate([np.arange(a, a + b) for a, b in zip(offv[sel], Mv[sel])])
d_v50 = dict(pos=idx[rows], rank=idx[rows], rkend=(np.repeat(Mv, Mv) - 1 - idx)[rows], y=yv[rows], fid=np.repeat(np.arange(len(sel)), Mv[sel]), M=np.repeat(Mv[sel], Mv[sel]), m=Mv[sel])
res["val_index_M50"] = table(d_v50, "rank", f"VAL M>=50 ({mk.sum()} feeds) by CANDIDATE INDEX (incl. last index)", lines, exclude_end=False)

# g_noend by raw pos on POOL (offset table candidate): log(rate(pos)/base), rkend 0 excluded; raw per-pos for pos < 30, binned + linear
# interpolation in log1p(pos) above; clamp at 299
y, x, keep = d_pool["y"], d_pool["pos"], d_pool["rkend"] != 0
base = y[keep].mean()
g = np.zeros(300); nraw = np.zeros(300, int); graw = np.full(300, np.nan)
for q in range(300):
    mk = keep & (x == q)
    nraw[q] = mk.sum()
    if nraw[q] > 0:
        graw[q] = np.log(max(y[mk].mean(), 1e-6) / base)
# smoothing: pos 0-19 raw (>= 10K rows each), then bins of growing width, interpolated in log1p(pos)
knots_x, knots_g = [], []
for q in range(20):
    knots_x.append(q); knots_g.append(graw[q])
for lo, hi in [(20, 24), (25, 29), (30, 39), (40, 49), (50, 69), (70, 99), (100, 139), (140, 199), (200, 299)]:
    mk = keep & (x >= lo) & (x <= hi)
    if mk.sum() >= 2000:
        knots_x.append(np.sqrt((lo + 1) * (hi + 1)) - 1); knots_g.append(np.log(max(y[mk].mean(), 1e-6) / base))
knots_x, knots_g = np.asarray(knots_x, float), np.asarray(knots_g, float)
g = np.interp(np.log1p(np.arange(300)), np.log1p(knots_x), knots_g)
lines.append(f"\ng_noend[pos] (POOL, rkend 0 excluded, base {base:.5f}): raw rows at pos 0/1/2/5/10/20/50/100/200: "
             f"{[int(nraw[q]) for q in (0, 1, 2, 5, 10, 20, 50, 100, 200)]}; g at pos 0/1/2/3/5/10/20/50/100/200/299: "
             f"{[round(float(g[q]), 3) for q in (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 299)]}; mean over POOL rows {float(g[np.clip(x, 0, 299)].mean()):+.4f}")
json.dump(dict(g_noend=g.tolist(), base_rate=float(base), knots_x=knots_x.tolist(), knots_g=knots_g.tolist(), n_raw=nraw.tolist(),
               source="POOL 134,118 feeds (p>0.5, M>=50, holdout excluded), rkend 0 excluded, raw pos 0-19 + bins, log1p interpolation"),
          open(f"{OUT}/g_noend_pos.json", "w"), indent=1)
lines.append(f"done in {time.time() - t0:.0f}s")
txt = "\n".join(lines)
open(f"{OUT}/premise_check.txt", "w").write(txt + "\n")
json.dump(res, open(f"{OUT}/premise_check.json", "w"), indent=1)
print(txt)
