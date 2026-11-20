"""WAVE 10 R2 (XOFF): the examination-offset table g_noend[pos] from the POOL rows (134,118 exploration train feeds = NN p_random > 0.5,
M >= 50, holdout excluded), as a WITHIN-FEED observed/expected lift by raw displayed position:
    lift(pos) = sum_{i at pos, rkend_i != 0} y_i / sum_{i at pos, rkend_i != 0} r_f(i),   r_f = positive rate of feed f over its non-end items,
    g_noend[pos] = log lift(pos)  (raw for pos 0-19, >= 129K rows each; bins above, interpolated in log1p(pos); clamp at 299).
The feed-level expectation removes the M-composition confound of the pooled curve (deep positions exist only in long feeds, which have
lower base rates: pooled lift 0.62 at pos 200-299 vs 0.84 within M >= 200 feeds). The last shown item (rkend 0; truncation spike,
rate 0.28) is excluded from both O and E, so the table has NO end-of-feed term (labels_loss A3: an end-of-feed offset on exploration
feeds costs -0.0011 holdout). Also written: the same O/E curve per M bin (information) and the rank-indexed curve.
Output: g_noend_pos_oe.json (the table the XOFF arm reads; md5 logged by train.py) + premise_g.txt.
"""
import hashlib, json, os, time
import numpy as np
import pyarrow.parquet as pq

ROOT = "<repo>"
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
t = pq.read_table(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
M, off = t.column("M").to_numpy().astype(np.int64), t.column("cand_off").to_numpy().astype(np.int64)
n = len(M)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
is_hold = np.zeros(n, bool); is_hold[hold] = True
pool = np.where((p > 0.5) & (M >= 50) & ~is_hold)[0]
assert len(pool) == 134118
cpos = np.load(f"{ROOT}/data/flat/train/cand_pos.npy", mmap_mode="r")
ctgt = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
o, m = off[pool], M[pool]
rows = np.concatenate([np.arange(a, a + b) for a, b in zip(o, m)])
pos = np.asarray(cpos[rows]).astype(np.int64)
y = np.asarray(ctgt[rows]).astype(np.float64)
fid = np.repeat(np.arange(len(pool)), m)
Mr = np.repeat(m, m)
order = np.lexsort((pos, fid))
rank = np.empty_like(pos)
start = np.concatenate([[0], np.cumsum(m)[:-1]])
rank[order] = np.arange(len(pos)) - np.repeat(start, m)
rkend = Mr - 1 - rank
keep = rkend != 0
# feed-level expected rate over non-end items
s = np.bincount(fid, y * keep, minlength=len(pool)); c = np.bincount(fid, keep.astype(np.float64), minlength=len(pool))
r_f = s / np.maximum(c, 1)
e = r_f[fid]
lines = [f"R2 XOFF offset table (O/E within-feed), {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; POOL feeds {len(pool)}, rows {len(y)}, non-end rows {int(keep.sum())}, "
         f"feeds with >= 1 non-end positive {int((s > 0).sum())}"]

def oe_curve(key, keep_mask, label):
    x = key
    O = np.bincount(np.clip(x, 0, 299), y * keep_mask, minlength=300)[:300]
    E = np.bincount(np.clip(x, 0, 299), e * keep_mask, minlength=300)[:300]
    N = np.bincount(np.clip(x, 0, 299), keep_mask.astype(np.float64), minlength=300)[:300]
    knots_x, knots_g, knots_n = [], [], []
    for q in range(20):
        if N[q] >= 2000:
            knots_x.append(q); knots_g.append(np.log(max(O[q], 0.5) / E[q])); knots_n.append(int(N[q]))
    for lo, hi in [(20, 24), (25, 29), (30, 39), (40, 49), (50, 69), (70, 99), (100, 139), (140, 199), (200, 299)]:
        Ob, Eb, Nb = O[lo:hi + 1].sum(), E[lo:hi + 1].sum(), N[lo:hi + 1].sum()
        if Nb >= 2000:
            knots_x.append(np.sqrt((lo + 1) * (hi + 1)) - 1); knots_g.append(np.log(max(Ob, 0.5) / Eb)); knots_n.append(int(Nb))
    kx, kg = np.asarray(knots_x, float), np.asarray(knots_g, float)
    g = np.interp(np.log1p(np.arange(300)), np.log1p(kx), kg)
    lines.append(f"[{label}] O/E lift by {('pos' if key is pos else 'rank')}: " + " ".join(f"{int(round(a))}:{np.exp(b):.3f}" for a, b in zip(kx, kg)))
    return g, kx, kg, knots_n, O, E, N

g, kx, kg, kn, O, E, N = oe_curve(pos, keep, "POOL all M")
g_rank, *_ = oe_curve(rank, keep, "POOL all M")
# SE of log lift at the knots: var(O/E) ~ sum y / E^2 (Poisson) -> se(log) ~ sqrt(O)/O
se_knots = []
for q, (a, b) in enumerate(zip(kx, kg)):
    pass
for lo, hi in [(50, 99), (100, 199), (200, 100000)]:
    mk = (Mr >= lo) & (Mr <= hi)
    oe_curve(pos, keep & mk, f"POOL M in [{lo},{min(hi, 300)}]")
lines.append(f"g_noend[pos] at 0/1/2/3/4/5/7/10/15/20/30/50/100/150/200/299: " + " ".join(f"{q}:{g[q]:+.3f}" for q in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 150, 200, 299)))
lines.append(f"rows at pos 0..4: {[int(N[q]) for q in range(5)]}; mean g over POOL rows {float(g[np.clip(pos, 0, 299)].mean()):+.4f}; "
             f"mean g over POOL positives {float(g[np.clip(pos[y > 0], 0, 299)].mean()):+.4f}; sd over rows {float(g[np.clip(pos, 0, 299)].std()):.4f}")
lines.append(f"half-strength offset 0.5*g: max {0.5 * g.max():+.3f} (pos 0), min {0.5 * g.min():+.3f}; pooled-curve table (premise_check.py) g at 0/100/299: "
             f"{[round(float(v), 3) for v in np.asarray(json.load(open(f'{OUT}/g_noend_pos.json'))['g_noend'])[[0, 100, 299]]]}")
tab = dict(g_noend=[float(v) for v in g], knots_x=kx.tolist(), knots_g=kg.tolist(), knots_n=kn, g_noend_by_rank=[float(v) for v in g_rank],
           source="POOL 134,118 exploration train feeds (NN p_random > 0.5, M >= 50, 30K holdout excluded); within-feed O/E lift by raw displayed pos, "
                  "last shown item (rkend 0) excluded from O and E; raw pos 0-19 + bins, log1p interpolation; clamp 299; written by premise_g.py",
           n_feeds=int(len(pool)), n_rows_nonend=int(keep.sum()), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
path = f"{OUT}/g_noend_pos_oe.json"
json.dump(tab, open(path, "w"), indent=1)
lines.append(f"wrote {path} md5 {hashlib.md5(open(path, 'rb').read()).hexdigest()} in {time.time() - t0:.0f}s")
open(f"{OUT}/premise_g.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
