"""Training-signal accounting and position decomposition (CPU, read-only). Prints tables; writes out/03_*.csv.

A. Composition of train by traffic class (production / pool-eligible random-like / holdout) x M group: feeds, positives,
   candidates; share of NN gradient units (each feed = 1 unit per pass with --loss_norm feed) in the main phase and in the
   m02 cooldown; K9-aux units.
B. Where train positives sit (shown within-feed rank / rank-from-end) by class.
C. RandTopN decomposition for M>=50: CTR-by-rank in random-like feeds = examination theta(k) (x E[gamma]);
   production CTR / random CTR at the same rank and M bin = relevance concentration of the production ranker.
   Log-share of the top-slot click excess that is examination vs relevance.
"""
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/insights/data_usage/out"
t = pl.read_parquet(OUT / "train_feeds.parquet")
M = t["M"].to_numpy(); cls = t["cls"].to_numpy(); npos = t["npos"].to_numpy(); prand_raw = t["prand_raw"].to_numpy()

print("=== A. composition")
grp = np.where(cls == 2, "holdout", np.where(cls == 1, "pool(p>0.5,M>=50)", np.where(M >= 50, "prod M>=50", "short M<50 (p:=0)")))
df = pl.DataFrame(dict(grp=grp, M=M, npos=npos, prand=np.nan_to_num(prand_raw, nan=-1)))
A = (df.group_by("grp").agg(feeds=pl.len(), pos=pl.col("npos").sum(), cands=pl.col("M").sum(), meanM=pl.col("M").mean(),
                            meanpos=pl.col("npos").mean()).sort("grp"))
A = A.with_columns((pl.col("feeds") / pl.col("feeds").sum()).alias("feed_share"), (pl.col("pos") / pl.col("pos").sum()).alias("pos_share"),
                   (pl.col("cands") / pl.col("cands").sum()).alias("cand_share"))
print(A)
A.write_csv(OUT / "03_composition.csv")
# expected number of exploration feeds hidden in the short group if pi(M<50) ~ pi(M>=50)
ms = M >= 50
pi50 = np.nanmean(np.where(ms, prand_raw, np.nan))
print(f"mean EDA posterior on M>=50 feeds: {pi50:.4f}; short feeds {np.sum(~ms)} -> if the same share, ~{pi50*np.sum(~ms):,.0f} "
      f"exploration feeds are trained as production (domain token 'production', K9 aux on a random order)")
ps = prand_raw[~ms]
print(f"  short feeds: EDA posterior mean {np.nanmean(ps):.3f} (unreliable), share >0.5 {np.nanmean(ps > 0.5):.3f}")

# NN gradient units: main phase visits every training feed once per epoch (8 epochs before the cooldown at 10 ep / 20%),
# cooldown 2 epochs at 317 feeds/step (19670 steps); m02 replaces 2% of steps by pool batches.
n_train = int(np.sum(cls != 2)); n_pool = int(np.sum(cls == 1))
poolM = M[cls == 1]
steps_cd = 19670; fps_reg = n_train * 2 / steps_cd
fps_pool = 32768 / poolM.mean()
fps_pool_b = min(512, fps_pool)
pool_steps = 0.02 * steps_cd
pool_units = pool_steps * fps_pool_b; reg_units = (steps_cd - pool_steps) * fps_reg
expl_main = n_pool / n_train
expl_cd = (reg_units * n_pool / n_train + pool_units) / (reg_units + pool_units)
print(f"\nfeeds/step regular {fps_reg:.0f}, pool batch ~{fps_pool_b:.0f} (mean pool M {poolM.mean():.0f})")
print(f"random-like (p>0.5) share of gradient units: main phase {expl_main:.4f}, m02 cooldown {expl_cd:.4f} "
      f"(pool passes in cooldown: regular {2.0:.1f} + replay {pool_units / n_pool:.2f})")
w_aux = 1 - t["prand_nn"].to_numpy()
print(f"K9 aux units (sum of 1-p over training feeds) / training feeds = {w_aux[cls != 2].sum() / n_train:.4f}")

print("\n=== B. where positives sit")
z = np.load(OUT / "train_pos_cands.npz")
pf, rk, re = z["fidx"], z["rank"].astype(np.int64), z["rkend"].astype(np.int64)
pc = cls[pf]; pM = M[pf]
def bucket(r):
    return np.select([r == 0, r <= 2, r <= 9, r <= 49], ["0", "1-2", "3-9", "10-49"], ">=50")
rows = []
for name, mk in [("all", np.ones(len(pf), bool)), ("prod M<50", (pc == 0) & (pM < 50)), ("prod M>=50", (pc == 0) & (pM >= 50)),
                 ("pool", pc == 1), ("holdout", pc == 2)]:
    b = bucket(rk[mk])
    d = {"class": name, "n_pos": int(mk.sum())}
    for k in ["0", "1-2", "3-9", "10-49", ">=50"]:
        d[f"rank {k}"] = float(np.mean(b == k))
    d["last item"] = float(np.mean(re[mk] == 0)); d["rkend 0-2"] = float(np.mean(re[mk] <= 2))
    rows.append(d)
B = pl.DataFrame(rows); print(B); B.write_csv(OUT / "03_positive_positions.csv")

print("\n=== C. RandTopN decomposition (M>=50), per M bin; rates exclude the last 3 ranks (truncation spike)")
yall = None
# rebuild rate by (class, M bin, rank) from per-candidate arrays
FL = ROOT / "data/flat/train"
pos = np.load(FL / "cand_pos.npy", mmap_mode="r")
fidc = np.repeat(np.arange(len(M)), M)
# within-feed rank for all candidates (needed for denominators)
order = np.lexsort((np.asarray(pos), fidc))
off = np.concatenate([[0], np.cumsum(M)[:-1]])
rank = np.empty(len(fidc), np.int32); rank[order] = (np.arange(len(fidc)) - off[fidc[order]]).astype(np.int32)
del order
rkend = (M[fidc] - 1 - rank).astype(np.int32)
y = np.load(FL / "cand_target.npy").astype(np.int8)
mbins = [50, 100, 150, 200, 301]
rb_edges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
cf = cls[fidc]; Mc = M[fidc]
res = []
for i in range(len(mbins) - 1):
    mk = (Mc >= mbins[i]) & (Mc < mbins[i + 1]) & (rkend >= 3)
    for c, cname in [(0, "prod"), (1, "rand")]:
        m2 = mk & ((cf == c) if c == 0 else (cf >= 1))  # random-like = pool + holdout
        rr = rank[m2]; yy = y[m2]
        base = yy.mean()
        for j in range(len(rb_edges) - 1):
            s = (rr >= rb_edges[j]) & (rr < rb_edges[j + 1])
            if s.sum() < 2000:
                continue
            res.append(dict(Mbin=f"[{mbins[i]},{mbins[i+1]})", cls=cname, rank=f"[{rb_edges[j]},{rb_edges[j+1]})",
                            n=int(s.sum()), rate=float(yy[s].mean()), base=float(base)))
C = pl.DataFrame(res)
Cw = C.pivot(on="cls", index=["Mbin", "rank"], values=["rate", "base", "n"])
Cw = Cw.with_columns(
    (pl.col("rate_rand") / pl.col("base_rand")).alias("theta_rel"),             # examination lift vs feed base
    (pl.col("rate_prod") / pl.col("base_prod")).alias("prod_lift"),              # total production lift
).with_columns(
    (pl.col("prod_lift") / pl.col("theta_rel")).alias("relevance_lift"),
    (pl.col("theta_rel").log() / pl.col("prod_lift").log()).alias("exam_logshare"),
)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20)
print(Cw.select(["Mbin", "rank", "rate_prod", "rate_rand", "theta_rel", "prod_lift", "relevance_lift", "exam_logshare", "n_rand"]))
Cw.write_csv(OUT / "03_randtopn.csv")

# rank-from-end spikes by class (M>=50)
print("\nrank-from-end rates (M>=50):")
for c, cname in [(0, "prod"), (1, "rand")]:
    mk = (Mc >= 50) & ((cf == c) if c == 0 else (cf >= 1))
    rr = rkend[mk]; yy = y[mk]
    print(cname, " ".join(f"e{k}:{yy[rr == k].mean():.4f}" for k in range(6)), f"base {yy.mean():.4f}")

# Examination theta curve (pooled over M>=50 random-like, relative to the feed base, excluding last 3) for later use
mk = (Mc >= 50) & (cf >= 1) & (rkend >= 3)
rr = np.minimum(rank[mk], 299); yy = y[mk]
num = np.bincount(rr, weights=yy, minlength=300); den = np.bincount(rr, minlength=300)
# per-feed base normalisation: divide by the mean rate of the class within the M>=50 random-like set
theta = (num / np.maximum(den, 1)) / yy.mean()
np.save(OUT / "theta_rand_rank.npy", theta)
print("\ntheta_rand (relative examination) at ranks 0,1,2,5,10,20,50,100,200:",
      " ".join(f"{theta[k]:.2f}" for k in [0, 1, 2, 5, 10, 20, 50, 100, 200]))
mk = (Mc >= 50) & (cf == 0) & (rkend >= 3)
rr = np.minimum(rank[mk], 299); yy = y[mk]
num = np.bincount(rr, weights=yy, minlength=300); den = np.bincount(rr, minlength=300)
pc_prod = (num / np.maximum(den, 1)) / yy.mean()
np.save(OUT / "ctr_prod_rank.npy", pc_prod)
print("prod CTR lift at ranks 0,1,2,5,10,20,50,100,200:", " ".join(f"{pc_prod[k]:.2f}" for k in [0, 1, 2, 5, 10, 20, 50, 100, 200]))
