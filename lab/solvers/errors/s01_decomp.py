"""(1) Loss decomposition of v11 on val: per-feed NDCG distribution, gap-to-1 by segment, positive rank distribution."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, ranks
import numpy as np
import polars as pl

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
nd = np.load(f"{SCR}/out/v11_nd.npy")
loss = np.load(f"{SCR}/out/v11_loss.npy")
r = np.load(f"{SCR}/out/v11_rank.npy")

# random baseline: mean over 20 random permutations
rng = np.random.default_rng(0)
ndr = np.mean([ndcg_per_feed(rng.random(len(y)), y, off, M) for _ in range(20)], 0)
np.save(f"{SCR}/out/rand_nd.npy", ndr)

uid = f["user_id"].to_numpy()
H = f["H"].to_numpy()
ts = f["timestamp"].to_numpy()
day = (ts + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
tr_users = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["user_id"])["user_id"].unique().to_numpy()
warm_user = np.isin(uid, tr_users) & (uid != 0)
u0 = uid == 0
cold_user = (~warm_user) & (~u0)
npos = np.bincount(fid, y, n)
cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(bool)
ncoldpos = np.bincount(fid, y * cold, n)
poscold = np.where(ncoldpos == npos, "all_cold", np.where(ncoldpos == 0, "all_warm", "mixed"))
coldshare = np.bincount(fid, cold, n) / M

pf = pl.DataFrame(dict(fidx=np.arange(n), uid=uid, M=M, H=H, ts=ts, day=day, u0=u0, cold_user=cold_user, warm_user=warm_user,
                       npos=npos, ncoldpos=ncoldpos, poscold=poscold, coldshare=coldshare, nd=nd, ndr=ndr))
pf.write_parquet(f"{SCR}/out/perfeed.parquet")

print(f"v11 {nd.mean():.5f}  random {ndr.mean():.5f}  gap to 1 = {1 - nd.mean():.5f}  random gap {1 - ndr.mean():.5f}")
qs = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
print("per-feed NDCG quantiles:", " ".join(f"{q:.2f}:{np.quantile(nd, q):.3f}" for q in qs))
print(f"share feeds nd==1: {(nd >= 0.99999).mean():.4f}; nd<random: {(nd < ndr - 1e-9).mean():.4f}; nd<0.2: {(nd < 0.2).mean():.4f}")
# NDCG histogram
h, e = np.histogram(nd, bins=10, range=(0, 1))
print("hist (0.1 bins):", " ".join(f"{x / n:.3f}" for x in h))


def seg_table(name, lab):
    labs = np.unique(lab)
    print(f"\n== {name} ==")
    print(f"{'seg':>14s} {'feeds%':>7s} {'NDCG':>7s} {'rand':>7s} {'lift':>7s} {'gainR':>6s} {'gapshare':>8s} {'pos/feed':>8s}")
    for l in labs:
        m = lab == l
        g = ((1 - nd[m]).sum()) / (1 - nd).sum()
        gr = (nd[m].mean() - ndr[m].mean()) / (1 - ndr[m].mean())
        print(f"{str(l):>14s} {m.mean() * 100:7.2f} {nd[m].mean():7.4f} {ndr[m].mean():7.4f} {nd[m].mean() - ndr[m].mean():7.4f} {gr:6.3f} {g * 100:8.2f} {npos[m].mean():8.2f}")


seg_table("npos", np.where(npos >= 4, "4+", np.where(npos >= 2, npos.astype(int).astype(str), "1")))
Mb = np.digitize(M, [10, 25, 50, 100, 200])
seg_table("M bucket", np.array(["<10", "10-24", "25-49", "50-99", "100-199", "200+"])[Mb])
Hb = np.digitize(H, [1, 11, 51, 129, 256])
seg_table("H bucket", np.array(["0", "1-10", "11-50", "51-128", "129-255", "256"])[Hb])
seg_table("user type", np.where(u0, "user0", np.where(cold_user, "new_user", "train_user")))
seg_table("positive coldness", poscold)
seg_table("day", np.array(["Tue", "Thu"])[day])
seg_table("npos x M<50", np.char.add(np.where(npos >= 2, "2+pos_", "1pos_"), np.where(M < 50, "M<50", "M>=50")))
cb = np.digitize(coldshare, [0.15, 0.3, 0.45, 0.6])
seg_table("feed cold share", np.array(["<.15", ".15-.3", ".3-.45", ".45-.6", ">=.6"])[cb])

# positive rank distribution
pi = y > 0
pct = r[pi] / np.maximum(M[fid[pi]] - 1, 1)
print(f"\npositives: {pi.sum()}; mean pct rank {pct.mean():.3f} (random 0.5)")
for thr, lab in ((0.5, "bottom half"), (0.75, "bottom 25%"), (0.9, "bottom 10%")):
    print(f"  {lab}: {(pct >= thr).mean():.4f}  (random {1 - thr:.2f})")
for thr, lab in ((0.1, "top 10%"), (0.25, "top 25%")):
    print(f"  {lab}: {(pct <= thr).mean():.4f}")
print("  rank 0 (top-1):", (r[pi] == 0).mean())
# by M bucket
for b, lab in enumerate(["<10", "10-24", "25-49", "50-99", "100-199", "200+"]):
    m = Mb[fid[pi]] == b
    print(f"  M {lab:>8s}: mean pct {pct[m].mean():.3f}  bottom half {(pct[m] >= 0.5).mean():.3f} bottom10 {(pct[m] >= 0.9).mean():.3f}  top10 {(pct[m] <= 0.1).mean():.3f}")
# loss share of positives in the bottom half
lp = loss[pi]
print(f"  loss share of bottom-half positives: {lp[pct >= 0.5].sum() / lp.sum():.3f}; bottom 10%: {lp[pct >= 0.9].sum() / lp.sum():.3f}")
# feed-level: fraction of feeds where best positive is in bottom half
bestr = np.full(n, 10 ** 9)
np.minimum.at(bestr, fid[pi], r[pi])
bp = bestr / np.maximum(M - 1, 1)
print(f"  feeds whose BEST positive is in bottom half: {(bp >= 0.5).mean():.4f}; their gap share {(1 - nd[bp >= 0.5]).sum() / (1 - nd).sum():.3f}")
print(f"  feeds whose best positive is top-1: {(bestr == 0).mean():.4f}; top-3: {(bestr < 3).mean():.4f}")
# multi-positive: for feeds with >=2 positives, how much loss from the worst positive
