"""Independent reviewer check (CPU only): TRUNC offset alignment on the REAL stage-E batch construction (seed 951)."""
import sys, time
import numpy as np, polars as pl, torch
sys.path.insert(0, "<repo>/nn_x/w5")
import cd_arms5 as C5
from data import SplitData
from cd_arms import exploration_pool
ROOT = "<repo>"
t0 = time.time()
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
M_np = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M_np)
pos = np.load(f"{ROOT}/data/flat/train/cand_pos.npy", mmap_mode="r")
y = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
class Tr: pass
tr = Tr(); tr.n = n; tr.n_cand = int(M_np.sum()); tr.M_np = M_np; tr.cand_off_np = off; tr.device = torch.device("cpu")
tr.M = torch.from_numpy(M_np); tr.cand_off = torch.from_numpy(off)
# train.py lines 236-246 (prand + holdout), verbatim logic
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(n)).all()
prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); prand_np[M_np < 50] = 0.0
cand = np.where((prand_np > 0.9) & (M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 951
rng = np.random.default_rng(seed)
feeds = np.setdiff1d(np.arange(n), hold)
prand = torch.from_numpy(prand_np)
trunc = C5.TruncOffset(1.0, tr, prand.numpy().astype(np.float64), hold, log=print)
p_soft = np.nan_to_num(pr["p_random"].astype(np.float64), nan=0.0)
L = C5.load_lifts()
print(f"setup {time.time()-t0:.0f}s")
ep0 = SplitData.batches(tr, feeds, 32768, 512, shuffle=True, rng=rng)
print("steps/epoch", len(ep0))
def gather(fb):  # data.py SplitData.gather lines 130-136 + target line 161 (CPU)
    fidx = torch.from_numpy(np.asarray(fb, dtype=np.int64))
    Mt = tr.M[fidx]
    Mx = int(-(-M_np[fb].max() // 16) * 16)
    ar = torch.arange(Mx)
    cmask = ar[None, :] < Mt[:, None]
    ci = tr.cand_off[fidx][:, None] + ar[None, :]
    ci = torch.where(cmask, ci, tr.cand_off[fidx][:, None])
    return dict(fidx=fidx, ci=ci, cmask=cmask, y=torch.from_numpy(np.asarray(y)[ci.numpy()].astype(np.float32)) * cmask)
holdset = set(hold.tolist())
def brute(fb, Mx):
    out = np.zeros((len(fb), Mx), np.float32)
    for i, fe in enumerate(fb):
        o, m = off[fe], M_np[fe]
        pf = np.asarray(pos[o:o + m]).astype(np.int64)
        # independent rkend: number of candidates of the feed shown strictly later, ties: later flat row counts as later
        idx = np.arange(m)
        rk = np.array([(pf > pf[k]).sum() + ((pf == pf[k]) & (idx > k)).sum() for k in range(m)]) if m <= 400 else None
        if m < 50: g = 0
        elif prand_np[fe] > 0.5: g = 2
        else: g = 1
        if g == 2 or fe in holdset: continue
        w = 1.0 - p_soft[fe]
        for k in range(m):
            if rk[k] <= 2:
                out[i, k] = np.float32(L[g, rk[k]] * w)
    return out
pick = list(range(0, 8)) + list(np.random.default_rng(7).choice(len(ep0), 40, replace=False))
bad = nzc = cells = 0; clicks = np.zeros((2, 4)); cnt = np.zeros((2, 4))
for bi in pick:
    fb = ep0[bi]
    b = gather(fb)
    o_ = trunc.offset(b).numpy()
    ex = brute(fb, o_.shape[1])
    bad += int((o_ != ex).sum()); nzc += int((ex != 0).sum()); cells += int(b["cmask"].sum())
    assert not np.isin(fb, hold).any()
    # label sanity: click rate at the max-offset cell vs others
    for i, fe in enumerate(fb):
        if fe in holdset or (M_np[fe] >= 50 and prand_np[fe] > 0.5): continue
        g = 0 if M_np[fe] < 50 else 1
        m = M_np[fe]; o = off[fe]; pf = np.asarray(pos[o:o+m]).astype(np.int64)
        r = np.argsort(np.argsort(-pf, kind="stable"), kind="stable")  # approx rank-from-end
        yy = np.asarray(y[o:o+m])
        for k in range(m):
            c = min(r[k], 3); clicks[g, c] += yy[k]; cnt[g, c] += 1
print(f"seed {seed}: {len(pick)} real epoch-0 batches, {cells} candidate cells, non-zero expected offsets {nzc}, MISMATCHES {bad}")
print("click rate by rkend 0/1/2/>=3 (prod M<50):", np.round(clicks[0] / cnt[0], 4), "-> log lift", np.round(np.log(clicks[0, :3] / cnt[0, :3] / (clicks[0, 3] / cnt[0, 3])), 3))
print("click rate by rkend 0/1/2/>=3 (prod M>=50):", np.round(clicks[1] / cnt[1], 4), "-> log lift", np.round(np.log(clicks[1, :3] / cnt[1, :3] / (clicks[1, 3] / cnt[1, 3])), 3))
# pool batches (cooldown m02): mix_rng = default_rng(seed + 1), pool = exploration_pool(feeds, prand_np, 0.5, hold)
pool = exploration_pool(feeds, prand_np, 0.5, hold)
mix_rng = np.random.default_rng(seed + 1)
expl = SplitData.batches(tr, pool, 32768, 512, shuffle=True, rng=mix_rng)
mx = 0.0
for fb in expl:
    mx = max(mx, float(trunc.offset(gather(fb)).abs().max()))
print(f"pool: {len(pool)} feeds, {len(expl)} batches, max |offset| over ALL pool batches {mx}; pool min M {M_np[pool].min()}; holdout in pool {np.isin(pool, hold).sum()}")
# holdout coef
print("holdout coef max |.|", float(trunc.coef[torch.from_numpy(hold)].abs().max()))
print(f"done {time.time()-t0:.0f}s")
