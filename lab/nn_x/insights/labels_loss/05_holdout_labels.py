"""(1d)+(2b) Label heterogeneity on the 30K train holdout (random-like, M>=50; positions are known there because
these are train feeds): how members rank positives by shown position, by end-of-feed (truncation) status and by the
event type the positive later shows up as (same user's next train feed; descriptive label enrichment, never an
input). Also: how much of the exploration-feed label variance a fixed examination offset g(pos)+h(rkend), fit on
the m02 pool feeds (holdout excluded), explains for the CE (variance-reduction potential of an exploration-only
position offset in the cooldown).  Writes out/05_holdout_labels.txt and out/pool_pos_offset.npz."""
import sys, os, time
import numpy as np
from numba import njit
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

OUT = "<repo>/nn_x/insights/labels_loss/out"
PR = "<repo>/preds"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

f, M, off = feed_meta("train")
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hm = np.load(f"{PR}/holdout/holdout_meta.npy")
assert (hm[0] == M[hold]).all(), "holdout reconstruction mismatch"
pool = (p > 0.5); pool[hold] = False
P(f"holdout reconstructed: {len(hold)} feeds; pool feeds {pool.sum()}")

cpos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
ctgt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
citem = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")

def rows(fids):
    return np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])

def within_rank(posv, Mv):
    """rank of pos within each feed (0 = top shown), and rank from the end."""
    n = len(Mv); fid = np.repeat(np.arange(n), Mv)
    o = np.lexsort((posv, fid))
    r = np.empty(len(posv), np.int64)
    start = np.repeat(np.concatenate([[0], np.cumsum(Mv)[:-1]]), Mv)
    r[o] = np.arange(len(posv)) - start
    return r, np.repeat(Mv, Mv) - 1 - r

# ---- fit examination offsets on pool feeds (sample 60K for speed) ----
rng = np.random.default_rng(0)
pf = np.sort(rng.choice(np.where(pool)[0], 60000, replace=False))
rp = rows(pf); yp = np.asarray(ctgt[rp]).astype(np.float64); posp = np.asarray(cpos[rp]).astype(np.int64)
rk, rke = within_rank(posp, M[pf])
base = yp.mean()
RMAX, EMAX = 120, 8
g = np.zeros(300); h = np.zeros(EMAX)
for it in range(6):  # backfitting of a multiplicative rate model  y ~ base * exp(g(rank) + h(rkend))
    eh = np.exp(h[np.minimum(rke, EMAX - 1)] * (rke < EMAX))
    for r in range(300):
        m = (np.minimum(rk, RMAX) == min(r, RMAX))
        if r > RMAX: g[r] = g[RMAX]; continue
        g[r] = np.log(max(yp[m].sum(), 1) / max((base * eh[m]).sum(), 1e-9))
    eg = np.exp(g[np.minimum(rk, 299)])
    for e in range(EMAX):
        m = rke == e
        h[e] = np.log(max(yp[m].sum(), 1) / max((base * eg[m]).sum(), 1e-9))
# smooth g beyond rank 30 (sparse) with a running mean
gs = g.copy()
for r in range(30, 300):
    lo, hi = max(30, r - 10), min(299, r + 10)
    gs[r] = g[lo:hi + 1].mean()
np.savez(f"{OUT}/pool_pos_offset.npz", g_rank=gs, h_rkend=h, base=base)
P("pool examination offsets (log rate ratio): g(rank) at 0,1,2,3,5,10,20,30,50,100: " +
  " ".join(f"{gs[r]:+.2f}" for r in [0, 1, 2, 3, 5, 10, 20, 30, 50, 100]))
P("h(rank from end) 0..7: " + " ".join(f"{x:+.2f}" for x in h))

# ---- holdout ----
rh = rows(hold); yh = np.asarray(ctgt[rh]).astype(np.float64); posh = np.asarray(cpos[rh]).astype(np.int64)
assert np.allclose(yh, np.load(f"{PR}/holdout/holdout_y.npy"))
Mh = M[hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
rkh, rkeh = within_rank(posh, Mh)
offset = gs[np.minimum(rkh, 299)] + np.where(rkeh < EMAX, h[np.minimum(rkeh, EMAX - 1)], 0.0)
MEM = [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"v28_lr1_m02_s41{i}" for i in range(1, 8)] + \
      [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)]
fid = np.repeat(np.arange(len(Mh)), Mh)

def ce(s):
    s = s.astype(np.float64)
    mx = np.maximum.reduceat(s, offh)
    e = np.exp(s - np.repeat(mx, Mh)); Z = np.add.reduceat(e, offh)
    lp = s - np.repeat(mx + np.log(Z), Mh)
    K = np.add.reduceat(yh, offh)
    return -np.bincount(fid, weights=lp * yh, minlength=len(Mh)) / K, lp

def zf(s):
    mu = np.bincount(fid, weights=s, minlength=len(Mh)) / Mh
    d = s - np.repeat(mu, Mh); sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=len(Mh)) / Mh)
    return d / np.repeat(sd + 1e-9, Mh)

ce0 = ce1 = ce_off = 0; S = np.zeros(len(yh)); corr = []
for mname in MEM:
    s = np.load(f"{PR}/holdout/nn_{mname}_holdout.npy").astype(np.float64)
    S += zf(s) / len(MEM)
    c0, _ = ce(s); c1, _ = ce(s + offset); c2, _ = ce(offset)
    ce0 += c0.mean() / len(MEM); ce1 += c1.mean() / len(MEM); ce_off = c2.mean()
    corr.append(np.corrcoef(zf(s), zf(rkh.astype(float)))[0, 1])
cu = np.log(Mh).mean()
P(f"\nholdout CE (nats/feed): uniform {cu:.4f}; offset only {ce_off:.4f}; member {ce0:.4f}; member + offset {ce1:.4f}"
  f"  -> offset removes {ce0-ce1:.4f} nats ({(ce0-ce1)/(cu-ce0)*100:.1f}% of the member's gain over uniform)")
P(f"within-feed corr(member z, shown rank) on random-like holdout: mean {np.mean(corr):+.4f} (should be ~0)")
# gradient share of positives at the end of the feed
_, lp = ce(S * 1.0)
ispos = yh == 1
endpos = ispos & (rkeh < 3)
P(f"share of holdout positives in the last 3 shown slots: {endpos.sum()/ispos.sum():.3f}; first 3 slots: "
  f"{(ispos & (rkh < 3)).sum()/ispos.sum():.3f}")

# ---- rank percentile of positives by shown position / end status (member-mean z) ----
o = np.lexsort((yh, -S, fid))  # pessimistic
rank = np.empty(len(S), np.int64); start = np.repeat(offh, Mh); rank[o] = np.arange(len(S)) - start
pct = rank / np.repeat(Mh - 1, Mh)
P("\nmember-mean rank percentile of holdout positives (0 = top) by SHOWN position (exploration feed, order random):")
for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 100), (100, 300)]:
    m = ispos & (rkh >= lo) & (rkh < hi) & (rkeh >= 3)
    P(f"  shown rank [{lo},{hi}) (not last 3): n {m.sum()}, mean pct {pct[m].mean():.3f}, P(top10%) {(pct[m]<0.1).mean():.3f}")
for e in range(3):
    m = ispos & (rkeh == e)
    P(f"  rank-from-end {e}: n {m.sum()}, mean pct {pct[m].mean():.3f}, P(top10%) {(pct[m]<0.1).mean():.3f}")

# ---- event type of the positive in the same user's next train feed (descriptive) ----
uid = f["user_id"].to_numpy(); ts = f["timestamp"].to_numpy(); hoff = f["hist_off"].to_numpy(); H = f["H"].to_numpy()
order = np.lexsort((ts, uid)); u_s, t_s = uid[order], ts[order]
nxt = np.full(len(M), -1, np.int64)
same = (u_s[1:] == u_s[:-1]) & (t_s[1:] > t_s[:-1]) & (u_s[1:] != 0)
nxt[order[:-1][same]] = order[1:][same]
hitem = np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r"); het = np.load(FLAT / "train/hist_event_type.npy", mmap_mode="r")
cls = np.full(len(yh), -1, np.int8)  # -1 n/a (no later feed), 0 not found, 1 type-1 only, 2 other types present
items_h = np.asarray(citem[rh])
for k, a in enumerate(hold):
    b = nxt[a]
    if b < 0: continue
    hs = np.asarray(hitem[hoff[b]:hoff[b] + H[b]]); es = np.asarray(het[hoff[b]:hoff[b] + H[b]])
    for j in range(offh[k], offh[k] + Mh[k]):
        if yh[j] != 1: continue
        mm = hs == items_h[j]
        if not mm.any(): cls[j] = 0
        else: cls[j] = 1 if (es[mm] == 1).all() else 2
P(f"\nholdout feeds with a later same-user feed: {(nxt[hold]>=0).sum()}; positives classified {(cls>=0).sum()}")
for c, nm in [(0, "not in later history"), (1, "type-1 only"), (2, "other event types (2..9)")]:
    m = cls == c
    if m.sum(): P(f"  {nm}: n {m.sum()}, mean pct {pct[m].mean():.3f}, P(top10%) {(pct[m]<0.1).mean():.3f}, "
                  f"share last-3 {(rkeh[m]<3).mean():.3f}")
open(f"{OUT}/05_holdout_labels.txt", "w").write("\n".join(lines) + "\n")
