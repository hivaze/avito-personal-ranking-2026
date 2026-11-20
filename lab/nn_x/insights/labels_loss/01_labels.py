"""(1) Label anatomy: positives per feed by M (train production / train random-like / val), share of positives in
multi-positive feeds, history event-type mix, and which history event types a train positive later shows up as in
the SAME user's next feed history (descriptive only: later snapshots are never used as inputs).
CPU only. Output: out/01_labels.txt + out/01_labels.json"""
import sys, json, time
import numpy as np, polars as pl
from numba import njit
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT

OUT = "<repo>/nn_x/insights/labels_loss/out"
import os; os.makedirs(OUT, exist_ok=True)
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

MB = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
res = {}

def npos_per_feed(split):
    f, M, off = feed_meta(split)
    y = np.load(FLAT / split / "cand_target.npy", mmap_mode="r")
    K = np.add.reduceat(np.asarray(y, dtype=np.int32), off) if len(off) else np.zeros(0)
    K[M == 0] = 0
    return f, M, off, K

pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)

t0 = time.time()
tabs = {}
for split in ["train", "val"]:
    f, M, off, K = npos_per_feed(split)
    tabs[split] = (f, M, off, K)
    groups = {"all": np.ones(len(M), bool)}
    if split == "train":
        pm = prand.copy(); pm[M < 50] = 0
        groups = {"prod(p<=0.5 or M<50)": pm <= 0.5, "random-like(p>0.5,M>=50)": pm > 0.5, "all": np.ones(len(M), bool)}
    for gname, g in groups.items():
        P(f"\n== {split} / {gname}: feeds {g.sum()}, positives {K[g].sum()}, mean K {K[g].mean():.3f}, "
          f"feeds K=0 {(K[g]==0).sum()}, share feeds K>=2 {(K[g]>=2).mean():.4f}, share of POSITIVES in K>=2 feeds "
          f"{K[g][K[g]>=2].sum()/max(K[g].sum(),1):.4f}")
        P("M bin | feeds% | meanK | P(K>=2) | pos-share in K>=2 | pos rate | mean 1/IDCG-weight K/IDCG(K)")
        idcg = np.concatenate([[1.0], np.cumsum(1.0 / np.log2(np.arange(2, 400)))])
        for lo, hi in zip(MB[:-1], MB[1:]):
            m = g & (M >= lo) & (M < hi)
            if m.sum() == 0: continue
            k = K[m]
            P(f"[{lo},{hi}) | {m.sum()/g.sum()*100:5.1f} | {k.mean():.3f} | {(k>=2).mean():.3f} | "
              f"{k[k>=2].sum()/max(k.sum(),1):.3f} | {k.sum()/M[m].sum()*100:5.2f}% | {(k/idcg[np.clip(k,0,398)]).mean():.3f}")
        kk = K[g]
        dist = {int(i): float((kk == i).mean()) for i in range(0, 7)}
        dist[">=7"] = float((kk >= 7).mean())
        P("K distribution:", json.dumps({str(a): round(b, 4) for a, b in dist.items()}))
        res[f"{split}/{gname}"] = dict(feeds=int(g.sum()), meanK=float(kk.mean()), Kdist=dist)
P(f"[{time.time()-t0:.0f}s]")

# ---------------- history event types / sources (all splits) ----------------
P("\n== history event_type mix (share of events) and per-feed presence")
for split in ["train", "val", "test"]:
    et = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    es = np.load(FLAT / split / "hist_event_source.npy", mmap_mode="r")
    if split == "train":  # 1 in 8 sample of events (contiguous blocks) is enough for shares
        et = np.asarray(et[::8]); es = np.asarray(es[::8])
    else:
        et = np.asarray(et); es = np.asarray(es)
    ce = np.bincount(et, minlength=11) / len(et)
    cs = np.bincount(es, minlength=31) / len(es)
    P(f"{split}: event_type shares " + " ".join(f"{i}:{ce[i]:.4f}" for i in range(11) if ce[i] > 0))
    top = np.argsort(-cs)[:10]
    P(f"{split}: top event_source shares " + " ".join(f"{i}:{cs[i]:.4f}" for i in top))
    res[f"hist_et/{split}"] = ce.tolist()

# ---------------- positives in the same user's NEXT train feed history ----------------
f, M, off, K = tabs["train"]
hoff = f["hist_off"].to_numpy(); H = f["H"].to_numpy()
uid = f["user_id"].to_numpy(); ts = f["timestamp"].to_numpy()
order = np.lexsort((ts, uid))
u_s, t_s = uid[order], ts[order]
nxt = np.full(len(M), -1, np.int64)
same = (u_s[1:] == u_s[:-1]) & (t_s[1:] > t_s[:-1]) & (u_s[1:] != 0)
nxt[order[:-1][same]] = order[1:][same]
gap = np.full(len(M), -1, np.int64); gap[order[:-1][same]] = t_s[1:][same] - t_s[:-1][same]
has = nxt >= 0
P(f"\n== train feeds with a later feed of the same (non-zero) user: {has.sum()} ({has.mean():.3f}); "
  f"median gap {np.median(gap[has])/3600:.1f} h")

citem = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
ctgt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
cpos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
hitem = np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r")
het = np.load(FLAT / "train/hist_event_type.npy", mmap_mode="r")
hes = np.load(FLAT / "train/hist_event_source.npy", mmap_mode="r")
hhp = np.load(FLAT / "train/hist_hpos.npy", mmap_mode="r")

@njit(cache=True)
def scan(a_idx, nxt, off, M, hoff, H, citem, ctgt, hitem, het, hes, out_pos, out_neg, etmask_pos, src_pos, minhpos):
    # out_pos[k]: per positive: 1 if found; etmask_pos: bitmask of event types of that item in the next feed history
    np_ = 0
    for a in a_idx:
        b = nxt[a]
        hs = hitem[hoff[b]:hoff[b] + H[b]]
        ets = het[hoff[b]:hoff[b] + H[b]]
        ess = hes[hoff[b]:hoff[b] + H[b]]
        srt = np.argsort(hs)
        hss = hs[srt]
        for j in range(off[a], off[a] + M[a]):
            it = citem[j]
            lo = np.searchsorted(hss, it)
            found = lo < len(hss) and hss[lo] == it
            if ctgt[j] == 1:
                mask = 0; src = -1; mh = 9999
                if found:
                    k = lo
                    while k < len(hss) and hss[k] == it:
                        e = srt[k]
                        mask |= (1 << ets[e])
                        if e < mh:
                            mh = e; src = ess[e]
                        k += 1
                out_pos[np_] = 1 if found else 0
                etmask_pos[np_] = mask
                src_pos[np_] = src
                minhpos[np_] = mh
                np_ += 1
            else:
                if found:
                    out_neg[0] += 1
                out_neg[1] += 1
    return np_

rng = np.random.default_rng(0)
a_idx = np.where(has)[0]
a_idx = np.sort(rng.choice(a_idx, size=min(400_000, len(a_idx)), replace=False))
npos_tot = int(K[a_idx].sum())
out_pos = np.zeros(npos_tot, np.int8); etm = np.zeros(npos_tot, np.int32); src = np.zeros(npos_tot, np.int32)
mh = np.zeros(npos_tot, np.int32); out_neg = np.zeros(2, np.int64)
t0 = time.time()
n = scan(a_idx, nxt, off, M.astype(np.int64), hoff, H.astype(np.int64), citem, ctgt, hitem, het, hes,
         out_pos, out_neg, etm, src, mh)
P(f"scanned {len(a_idx)} feeds / {n} positives in {time.time()-t0:.0f}s")
fnd = out_pos == 1
P(f"P(positive item in next-feed history) = {fnd.mean():.4f}; P(negative in next history) = {out_neg[0]/out_neg[1]:.5f}")
P("event types (bit set) among found positives: share of found positives having type k at least once:")
P("  " + " ".join(f"et{k}:{((etm[fnd] >> k) & 1).mean():.3f}" for k in range(11)))
# exclusive combos
combos = {}
for m_ in etm[fnd]:
    combos[int(m_)] = combos.get(int(m_), 0) + 1
top = sorted(combos.items(), key=lambda x: -x[1])[:12]
P("most common event-type sets (types in the set: share of found positives):")
for m_, c in top:
    P(f"  {{{','.join(str(k) for k in range(11) if (m_ >> k) & 1)}}}: {c/fnd.sum():.4f}")
cs = np.bincount(src[fnd], minlength=31) / fnd.sum()
P("event_source of the newest matching event: " + " ".join(f"{i}:{cs[i]:.3f}" for i in np.argsort(-cs)[:8]))
P(f"hpos of newest matching event: median {np.median(mh[fnd]):.0f}, P(hpos<5) {(mh[fnd]<5).mean():.3f}, "
  f"P(hpos<20) {(mh[fnd]<20).mean():.3f}")
# found share by gap bucket
gp = np.repeat(gap[a_idx], K[a_idx])
for lo, hi in [(0, 3600), (3600, 6*3600), (6*3600, 24*3600), (24*3600, 3*86400), (3*86400, 30*86400)]:
    m_ = (gp >= lo) & (gp < hi)
    if m_.sum():
        P(f"  gap [{lo/3600:.0f}h,{hi/3600:.0f}h): positives {m_.sum()}, found {fnd[m_].mean():.3f}, "
          f"has et>1 among found {((etm[m_ & fnd] >> 2) > 0).mean():.3f}")
# position of positives with a strong (non-type-1) event vs type-1 only (production feeds only, descriptive)
pos_of_pos = np.asarray(cpos)[np.concatenate([np.arange(off[a], off[a] + M[a]) for a in a_idx])]
tg = np.asarray(ctgt)[np.concatenate([np.arange(off[a], off[a] + M[a]) for a in a_idx])]
pp = pos_of_pos[tg == 1]
Mrep = np.repeat(M[a_idx], K[a_idx]); last = pp == (np.repeat(np.asarray([cpos[off[a]:off[a]+M[a]].max() for a in a_idx]), K[a_idx]))
only1 = fnd & (etm == 2)
other = fnd & (etm != 2)
P(f"positives found with type-1 only: {only1.sum()} | with other types: {other.sum()}")
P(f"  median pos: type1-only {np.median(pp[only1]):.0f}, other {np.median(pp[other]):.0f}, not found {np.median(pp[~fnd]):.0f}")
P(f"  share 'last shown item': type1-only {last[only1].mean():.3f}, other {last[other].mean():.3f}, not found {last[~fnd].mean():.3f}")
res["next_hist"] = dict(found=float(fnd.mean()), neg=float(out_neg[0] / out_neg[1]),
                        et_share={k: float(((etm[fnd] >> k) & 1).mean()) for k in range(11)})
json.dump(res, open(f"{OUT}/01_labels.json", "w"), indent=1)
open(f"{OUT}/01_labels.txt", "w").write("\n".join(lines) + "\n")
