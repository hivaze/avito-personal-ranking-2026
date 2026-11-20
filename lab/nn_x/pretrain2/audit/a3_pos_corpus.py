"""Pre-training audit A3 (CPU): val positive vs negative coverage by train history / ID table; history events without a
content vector (by event type) and how many a global item table recovers; within-sequence repeats; corpus composition."""
import time, numpy as np, polars as pl
from pathlib import Path
R = Path("<repo>"); F = R / "data/flat"; P = R / "data/pretrain"
t0 = time.time()
def L(s, n): return np.load(F / s / f"{n}.npy", mmap_mode="r")
cnt = np.load(P / "item_cnt.npy", mmap_mode="r"); G = len(cnt)
idf = np.asarray(cnt) >= 3
trh = np.zeros(G, np.bool_); trh[np.asarray(L("train", "hist_item_id"))] = True
trc = np.zeros(G, np.bool_); trc[np.asarray(L("train", "cand_item_id"))] = True
has2 = np.load(P / "item_has2.npy", mmap_mode="r")
vi = np.asarray(L("val", "cand_item_id")); vt = np.asarray(L("val", "cand_target")) == 1
print(f"val positives {vt.sum():,} of {len(vt):,} rows ({vt.mean():.4f})")
for lab, b in [("ID row (cnt>=3)", idf), ("in train hist", trh), ("in train cand", trc), ("in train cand|hist", trh | trc)]:
    print(f"  {lab:20s}: positives {b[vi[vt]].mean():.4f} negatives {b[vi[~vt]].mean():.4f}; pos rate if covered {vt[b[vi]].mean():.4f} / not {vt[~b[vi]].mean():.4f}")
# feeds whose positives are all cold (not in train cand|hist)
fe = pl.read_parquet(F / "val/feeds.parquet"); M = fe["M"].to_numpy(); rf = np.repeat(np.arange(len(M)), M)
cold = ~(trh | trc)[vi]
npos = np.bincount(rf, weights=vt, minlength=len(M)); ncoldpos = np.bincount(rf, weights=vt & cold, minlength=len(M))
haspos = npos > 0
print(f"  val feeds with >=1 positive {haspos.mean():.4f}; all positives cold {np.mean(ncoldpos[haspos]==npos[haspos]):.4f}; any cold positive {np.mean(ncoldpos[haspos]>0):.4f}")
# history events without content vector
for s in ["val", "test"]:
    he = np.asarray(L(s, "hist_emb_idx")); hi = np.asarray(L(s, "hist_item_id")); et = np.asarray(L(s, "hist_event_type"))
    miss = he < 0
    print(f"== {s} history: emb missing {miss.mean():.4f}; recovered by global table item_has2 {has2[hi[miss]].mean():.4f}; missing by et: " +
          " ".join(f"et{k}:{miss[et==k].mean():.3f}" for k in range(1, 11)))
    print(f"   missing-emb events with an ID row (cnt>=3): {idf[hi[miss]].mean():.4f}; with ID row among all events {idf[hi].mean():.4f}")
# within-sequence repeats (val/test full, train 200K-feed sample): share of events whose item occurs at an older position
def repeats(s, feeds=None):
    f = pl.read_parquet(F / s / "feeds.parquet"); off = f["hist_off"].to_numpy(); H = f["H"].to_numpy()
    ids = np.arange(len(f)) if feeds is None else feeds
    hi = L(s, "hist_item_id"); rep = 0; tot = 0; uniq_share = []
    for i in ids:
        a = np.asarray(hi[off[i]:off[i] + H[i]])
        if len(a) == 0: continue
        u = len(np.unique(a)); rep += len(a) - u; tot += len(a); uniq_share.append(u / len(a))
    return rep / tot, np.median(uniq_share)
rng = np.random.default_rng(0)
for s, fd in [("val", None), ("test", None), ("train", rng.choice(3151208, 100_000, replace=False))]:
    r, u = repeats(s, fd); print(f"== {s}{' (100K-feed sample)' if fd is not None else ''}: share of history events repeating an item already in the snapshot {r:.4f}; median distinct-item share per snapshot {u:.3f}")
# corpus composition
off = np.load(P / "seq_off.npy"); sp = np.load(P / "seq_split.npy"); Ls = np.diff(off)
for k, nm in enumerate(["train", "val", "test"]):
    m = sp == k; print(f"== corpus {nm}: seqs {m.sum():,} tokens {Ls[m].sum():,} len q50 {np.median(Ls[m]):.0f} share len=256 {np.mean(Ls[m]==256):.3f}")
tf = np.load(P / "tgt_fe.npy", mmap_mode="r"); tn = np.load(P / "tgt_new.npy", mmap_mode="r")
smp = rng.choice(len(tf), 20_000_000, replace=False); smp.sort()
print(f"   positions with a valid fe target {np.mean(np.asarray(tf[smp])>=0):.4f}; new target {np.mean(np.asarray(tn[smp])>=0):.4f} (20M-position sample)")
fe_items = np.asarray(tf[smp]); fe_items = fe_items[fe_items >= 0]
print(f"   fe targets whose item has an ID row {idf[fe_items].mean():.4f}")
print(f"done {time.time()-t0:.0f}s")
