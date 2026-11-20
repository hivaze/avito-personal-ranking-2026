"""Pre-training audit A2 (CPU): history lengths, user overlap, time gaps, event composition, snapshot alignment
(how many val/test history events are newer than the user's latest train snapshot = not in the train part of the
pretraining corpus; how many are in no corpus sequence at all), user-embedding drift.  Full data unless stated."""
import time, numpy as np, polars as pl, numba as nb
from pathlib import Path
nb.set_num_threads(12)
R = Path("<repo>"); F = R / "data/flat"; P = R / "data/pretrain"
BOUND = 1790497920
t0 = time.time()
def L(s, n): return np.load(F / s / f"{n}.npy", mmap_mode="r")
fe = {s: pl.read_parquet(F / s / "feeds.parquet") for s in ["train", "val", "test"]}
pr = np.nan_to_num(np.load(R / "eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
fe["train"] = fe["train"].with_columns(pl.Series("p_random", pr))
def day(ts): return (np.asarray(ts, np.int64) + 3 * 3600) // 86400
import datetime as dt
def dstr(ts): return dt.datetime.utcfromtimestamp(int(ts) + 3 * 3600).strftime("%m-%d %H:%M MSK")
print("== time ranges (MSK)")
for s in fe:
    ts = fe[s]["timestamp"].to_numpy(); print(f"  {s}: {dstr(ts.min())} .. {dstr(ts.max())}; distinct ts {len(np.unique(ts))}; ts granularity gcd-ish: {np.min(np.diff(np.unique(ts)))}s")
tr_ts = fe["train"]["timestamp"].to_numpy(); main_end = np.sort(tr_ts)[-35]  # ignore 34 boundary feeds stamped Sep 20 00:00
print(f"  train end (excl. 34 Sep-20 00:00 feeds) {dstr(main_end)}; public/private boundary {dstr(BOUND)}")
for s in ["val", "test"]:
    ts = fe[s]["timestamp"].to_numpy(); print(f"  {s} start - train end = {(ts.min()-main_end)/86400:.2f} d; {s} end - train end = {(ts.max()-main_end)/86400:.2f} d")
te_ts = fe["test"]["timestamp"].to_numpy(); print(f"  public feeds {np.sum(te_ts<BOUND)}, private {np.sum(te_ts>=BOUND)}; private days {sorted(set(day(te_ts[te_ts>=BOUND]).tolist()))}")
# ---- H distributions
def hq(H, lab):
    H = np.asarray(H); q = np.quantile(H, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"  {lab:34s} n {len(H):>9,} mean {H.mean():6.1f} q10/25/50/75/90 {q.astype(int).tolist()} H=0 {np.mean(H==0):.4f} H<10 {np.mean(H<10):.4f} H<64 {np.mean(H<64):.4f} H=256 {np.mean(H==256):.4f}")
print("== history length H (events per feed snapshot, cap 256)")
tr = fe["train"]
hq(tr["H"], "train all feeds")
hq(tr.filter(pl.col("p_random") > 0.9)["H"], "train random-like (p_random>0.9)")
lat = tr.filter(pl.col("user_id") != 0).sort("timestamp").group_by("user_id").agg(pl.col("H").last(), pl.col("fidx").last(), pl.len().alias("nf"), pl.col("timestamp").first().alias("t0"), pl.col("timestamp").last().alias("t1"))
hq(lat["H"], "train latest snapshot per user")
hq(fe["val"]["H"], "val"); hq(fe["test"]["H"], "test")
hq(fe["test"].filter(pl.col("timestamp") < BOUND)["H"], "test public"); hq(fe["test"].filter(pl.col("timestamp") >= BOUND)["H"], "test private")
for s in fe:
    u = fe[s]["user_id"].to_numpy(); print(f"  {s}: user_id==0 feed share {np.mean(u==0):.4f}; distinct non-zero users {len(np.unique(u[u>0])):,}")
nf = lat["nf"].to_numpy(); span = (lat["t1"].to_numpy() - lat["t0"].to_numpy()) / 86400
print(f"== train feeds per non-zero user: mean {nf.mean():.3f}; share >=2 {np.mean(nf>=2):.4f} >=3 {np.mean(nf>=3):.4f} >=5 {np.mean(nf>=5):.4f}; max {nf.max()}; span(first..last) among >=2: q50 {np.median(span[nf>=2]):.2f} d q90 {np.quantile(span[nf>=2],0.9):.2f} d")
# ---- user overlap
tru = set(lat["user_id"].to_list())
latd = dict(zip(lat["user_id"].to_list(), zip(lat["fidx"].to_list(), lat["H"].to_list(), lat["t1"].to_list(), lat["nf"].to_list())))
vu = set(fe["val"].filter(pl.col("user_id") != 0)["user_id"].to_list())
print("== user overlap (non-zero users; feed-weighted)")
for lab, f in [("val", fe["val"]), ("test", fe["test"]), ("test public", fe["test"].filter(pl.col("timestamp") < BOUND)), ("test private", fe["test"].filter(pl.col("timestamp") >= BOUND))]:
    u = f["user_id"].to_numpy(); ts = f["timestamp"].to_numpy(); H = f["H"].to_numpy(); nz = u > 0
    intr = np.array([x in tru for x in u]) & nz
    gaps = np.array([(t - latd[x][2]) / 86400 for x, t in zip(u[intr], ts[intr])])
    trH = np.array([latd[x][1] for x in u[intr]]); trn = np.array([latd[x][3] for x in u[intr]])
    print(f"  {lab:12s}: feeds {len(u):,}; user0 {np.mean(~nz):.4f}; non-zero user WITH train feed {intr.mean():.4f} (of non-zero {intr[nz].mean():.4f}); without {np.mean(nz & ~intr):.4f}")
    print(f"     returning users: train feeds/user mean {trn.mean():.2f}; latest-train H q50 {np.median(trH):.0f} mean {trH.mean():.1f}; own H q50 {np.median(H[intr]):.0f}; new users' own H q50 {np.median(H[nz & ~intr]) if (nz & ~intr).any() else -1:.0f} H=0 {np.mean(H[nz & ~intr]==0) if (nz & ~intr).any() else -1:.3f}")
    print(f"     gap feed ts - user's last train feed ts (days): q10 {np.quantile(gaps,0.1):.2f} q50 {np.median(gaps):.2f} q90 {np.quantile(gaps,0.9):.2f} max {gaps.max():.2f}")
    if lab == "test":
        print(f"     test users also in val: {np.mean([x in vu for x in u[nz]]):.4f} of non-zero test feeds")
# ---- event composition
print("== history event composition (share of events)")
comp = {}
for s in ["train", "val", "test"]:
    et = np.asarray(L(s, "hist_event_type")).astype(np.int64); es = np.asarray(L(s, "hist_event_source")).astype(np.int64)
    comp[s] = (np.bincount(et, minlength=11) / len(et), np.bincount(es, minlength=31) / len(es))
    pair = np.bincount(et * 32 + es, minlength=11 * 32) / len(et)
    top = np.argsort(-pair)[:8]
    print(f"  {s}: events {len(et):,}; et1&es1 (fe-target type) share {pair[1*32+1]:.4f}; top (et,es): " + ", ".join(f"({k//32},{k%32}) {pair[k]:.3f}" for k in top))
    hp = np.asarray(L(s, "hist_hpos")).astype(np.int64)
    bks = [(0, 32), (32, 128), (128, 192), (192, 256)]
    print("     et share by hpos bucket (et1/et2/et3/et4/et>=5): " + " | ".join(
        f"hpos {a}-{b-1}: " + "/".join(f"{np.mean(et[(hp>=a)&(hp<b)]==k):.3f}" for k in [1, 2, 3, 4]) + f"/{np.mean(et[(hp>=a)&(hp<b)]>=5):.3f}" for a, b in bks))
    del et, es, hp
print("  event_type share train/val/test: " + "  ".join(f"et{k}: {comp['train'][0][k]:.4f}/{comp['val'][0][k]:.4f}/{comp['test'][0][k]:.4f}" for k in range(11) if comp['train'][0][k] + comp['val'][0][k] > 1e-4))
print("  event_source share (>0.5% anywhere) train/val/test: " + "  ".join(f"es{k}: {comp['train'][1][k]:.4f}/{comp['val'][1][k]:.4f}/{comp['test'][1][k]:.4f}" for k in range(31) if max(comp['train'][1][k], comp['val'][1][k], comp['test'][1][k]) > 0.005))
print(f"  ({time.time()-t0:.0f}s)")
# ---- snapshot alignment
@nb.njit(cache=True)
def align(new, old, m):
    # new/old newest-first key arrays; smallest k with new[k:k+mm]==old[:mm]; returns k or -1
    if len(old) == 0:
        return len(new)
    for k in range(len(new)):
        mm = min(m, len(old), len(new) - k)
        ok = True
        for j in range(mm):
            if new[k + j] != old[j]:
                ok = False; break
        if ok:
            return k
    return -1
@nb.njit(parallel=True, cache=True)
def batch(keyA, offA, HA, keyB, offB, HB, m, out_k, out_setnew):
    for i in nb.prange(len(offA)):
        a = keyA[offA[i]:offA[i] + HA[i]]; b = keyB[offB[i]:offB[i] + HB[i]]
        out_k[i] = align(a, b, m)
        sb = set(b[:1]) if len(b) > 0 else set([np.int64(-1)])
        for x in b:
            sb.add(x)
        c = 0
        for x in a:
            if x not in sb:
                c += 1
        out_setnew[i] = c
def keys(s):
    it = np.asarray(L(s, "hist_item_id")).astype(np.int64); et = np.asarray(L(s, "hist_event_type")).astype(np.int64); es = np.asarray(L(s, "hist_event_source")).astype(np.int64)
    return it * 512 + et * 32 + es
ktr = keys("train"); print(f"  train keys built ({time.time()-t0:.0f}s)")
tr_off = tr["hist_off"].to_numpy(); tr_H = tr["H"].to_numpy()
print("== val/test history events vs the pretraining corpus (alignment on (item, event_type, event_source), newest-first, m=4)")
corp_f = {1: set(np.load(P / "seq_fidx.npy")[np.load(P / "seq_split.npy") == 1].tolist()), 2: set(np.load(P / "seq_fidx.npy")[np.load(P / "seq_split.npy") == 2].tolist())}
for sid, s in [(1, "val"), (2, "test")]:
    f = fe[s]; kv = keys(s); u = f["user_id"].to_numpy(); off = f["hist_off"].to_numpy(); H = f["H"].to_numpy(); ts = f["timestamp"].to_numpy()
    has = np.array([(x != 0) and (x in latd) for x in u])
    idx = np.where(has)[0]; trf = np.array([latd[x][0] for x in u[idx]], np.int64)
    outk = np.empty(len(idx), np.int64); outs = np.empty(len(idx), np.int64)
    batch(kv, off[idx].astype(np.int64), H[idx].astype(np.int64), ktr, tr_off[trf].astype(np.int64), tr_H[trf].astype(np.int64), 4, outk, outs)
    al = outk >= 0
    newev = np.where(al, outk, outs)
    tot = H.sum(); new_returning = newev.sum(); new_cold = H[~has].sum()
    print(f"  {s}: feeds with a train snapshot {has.mean():.4f}; contiguous alignment found {al.mean():.4f}; new events per returning feed (k) q25/50/75/90 {np.quantile(outk[al],[0.25,0.5,0.75,0.9]).astype(int).tolist()} mean {outk[al].mean():.1f}; set-based new q50 {np.median(outs):.0f}")
    print(f"     k==0 (no new event since train) {np.mean(outk[al]==0):.4f}; k==H (no overlap) {np.mean(outk[al]==H[idx][al]):.4f}")
    print(f"     share of ALL {s} history events NOT in any train snapshot (new since train + users w/o train feed): {(new_returning+new_cold)/tot:.4f}  [returning-user new {new_returning/tot:.4f}; no-train-feed users {new_cold/tot:.4f}]")
    pub = ts < BOUND
    if s == "test":
        for lab, msk in [("public", pub), ("private", ~pub)]:
            hm = has & msk; nr = newev[msk[idx]].sum(); nc = H[(~has) & msk].sum()
            print(f"     {lab}: returning {hm.sum()/msk.sum():.4f}; median k {np.median(outk[msk[idx] & al]):.0f}; share events not in any train snapshot {(nr+nc)/H[msk].sum():.4f}")
    # events in NO corpus sequence: feeds not selected (later snapshots of multi-feed users) vs that user's selected snapshot
    fid = f["fidx"].to_numpy(); notsel = np.array([x not in corp_f[sid] for x in fid])
    print(f"     {s} feeds not themselves in the corpus (later val/test snapshots of the same user): {notsel.sum()} ({notsel.mean():.4f}); their events {H[notsel].sum():,} of {tot:,}")
# combined val+test 'not in corpus' events: align non-selected feeds to the user's selected (earliest val/test) snapshot
vt = pl.concat([fe["val"].with_columns(pl.lit(1).alias("sp")), fe["test"].with_columns(pl.lit(2).alias("sp"))]).filter(pl.col("user_id") != 0).sort(["timestamp", "sp", "fidx"])
first = vt.group_by("user_id", maintain_order=True).agg(pl.col("fidx").first().alias("f0"), pl.col("sp").first().alias("s0"))
later = vt.join(first, on="user_id").filter(~((pl.col("fidx") == pl.col("f0")) & (pl.col("sp") == pl.col("s0"))))
kk = {1: keys("val"), 2: keys("test")}; offs = {1: fe["val"]["hist_off"].to_numpy(), 2: fe["test"]["hist_off"].to_numpy()}; Hs = {1: fe["val"]["H"].to_numpy(), 2: fe["test"]["H"].to_numpy()}
tot_not = 0
for r in later.iter_rows(named=True):
    a = kk[r["sp"]][offs[r["sp"]][r["fidx"]]:offs[r["sp"]][r["fidx"]] + Hs[r["sp"]][r["fidx"]]]
    b = kk[r["s0"]][offs[r["s0"]][r["f0"]]:offs[r["s0"]][r["f0"]] + Hs[r["s0"]][r["f0"]]]
    k = align(a, b, 4); tot_not += (k if k >= 0 else len(np.setdiff1d(a, b)))
print(f"  later val/test snapshots of the same user: {len(later)} feeds; their events newer than the user's corpus snapshot: {tot_not:,} = {tot_not/(Hs[1].sum()+Hs[2].sum()):.5f} of all val+test history events")
# ---- train: events lost by keeping only the latest train snapshot (earliest vs latest, users with >=2 train feeds; 100K-user sample)
first_tr = tr.filter(pl.col("user_id") != 0).sort("timestamp").group_by("user_id").agg(pl.col("fidx").first().alias("f0"), pl.col("fidx").last().alias("f1"), pl.len().alias("nf")).filter(pl.col("nf") >= 2)
smp = first_tr.sample(n=min(100_000, len(first_tr)), seed=0)
f0 = smp["f0"].to_numpy().astype(np.int64); f1 = smp["f1"].to_numpy().astype(np.int64)
outk = np.empty(len(f0), np.int64); outs = np.empty(len(f0), np.int64)
batch(ktr, tr_off[f1].astype(np.int64), tr_H[f1].astype(np.int64), ktr, tr_off[f0].astype(np.int64), tr_H[f0].astype(np.int64), 4, outk, outs)
al = outk >= 0; H0 = tr_H[f0]; H1 = tr_H[f1]
dropped = np.where(al, np.maximum(0, H0 - (H1 - outk)), 0)
print(f"== train users with >=2 feeds ({len(first_tr):,}; sample {len(f0):,}): earliest->latest new events k q50 {np.median(outk[al]):.0f} q90 {np.quantile(outk[al],0.9):.0f}; aligned {al.mean():.4f}")
print(f"   events of the EARLIEST snapshot that fell out of the latest 256-window (lost in a latest-only corpus): mean {dropped[al].mean():.1f}/user, share of earliest-snapshot events {dropped[al].sum()/H0[al].sum():.4f}; users with any loss {np.mean(dropped[al]>0):.4f}")
# ---- user embedding drift
ue_tr = np.load(F / "train/user_emb.npy", mmap_mode="r")
a = np.asarray(ue_tr[np.sort(f0[:20000])]).astype(np.float32); o = np.argsort(f0[:20000]); b_idx = f1[:20000][o]
b = np.asarray(ue_tr[b_idx]).astype(np.float32)
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
print(f"== organisers' user_emb: same user earliest vs latest train feed (20K users): identical {np.mean(np.all(a==b,1)):.4f}; cos q10/50/90 {np.quantile(cos,[0.1,0.5,0.9]).round(4).tolist()}")
ue_v = np.load(F / "val/user_emb.npy"); u = fe["val"]["user_id"].to_numpy(); has = np.array([(x != 0) and (x in latd) for x in u])
vi = np.where(has)[0]; ti = np.array([latd[x][0] for x in u[vi]])
A = ue_v[vi].astype(np.float32); B = np.asarray(ue_tr[np.sort(ti)]).astype(np.float32)[np.argsort(np.argsort(ti))]
cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
print(f"   val feed vs same user's latest train feed ({len(vi)}): identical {np.mean(np.all(A==B,1)):.4f}; cos q10/50/90 {np.quantile(cos,[0.1,0.5,0.9]).round(4).tolist()}")
print(f"done {time.time()-t0:.0f}s")
