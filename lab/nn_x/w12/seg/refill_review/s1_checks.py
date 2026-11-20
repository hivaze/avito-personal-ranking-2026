"""Reviewer step 1 (label-free): provenance of the study's recovered events, my own aligner on all qualifying feeds,
300-feed comparison, positive check, item-restore shares, parity by day.   python s1_checks.py > out_s1.txt
"""
import sys
import time
from collections import Counter

import numpy as np

from rr_lib import BOUND, HERE, ROOT, SPLITS, Tables, pair, timeline

STUDY = f"{ROOT}/nn_x/w12/seg/refill/out"
t0 = time.time()
T = Tables()
hold, prand = T.holdout()
SETS = (("holdout", "train", hold), ("val", "val", np.arange(T.n["val"])), ("test", "test", np.arange(T.n["test"])))


def log(*a):
    print(*a, flush=True)


def rows_take(arr, rows):
    o = np.argsort(rows, kind="stable")
    out = np.empty(len(rows), arr.dtype)
    out[o] = arr[rows[o]]
    return out


def keys_of(sc, row):
    """key and item of events given as (split code, history row) arrays."""
    key, item = np.zeros(len(sc), np.int64), np.zeros(len(sc), np.int64)
    for c in range(3):
        m = sc == c
        if m.any():
            s = SPLITS[c]
            it = rows_take(T.item[s], row[m]).astype(np.int64)
            key[m] = it * 100000 + rows_take(T.et[s], row[m]).astype(np.int64) * 1000 + rows_take(T.es[s], row[m]).astype(np.int64)
            item[m] = it
    return key, item


log("## 0. tables")
for s in SPLITS:
    log(f"   {s}: feeds {T.n[s]}, timestamp range {T.ts[s].min()}..{T.ts[s].max()} (resolution 128 s), user_id 0 share {100 * (T.uid[s] == 0).mean():.2f}%, H = 256 share {100 * (T.H[s] == 256).mean():.2f}%")
log(f"   train max ts < val min ts: {T.ts['train'].max() < T.ts['val'].min()}; val max ts < test min ts: {T.ts['val'].max() < T.ts['test'].min()}")

# ------------------------------------------------------------------ 1. provenance of the study's recovered rows (ALL feeds)
log("\n## 1. PROVENANCE of the study's recovered events (all qualifying feeds; from raw tables via hist_fidx)")
Z = {}
for name, s, ids in SETS:
    z = np.load(f"{STUDY}/align_{name}.npz")
    Z[name] = z
    q = z["fid"]
    myq = T.qualifying(s, ids)
    log(f"   {name}: qualifying mine {len(myq)} = {100 * len(myq) / len(ids):.2f}% of {len(ids)}; the study's list identical: {np.array_equal(np.sort(myq), np.sort(q))}; "
        f"user_id 0 among them: {(T.uid[s][q] == 0).sum()}; H != 256 among them: {(T.H[s][q] != 256).sum()}")
    for tag in ("T", "U"):
        sc, row = z[tag + "_sc"], z[tag + "_row"]
        m = sc >= 0
        fi = np.repeat(np.arange(len(q)), m.sum(1))
        scv, rowv = sc[m].astype(np.int64), row[m]
        bad_u = bad_t = bad_r = 0
        gaps = []
        src_split = np.zeros(3, np.int64)
        for c in range(3):
            mc = scv == c
            if not mc.any():
                continue
            ss = SPLITS[c]
            g = rows_take(T.hf[ss], rowv[mc]).astype(np.int64)
            # the row must lie inside the history of feed g
            inside = (rowv[mc] >= T.hoff[ss][g]) & (rowv[mc] < T.hoff[ss][g] + T.H[ss][g])
            bad_r += int((~inside).sum())
            f = q[fi[mc]]
            bad_u += int((T.uid[ss][g] != T.uid[s][f]).sum())
            dt = T.ts[s][f] - T.ts[ss][g]
            bad_t += int((dt <= 0).sum())
            gaps.append(dt)
            src_split[c] = mc.sum()
        gaps = np.concatenate(gaps) if gaps else np.zeros(1)
        log(f"      {tag}: recovered rows {int(m.sum())} (from train / val / test tables: {src_split.tolist()}); row outside its feed's history {bad_r}; other user_id {bad_u}; "
            f"source snapshot NOT strictly earlier {bad_t}; smallest gap {gaps.min()} s, median {np.median(gaps) / 86400:.2f} d")
        # are recovered events also stored (same key) in the feed's own 256? (U is 'old-only' by definition)
    A, B, S = z["A"], z["B"], z["stats"]
    log(f"      study flags: A {int(A.sum())}, B {int(B.sum())}, R_T (A, B, n_T >= 1) {int((A & B & (S[:, 6] >= 1)).sum())}, R_U (A, n_U >= 1) {int((A & (S[:, 7] >= 1)).sum())}")

# ------------------------------------------------------------------ 2. my aligner on all qualifying feeds
log("\n## 2. MY ALIGNER (patience anchors + multiset gaps) on all qualifying feeds")
MINE = {}
for name, s, ids in SETS:
    z = Z[name]
    q = z["fid"]
    n = len(q)
    A, Bm, nT, nU = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.int64)
    T_sc, T_row, U_sc, U_row = (np.full((n, 128), -1, np.int64) for _ in range(4))
    gap1 = np.zeros(n)
    order = []
    for i, f in enumerate(q):
        snaps = T.earlier(s, f)
        gap1[i] = (T.ts[s][f] - snaps[0][2]) / 86400
        U, st = timeline(T, s, f, snaps=snaps)
        A[i] = st["n_used"] >= 1
        Bm[i] = A[i] and st["deep"] >= 248
        old = U["lpos"] < 0
        nT[i], nU[i] = st["n_T"], st["n_U"]
        tail = np.where(old)[0]
        tail = tail[tail > st["last"]][:128]
        T_sc[i, :len(tail)], T_row[i, :len(tail)] = U["sc"][tail], U["row"][tail]
        allo = np.where(old)[0][:128]
        U_sc[i, :len(allo)], U_row[i, :len(allo)] = U["sc"][allo], U["row"][allo]
        if i < 1500:  # order evidence against the most recent earlier snapshot
            lk, _, _ = T.hist(SPLITS.index(s), f)
            ek, _, _ = T.hist(snaps[0][0], snaps[0][1])
            mL, mE, anc = pair(lk, ek)
            if len(anc) >= 16:
                order.append([anc[0][0], 255 - anc[-1][0], anc[0][1], len(ek) - 1 - anc[-1][1]])
    o = np.array(order)
    log(f"   {name}: order evidence ({len(o)} feeds, most recent earlier snapshot, >= 16 anchors): stored-only stretch BEFORE the first anchor {o[:, 0].mean():.1f} events, AFTER the last anchor {o[:, 1].mean():.1f}; "
        f"earlier-snapshot stretch before its first anchor {o[:, 2].mean():.1f}, after its last anchor {o[:, 3].mean():.1f}  -> index 0 = newest")
    rT, rU = A & Bm & (nT >= 1), A & (nU >= 1)
    sA, sB, sS = z["A"], z["B"], z["stats"]
    srT, srU = sA & sB & (sS[:, 6] >= 1), sA & (sS[:, 7] >= 1)
    pc = lambda x: " / ".join(f"{v:.0f}" for v in np.percentile(x, (5, 25, 50, 75, 95))) if len(x) else "-"
    log(f"   {name}: my A {int(A.sum())} ({100 * A.mean():.1f}%; study {int(sA.sum())}, both {int((A & sA).sum())}); my B {int(Bm.sum())} (study {int(sB.sum())}, both {int((Bm & sB).sum())}); "
        f"my R_T {int(rT.sum())} (study {int(srT.sum())}, both {int((rT & srT).sum())}); my R_U {int(rU.sum())} (study {int(srU.sum())}, both {int((rU & srU).sum())})")
    log(f"      n_T on my R_T (5/25/50/75/95%): {pc(nT[rT])}, >= 128: {int((nT[rT] >= 128).sum())}; study on its R_T: {pc(sS[srT, 6])}, >= 128: {int((sS[srT, 6] >= 128).sum())}; "
        f"corr(n_T mine, study) on common R_T {np.corrcoef(nT[rT & srT], sS[rT & srT, 6])[0, 1]:.3f}; mean fill mine {np.minimum(nT[rT], 128).mean():.1f}, study {np.minimum(sS[srT, 6], 128).mean():.1f}")
    log(f"      n_U on my R_U: {pc(nU[rU])}, >= 128: {int((nU[rU] >= 128).sum())}; study on its R_U: {pc(sS[srU, 7])}, >= 128: {int((sS[srU, 7] >= 128).sum())}; "
        f"corr {np.corrcoef(nU[rU & srU], sS[rU & srU, 7])[0, 1]:.3f}; mean fill mine {np.minimum(nU[rU], 128).mean():.1f}, study {np.minimum(sS[srU, 7], 128).mean():.1f}")
    log(f"      gap to the most recent earlier snapshot, days (5/25/50/75/95%): " + " / ".join(f"{v:.1f}" for v in np.percentile(gap1, (5, 25, 50, 75, 95))))
    if name == "test":
        pub = T.ts["test"][q] < BOUND
        npub, npri = (T.ts["test"] < BOUND).sum(), (T.ts["test"] >= BOUND).sum()
        log(f"      test shares: qualifying public-like {100 * pub.sum() / npub:.2f}%, private-like {100 * (~pub).sum() / npri:.2f}%; my R_T {100 * (rT & pub).sum() / npub:.2f}% / {100 * (rT & ~pub).sum() / npri:.2f}%; "
            f"my R_U {100 * (rU & pub).sum() / npub:.2f}% / {100 * (rU & ~pub).sum() / npri:.2f}%; study R_T {100 * (srT & pub).sum() / npub:.2f}% / {100 * (srT & ~pub).sum() / npri:.2f}%; study R_U {100 * (srU & pub).sum() / npub:.2f}% / {100 * (srU & ~pub).sum() / npri:.2f}%")
    if name == "val":
        log(f"      val shares of all val feeds: my R_T {100 * rT.sum() / T.n['val']:.2f}%, my R_U {100 * rU.sum() / T.n['val']:.2f}%; study R_T {100 * srT.sum() / T.n['val']:.2f}%, R_U {100 * srU.sum() / T.n['val']:.2f}%")
    MINE[name] = dict(fid=q, A=A, B=Bm, nT=nT, nU=nU, T_sc=T_sc, T_row=T_row, U_sc=U_sc, U_row=U_row)
    np.savez(f"{HERE}/out/myalign_{name}.npz", **MINE[name])
    log(f"   [{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ 3. overlap of the study's events with mine + item-restore shares (all feeds of holdout / val / test)
log("\n## 3. OVERLAP study vs mine (key multisets, first min(n, 128) events) and ITEM-RESTORE shares (label-free)")
for name, s, ids in SETS:
    z, my = Z[name], MINE[name]
    q = z["fid"]
    sc0 = SPLITS.index(s)
    for tag in ("T", "U"):
        sS = z["stats"]
        if tag == "T":
            rs, rm = z["A"] & z["B"] & (sS[:, 6] >= 1), my["A"] & my["B"] & (my["nT"] >= 1)
        else:
            rs, rm = z["A"] & (sS[:, 7] >= 1), my["A"] & (my["nU"] >= 1)
        both = np.where(rs & rm)[0]
        inter = ns = nm = 0
        same_rows = 0
        cls_s, cls_m = np.zeros(4), np.zeros(4)  # item in stored 129-256 | in newest 128 only | novel | total
        key_in_stored = 0
        for src, rd, cls in (("s", rs, cls_s), ("m", rm, cls_m)):
            for i in np.where(rd)[0]:
                if src == "s":
                    sc, row = z[tag + "_sc"][i].astype(np.int64), z[tag + "_row"][i]
                else:
                    sc, row = my[tag + "_sc"][i], my[tag + "_row"][i]
                m = sc >= 0
                k, it = keys_of(sc[m], row[m])
                lk, _, li = T.hist(sc0, q[i])
                old_items, new_items = set(li[128:].tolist()), set(li[:128].tolist())
                in_old = np.array([x in old_items for x in it.tolist()])
                in_new = np.array([x in new_items for x in it.tolist()])
                cls += [in_old.sum(), (in_new & ~in_old).sum(), (~in_new & ~in_old).sum(), len(it)]
                if src == "s":
                    sk = Counter(lk.tolist())
                    key_in_stored += sum(1 for x in k.tolist() if sk.get(x, 0) > 0)
        for i in both:
            a_sc, a_row = z[tag + "_sc"][i].astype(np.int64), z[tag + "_row"][i]
            b_sc, b_row = my[tag + "_sc"][i], my[tag + "_row"][i]
            ka, _ = keys_of(a_sc[a_sc >= 0], a_row[a_sc >= 0])
            kb, _ = keys_of(b_sc[b_sc >= 0], b_row[b_sc >= 0])
            ca, cb = Counter(ka.tolist()), Counter(kb.tolist())
            inter += sum((ca & cb).values())
            ns += len(ka)
            nm += len(kb)
            same_rows += len(set(zip(a_sc[a_sc >= 0].tolist(), a_row[a_sc >= 0].tolist())) & set(zip(b_sc[b_sc >= 0].tolist(), b_row[b_sc >= 0].tolist())))
        log(f"   {name} {tag}: feeds in both read sets {len(both)}; study events {ns}, mine {nm}, common (key multiset) {inter} = {100 * inter / max(ns, 1):.1f}% of the study's, {100 * inter / max(nm, 1):.1f}% of mine; identical (table, row) {100 * same_rows / max(ns, 1):.1f}% of the study's")
        log(f"      ITEM-RESTORE, study events on its read set ({int(cls_s[3])} events): item occurs in the stored events 129-256 (removed by pass b) {100 * cls_s[0] / cls_s[3]:.1f}% | only in the newest 128 {100 * cls_s[1] / cls_s[3]:.1f}% | item not in the stored 256 {100 * cls_s[2] / cls_s[3]:.1f}% "
            f"|| same (item, type, source) key present in the stored 256: {100 * key_in_stored / cls_s[3]:.1f}%")
        log(f"      ITEM-RESTORE, my events on my read set ({int(cls_m[3])} events): in stored 129-256 {100 * cls_m[0] / cls_m[3]:.1f}% | only newest 128 {100 * cls_m[1] / cls_m[3]:.1f}% | novel {100 * cls_m[2] / cls_m[3]:.1f}%")
    log(f"   [{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ 4. 300-feed checks
log("\n## 4. 300 RANDOM QUALIFYING FEEDS per set (rng 4242): aligner-free order test of the study's T events; positive check with my aligner")
for name, s, ids in SETS:
    z = Z[name]
    q = z["fid"]
    sc0 = SPLITS.index(s)
    pick = np.sort(np.random.default_rng(4242).choice(len(q), 300, replace=False))
    # (ii) aligner-free order test: a T event taken from snapshot g at position p must be deeper in g than every key common to the stored history and g (unique in both)
    n_ev = n_viol = n_feed = 0
    mass = []
    n_bad_feed = 0
    for i in pick:
        f = q[i]
        sc, row = z["T_sc"][i].astype(np.int64), z["T_row"][i]
        m = sc >= 0
        if not (z["A"][i] and z["B"][i]) or m.sum() == 0:
            continue
        n_feed += 1
        lk, _, _ = T.hist(sc0, f)
        cl = Counter(lk.tolist())
        feed_bad = False
        for c in np.unique(sc[m]):
            ss = SPLITS[c]
            r = row[m][sc[m] == c]
            g = rows_take(T.hf[ss], r).astype(np.int64)
            for gg in np.unique(g):
                ek, er, _ = T.hist(int(c), int(gg))
                ce = Counter(ek.tolist())
                common_pos = np.array([j for j, k in enumerate(ek.tolist()) if ce[k] == 1 and cl.get(k, 0) == 1])
                p = r[g == gg] - T.hoff[ss][gg]
                if len(common_pos) == 0:
                    continue
                deeper = (common_pos[None, :] > p[:, None]).sum(1)
                n_ev += len(p)
                n_viol += int((deeper > 0).sum())
                mass.append(deeper / len(common_pos))
                feed_bad |= bool((deeper > 0).any())
        n_bad_feed += feed_bad
    mass = np.concatenate(mass) if mass else np.zeros(1)
    log(f"   {name}: order test on {n_feed} of the 300 feeds (study's R_T): T events {n_ev}; events with >= 1 common stored event DEEPER in the source snapshot {n_viol} = {100 * n_viol / max(n_ev, 1):.2f}%; "
        f"mean share of the common events that are deeper {100 * mass.mean():.3f}%; feeds with any such event {n_bad_feed}")
    # (iv) positive check with my aligner
    for lab, cs in (("snapshots cut to 192", 192), ("snapshots at stored length", 256)):
        tot = np.zeros(5)
        full_share = []
        n_rec_feed = 0
        for i in pick:
            f = q[i]
            U, st = timeline(T, s, f, cap=192, cap_snap=cs)
            if st["n_used"] < 1 or st["deep"] < 184 or st["n_T"] < 1:
                continue
            n_rec_feed += 1
            old = np.where(U["lpos"] < 0)[0]
            tail = old[old > st["last"]][:64]
            rec = U["key"][tail]
            fk, _, _ = T.hist(sc0, f)
            newc, truth = Counter(fk[:192].tolist()), fk[192:256]
            tc = Counter(truth.tolist())
            c = [0, 0, 0]
            for k in rec.tolist():
                if tc.get(k, 0) > 0:
                    tc[k] -= 1
                    c[0] += 1
                elif newc.get(k, 0) > 0:
                    newc[k] -= 1
                    c[1] += 1
                else:
                    c[2] += 1
            exact = int((rec == truth[:len(rec)]).sum())
            tot += [c[0], c[1], c[2], exact, len(rec)]
            if len(rec) == 64:
                full_share.append(c[0] / 64)
        log(f"   {name} positive check, my aligner, {lab}: feeds with >= 1 recovered event {n_rec_feed} of 300 ({len(full_share)} with all 64); recovered events {int(tot[4])}: exact position {100 * tot[3] / tot[4]:.1f}% | is a true event 193-256 {100 * tot[0] / tot[4]:.1f}% | "
            f"stored among the newest 192 (placement error) {100 * tot[1] / tot[4]:.1f}% | not stored in the 256 {100 * tot[2] / tot[4]:.1f}% || feeds with 64 recovered: share of the 64 true events found {100 * np.mean(full_share) if full_share else float('nan'):.1f}%")
    log(f"   [{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ 5. parity by day (label-free)
log("\n## 5. PARITY by day: capped, user_id != 0 feeds with a strictly earlier snapshot of the user (train feeds: TRAIN snapshots only)")
for s, allowed in (("train", ("train",)), ("val", ("train", "val")), ("test", ("train", "val", "test"))):
    u = np.concatenate([T.uid[a] for a in allowed])
    t = np.concatenate([T.ts[a] for a in allowed])
    o = np.argsort(u, kind="stable")
    us, ts_ = u[o], t[o]
    ub, st = np.unique(us, return_index=True)
    tmin = np.minimum.reduceat(ts_, st)
    cap = (T.H[s] == 256) & (T.uid[s] != 0)
    first = tmin[np.searchsorted(ub, T.uid[s])]
    has = cap & (first < T.ts[s])
    day = (T.ts[s] - T.ts[s].min()) // 86400
    # gap to the most recent strictly earlier snapshot for qualifying feeds (sample up to 20000)
    idx = np.where(has)[0]
    if len(idx) > 20000:
        idx = np.sort(np.random.default_rng(1).choice(idx, 20000, replace=False))
    key = us.astype(np.float64)
    gaps = []
    for f in idx:
        lo, hi = np.searchsorted(us, T.uid[s][f], "left"), np.searchsorted(us, T.uid[s][f], "right")
        tt = ts_[lo:hi]
        gaps.append(T.ts[s][f] - tt[tt < T.ts[s][f]].max())
    gaps = np.array(gaps) / 86400
    log(f"   {s} (snapshots from {'+'.join(allowed)}): feeds {T.n[s]}; capped and user_id != 0 {100 * cap.mean():.2f}%; qualifying {100 * has.mean():.2f}% of feeds = {100 * has.sum() / max(cap.sum(), 1):.1f}% of capped; "
        f"gap to the most recent earlier snapshot, days (5/25/50/75/95%): " + " / ".join(f"{v:.2f}" for v in np.percentile(gaps, (5, 25, 50, 75, 95))))
    log("      by day: " + " | ".join(f"d{d}: {100 * has[day == d].mean():.1f}% of feeds ({100 * has[day == d].sum() / max(cap[day == d].sum(), 1):.0f}% of capped)" for d in np.unique(day)))
log(f"[{time.time() - t0:.0f}s] done")
