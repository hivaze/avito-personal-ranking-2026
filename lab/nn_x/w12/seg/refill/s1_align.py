"""Step 1 (label-free): qualifying feeds, alignment, recovered events, ambiguity criterion, positive check.
  python s1_align.py            -> out/align_{holdout,val,test}.npz, out_s1.txt (stdout)
No label is read. Definitions (see align_lib.py and DECLARATION.txt):
  T events = events of the union timeline OLDER than the stored 256th event (after the last stored event), newest first.
  U events = all events the earlier snapshots hold and the stored history lacks (old-only), newest first (T is its tail).
  criterion A (aligned)      : >= 1 earlier snapshot merged (LCS >= 16 anchors and an exact contiguous run >= 8).
  criterion B (end-anchored) : the deepest stored event that is an LCS anchor has index >= 248 (<= 7 stored events behind it).
"""
import sys
import time

import numpy as np

from align_lib import BOUND, OUT, SPL, Flat, lcs, union_timeline

KEEP = 128
t0 = time.time()
F = Flat()
hold, _ = F.holdout()
STATS = ("n_snap", "n_merged", "first_anchor", "first_run", "deep_anchor", "n_anch_L", "n_T", "n_U", "n_swap")


def pct(x, q=(5, 25, 50, 75, 95)):
    return " / ".join(f"{v:.0f}" for v in np.percentile(x, q))


def qualifying(split, ids):
    H, u = F.H[split], F.uid[split]
    cap = ids[H[ids] == 256]
    cap_u = cap[u[cap] != 0]
    q = np.array([len(F.earlier(split, f)) > 0 for f in cap_u], bool)
    return cap, cap_u, cap_u[q]


def types_of(sc, row):
    out = np.zeros(len(sc), np.int64)
    for i in range(3):
        m = sc == i
        if m.any():
            out[m] = np.asarray(F.et[SPL[i]][np.sort(row[m])])[np.argsort(np.argsort(row[m]))]
    return out


def run_set(name, split, ids_all):
    cap, cap_u, q = qualifying(split, ids_all)
    n = len(q)
    print(f"\n## {name}: feeds {len(ids_all)} | capped (H = 256) {len(cap)} = {100 * len(cap) / len(ids_all):.2f}% | capped, user_id != 0 {len(cap_u)} | "
          f"capped, user_id != 0, >= 1 strictly earlier snapshot of the user: {n} = {100 * n / len(ids_all):.2f}% of feeds", flush=True)
    S = np.zeros((n, len(STATS)), np.int64)
    T_sc, T_row = np.full((n, KEEP), -1, np.int8), np.full((n, KEEP), -1, np.int64)
    U_sc, U_row = np.full((n, KEEP), -1, np.int8), np.full((n, KEEP), -1, np.int64)
    snap_split = np.zeros((n, 3), np.int64)
    t1share, ord_stats = [], []
    for i, f in enumerate(q):
        snaps = F.earlier(split, f)
        for sc, _ in snaps:
            snap_split[i, sc] += 1
        U, st = union_timeline(F, split, f, snaps=snaps)
        S[i] = [st[k] for k in STATS]
        old = U["lpos"] < 0
        iL = st["iL"]
        tsc, trow = U["sc"][iL + 1:], U["row"][iL + 1:]
        k = min(len(tsc), KEEP)
        T_sc[i, :k], T_row[i, :k] = tsc[:k], trow[:k]
        usc, urow = U["sc"][old], U["row"][old]
        k = min(len(usc), KEEP)
        U_sc[i, :k], U_row[i, :k] = usc[:k], urow[:k]
        if i < 1500:  # order evidence and type composition on the first 1500 feeds (label-free)
            sc1, f1 = snaps[0]
            ek, _ = F.seq(sc1, f1)
            lk, _ = F.seq(SPL.index(split), f)
            ia, ib = lcs(lk, ek)
            if len(ia) >= 16:
                ord_stats.append([ia[0], 255 - ia[-1], ib[0], len(ek) - 1 - ib[-1]])
            et_st = np.asarray(F.et[split][F.hoff[split][f] + 128:F.hoff[split][f] + 256])
            row = [(et_st == 1).mean(), np.nan, np.nan]
            if st["n_T"] > 0:
                row[1] = (types_of(T_sc[i][T_sc[i] >= 0], T_row[i][T_sc[i] >= 0]) == 1).mean()
            if st["n_U"] > 0:
                row[2] = (types_of(U_sc[i][U_sc[i] >= 0], U_row[i][U_sc[i] >= 0]) == 1).mean()
            t1share.append(row)
    A = S[:, 1] >= 1
    B = A & (S[:, 4] >= 248)
    nT, nU = S[:, 6], S[:, 7]
    print(f"earlier snapshots per feed (5/25/50/75/95%): {pct(S[:, 0])}; by split of the snapshot: train {snap_split[:, 0].sum()}, val {snap_split[:, 1].sum()}, test {snap_split[:, 2].sum()}")
    print(f"criterion A (aligned: >= 1 snapshot with >= 16 LCS anchors and an exact run >= 8): pass {A.sum()} = {100 * A.mean():.1f}%, FAIL {100 * (1 - A.mean()):.1f}% (dropped from everything)")
    print(f"  first merged snapshot, anchors (5/25/50/75/95%): {pct(S[A, 2])}; longest exact run: {pct(S[A, 3])}; tie swaps repaired per feed: {pct(S[A, 8])}")
    print(f"criterion B (end-anchored: deepest anchored stored event >= index 248), among A: pass {B.sum()} = {100 * B.sum() / max(A.sum(), 1):.1f}% of A = {100 * B.mean():.1f}% of qualifying; A-and-not-B {100 * (A & ~B).mean():.1f}% (kept for U, dropped from T)")
    for nm, m, x in (("T (older than the stored 256th event), feeds A and B", B, nT), ("U (all old-only events), feeds A", A, nU)):
        print(f"  {nm}: n = {m.sum()}; recovered events (5/25/50/75/95%): {pct(x[m])}; mean {x[m].mean():.1f}; "
              + "; ".join(f">= {k}: {(x[m] >= k).sum()} ({100 * (x[m] >= k).mean():.1f}% of them, {100 * (m & (x >= k)).sum() / n:.1f}% of qualifying)" for k in (1, 32, 64, 128)))
    o = np.array(ord_stats)
    print(f"order evidence (first {len(o)} feeds with >= 16 anchors to the most recent earlier snapshot): stored-only events BEFORE the first anchor (index-0 side) mean {o[:, 0].mean():.1f}, "
          f"AFTER the last anchor {o[:, 1].mean():.1f}; earlier-snapshot-only events before its first anchor {o[:, 2].mean():.1f}, after its last anchor {o[:, 3].mean():.1f}")
    ts = np.array(t1share)
    print(f"type-1 (view) share: stored events 129-256 {np.nanmean(ts[:, 0]):.3f} | T events {np.nanmean(ts[:, 1]):.3f} | U events {np.nanmean(ts[:, 2]):.3f} (first 1500 feeds)")
    np.savez(OUT / f"align_{name}.npz", fid=q, stats=S, A=A, B=B, T_sc=T_sc, T_row=T_row, U_sc=U_sc, U_row=U_row, n_all=len(ids_all), cap=cap)
    return q, S, A, B


def positive_check(name, split, q, cap_snap, label):
    """pretend the cap is 192: stored history cut to 192, recover events 193-256, compare with the truth."""
    rows = []
    for f in q:
        U, st = union_timeline(F, split, f, cap=192, cap_snap=cap_snap)
        if st["n_merged"] < 1 or st["deep_anchor"] < 184 or st["n_T"] < 1:
            rows.append([st["n_merged"] >= 1, st["deep_anchor"] >= 184, 0, 0, 0, 0, 0])
            continue
        iL = st["iL"]
        rec = U["key"][iL + 1:][:64]
        truth, _ = F.seq(SPL.index(split), f)
        truth = truth[192:256]
        m = len(rec)
        pos = int((rec == truth[:m]).sum())
        # order-free: multiset intersection with the true events 193..192+m and with all 64 true events
        def inter(a, b):
            ua, ca = np.unique(a, return_counts=True)
            ub, cb = np.unique(b, return_counts=True)
            _, xa, xb = np.intersect1d(ua, ub, return_indices=True)
            return int(np.minimum(ca[xa], cb[xb]).sum())
        rows.append([True, True, m, pos, inter(rec, truth[:m]), inter(rec, truth), int(pos == m)])
    r = np.array(rows, dtype=np.int64)
    ok = r[:, 2] > 0
    ev = r[ok, 2].sum()
    full = ok & (r[:, 2] == 64)
    print(f"positive check [{name}, {label}]: feeds {len(q)}; aligned {r[:, 0].sum()}, end-anchored (index >= 184) {r[:, 1].sum()}, with >= 1 recovered event {ok.sum()} ({full.sum()} with all 64); "
          f"recovered events compared {ev}: exact match at the same position {100 * r[ok, 3].sum() / ev:.1f}%; in the true events of the same span (any order) {100 * r[ok, 4].sum() / ev:.1f}%; "
          f"in the true events 193-256 (any order) {100 * r[ok, 5].sum() / ev:.1f}%; feeds with every recovered event exact {100 * r[ok, 6].mean():.1f}%; "
          f"feeds with 64 recovered: position-exact {100 * r[full, 3].sum() / max(64 * full.sum(), 1):.1f}%, any order {100 * r[full, 5].sum() / max(64 * full.sum(), 1):.1f}%", flush=True)


res = {}
for name, split, ids in (("holdout", "train", hold), ("val", "val", np.arange(len(F.uid["val"]))), ("test", "test", np.arange(len(F.uid["test"])))):
    res[name] = run_set(name, split, ids)
    print(f"[{time.time() - t0:.0f}s]", flush=True)
    if name == "test":
        q, S, A, B = res[name]
        tst = F.ts["test"]
        pub = tst < BOUND
        nt = len(tst)
        for nm, m in (("test all", np.ones(nt, bool)), ("public-like days", pub), ("private-like days", ~pub)):
            mq = m[q]
            cap = (F.H["test"] == 256) & m
            print(f"  {nm}: feeds {m.sum()}; capped {100 * cap.sum() / m.sum():.2f}%; qualifying {100 * mq.sum() / m.sum():.2f}%; aligned (A) {100 * (mq & A).sum() / m.sum():.2f}%; "
                  + "; ".join(f"T >= {k} (A, B): {100 * (mq & B & (S[:, 6] >= k)).sum() / m.sum():.2f}%" for k in (1, 64, 128)) + "; "
                  + "; ".join(f"U >= {k} (A): {100 * (mq & A & (S[:, 7] >= k)).sum() / m.sum():.2f}%" for k in (1, 64, 128))
                  + f"; mean T fill (cap 128) per feed of the segment {np.minimum(S[:, 6], 128)[mq & B].sum() / m.sum():.2f}; mean U fill {np.minimum(S[:, 7], 128)[mq & A].sum() / m.sum():.2f}")
    if name == "val":
        q, S, A, B = res[name]
        nv = len(F.uid["val"])
        print(f"  val: qualifying {100 * len(q) / nv:.2f}% of feeds; aligned (A) {100 * A.sum() / nv:.2f}%; "
              + "; ".join(f"T >= {k} (A, B): {100 * (B & (S[:, 6] >= k)).sum() / nv:.2f}%" for k in (1, 64, 128)) + "; "
              + "; ".join(f"U >= {k} (A): {100 * (A & (S[:, 7] >= k)).sum() / nv:.2f}%" for k in (1, 64, 128))
              + f"; mean T fill per val feed {np.minimum(S[:, 6], 128)[B].sum() / nv:.2f}; mean U fill {np.minimum(S[:, 7], 128)[A].sum() / nv:.2f}")
print("\n## positive check of the method (pretend the cap is 192; truth = stored events 193-256)")
for name, split in (("holdout", "train"), ("val", "val")):
    q = res[name][0]
    positive_check(name, split, q, 192, "earlier snapshots also cut to 192 (faithful)")
    positive_check(name, split, q, 256, "earlier snapshots at their stored length")
print(f"[{time.time() - t0:.0f}s] done")
