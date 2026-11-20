"""Step 1 (label-free; no label file is opened): align EVERY feed of train / val / test (holdout feeds are train rows) with
the most recent admissible strictly earlier snapshot of the same user that is at least k days older, for each k in KS,
and keep the events older than the feed's own oldest stored event (T, <= 128). See hx_lib.py for the definitions.

  python s1_align_all.py [--splits train,val,test] [--limit N] [--workers 11]
  -> out/align_<split>.npz : per (feed, k) stats + concatenated T events (source split code, source global history row)

CPU only: 11 single-threaded worker processes (+ an idle parent), so at most 12 busy threads.
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="train,val,test")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=11)
ap.add_argument("--chunk", type=int, default=1500)
ARGS = ap.parse_args()
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "1"  # per worker; ARGS.workers processes in total
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import OUTD, SPL, Flat, extend_one, pick_snapshot  # noqa: E402

KS = (0.0, 3.0, 5.0, 7.0, 10.0)
STAT = ("has", "sc", "fe", "gap", "A", "B", "n_anchor", "deep", "nT", "nU", "lenE")
F = None


def work(job):
    split, ids = job
    nk = len(KS)
    S = np.zeros((len(ids), nk, len(STAT)), np.float64)
    S[:, :, 1:3] = -1
    ev_sc, ev_row, ev_cnt = [], [], np.zeros((len(ids), nk), np.int32)
    for i, f in enumerate(ids):
        if F.H[split][f] == 0:
            continue
        cache = {}
        for j, k in enumerate(KS):
            snap = pick_snapshot(F, split, f, k)
            if snap is None:
                continue
            key = (snap[0], snap[1])
            if key not in cache:
                cache[key] = extend_one(F, split, f, snap)
            st, tsc, trow = cache[key]
            S[i, j] = [1, st["sc"], st["fe"], st["gap"], st["A"], st["B"], st["n_anchor"], st["deep"], st["nT"], st["nU"], st["lenE"]]
            ev_sc.append(tsc)
            ev_row.append(trow)
            ev_cnt[i, j] = len(tsc)
    sc = np.concatenate(ev_sc) if ev_sc else np.zeros(0, np.int8)
    row = np.concatenate(ev_row) if ev_row else np.zeros(0, np.int64)
    return S, ev_cnt, sc, row


def main():
    global F
    t0 = time.time()
    F = Flat()
    from align_lib import lcs
    lcs(np.arange(5, dtype=np.int64), np.arange(5, dtype=np.int64))  # compile once in the parent (inherited by fork)
    os.makedirs(OUTD, exist_ok=True)
    ctx = mp.get_context("fork")
    for split in ARGS.splits.split(","):
        n = len(F.uid[split])
        ids = np.arange(n)
        if ARGS.limit:
            ids = np.random.default_rng(1117).choice(n, ARGS.limit, replace=False)
            ids.sort()
        # only feeds that can have an admissible earlier snapshot need work; others keep zeros
        jobs = [(split, ids[i:i + ARGS.chunk]) for i in range(0, len(ids), ARGS.chunk)]
        res = []
        with ctx.Pool(ARGS.workers) as pool:
            for r_i, r in enumerate(pool.imap(work, jobs)):
                res.append(r)
                if r_i % 100 == 0:
                    print(f"[{time.time() - t0:6.0f}s] {split}: chunk {r_i + 1}/{len(jobs)}", flush=True)
        S = np.concatenate([r[0] for r in res])
        cnt = np.concatenate([r[1] for r in res])
        sc = np.concatenate([r[2] for r in res])
        row = np.concatenate([r[3] for r in res])
        # event order in sc/row: feed-major, then k (as appended in work)
        suffix = f"_limit{ARGS.limit}" if ARGS.limit else ""
        np.savez(f"{OUTD}/align_{split}{suffix}.npz", fid=ids, S=S, cnt=cnt, ev_sc=sc, ev_row=row, KS=np.array(KS),
                 STAT=np.array(STAT))
        has = S[:, 0, 0] > 0
        print(f"[{time.time() - t0:6.0f}s] {split}: {len(ids)} feeds, with an admissible earlier snapshot (k=0) {has.mean():.4f}, "
              f"extended (k=0) {(cnt[:, 0] > 0).mean():.4f}, events stored {len(sc)}", flush=True)


if __name__ == "__main__":
    main()
