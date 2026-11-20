"""gap1117 / lifelong: shared helpers (label-free). The aligner is the reviewed refill aligner
(lab/nn_x/w12/seg/refill/align_lib.py: LCS on (item, type, source) + tie-swap repair; reviewed in
lab/reports/SEG_1115/refill_review.md: 0 provenance violations, placement error 0.4-0.5%), imported READ-ONLY
(sys.dont_write_bytecode, nothing is written into that folder).

Extension block of a feed (one snapshot only, as the task specifies):
  snapshot  = the most recent snapshot of the same user_id (never user 0) with a STRICTLY smaller timestamp that is
              at least k days older than the feed, from an allowed split
              (train rows: train; val rows: train + val, never test; test rows: train + val + test);
  alignment = refill align_lib.union_timeline(..., snaps=[that snapshot]);
  criterion A (aligned)       : LCS >= 16 anchors and an exact contiguous run >= 8 (align_lib.MIN_ANCH / MIN_RUN);
  criterion B (end-anchored)  : the deepest stored event that is an anchor has index >= H - 8 (= 248 when H = 256);
  T events                    : events of the snapshot placed after the stored oldest event (older than it), newest first,
                                at most 128 kept.
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
sys.path.insert(0, "<repo>/lab/nn_x/w12/seg/refill")
import numpy as np  # noqa: E402

import align_lib  # noqa: E402
from align_lib import SPL, Flat, union_timeline  # noqa: E402,F401

HERE = "<repo>/lab/nn_x/w12/gap1117/lifelong"
OUTD = f"{HERE}/out"
DATA_OUT = "<repo>/artifacts/data/feats_new/histext_1117"
KEEP = 128
B_TOL = 8
ALLOWED = {"train": (0,), "val": (0, 1), "test": (0, 1, 2)}  # split codes of admissible snapshots per target split
DAY = 86400.0
BOUND = align_lib.BOUND  # public-like / private-like test boundary


def pick_snapshot(F, split, f, k_days):
    """most recent admissible strictly earlier snapshot at least k_days older: (sc, fidx, gap_days) or None."""
    t = F.ts[split][f]
    for sc, fe in F.earlier(split, f):  # most recent first; [] for user 0; strictly earlier timestamps only
        if sc not in ALLOWED[split]:
            continue
        g = (t - F.ts[SPL[sc]][fe]) / DAY
        assert g > 0
        if g >= k_days:
            return sc, fe, g
    return None


def extend_one(F, split, f, snap):
    """align feed f of `split` with one snapshot; returns (stats dict, T_sc int8[n], T_row int64[n]) with n <= KEEP."""
    sc, fe, g = snap
    H = int(F.H[split][f])
    U, st = union_timeline(F, split, f, snaps=[(sc, fe)])
    A = st["n_merged"] >= 1
    B = A and st["deep_anchor"] >= H - B_TOL
    iL = st["iL"]
    old = U["lpos"] < 0
    tm = old.copy()
    tm[: iL + 1] = False
    tsc, trow = U["sc"][tm], U["row"][tm]
    assert (tsc == sc).all()
    out = dict(sc=sc, fe=fe, gap=g, A=A, B=B, n_anchor=st["first_anchor"] if A else 0, deep=st["deep_anchor"],
               nT=int(st["n_T"]), nU=int(st["n_U"]), lenE=int(F.H[SPL[sc]][fe]))
    if not (A and B):
        return out, np.zeros(0, np.int8), np.zeros(0, np.int64)
    return out, tsc[:KEEP].astype(np.int8), trow[:KEEP].astype(np.int64)
