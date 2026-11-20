"""hzcv: horizon-matched co-visitation for train rows (featrev F1 + F2 offsets).

Files (float32, aligned with data/flat/<split>/cand_*.npy rows) in data/feats_new/<split>/:
  seq_cvlag_invd20, seq_cvlag_pop, seq_cvlag_norm
      TRAIN row (feed of day d, ts t, user u, lag L, cutoff c = hz_common.cutoff_per_feed): the UNCHANGED
      feats_v2/seq_/feat_covis.py _covis kernel (CAP 3000, +-20 window, 0.95^hpos weight, same non-zero user's
      sequences skipped, constrained-part sequences with ts >= t skipped) over
        Index([("train", source_fidx(day(c)), False),             # dense: latest train feed of each non-zero user
                                                                  #  with day < day(c) + every user-0 feed day < day(c)
               ("train", {train feeds: s = 1 and ts >= c}, True)])  # sparse 1/37 window, ts < t enforced in kernel
      One Index per distinct cutoff day; every cutoff day <= the first train day (Sep 6) has an empty dense part and
      the same sparse part, so those share one Index. Each train feed is queried once, with its own L.
      VAL/TEST rows: exact copies of data/feats/<split>/seq_cv_{invd20,pop,norm} (all-train dense source before
      2026-09-20 + every val/test feed with ts < t as the sparse part: the same structure).
  hzcv_expo_slog  sign(E)*log1p(|E|/24), E = (c - t_post)/3600 h, 1e5 for capped f4 (val/test == src_expo_slog)
  hzcv_censored   1[E <= 0]                                                        (val/test == src_censored)
  hzcv_stale_d    clip((t - c)/86400, 0, 9)/10, feed level
  train/hz_lag    L in days (row-aligned, train only)
No target / pos is read anywhere. No other split's labels. Same-user sequences skipped (any time), and every
source sequence is strictly earlier than the feed (dense: day < day(c) <= own day; sparse: ts < t).
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/hzcv.py {vt|train|check} [threads<=16]
"""
import sys, time
from pathlib import Path
import numpy as np
import numba as nb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_common as hz  # noqa: E402

sys.path.insert(0, "<repo>/feats_v2/seq_")
from feat_covis import Index, run_queries, NAMES as CV_NAMES  # noqa: E402  (unchanged kernel + index)
from sources import source_fidx  # noqa: E402

T0 = time.time()
KEEP = {"seq_cv_invd20": "seq_cvlag_invd20", "seq_cv_pop": "seq_cvlag_pop", "seq_cv_norm": "seq_cvlag_norm"}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def offsets(split):
    ts, cut, L = hz.cutoff_per_feed(split)
    E = hz.expo_hours(split, cut, ts)
    fi = np.load(hz.FLAT / split / "cand_fidx.npy", mmap_mode="r")
    sd = hz.stale_d(ts, cut)
    out = {"hzcv_expo_slog": hz.slog_days(E), "hzcv_censored": (E <= 0).astype(np.float32),
           "hzcv_stale_d": sd[np.asarray(fi)]}
    if split == "train":
        out["hz_lag"] = L.astype(np.float32)[np.asarray(fi)]
    return out, E


def run_vt(split):
    for a, b in KEEP.items():
        x = np.load(hz.FEATS / split / f"{a}.npy")
        hz.save_new(split, b, x)
    out, E = offsets(split)
    ref_h = np.load(hz.FEATS_NEW / split / "src_expo_h.npy", mmap_mode="r")
    ref_s = np.load(hz.FEATS_NEW / split / "src_expo_slog.npy", mmap_mode="r")
    ref_c = np.load(hz.FEATS_NEW / split / "src_censored.npy", mmap_mode="r")
    assert np.array_equal(E, ref_h), "expo hours differ from src_expo_h"
    assert np.array_equal(out["hzcv_expo_slog"], ref_s), "slog differs from src_expo_slog"
    assert np.array_equal(out["hzcv_censored"], ref_c), "censored differs from src_censored"
    for n, x in out.items():
        hz.save_new(split, n, x)
    log(split, "censored %.4f stale_d range %.3f-%.3f" % (out["hzcv_censored"].mean(), out["hzcv_stale_d"].min(),
                                                          out["hzcv_stale_d"].max()))


def groups_train():
    """Map every train feed to its index key = max(cutoff day, first train day)."""
    f = hz.feeds("train")
    ts, cut, L = hz.cutoff_per_feed("train")
    cday = hz.day_of(cut)
    assert np.all(cut == cday * hz.DAY - hz.MSK)
    d0 = hz.train_days()[0]
    key = np.maximum(cday, d0)
    return f, ts, cut, L, key, d0


def build_index(key, d0, sp_fidx, sp_ts):
    c_key = key * hz.DAY - hz.MSK
    parts = []
    if key > d0:
        parts.append(("train", source_fidx(int(key)), False))
    sp = sp_fidx[sp_ts >= c_key]
    parts.append(("train", sp, True))
    return Index(parts), (len(parts[0][1]) if key > d0 else 0), len(sp)


def run_train(nthr):
    f, ts, cut, L, key, d0 = groups_train()
    N = len(np.load(hz.FLAT / "train/cand_fidx.npy", mmap_mode="r"))
    sp_mask = hz.sparse_per_feed("train")
    sp_fidx = np.nonzero(sp_mask)[0].astype(np.int64)
    sp_ts = ts[sp_fidx]
    out = np.zeros((N, len(CV_NAMES)), np.float32)
    done = np.zeros(len(f), np.bool_)
    for k in np.unique(key):
        sel = np.nonzero(key == k)[0].astype(np.int64)
        idx, nd, ns = build_index(int(k), d0, sp_fidx, sp_ts)
        log(f"key day {k - d0 + 6:3d} (Sep): dense {nd} seqs, sparse {ns} seqs; queries {len(sel)} "
            f"(L counts {np.bincount(L[sel], minlength=9)[[0, 4, 8]].tolist()})")
        run_queries(idx, "train", sel, out)
        done[sel] = True
        del idx
    assert done.all()
    for a, b in KEEP.items():
        hz.save_new("train", b, out[:, CV_NAMES.index(a)])
    del out
    o, _ = offsets("train")
    for n, x in o.items():
        hz.save_new("train", n, x)
    log("train done; censored share by L:",
        {int(l): float(o["hzcv_censored"][o["hz_lag"] == l].mean()) for l in (0, 4, 8)})


def check(nthr):
    """Pipeline check: with L = 0 and NO sparse part, the same Index/run_queries call must reproduce the as-built
    data/feats/train/seq_cv_* for a sample of train feeds (2 days)."""
    f = hz.feeds("train")
    ts = f["timestamp"].to_numpy().astype(np.int64)
    d = hz.day_of(ts)
    days = hz.train_days()
    N = len(np.load(hz.FLAT / "train/cand_fidx.npy", mmap_mode="r"))
    out = np.zeros((N, len(CV_NAMES)), np.float32)
    rng = np.random.default_rng(0)
    rows_all = []
    off = f["cand_off"].to_numpy(); M = f["M"].to_numpy()
    for day in (days[4], days[10]):
        sel = np.sort(rng.choice(np.nonzero(d == day)[0], 1500, replace=False)).astype(np.int64)
        idx = Index([("train", source_fidx(int(day)), False)])
        run_queries(idx, "train", sel, out)
        rows_all.append(np.concatenate([np.arange(off[i], off[i] + M[i]) for i in sel]))
        del idx
    rows = np.concatenate(rows_all)
    for a in KEEP:
        ref = np.load(hz.FEATS / "train" / f"{a}.npy", mmap_mode="r")[rows]
        x = out[rows, CV_NAMES.index(a)]
        log("check", a, "max|diff|", float(np.max(np.abs(x - ref))), "rows", len(rows), "nonzero", float((ref != 0).mean()))


if __name__ == "__main__":
    mode = sys.argv[1]
    nthr = min(int(sys.argv[2]) if len(sys.argv) > 2 else 16, 16)
    nb.set_num_threads(nthr)
    if mode == "vt":
        for s in ("val", "test"):
            run_vt(s)
    elif mode == "train":
        run_train(nthr)
    elif mode == "check":
        check(nthr)
