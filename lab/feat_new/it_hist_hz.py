"""it_hist_hz_*: age-gated, density-matched leave-user-later-out (LUO) item popularity for TRAIN rows.
Val/test values are byte-identical copies of the as-built it_hist_*_luo (src/features_luo.py).

Files (float32, aligned with data/flat/<split>/cand_*.npy rows), written to data/feats_new/<split>/:
  it_hist_hz_all, it_hist_hz_et1 .. it_hist_hz_et4   (event type all / 1..4, as it_hist_*_luo)

Rule for a TRAIN row (item j, feed time t, user u):
  t_post = t - age(f4)                    (exact listing creation time, feats_v2/life_ knots, feat_life_ sec 1)
  c      = day_cut(t) - L * 86400         (day_cut = 00:00 MSK of the latest full train day <= own day, i.e. the
                                            seq_cv_* / feat_new/research_expo.py convention; L in {0, 4, 8} days =
                                            the SHARED hz_common.lag_raw(feed_id) hash, one third each; the 30K
                                            nn/train.py holdout feeds get L = 0; all from feat_new/hz_common.py)
  E      = (c - t_post) in hours          (identical formula to feat_new/research_expo.py src_expo_h; capped f4 = old)
  - E > 0 (listing existed before the cutoff, incl. capped f4):   it_hist_hz_X = it_hist_X_luo (as built)
  - E <= 0 (listing created at or after the cutoff):               it_hist_hz_X = # occurrences of j (event type X)
        in the histories of snapshots S in  {all val/test feeds} U {train feeds with s(S) = 1},
        s(S) = 1  <=>  hz_common.sparse_flag(feed_id)   (1/37 hash sample = exploration sampling density),
        excluding snapshots S of the same NON-ZERO user with ts(S) > t (as-built LUO rule; user 0 keeps the as-built
        handling: nothing removed for user-0 rows, user-0 snapshots counted like any other user's).
Rule for VAL/TEST rows: it_hist_X_luo as built (= the same rule with c = 2026-09-20 00:00 MSK, the end of dense data:
  listings created after it can only be counted in val/test snapshots, which are already a ~1/37 sample).

No labels are read. Same-user later snapshots are never counted for u != 0. The thinning uses only feed_id hashes; the
gate uses only the row's own f4, timestamp and the fixed cutoff rule. Every counted occurrence is a subset of the
already-accepted it_hist_*_luo occurrences (checked below).

Self-checks (printed):
  V1 val/test young rows (c = Sep 20): recount with this code == as-built luo (up to t_post boundary noise).
  V2 train young rows: hz <= as-built luo for every row and event type.
  V3 a 1M-row sample of train young rows recounted WITHOUT thinning (all train snapshots) == as-built luo exactly.

Usage: CUDA_VISIBLE_DEVICES="" python feat_new/it_hist_hz.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "16")
import sys, time
from pathlib import Path
import numpy as np
import polars as pl
import numba as nb

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
OUT = ROOT / "data/feats_new"
MSK = 3 * 3600
CUT_VT = 1789851600  # 2026-09-20 00:00 MSK
K = 64
CAP = 63 / 64
RHO = 37
LAGS = np.array([0, 4, 8], np.int64)
NAMES = ["all", "et1", "et2", "et3", "et4"]
NI = 111_400_000


# Shared horizon-matching definitions (feat_new/hz_common.py, also used by hzcv / ixh): per-feed lag L, the 1/37
# sparse train-feed sample s(S), the train cutoff and the 30K holdout (L = 0). Imported, not copied, so all hz*
# families use the same L and the same sparse sample for every feed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_common as hz  # noqa: E402


def sample_s(feed_id):
    """s(S) = 1 for the 1/37 hash sample of train feeds (hz_common.sparse_flag)."""
    return hz.sparse_flag(feed_id)


def holdout_feeds():
    """Exactly nn/train.py --holdout 30000 (hz_common.holdout_feeds)."""
    return hz.holdout_feeds()


def lag_days(feed_id, fidx_hold=None):
    """Per-feed lag in days (hz_common.lag_raw), holdout feeds forced to 0."""
    L = hz.lag_raw(feed_id).copy()
    if fidx_hold is not None:
        L[fidx_hold] = 0
    return L


def cutoff_per_feed(split):
    """(ts, cutoff c) per feed (hz_common.cutoff_per_feed, lag=True). Train: day_cut(t) - L days; val/test: Sep 20."""
    ts, cut, _ = hz.cutoff_per_feed(split, lag=True)
    return ts, cut


def age_hours(x, knots):
    """Piecewise-linear f4 -> listing age in hours (copy of feats_v2/life_/post_time.py::age_hours)."""
    x = x.astype(np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


def young_rows(split, knots, step=20_000_000):
    """Rows with E = c - t_post <= 0 (listing created at/after the cutoff). Capped f4 -> old."""
    ts, cut = cutoff_per_feed(split)
    fi = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
    f4 = np.load(FLAT / split / "cand_f4.npy", mmap_mode="r")
    out = []
    for s in range(0, len(fi), step):
        ff = np.asarray(fi[s:s + step])
        a = age_hours(np.asarray(f4[s:s + step]), knots)
        e = (cut[ff] - ts[ff]) / 3600.0 + a
        out.append(np.flatnonzero(e <= 0) + s)  # NaN (capped) compares False -> old
    return np.concatenate(out).astype(np.int64)


def gather_hist(split, fids, need):
    """History entries (item, snapshot ts, snapshot user, event type) of the given feeds, restricted to need[item]."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    ts = f["timestamp"].to_numpy().astype(np.int64); us = f["user_id"].to_numpy().astype(np.int64)
    ho = f["hist_off"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    hi = np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r")
    he = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    out = ([], [], [], [])
    CH = 100_000
    for s in range(0, len(fids), CH):
        fb = np.asarray(fids[s:s + CH], np.int64)
        Hb = H[fb]
        rows = np.repeat(ho[fb] - np.concatenate([[0], np.cumsum(Hb)[:-1]]), Hb) + np.arange(Hb.sum())
        ff = np.repeat(fb, Hb)
        a = np.asarray(hi[rows])
        k = need[a]
        out[0].append(a[k].astype(np.int32)); out[1].append(ts[ff[k]]); out[2].append(us[ff[k]])
        out[3].append(np.asarray(he[rows[k]]).astype(np.int8))
    return tuple(np.concatenate(o) for o in out)


def build_index(parts):
    it = np.concatenate([p[0] for p in parts]); t = np.concatenate([p[1] for p in parts])
    u = np.concatenate([p[2] for p in parts]); e = np.concatenate([p[3] for p in parts])
    o = np.argsort(it, kind="stable")
    it, t, u, e = it[o], t[o], u[o], e[o]
    uit, st, cn = np.unique(it, return_index=True, return_counts=True)
    return uit.astype(np.int32), st.astype(np.int64), cn.astype(np.int64), t, u, e


@nb.njit(parallel=True, cache=True)
def count_rows(citem, cts, cuser, uit, ust, ucn, ht, hu, he, out):
    """out[r, 0] = all event types, out[r, k] = event type k (1..4); same non-zero user with ts(S) > t removed."""
    for r in nb.prange(len(citem)):
        for c in range(5):
            out[r, c] = 0
        k = np.searchsorted(uit, citem[r])
        if k >= len(uit) or uit[k] != citem[r]:
            continue
        u = cuser[r]; t = cts[r]
        for i in range(ust[k], ust[k] + ucn[k]):
            if u != 0 and hu[i] == u and ht[i] > t:
                continue
            out[r, 0] += 1
            e = he[i]
            if 1 <= e <= 4:
                out[r, e] += 1


def cand_cols(split, rows):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    ts = f["timestamp"].to_numpy().astype(np.int64); us = f["user_id"].to_numpy().astype(np.int64)
    fi = np.asarray(np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")[rows]).astype(np.int64)
    item = np.asarray(np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")[rows]).astype(np.int32)
    return item, ts[fi], us[fi]


def luo_rows(split, rows):
    return np.stack([np.asarray(np.load(FEATS / split / f"it_hist_{n}_luo.npy", mmap_mode="r")[rows], np.float64)
                     for n in NAMES], 1)


def main():
    T0 = time.time()
    lg = lambda *a: print(*a, f"[{time.time()-T0:.0f}s]", flush=True)
    knots = np.load(FEATS / "life_f4_age_knots.npy")
    ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
    fid_tr = ftr["feed_id"].to_numpy()
    s1 = np.flatnonzero(sample_s(fid_tr))
    L = lag_days(fid_tr, holdout_feeds())
    lg(f"train feeds {len(fid_tr)}, s=1 feeds {len(s1)} (1/{len(fid_tr)/len(s1):.2f}); "
       f"L share 0/4/8 = {[round(float((L == v).mean()), 4) for v in LAGS]}")
    yr = {sp: young_rows(sp, knots) for sp in ("train", "val", "test")}
    ncand = {sp: len(np.load(FLAT / sp / "cand_fidx.npy", mmap_mode="r")) for sp in yr}
    lg("young rows (E <= 0):", {sp: (len(v), round(len(v) / ncand[sp], 4)) for sp, v in yr.items()})
    cand = {sp: cand_cols(sp, yr[sp]) for sp in yr}
    need = np.zeros(NI, np.bool_)
    for sp in yr:
        need[cand[sp][0]] = True
    lg("needed items", int(need.sum()))
    parts = [gather_hist("train", s1, need)]
    for sp in ("val", "test"):
        parts.append(gather_hist(sp, np.arange(len(pl.read_parquet(FLAT / sp / "feeds.parquet"))), need))
    lg("occurrences: train s=1", len(parts[0][0]), "val", len(parts[1][0]), "test", len(parts[2][0]))
    idx = build_index(parts)
    del parts
    lg("index items", len(idx[0]))
    cnt = {}
    for sp in yr:
        o = np.zeros((len(yr[sp]), 5), np.int64)
        count_rows(*cand[sp], *idx, o)
        cnt[sp] = o
    lg("counted")
    # V1: val/test young rows -> must reproduce the as-built LUO (only val/test snapshots can contain them)
    for sp in ("val", "test"):
        ref = luo_rows(sp, yr[sp])
        d = ref - cnt[sp]
        lg(f"V1 {sp}: young rows {len(d)}, exact match share per type {[round(float((d[:, i] == 0).mean()), 5) for i in range(5)]}, "
           f"rows with any diff {int((d != 0).any(1).sum())}, max |diff| {float(np.abs(d).max()) if len(d) else 0}")
    # V2: train young rows -> subset of as-built LUO occurrences
    ref_tr = luo_rows("train", yr["train"])
    d = ref_tr - cnt["train"]
    lg(f"V2 train: young rows {len(d)}, min(luo - hz) per type {d.min(0).tolist()}, "
       f"share hz < luo (all) {float((d[:, 0] > 0).mean()):.4f}, zero share luo {float((ref_tr[:, 0] == 0).mean()):.4f} "
       f"-> hz {float((cnt['train'][:, 0] == 0).mean()):.4f}")
    assert d.min() >= 0, "hz count exceeds as-built luo count (not a subset)"
    del ref_tr
    # V3: unthinned recount (all train snapshots + val/test) on a 1M-row sample == as-built LUO exactly
    rng = np.random.default_rng(0)
    smp = np.sort(rng.choice(len(yr["train"]), min(1_000_000, len(yr["train"])), replace=False))
    need3 = np.zeros(NI, np.bool_); need3[cand["train"][0][smp]] = True
    p3 = [gather_hist("train", np.arange(len(fid_tr)), need3)]
    for sp in ("val", "test"):
        p3.append(gather_hist(sp, np.arange(len(pl.read_parquet(FLAT / sp / "feeds.parquet"))), need3))
    idx3 = build_index(p3); del p3
    o3 = np.zeros((len(smp), 5), np.int64)
    count_rows(cand["train"][0][smp], cand["train"][1][smp], cand["train"][2][smp], *idx3, o3)
    d3 = luo_rows("train", yr["train"][smp]) - o3
    lg(f"V3 train sample {len(smp)} unthinned recount: exact match share per type "
       f"{[round(float((d3[:, i] == 0).mean()), 6) for i in range(5)]}, max |diff| {float(np.abs(d3).max())}")
    del idx3, o3
    # write
    for sp in ("train", "val", "test"):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(NAMES):
            x = np.load(FEATS / sp / f"it_hist_{n}_luo.npy").astype(np.float32)
            if sp == "train":
                x[yr[sp]] = cnt[sp][:, i].astype(np.float32)
            np.save(OUT / sp / f"it_hist_hz_{n}.npy", x)
            del x
        lg("wrote", sp)
    sc = Path("/tmp/scratchpad/featrev/it_hist_hz")
    sc.mkdir(parents=True, exist_ok=True)
    np.save(sc / "young_rows_train.npy", yr["train"]); np.save(sc / "L_train_feeds.npy", L.astype(np.int8))
    lg("done")


if __name__ == "__main__":
    main()
