"""Diagnostic variant for the hzcv gate (NOT a deliverable feature): capped lag.

Under the spec lag (hz_common), 43% of train feeds (L=4: 46%, L=8: 76%) have cutoff day <= Sep 6, i.e. an EMPTY dense
co-vis source (train data start on Sep 6), whereas every val/test row has the full 13-day dense source.
Capped lag (pre-declared): L_cap = largest l in {0, 4, 8} with l <= L_hash and day(c_l) >= Sep 9 (>= 3 dense days),
else 0. Holdout feeds keep L = 0. Only the 50K gate training feeds (review train_sample_feeds.npy) are rebuilt,
with the same unchanged kernel/Index as hzcv.py; offsets are recomputed with the capped cutoff.
Output: scratchpad/featrev/hzcv/lagcap_trs.npz (rows, changed, seq_cvlag_*, hzcv_*).
"""
import sys, time
from pathlib import Path
import numpy as np
import numba as nb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_common as hz  # noqa: E402
sys.path.insert(0, "<repo>/feats_v2/seq_")
from feat_covis import Index, run_queries, NAMES as CV_NAMES  # noqa: E402
from sources import source_fidx  # noqa: E402
SCRR = "/tmp/scratchpad"
sys.path.insert(0, SCRR + "/review/leakage")
from common_rows import rows_of, SCR  # noqa: E402

OUT = Path(SCRR) / "featrev/hzcv"
T0 = time.time()


def main():
    nb.set_num_threads(16)
    trs = np.load(SCR / "train_sample_feeds.npy")
    ts, c_spec, L = hz.cutoff_per_feed("train")
    c0 = c_spec + hz.DAY * L
    d0 = hz.train_days()[0]
    min_day = d0 + 3  # Sep 9
    Lc = np.zeros_like(L)
    for l in (4, 8):
        ok = (L >= l) & (hz.day_of(c0 - hz.DAY * l) >= min_day)
        Lc[ok] = l
    cut = c0 - hz.DAY * Lc
    rows, off, M = rows_of("train", trs)
    fr = np.repeat(trs, M)
    changed_f = Lc[trs] != L[trs]
    print("gate feeds", len(trs), "changed", int(changed_f.sum()), "L_cap dist", np.bincount(Lc[trs])[[0, 4, 8]],
          flush=True)
    res = {n: np.asarray(np.load(hz.FEATS_NEW / "train" / f"{n}.npy", mmap_mode="r")[rows], np.float32)
           for n in ("seq_cvlag_invd20", "seq_cvlag_pop", "seq_cvlag_norm")}
    N = len(np.load(hz.FLAT / "train/cand_fidx.npy", mmap_mode="r"))
    out = np.zeros((N, len(CV_NAMES)), np.float32)
    sp_fidx = np.nonzero(hz.sparse_per_feed("train"))[0].astype(np.int64)
    sp_ts = ts[sp_fidx]
    q = trs[changed_f]
    key = hz.day_of(cut[q])
    for k in np.unique(key):
        sel = np.sort(q[key == k]).astype(np.int64)
        ck = k * hz.DAY - hz.MSK
        idx = Index([("train", source_fidx(int(k)), False), ("train", sp_fidx[sp_ts >= ck], True)])
        run_queries(idx, "train", sel, out)
        print(f"[{time.time()-T0:.0f}s] key Sep {k - d0 + 6}: queries {len(sel)}", flush=True)
        del idx
    chg_rows = np.repeat(changed_f, M)
    for a, b in (("seq_cv_invd20", "seq_cvlag_invd20"), ("seq_cv_pop", "seq_cvlag_pop"), ("seq_cv_norm", "seq_cvlag_norm")):
        v = out[rows, CV_NAMES.index(a)]
        res[b] = np.where(chg_rows, v, res[b]).astype(np.float32)
    E = hz.expo_hours("train", cut, ts)[rows]
    res["hzcv_expo_slog"] = hz.slog_days(E)
    res["hzcv_censored"] = (E <= 0).astype(np.float32)
    res["hzcv_stale_d"] = hz.stale_d(ts, cut)[fr]
    # unchanged feeds must reproduce the deliverable files exactly
    for n in ("hzcv_expo_slog", "hzcv_censored", "hzcv_stale_d"):
        ref = np.asarray(np.load(hz.FEATS_NEW / "train" / f"{n}.npy", mmap_mode="r")[rows], np.float32)
        assert np.array_equal(res[n][~chg_rows], ref[~chg_rows]), n
    print("censored share capped", float(res["hzcv_censored"].mean()), "cvpop0 capped",
          float((res["seq_cvlag_pop"] == 0).mean()), flush=True)
    np.savez(OUT / "lagcap_trs.npz", rows=rows, changed=chg_rows, **res)


if __name__ == "__main__":
    main()
