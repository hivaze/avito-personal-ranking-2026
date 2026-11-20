"""Second batch of derived features (own feed only; identical for val and test):
  d_cat_run        run length of the candidate's category from the newest history event (0 if the newest event differs)
  d_cat_run_e1     same, counting only event_type 1 events (non-type-1 events are skipped, not breaking the run)
  d_cat_pur20      share of the 20 newest events in the candidate's category (cat_cnt_20 / min(20, H))
  d_cat_pur5       share of the 5 newest events
  d_top_cat20      1 if the candidate's category is the modal category of the 20 newest events
  d_f3_resid       f3 minus the split-wide mean of f3 in the same (cat_cnt_20 bucket, category) cell (affinity not explained by
                   the visible recent history; unsupervised, labels not used)
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

OUT = Path("<repo>/eda/errors/missed_positives/derived")
NAMES = ["d_cat_run", "d_cat_run_e1", "d_cat_pur20", "d_cat_pur5", "d_top_cat20"]


@nb.njit(parallel=True, cache=True)
def kern(coff, M, hoff, H, ccat, hcat, het, hpos, out):
    for f in nb.prange(len(M)):
        o, m, ho, h = coff[f], M[f], hoff[f], H[f]
        # order history by hpos ascending (newest first)
        order = np.argsort(hpos[ho:ho + h])
        cats = hcat[ho:ho + h][order]; ets = het[ho:ho + h][order]
        c20 = np.zeros(128, np.int64); c5 = np.zeros(128, np.int64)
        n20 = min(20, h); n5 = min(5, h)
        for j in range(n20):
            c20[cats[j]] += 1
            if j < 5:
                c5[cats[j]] += 1
        mode = -1; best = 0
        for c in range(128):
            if c20[c] > best:
                best = c20[c]; mode = c
        for i in range(m):
            cc = ccat[o + i]
            run = 0
            for j in range(h):
                if cats[j] == cc:
                    run += 1
                else:
                    break
            run1 = 0
            for j in range(h):
                if ets[j] != 1:
                    continue
                if cats[j] == cc:
                    run1 += 1
                else:
                    break
            out[o + i, 0] = run; out[o + i, 1] = run1
            out[o + i, 2] = c20[cc] / n20 if n20 > 0 else np.nan
            out[o + i, 3] = c5[cc] / n5 if n5 > 0 else np.nan
            out[o + i, 4] = 1.0 if (cc == mode and best > 0) else 0.0


def run(split):
    f, M, off = feed_meta(split)
    H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
    d = FLAT / split
    L = lambda n: np.load(d / f"{n}.npy").astype(np.int64)
    out = np.zeros((int(M.sum()), len(NAMES)), np.float32)
    kern(off.astype(np.int64), M, hoff, H, L("cand_category_id"), L("hist_category_id"), L("hist_event_type"), L("hist_hpos"), out)
    for i, n in enumerate(NAMES):
        np.save(OUT / split / f"{n}.npy", out[:, i])
    # f3 residual given (cat_cnt_20 bucket, category)
    f3 = np.load(FEATS / split / "f3.npy").astype(np.float64)
    c20 = np.load(FEATS / split / "cat_cnt_20.npy").astype(np.float64)
    cat = np.load(d / "cand_category_id.npy").astype(np.int64)
    bk = np.digitize(c20, [0.5, 1.5, 2.5, 4.5, 8.5, 14.5])
    key = bk * 128 + cat
    s = np.bincount(key, weights=f3, minlength=7 * 128); n = np.bincount(key, minlength=7 * 128)
    np.save(OUT / split / "d_f3_resid.npy", (f3 - (s / np.maximum(n, 1))[key]).astype(np.float32))
    print(split, {nm: float(np.nanmean(out[:, i])) for i, nm in enumerate(NAMES)}, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["val", "test"]:
        run(s)
