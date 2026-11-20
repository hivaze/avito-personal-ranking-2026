"""featrev research: censoring-aware helpers for the causal collaborative inputs + a clip-safe prew rescale.

Writes float32 arrays aligned with data/flat/<split>/cand_*.npy to data/feats_new/<split>/:

  src_expo_h    hours the candidate listing existed before the cutoff of the causal co-visitation source (seq_cv_*):
                E = cutoff - t_post, t_post = t_feed - age(f4) (feats_v2/life_ knots: f4 is the exact listing age).
                cutoff: train rows -> start (00:00 MSK) of the latest full train day <= own day, i.e. exactly the
                seq_cv_* expanding-window convention (feats_v2/seq_/sources.py feed_day_index);
                val/test rows -> 2026-09-20 00:00 MSK (end of train, the seq_cv_* all-train source).
                f4 at its cap (>= 63/64, age >= 4.6 y) -> E = +1e5 h. NaN never occurs.
                NN scale: sign(E) * log1p(|E| / 24)  (days, signed log) -> written as src_expo_slog.
  src_censored  1.0 if E <= 0 (listing created after the source cutoff: co-vis / popularity evidence structurally
                absent), else 0.0.
  prew_fe_c     (prew_fe - 15) / 2, NaN -> -10   (clip-safe version of prew_fe for nn/data.py attach_xt, whose
  prew_new_c    (prew_new - 14) / 2, NaN -> -10   generic clip(+-10) saturates 79-87% of warm rows of the raw columns;
                                                  NaN keeps the bottom position it has today via NaN -> 0 < 10).
No labels, no other feeds, no same-user information are read. Identical code path for all splits.
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/research_expo.py [splits]
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
OUT = ROOT / "data/feats_new"
MSK = 3 * 3600
CUT_VT = 1789851600  # 2026-09-20 00:00 MSK
K = 64
CAP = 63 / 64


def age_hours(x, knots):
    """Piecewise-linear f4 -> listing age in hours (copy of feats_v2/life_/post_time.py::age_hours)."""
    x = x.astype(np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


def cutoff_per_feed(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    ts = f["timestamp"].to_numpy().astype(np.int64)
    if split != "train":
        return ts, np.full(len(ts), CUT_VT, np.int64)
    d = (ts + MSK) // 86400
    u, c = np.unique(d, return_counts=True)
    days = u[c > 1000]  # same rule as feats_v2/seq_/sources.py::train_days
    idx = np.searchsorted(days, d, side="right") - 1
    assert idx.min() >= 0
    return ts, days[idx] * 86400 - MSK


def main():
    splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["train", "val", "test"]
    knots = np.load(FEATS / "life_f4_age_knots.npy")
    for split in splits:
        (OUT / split).mkdir(parents=True, exist_ok=True)
        ts, cut = cutoff_per_feed(split)
        fi = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
        f4 = np.load(FLAT / split / "cand_f4.npy", mmap_mode="r")
        n = len(fi)
        E = np.empty(n, np.float32)
        step = 20_000_000
        for s in range(0, n, step):
            ff = np.asarray(fi[s:s + step])
            a = age_hours(np.asarray(f4[s:s + step]), knots)
            e = (cut[ff] - ts[ff]) / 3600.0 + a  # = (cutoff - t_post) in hours
            E[s:s + step] = np.where(np.isnan(e), 1e5, e)
        np.save(OUT / split / "src_expo_h.npy", E)
        np.save(OUT / split / "src_censored.npy", (E <= 0).astype(np.float32))
        np.save(OUT / split / "src_expo_slog.npy",
                (np.sign(E) * np.log1p(np.abs(E) / 24.0)).astype(np.float32))
        for name, c in (("prew_fe", 15.0), ("prew_new", 14.0)):
            x = np.load(FEATS / split / f"{name}.npy", mmap_mode="r")
            out = np.empty(len(x), np.float32)
            for s in range(0, len(x), step):
                xx = np.asarray(x[s:s + step], np.float32)
                out[s:s + step] = np.where(np.isnan(xx), -10.0, (xx - c) / 2.0)
            np.save(OUT / split / f"{name}_c.npy", out)
        print(split, n, "censored share %.4f" % float((E <= 0).mean()), flush=True)


if __name__ == "__main__":
    main()
