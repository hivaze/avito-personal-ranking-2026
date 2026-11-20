"""Nested (honest) greedy forward selection of a sparse linear residual, NDCG-direct.

Inside each outer fold k: on the 4 training folds pick the (feature, weight) with the best mean NDCG, add it to the
score, repeat up to STEPS terms; evaluate each step on fold k. This is the cross-fitted value of "scan everything,
keep the best few terms" -- i.e. of the distillation procedure itself, with all its multiplicity inside the fit.
"""
import numpy as np
import polars as pl

import prep
from prep import kit

STEPS = 3
df = prep.derive(kit.load())
y = df["label"].to_numpy()
s0 = df["score"].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
F = prep.FastNDCG(y, off, sz)
fold = kit.folds(df)
ffold = fold[off]
const = {"M", "H", "guest", "uvec_missing", "hour", "weekday", "newest_type", "newest_src", "newest_cat",
         "n_hist_cats32", "n_feed_cats", "score", "s_to", "fidx", "rank", "gap_top", "gap_next", "rank_pct"}
names = [n for n in prep.feature_names(df) if n not in const and not n.startswith("fm_")]
Xs = []
for n in names:
    x = df[n].cast(pl.Float64).fill_nan(None).to_numpy()
    nan = ~np.isfinite(x)
    x = np.where(nan, np.nanmedian(x), x)
    Xs.append(((x - x.mean()) / (x.std() + 1e-12)).astype(np.float32))
W = np.array([-0.1, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05, 0.1])
base_nd = F(s0)
cache = {}


def scan(cur_key, cur):
    """Per-feed NDCG change for every (feature, weight) on top of the current score; cached by selection path."""
    if cur_key not in cache:
        b = F(cur)
        cache[cur_key] = np.stack([np.stack([F(cur + w * x) - b for w in W]) for x in Xs]), b
    return cache[cur_key]


res = np.zeros((STEPS, len(sz)))
paths = []
for k in range(5):
    trm = ffold != k
    cur, key, path = s0.copy(), (), []
    for st in range(STEPS):
        nd, b = scan(key, cur)
        m = nd[:, :, trm].mean(2)
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[i, j] <= 0:
            res[st, ~trm] = (F(cur) - base_nd)[~trm]
            path.append(None)
            continue
        cur = cur + W[j] * Xs[i]
        key = key + ((i, j),)
        path.append((names[i], float(W[j]), float(m[i, j])))
        res[st, ~trm] = (F(cur) - base_nd)[~trm]
    paths.append(path)
    print(k, path, flush=True)
with open("cv_log.txt", "a") as fh:
    for st in range(STEPS):
        d = res[st]
        line = (f"nested_fs step={st + 1}: delta={d.mean():+.5f} se={d.std(ddof=1) / np.sqrt(len(d)):.5f} "
                f"z={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):.2f}")
        print(line)
        fh.write(line + "\n")
