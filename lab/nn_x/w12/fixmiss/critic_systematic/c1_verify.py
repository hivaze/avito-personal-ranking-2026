"""Critic check 1 (CPU): independent re-check of the systematic stream's base numbers.
- own NDCG (pessimistic ties, per-feed loop) on the stored best file vs src/metrics
- alignment of the stream's val_table (score / label / row order) with preds + flat labels
- user halves: user-disjoint? sizes
- per-feed spread of the final score (is it a z-score => per-feed temperature partly a z artefact?)
"""
import sys
import numpy as np
import polars as pl
import numba as nb

ROOT = "<repo>"
sys.path.insert(0, ROOT + "/src")
from metrics import ndcg_per_feed  # noqa: E402

F = pl.read_parquet(ROOT + "/data/flat/val/feeds.parquet")
M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
assert (off == np.concatenate([[0], np.cumsum(M)[:-1]])).all(), "offsets must be contiguous"
y = np.load(ROOT + "/data/flat/val/cand_target.npy").astype(np.float64)
ft = np.load(ROOT + "/preds/ens_FD16P20_m04_02kd_ft_val.npy").astype(np.float64)
to = np.load(ROOT + "/preds/ens_FD16P20_m04_02kd_to_val.npy").astype(np.float64)
s = 0.5 * ft + 0.5 * to


@nb.njit(cache=True)
def my_ndcg(s, y, off, M):
    out = np.ones(len(M))
    for f in range(len(M)):
        o = off[f]; m = M[f]
        npos = 0
        for i in range(o, o + m):
            if y[i] > 0:
                npos += 1
        if npos == 0:
            continue
        dcg = 0.0
        for i in range(o, o + m):
            if y[i] > 0:
                # pessimistic: rank = #items with score > s_i  + #negatives with score == s_i  (+ positives tied
                # are placed after negatives; among tied positives order does not matter for binary gain)
                r = 0
                rp = 0
                for j in range(o, o + m):
                    if s[j] > s[i]:
                        r += 1
                    elif s[j] == s[i] and j != i:
                        if y[j] == 0:
                            r += 1
                        elif j < i:
                            rp += 1
                dcg += 1.0 / np.log2(r + rp + 2.0)
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


a = my_ndcg(s, y, off, M); b = ndcg_per_feed(s, y, off, M); bf = ndcg_per_feed(s.astype(np.float32), y, off, M)
print(f"own NDCG {a.mean():.6f}  src/metrics f64 {b.mean():.6f}  f32 {bf.mean():.6f}  max|own-src| {np.abs(a - b).max():.2e}")
fid = np.repeat(np.arange(len(M)), M)
# ties inside feeds
o = np.lexsort((-s, fid)); ss = s[o]; ff = fid[o]
ties = ((ss[1:] == ss[:-1]) & (ff[1:] == ff[:-1])).sum()
o32 = np.lexsort((-s.astype(np.float32), fid)); s32 = s.astype(np.float32)[o32]
ties32 = ((s32[1:] == s32[:-1]) & (fid[o32][1:] == fid[o32][:-1])).sum()
print(f"adjacent within-feed ties: f64 {ties}  f32 {ties32}")

D = pl.read_parquet(ROOT + "/lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet",
                    columns=["row", "fidx", "score", "s_ft", "s_to", "label", "user_id", "M"])
print("table aligned: row==arange", bool((D["row"].to_numpy() == np.arange(len(s))).all()),
      " fidx==fid", bool((D["fidx"].to_numpy() == fid).all()),
      " max|score-s|", float(np.abs(D["score"].to_numpy() - s).max()),
      " label==y", bool((D["label"].to_numpy() == y).all()),
      " M ok", bool((D["M"].to_numpy() == M[fid]).all()))
half = np.load(ROOT + "/lab/nn_x/w12/fixmiss/systematic/data/half.npy")
u = F["user_id"].to_numpy()
du = pl.DataFrame({"u": u, "h": half}).filter(pl.col("u") != 0).group_by("u").agg(pl.col("h").n_unique().alias("k"))
print(f"halves: A {int((half == 0).sum())} B {int((half == 1).sum())}; users in both halves: {int((du['k'] > 1).sum())} of {du.height}; "
      f"guest feeds {int((u == 0).sum())}; users with >1 val feed {int(pl.DataFrame({'u': u}).filter(pl.col('u') != 0).group_by('u').len().filter(pl.col('len') > 1).height)}")
# per-feed spread of the final score and of each branch
for name, x in [("final", s), ("ft", ft), ("to", to)]:
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(M)) / M - mu ** 2, 0))
    print(f"{name}: per-feed sd mean {sd.mean():.4f} sd-of-sd {sd.std():.4f} corr(sd, logM) {np.corrcoef(sd, np.log(M))[0, 1]:+.3f}")
