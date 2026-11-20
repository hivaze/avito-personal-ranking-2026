"""Feature building for the residual model (shared by the analysis scripts)."""
from __future__ import annotations

import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "12")

sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import kit  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

RAW = ["ucos", "hmax", "hmax_idx", "hmax_type", "hmax_src", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95",
       "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch", "cat_views", "cat_actions",
       "cat_newest_idx", "cat_user_rank", "same_cat_newest", "cat_hist_share", "cand_maxcos", "n_cand_dup95",
       "cat_n_in_feed", "rank", "cat_rank_in_feed", "gap_next", "gap_top", "f2_home", "f1", "f2", "f3", "f4",
       "f5", "cat", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
       "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "M", "H", "guest",
       "uvec_missing", "hour", "weekday", "newest_type", "newest_src", "newest_cat", "n_hist_cats32",
       "n_feed_cats", "score", "s_ft", "s_to"]
ZCOLS = ["ucos", "hmax", "h1", "h5", "h32", "hmean5", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo",
         "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50",
         "seq_cv_invd20", "cand_maxcos", "c_contact", "c_vfeed", "c_vsearch"]


def derive(df: pl.DataFrame) -> pl.DataFrame:
    """Add within-feed z-scores / percentile ranks and a few interaction columns."""
    ex = []
    for c in ZCOLS:
        x = pl.col(c).cast(pl.Float64).fill_nan(None)
        mu = x.mean().over("fidx")
        sd = x.std().over("fidx")
        ex.append(((x - mu) / (sd + 1e-6)).fill_null(0.0).alias(f"z_{c}"))
        ex.append((x.rank("average").over("fidx") / pl.col("M")).fill_null(-1.0).alias(f"p_{c}"))
        ex.append(mu.fill_null(-9).alias(f"fm_{c}"))
    ex += [
        ((pl.col("rank") - 1) / (pl.col("M") - 1).clip(1)).alias("rank_pct"),
        (pl.col("s_ft") - pl.col("s_to")).alias("br_diff"),
        (pl.col("cat_n_in_feed") / pl.col("M")).alias("cat_feed_share"),
        (pl.col("cat_hist_share") - pl.col("cat_n_in_feed") / pl.col("M")).alias("cat_hist_minus_feed"),
        (pl.col("cat_rank_in_feed") / pl.col("cat_n_in_feed")).alias("cat_rank_pct"),
    ]
    return df.with_columns(ex)


def feature_names(df: pl.DataFrame) -> list[str]:
    der = [c for c in df.columns if c.startswith(("z_", "p_", "fm_"))] + [
        "rank_pct", "br_diff", "cat_feed_share", "cat_hist_minus_feed", "cat_rank_pct"]
    return RAW + der


def matrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in names]).to_numpy()


def softmax_scale(score: np.ndarray, y: np.ndarray, off: np.ndarray, sz: np.ndarray) -> float:
    """MLE of c in a per-feed softmax P(i) ~ exp(c * score_i) (positives as multinomial draws)."""
    fid = np.repeat(np.arange(len(sz)), sz)

    def nll(c):
        e = np.exp(c * score - np.maximum.reduceat(c * score, off)[fid])
        lz = np.log(np.add.reduceat(e, off))[fid] + np.maximum.reduceat(c * score, off)[fid]
        return -(y * (c * score - lz)).sum()
    cs = np.linspace(0.5, 4, 36)
    v = [nll(c) for c in cs]
    return float(cs[int(np.argmin(v))])


class FastNDCG:
    """Vectorised per-feed NDCG identical to kit.ndcg_per_feed (pessimistic ties) for a fixed feed layout."""

    def __init__(self, y: np.ndarray, off: np.ndarray, sz: np.ndarray):
        self.y = y.astype(np.float64)
        self.off, self.sz = off, sz
        self.fid = np.repeat(np.arange(len(sz)), sz)
        self.pos_base = np.repeat(off, sz)
        disc = 1.0 / np.log2(np.arange(2, sz.max() + 2))
        self.disc = disc
        npos = np.add.reduceat(self.y, off).astype(int)
        cs = np.r_[0, np.cumsum(disc)]
        self.idcg = cs[np.maximum(npos, 1)]
        self.nopos = npos == 0

    def __call__(self, s: np.ndarray) -> np.ndarray:
        order = np.lexsort((self.y, -s, self.fid))
        pos = np.arange(len(s)) - self.pos_base
        dcg = np.add.reduceat(self.y[order] * self.disc[pos], self.off)
        out = dcg / self.idcg
        out[self.nopos] = 1.0
        return out
