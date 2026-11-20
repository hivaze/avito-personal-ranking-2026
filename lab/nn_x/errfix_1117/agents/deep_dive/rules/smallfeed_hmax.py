"""Rule smallfeed_hmax (deep_dive agent, 2026-11-17). WEAK CANDIDATE - probably selection noise, see notes.txt.

In small feeds (M < 40 candidates) add 0.05 x the within-feed z-score of hmax (the candidate's max cosine to any of
the user's 256 history events). Elsewhere the score is unchanged.

Discovery, cross-fitted over kit.folds (w chosen on 4 folds, evaluated on the 5th): +0.00074, SE 0.00031, z 2.39;
w = 0.05 chosen in all 5 folds; per-fold OOF deltas -0.00007 / +0.00135 / +0.00076 / +0.00096 / +0.00076.
Selected as the best of 144 segment x feature pairs (12 behavioural segments x 12 features, 8 weights each),
so the nominal z is not selection-adjusted. The weight curve is peaked (w=0.035: +0.00044, w=0.075: +0.00025).
Uses only hmax, M, score and fidx; no s_ft, so it runs unchanged on the train-period holdout.
"""
import numpy as np
import polars as pl

W = 0.05          # weight on the within-feed z-score of hmax
M_MAX = 40        # applies to feeds with fewer than this many candidates
HMAX_FILL = 0.80  # constant used where hmax is NaN (no history event with a vector); ~ discovery mean


def adjust(df: pl.DataFrame) -> np.ndarray:
    h = pl.col("hmax").cast(pl.Float64)
    h = pl.when(h.is_nan() | h.is_null()).then(pl.lit(HMAX_FILL)).otherwise(h)
    d = df.select(pl.col("fidx"), pl.col("M"), pl.col("score").cast(pl.Float64).alias("s"), h.alias("h"))
    d = d.with_columns(pl.col("h").mean().over("fidx").alias("hm"),
                       ((pl.col("h") * pl.col("h")).mean().over("fidx") - pl.col("h").mean().over("fidx") ** 2).alias("hv"))
    z = ((d["h"] - d["hm"]) / np.sqrt(np.maximum(d["hv"].to_numpy(), 1e-6))).to_numpy()
    small = (d["M"].to_numpy() < M_MAX).astype(np.float64)
    s = d["s"].to_numpy()
    new = s + W * small * z
    new = np.where(np.isfinite(new), new, s)
    return new
