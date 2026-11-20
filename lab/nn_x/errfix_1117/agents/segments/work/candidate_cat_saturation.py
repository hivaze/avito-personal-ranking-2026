"""NOT SUBMITTED (near-miss, below the +0.0003 bar). Within-category saturation:
demote the k-th item of a category in the feed (k = cat_rank_in_feed, the model's rank among same-category
candidates) by W * (k - 1) ** 0.3. Category leaders are untouched.
Cross-fitted on discovery: +0.00023 (se 0.00009, z 2.5); in-sample +0.00023. Head-restricted conditional-logit
likelihood does NOT corroborate it (follower coefficient ~0 at ranks <= 20)."""
import numpy as np
import polars as pl

W = 0.02
P = 0.3


def adjust(df: pl.DataFrame) -> np.ndarray:
    s = df["score"].to_numpy().astype(np.float64)
    cr = df["cat_rank_in_feed"].to_numpy().astype(np.float64)
    new = s - W * np.clip(cr - 1.0, 0.0, None) ** P
    return new + 1e-6 * s
