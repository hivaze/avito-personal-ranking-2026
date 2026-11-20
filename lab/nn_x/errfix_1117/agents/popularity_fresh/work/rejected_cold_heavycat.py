"""NOT SUBMITTED (near-miss, kept for reference). Boost cold listings (no train impressions) in categories the
user has browsed heavily (>= 50 history events in the candidate's category).

Behavioural story: a heavy browser of a category has probably already been shown the established (exposed)
listings there; new, never-shown listings are novel to them. Conditional-logit beta = +0.139 (se 0.041).

Why it is not submitted (popularity_fresh agent, 2026-11-17):
  - own-family cross-fit +0.00068 (se 0.00027, z 2.52, weight 0.09 chosen in all folds), BUT
  - nested selection over the 25 families tried in this lens: -0.00011 (seed 0), -0.00087, -0.00024, -0.00063;
  - 10 feeds carry essentially all of the in-sample net gain (533 feeds worse, 386 better);
  - jagged response: cat_views threshold 20 -> +0.00011, 50 -> +0.00054, 100 -> +0.00006; w 0.12 -> +0.00024;
  - even if the effect were fully real at the in-sample beta, simulated expected NDCG gain is +0.00027 (sd 0.00033);
  - the young-cold part (f4 < .2) is negative (-0.00018); on test the cold set is larger (staler popularity).
Columns read: xp_imp_c50 (cold status only), cat_views, score.
"""
import numpy as np
import polars as pl

W = 0.09
MIN_CAT_VIEWS = 50


def adjust(df: pl.DataFrame) -> np.ndarray:
    s = df["score"].to_numpy().astype(np.float64)
    cold = df["xp_imp_c50"].to_numpy() == 0
    heavy = df["cat_views"].to_numpy() >= MIN_CAT_VIEWS
    return s + W * (cold & heavy) + 1e-7 * s
