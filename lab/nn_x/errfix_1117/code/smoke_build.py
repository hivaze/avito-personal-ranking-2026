"""Smoke test of build_tables on the first 300 validation feeds."""
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import build_tables as bt  # noqa: E402
from avito_ranker.ensemble.build import final_score, load_pool  # noqa: E402
from avito_ranker.ensemble.feeds import load_feed_index  # noqa: E402

if __name__ == "__main__":
    F = load_feed_index("val"); b = load_pool("FD16P20kd", "val"); sc = final_score(b, F, 0.5)
    n = 300; e = int(F.offsets[n])
    t = time.time()
    df = bt.build("val", np.arange(n), {"score": sc[:e]}, F.offsets[:n], F.sizes[:n],
                  np.load("<repo>/artifacts/data/flat/val/cand_target.npy")[:e])
    print("300 feeds in", round(time.time() - t, 1), "s", df.shape)
    print(df.select(["fidx", "rank", "score", "hmax", "hmax_idx", "h5", "c_contact", "cat_views", "cat_newest_idx",
                     "cand_maxcos", "cat_rank_in_feed", "f2_home", "label"]).head(5))
    print("nulls", int(df.null_count().sum_horizontal()[0]))
    print(df.select([pl.col(c).is_nan().mean().alias(c) for c, t_ in df.schema.items() if t_ in (pl.Float32, pl.Float64)]))
