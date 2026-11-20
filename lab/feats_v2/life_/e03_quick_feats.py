"""Exploration step 3: write a quick exploratory life_ set for val (from e02 parquet) to run the harness."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
SCR = Path("/tmp/scratchpad")
V = pl.read_parquet(SCR / "e02_val.parquet")
cnt = ["sp_c_fut", "sp_c_past", "sp_h_fut", "sp_h_past", "sp_c_fut24", "sp_h_fut24", "sp_h1_fut24"]
V = V.with_columns([pl.col(c).fill_null(0) for c in cnt])
keep = cnt + ["sp_prev_dt", "sp_next_dt", "sp_h_fut_minhpos", "trc_ndays", "trc_nlast1d", "trh_mean_hpos", "tr_age_h",
              "tr_lastgap_h", "trc_lastgap_h", "trh_lastfresh_gap_h", "trc_span_h", "trh_firstfresh_age_h", "trh_nusers"]
out = FEATS / "val"
for c in keep:
    np.save(out / f"life_x{c}.npy", V[c].cast(pl.Float32).to_numpy())
print("saved", len(keep))
