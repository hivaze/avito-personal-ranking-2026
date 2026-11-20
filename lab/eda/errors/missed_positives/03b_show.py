import polars as pl
df = pl.read_parquet("<repo>/eda/errors/missed_positives/screen.parquet").filter(pl.col("t_best").is_not_nan())
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(4)
print("n feats with within-feed variation:", df.height)
cols = ["feat", "t_best", "t_clean", "dll_best", "wr_MP_FA3", "wr_MP_se", "wr_TP3_FA3", "med_MP", "med_TP3", "med_FA3"]
print(df.sort(pl.col("t_best").abs(), descending=True).head(70).select(cols))
