"""Join the errfix_1117 per-candidate columns (label-free, own-feed only) for all val rows, in flat row order.
Writes ../out/valfeat.parquet (no label column, no score columns)."""
import ph, numpy as np, polars as pl
cols_drop = ["score", "s_ft", "s_to", "label"]
a = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/data/disc.parquet")
b = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/private/confirm.parquet")
d = pl.concat([a, b]).drop(cols_drop).sort("row")
D = ph.Data("val")
assert len(d) == len(D.y) and (d["row"].to_numpy() == np.arange(len(D.y))).all()
assert (d["fidx"].to_numpy() == D.fid).all()
d.write_parquet("../out/valfeat.parquet")
print(d.shape, "ok")
t = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/private/test.parquet").drop([c for c in cols_drop if c in pl.read_parquet_schema("<repo>/lab/nn_x/errfix_1117/private/test.parquet")]).sort("row")
T = ph.Data("test")
assert len(t) == len(T.fid) and (t["row"].to_numpy() == np.arange(len(T.fid))).all() and (t["fidx"].to_numpy() == T.fid).all()
t.write_parquet("../out/testfeat.parquet"); print(t.shape, "test ok")
