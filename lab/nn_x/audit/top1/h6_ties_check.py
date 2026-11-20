"""H6: do any of our candidate / submission files contain tied scores inside a feed (ties place negatives first)?"""
import glob
import numpy as np, polars as pl
files = sorted(glob.glob("<repo>/subs/*.parquet"))
for fn in files:
    d = pl.read_parquet(fn)
    s = d["score"].cast(pl.Float32).to_numpy(); fid = d["feed_id"].to_numpy()
    o = np.lexsort((s, fid)); fs, ss = fid[o], s[o]
    ties = int(np.sum((fs[1:] == fs[:-1]) & (ss[1:] == ss[:-1])))
    print(f"{fn.split('/')[-1]:45s} rows {len(d):8d} ties {ties:6d} finite {bool(np.isfinite(s).all())} range [{s.min():.3g}, {s.max():.3g}]")
