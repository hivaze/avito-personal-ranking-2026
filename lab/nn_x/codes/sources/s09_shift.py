"""s09: train-vs-eval shift of the source mix: train sample (300K feeds) split by exploration-likeness (p_random>0.5)
vs production feeds, against val/test. Shares of all events; type-1 share; P(source | type 1)."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, feeds
rows = []
tr, fs = load_hist("train", train_sample(300_000, seed=9))
pr = feeds("train").select("fidx", "p_random")
tr = tr.join(pr, on="fidx")
parts = {"train_prod(p<0.5)": tr.filter(pl.col("p_random") < 0.5), "train_expl(p>0.5)": tr.filter(pl.col("p_random") > 0.5)}
for sp in ["val", "test"]:
    parts[sp], _ = load_hist(sp)
for name, d in parts.items():
    n = d.height; nf = d["fidx"].n_unique()
    v = d.filter(pl.col("et") == 1)
    r = dict(part=name, feeds=nf, H_mean=n / nf, type1_share=v.height / n, anon_feed_share=float(d.filter(pl.col("user_id") == 0)["fidx"].n_unique() / nf))
    for s in [0, 1, 2, 3, 4, 5, 6, 7]:
        r[f"s{s}_all"] = float((d["es"] == s).mean())
    for s in [0, 1, 2, 4, 5, 6, 7]:
        r[f"s{s}|t1"] = float((v["es"] == s).mean())
    rows.append(r)
pshow(pl.DataFrame(rows))
