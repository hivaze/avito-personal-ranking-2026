"""Verifier: independent recomputation of the 4 history-count KEEP features on val/test with polars (no numba kernel),
plus hist ordering check (row j of a feed's history == hpos j)."""
import numpy as np, polars as pl, sys
FL = "<repo>/data/flat"; F = "<repo>/data/feats"; W = "<repo>/data/clu_work"
for sp in ["val", "test"]:
    fe = pl.read_parquet(f"{FL}/{sp}/feeds.parquet")
    hf = np.load(f"{FL}/{sp}/hist_fidx.npy"); hp = np.load(f"{FL}/{sp}/hist_hpos.npy")
    ho = fe["hist_off"].to_numpy()
    print(sp, "hist row j == hpos:", bool(np.all(hp == np.arange(len(hf)) - ho[hf])))
    he = np.load(f"{FL}/{sp}/hist_emb_idx.npy"); ce = np.load(f"{FL}/{sp}/cand_emb_idx.npy")
    het = np.load(f"{FL}/{sp}/hist_event_type.npy"); cf = np.load(f"{FL}/{sp}/cand_fidx.npy")
    for K, feats in ((1024, ["hcnt", "hmin"]), (4096, ["hrw", "het1"])):
        tab = np.load(f"{W}/item_cl_{K}_{sp}.npy")
        hc = np.where(he >= 0, tab[np.maximum(he, 0)], -1).astype(np.int64)
        cc = np.where(ce >= 0, tab[np.maximum(ce, 0)], -1).astype(np.int64)
        h = pl.DataFrame(dict(f=hf, c=hc, hp=hp, e1=(het == 1).astype(np.int32))).filter(pl.col("c") >= 0)
        g = h.group_by(["f", "c"]).agg(pl.len().alias("hcnt"), pl.col("hp").min().alias("hmin"),
                                         (0.5 ** (pl.col("hp") / 5.0)).sum().alias("hrw"), pl.col("e1").sum().alias("het1"))
        c = pl.DataFrame(dict(f=cf, c=cc, r=np.arange(len(cf)))).join(g, on=["f", "c"], how="left").sort("r")
        for nm in feats:
            ref = c[nm].to_numpy().astype(np.float64)
            fill = 999.0 if nm == "hmin" else 0.0
            ref = np.where(np.isnan(ref), fill, ref); ref[cc < 0] = np.nan
            got = np.load(f"{F}/{sp}/clu_{nm}{K}.npy").astype(np.float64)
            ok = ~np.isnan(ref)
            print(f"  {sp} clu_{nm}{K}: nan-pattern equal={bool(np.array_equal(np.isnan(ref), np.isnan(got)))} "
                  f"max|diff|={np.nanmax(np.abs(ref[ok]-got[ok])):.2e}")
