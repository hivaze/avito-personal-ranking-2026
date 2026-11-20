"""07: final tables. out/mapping.csv (per code: Avito name, vertical, confidence, key evidence stats), out/vertical_stats.csv
(per vertical: candidate / positive shares and CTR on val, train exploration-like, train production-like), out/code_vertical.json."""
from common_cat import *
import json
mp = pl.read_parquet(OUT / "mapping.parquet")
cs = pl.read_parquet(OUT / "cand_stats.parquet")
CONF = {  # (vertical confidence, name confidence)
    2: "high/high", 3: "high/high", 4: "high/medium", 5: "high/medium", 26: "high/medium",
    6: "high/high", 7: "high/medium", 8: "high/medium", 32: "medium/medium", 27: "medium/medium", 47: "medium/medium",
    9: "high/medium", 10: "high/high", 11: "high/medium", 12: "high/medium", 25: "medium/medium", 30: "high/medium", 31: "high/high",
    13: "high/high", 14: "medium/medium", 15: "high/high", 16: "high/medium", 33: "medium/medium",
    17: "high/medium", 18: "high/medium", 29: "high/high", 40: "high/medium", 41: "high/medium", 42: "high/medium", 43: "high/medium", 44: "high/medium", 46: "high/medium",
    19: "medium/medium", 20: "medium/medium", 21: "medium/medium", 22: "medium/medium", 23: "medium/medium", 28: "medium/medium", 45: "medium/medium",
    34: "high/medium", 35: "high/medium", 36: "high/medium", 37: "high/medium", 38: "high/medium", 39: "high/high",
    48: "high/high", 49: "medium/low", 53: "medium/medium", 50: "high/high", 24: "medium/medium", 51: "medium/medium", 52: "high/high",
}
mp = mp.with_columns(pl.col("code").map_elements(lambda c: CONF[c], return_dtype=pl.Utf8).alias("confidence_vertical_name"))
mp.write_csv(OUT / "mapping.csv")
json.dump({int(c): v for c, v in zip(mp["code"], mp["vertical"])}, open(OUT / "code_vertical.json", "w"), indent=0)
vmap = dict(zip(mp["code"].to_list(), mp["vertical"].to_list()))
d = cs.filter(pl.col("code") >= 2).with_columns(pl.col("code").replace_strict(vmap, default="none").alias("vertical"))
agg = d.group_by("vertical").agg(pl.len().alias("n_codes"), pl.col("n_val").sum(), pl.col("pos_val").sum(), pl.col("n_trR").sum(), pl.col("pos_trR").sum(),
                                 pl.col("n_trP").sum(), pl.col("pos_trP").sum(), pl.col("items_train").sum())
agg = agg.with_columns((pl.col("n_val") / pl.col("n_val").sum()).alias("share_val"), (pl.col("pos_val") / pl.col("pos_val").sum()).alias("pos_share_val"),
                       (pl.col("pos_val") / pl.col("n_val")).alias("ctr_val"), (pl.col("pos_trR") / pl.col("n_trR")).alias("ctr_trR"),
                       (pl.col("pos_trP") / pl.col("n_trP")).alias("ctr_trP"), (pl.col("n_trP") / pl.col("n_trP").sum()).alias("share_trP"),
                       (pl.col("n_trR") / pl.col("n_trR").sum()).alias("share_trR")).sort("share_val", descending=True)
agg.write_csv(OUT / "vertical_stats.csv")
print(agg.select(["vertical", "n_codes", "share_val", "share_trR", "share_trP", "pos_share_val", "ctr_val", "ctr_trR", "ctr_trP", "items_train"]).with_columns(pl.col(pl.Float64).round(4)))
