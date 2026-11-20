"""2x2 error map: intent regime (positive in f3-top category or not) x positive coldness (val, leak-masked)."""
import numpy as np, polars as pl
OUT = "<repo>/eda/errors/segments"
d = pl.read_parquet(f"{OUT}/feed_table.parquet").filter(pl.col("leak").not_())
N = len(d)
reg = np.where(d["pos_f3_catrank"].to_numpy() == 0, "f3-top", "off-f3")
cold = np.where(d["pos_cold_all"].to_numpy(), "pos all-cold", np.where(d["pos_cold_any"].to_numpy(), "pos mixed", "pos warm"))
nd = {k: d[f"ndcg_{k}"].to_numpy() for k in ["nn_oof", "nn_tr", "stk", "cb", "blend", "catorc_nn_oof", "withinorc_nn_oof"]}
L = (1 - nd["nn_oof"]).sum(); CG = (nd["catorc_nn_oof"] - nd["nn_oof"]).sum(); WG = (nd["withinorc_nn_oof"] - nd["nn_oof"]).sum()
def pse(x): return f"{x.mean():+.4f} ({x.std(ddof=1)/np.sqrt(len(x)):.4f})"
print("| regime | positive | share | NN oof | loss share | share of cat-orc headroom | share of within-orc headroom | oof-tr | stk-tr | CB-oof | blend-oof |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in ["f3-top", "off-f3"]:
    for c in ["pos warm", "pos mixed", "pos all-cold", "ALL"]:
        m = (reg == r) & ((cold == c) | (c == "ALL"))
        print(f"| {r} | {c} | {100*m.mean():.1f}% | {nd['nn_oof'][m].mean():.4f} | {100*(1-nd['nn_oof'][m]).sum()/L:.1f}% | "
              f"{100*(nd['catorc_nn_oof'][m]-nd['nn_oof'][m]).sum()/CG:.1f}% | {100*(nd['withinorc_nn_oof'][m]-nd['nn_oof'][m]).sum()/WG:.1f}% | "
              f"{pse(nd['nn_oof'][m]-nd['nn_tr'][m])} | {pse(nd['stk'][m]-nd['nn_tr'][m])} | {pse(nd['cb'][m]-nd['nn_oof'][m])} | {pse(nd['blend'][m]-nd['nn_oof'][m])} |")
print(f"\ntotal headroom: 1-NDCG {L/N:.4f}; cat-oracle gain {CG/N:.4f}; within-oracle gain {WG/N:.4f}")
