"""Per-segment decomposition of the val error (leak-masked: 117 later-user feeds excluded).

For each segmentation and bucket: share of feeds (val; test for observable ones), mean NDCG per model, share of total
loss (1-NDCG) of nn_oof and nn_tr, positive-category-oracle gap (cross-category part) and within-category-oracle gap,
paired deltas with SE: stk-nn_tr, nn_oof-nn_tr, cb-nn_oof, blend-nn_oof, and the NN-vs-CB best-of headroom.
Writes segments.csv and prints markdown tables.
"""
import numpy as np, polars as pl

OUT = "<repo>/eda/errors/segments"
d = pl.read_parquet(f"{OUT}/feed_table.parquet").filter(~pl.col("leak"))
te = pl.read_parquet(f"{OUT}/test_segments.parquet")


from segdefs import cut, seg_defs


S = seg_defs(d)
St = seg_defs(te, test=True)
N = len(d)
nd = {k: d[f"ndcg_{k}"].to_numpy() for k in ["nn_oof", "nn_tr", "stk", "cb", "lgb", "blend", "f1", "f3",
                                            "catorc_nn_oof", "withinorc_nn_oof"]}
L_oof = (1 - nd["nn_oof"]).sum(); L_tr = (1 - nd["nn_tr"]).sum()


def pse(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


rows = []
for sname, lab in S.items():
    labs_t = St.get(sname)
    for b in sorted(set(lab), key=lambda s: str(s)):
        m = lab == b
        r = dict(seg=sname, bucket=b, n=int(m.sum()), share=m.mean())
        r["share_test"] = float((labs_t == b).mean()) if (labs_t is not None and sname not in ("day",)) else np.nan
        for k in ["nn_oof", "nn_tr", "stk", "cb", "blend", "f1"]:
            r[k] = nd[k][m].mean()
        r["loss_share_oof"] = (1 - nd["nn_oof"][m]).sum() / L_oof
        r["loss_share_tr"] = (1 - nd["nn_tr"][m]).sum() / L_tr
        r["lift"] = r["loss_share_oof"] / r["share"]
        r["catorc_gap"] = (nd["catorc_nn_oof"][m] - nd["nn_oof"][m]).mean()
        r["withinorc_gap"] = (nd["withinorc_nn_oof"][m] - nd["nn_oof"][m]).mean()
        for name, a, bb in [("stk_m_tr", "stk", "nn_tr"), ("oof_m_tr", "nn_oof", "nn_tr"), ("cb_m_oof", "cb", "nn_oof"),
                            ("blend_m_oof", "blend", "nn_oof"), ("cb_m_tr", "cb", "nn_tr")]:
            mu, se = pse(nd[a][m] - nd[bb][m]); r[name] = mu; r[name + "_se"] = se
        r["bestof_nn_cb"] = (np.maximum(nd["nn_oof"][m], nd["cb"][m]) - nd["nn_oof"][m]).mean()
        r["rho_nn_cb"] = d["rho_nn_cb"].to_numpy()[m].mean()
        rows.append(r)
T = pl.DataFrame(rows)
T.write_csv(f"{OUT}/segments.csv")

pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(400)
pl.Config.set_fmt_str_lengths(30)
print(f"N clean feeds = {N}; nn_oof {nd['nn_oof'].mean():.4f} nn_tr {nd['nn_tr'].mean():.4f} stk {nd['stk'].mean():.4f} "
      f"cb {nd['cb'].mean():.4f} blend {nd['blend'].mean():.4f}; catorc {nd['catorc_nn_oof'].mean():.4f} "
      f"withinorc {nd['withinorc_nn_oof'].mean():.4f}")


def fmt(T):
    c = []
    for col in T.columns:
        if T[col].dtype in (pl.Float64, pl.Float32):
            c.append(pl.col(col).round(4))
        else:
            c.append(pl.col(col))
    return T.select(c)


for sname in S:
    t = T.filter(pl.col("seg") == sname)
    print(f"\n### {sname}")
    print(fmt(t.select("bucket", "n", "share", "share_test", "nn_oof", "nn_tr", "stk", "cb", "blend", "loss_share_oof",
                       "lift", "catorc_gap", "withinorc_gap", "oof_m_tr", "oof_m_tr_se", "stk_m_tr", "stk_m_tr_se",
                       "cb_m_oof", "cb_m_oof_se", "blend_m_oof", "blend_m_oof_se", "bestof_nn_cb", "rho_nn_cb")))
