import polars as pl, sys
T = pl.read_csv("<repo>/eda/errors/segments/segments.csv")
order = {"M": ["M5-10","M11-25","M26-50","M51-100","M101-200","M201-300"],
         "H": ["H0","H1-19","H20-63","H64-191","H192-255","H256"],
         "user": ["uid0","tr0","tr1","tr2-4","tr5+"]}
def f(x, p=3):
    return "" if x is None else f"{x:.{p}f}"
for seg in T["seg"].unique(maintain_order=True):
    t = T.filter(pl.col("seg") == seg)
    if seg in order:
        t = t.with_columns(pl.col("bucket").replace_strict({b: i for i, b in enumerate(order[seg])}, default=99).alias("o")).sort("o")
    print(f"\n#### {seg}\n")
    print("| bucket | n | share val | share test | NN oof | NN tr | stk | CB | loss share | lift | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) | rho NN/CB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t.iter_rows(named=True):
        st = "" if r["share_test"] is None or r["share_test"] != r["share_test"] else f"{100*r['share_test']:.1f}%"
        print(f"| {r['bucket']} | {r['n']} | {100*r['share']:.1f}% | {st} | {f(r['nn_oof'],4)} | {f(r['nn_tr'],4)} | {f(r['stk'],4)} | {f(r['cb'],4)} | "
              f"{100*r['loss_share_oof']:.1f}% | {f(r['lift'],2)} | {f(r['catorc_gap'])} | {f(r['withinorc_gap'])} | "
              f"{r['oof_m_tr']:+.4f} ({r['oof_m_tr_se']:.4f}) | {r['stk_m_tr']:+.4f} ({r['stk_m_tr_se']:.4f}) | {r['cb_m_oof']:+.4f} ({r['cb_m_oof_se']:.4f}) | "
              f"{r['blend_m_oof']:+.4f} ({r['blend_m_oof_se']:.4f}) | {f(r['rho_nn_cb'],2)} |")
