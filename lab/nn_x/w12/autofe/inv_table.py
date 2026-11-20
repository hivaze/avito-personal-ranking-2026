import polars as pl
d = pl.read_csv("<repo>/nn_x/w12/autofe/inventory.csv", infer_schema_length=None)
print("| input | NN scaling | zeros | NaN val / train | mode mass | skew / kurt (as fed) | max abs z test | PSI trn→val | PSI val→test | between-feed var | rel. version in model | flags |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in d.iter_rows(named=True):
    n = r["name"]; fl = []
    sent = n.startswith(("cos_het", "seq_max", "maxsim", "seq_top", "cos_hall")) and r["nan_val"] > 0
    if sent: fl.append("sentinel −1.5 mixed with values")
    if n.startswith("pre2") and r["nan_trs"] > 0.1: fl.append("NaN→0 on 28% of train (blank regime), never on val/test")
    if r["zero"] > 0.8 and r["kurt_nn"] > 50: fl.append("sparse + heavy tail")
    elif r["kurt_nn"] > 20 and not sent: fl.append("heavy tail")
    if r["psi_trs_val"] > 0.1: fl.append("train→val drift")
    if r["between_var"] is not None and r["between_var"] > 0.75 and not r["has_rel_in_model"]: fl.append("mostly feed-level, no relative version")
    if n == "item_norm": fl = ["stand-in only (model uses log1p(norm)·has_emb)"]
    bv = "" if r["between_var"] is None else f"{r['between_var']:.2f}"
    rel = "" if r["has_rel_in_model"] is None else ("yes" if r["has_rel_in_model"] else "no")
    print(f"| {n} | {r['nn_scaling']} | {r['zero']:.3f} | {r['nan_val']:.3f} / {r['nan_trs']:.3f} | {r['mode_mass']:.3f} | {r['skew_nn']:+.1f} / {r['kurt_nn']:.0f} | {r['maxabsz_test']:.1f} | {r['psi_trs_val']:.3f} | {r['psi_val_test']:.3f} | {bv} | {rel} | {'; '.join(fl)} |")
