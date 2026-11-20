"""NDCG levels of every pass on the read sets (pool4 and the two families), markdown rows for the report. Post-read, descriptive."""
import numpy as np
import s3_read as R

S = {k: R.Set(k) for k in ("holdout", "val")}
for tag in "TU":
    print(f"\n### design {tag}: mean NDCG on the read set (holdout n {int(S['holdout'].read[tag].sum())}, val n {int(S['val'].read[tag].sum())})")
    hdr = ["pass", "holdout pool4", "holdout distilled", "holdout hard", "val pool4", "val distilled", "val hard"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for lab, p in (("a stored 256", "a"), ("b newest 128", "b"), ("c newest 128 + recovered", "c" + tag), ("am stored 128 + n", "am" + tag), ("e placebo", "e" + tag), ("b192 newest 192", "b192"),
                   ("c192 newest 192 + recovered", "c192" + tag), ("g 128 + stored 193-256", "g"), ("d1 256 + recovered (dense held)", "d1" + tag), ("d2 256 + recovered (dense recomputed)", "d2" + tag), ("dP 256 + other user's", "dP" + tag)):
        row = [lab]
        for name in ("holdout", "val"):
            s = S[name]
            for who in ("pool4", "distilled", "hard"):
                x = s.ndcg(p, who)[s.read[tag]]
                row.append(f"{np.nanmean(x):.5f}" + ("" if not np.isnan(x).any() else f" (n {int((~np.isnan(x)).sum())})"))
        print("| " + " | ".join(row) + " |")
    print("fill: holdout mean %.1f events (full 128: %d feeds); val mean %.1f (full: %d)" % (S["holdout"].nf[tag][S["holdout"].read[tag]].mean(), (S["holdout"].nf[tag][S["holdout"].read[tag]] == 128).sum(),
                                                                                              S["val"].nf[tag][S["val"].read[tag]].mean(), (S["val"].nf[tag][S["val"].read[tag]] == 128).sum()))
