"""Family-level training-curve analysis (CPU, read-only on parsed logs)."""
import json
import os
import re

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
C = [json.loads(l) for l in open(f"{OUT}/curves.jsonl")]
# keep the last complete segment per variant
byv = {}
for c in C:
    if c["steps"] and c["steps"][-1] == c["total"]:
        byv[c["variant"]] = c
FAM = {
    "F5 lr2e-3 mix0.3 cvnd/prew": [f"v12_all_{s}" for s in ("e10", "s81", "s82")] + [f"v12_allprew_s{s}" for s in (83, 84, 85)],
    "F6 lr1e-3 mix0.1 cvnd": [f"v14_s{s}" for s in (92, 93, 94, 95, 96)] + ["v13_lr1_mix1"],
    "F7 lr2e-3 mix0.1 pre2s": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F9 lr1e-3 mix0.1 pre2s": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F8 2ep lr1e-3 mix0.1 pre2s": [f"v20_pre2s_e2_s{s}" for s in range(301, 313)],
    "sx 2ep base (s23,s24)": ["sx_base_s23", "sx_base_s24"],
    "AdamW cos 10ep cvnd (v9_e10 x4)": ["v9_e10", "v9_e10_s61", "v9_e10_s62", "v9_e10_s63"],
    "F5cd 1ep continuation": [f"v15_cd_all_{s}" for s in ("e10", "s81", "s82")] + [f"v15_cd_allprew_s{s}" for s in (83, 84, 85)],
    "F6cd 1ep continuation": [f"v21_cd_v14_s{s}" for s in (92, 94, 96)],
}
SEED23 = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr1_mix3", "v13_lr2_mix1", "v12_all_e10", "v9_e10"]


def fam_table(names):
    cs = [byv[n] for n in names if n in byv]
    steps = sorted(set.intersection(*[set(c["steps"]) for c in cs]))
    V = np.array([[c["val"][c["steps"].index(s)] for s in steps] for c in cs])
    H = np.array([[c["hold"][c["steps"].index(s)] for s in steps] for c in cs])
    return cs, np.array(steps), V, H


lines = []
P = lines.append
for fam, names in FAM.items():
    cs, st, V, H = fam_table(names)
    tot = cs[0]["total"]
    P(f"\n### {fam}: n={len(cs)} total={tot}")
    pick = [s for s in st if s % (2500 if tot > 50000 else 1000) == 0 or s == tot]
    if tot <= 10000:
        pick = list(st)
    P("step  " + " ".join(f"{s:>7d}" for s in pick))
    idx = [list(st).index(s) for s in pick]
    P("val   " + " ".join(f"{V[:, i].mean():.5f}" for i in idx))
    P("hold  " + " ".join(f"{H[:, i].mean():.5f}" for i in idx))
    P("valsd " + " ".join(f"{V[:, i].std(ddof=1):.5f}" if len(cs) > 1 else "   -   " for i in idx))
    # per-run summary
    rows = []
    for c in cs:
        s = np.array(c["steps"]); v = np.array(c["val"]); h = np.array(c["hold"])
        j = int(np.nanargmax(h)); i = int(np.nanargmax(v))
        tc = int(round(0.8 * c["total"])) if c["total"] > 50000 else None
        r = dict(variant=c["variant"], sel_step=int(s[j]), sel_val=v[j], sel_hold=h[j], max_val=v[i], max_val_step=int(s[i]),
                 last_val=v[-1], last_hold=h[-1])
        if tc:
            k = int(np.searchsorted(s, tc)) - 1  # last eval before the cooldown starts
            r.update(pre_cd_step=int(s[k]), pre_cd_val=v[k], pre_cd_hold=h[k],
                     plateau_best_hold=np.nanmax(h[: k + 1]), plateau_best_val_at=v[int(np.nanargmax(h[: k + 1]))],
                     cd_max_val=np.nanmax(v[k + 1:]), cd_max_hold=np.nanmax(h[k + 1:]),
                     cd_argmax_hold=int(s[k + 1 + int(np.nanargmax(h[k + 1:]))]),
                     val_20k=v[int(np.searchsorted(s, 20000))], hold_20k=h[int(np.searchsorted(s, 20000))])
        rows.append(r)
    df = pd.DataFrame(rows)
    P(df.round(5).to_string(index=False))
    if "pre_cd_val" in df:
        P(f"MEAN sel_val {df.sel_val.mean():.5f} sd {df.sel_val.std(ddof=1):.5f} | sel_hold {df.sel_hold.mean():.5f} sd {df.sel_hold.std(ddof=1):.5f}"
          f" | plateau 20k->pre-cd val {df.val_20k.mean():.5f}->{df.pre_cd_val.mean():.5f} hold {df.hold_20k.mean():.5f}->{df.pre_cd_hold.mean():.5f}"
          f" | cd max-hold gain {np.mean(df.cd_max_hold - df.plateau_best_hold):+.5f} | cd max-val - pre-cd val {np.mean(df.cd_max_val - df.pre_cd_val):+.5f}"
          f" | end - sel: val {np.mean(df.last_val - df.sel_val):+.5f} hold {np.mean(df.last_hold - df.sel_hold):+.5f}"
          f" | sel in cooldown {int((df.sel_step > df.pre_cd_step).sum())}/{len(df)}")
    else:
        P(f"MEAN sel_val {df.sel_val.mean():.5f} sd {df.sel_val.std(ddof=1) if len(df) > 1 else float('nan'):.5f} | sel_hold {df.sel_hold.mean():.5f}"
          f" | last_val {df.last_val.mean():.5f} last_hold {df.last_hold.mean():.5f} | sel==last {int((df.sel_step == cs[0]['total']).sum())}/{len(df)}")

P("\n### seed-23 cvnd matched set (identical up to the cooldown within the same lr)")
for n in SEED23:
    c = byv.get(n)
    if not c:
        continue
    s = np.array(c["steps"]); v = np.array(c["val"]); h = np.array(c["hold"])
    pk = [20000, 40000, 60000, 77500, 80000, 82500, 85000, 87500, 90000, 95000, 98350]
    P(f"{n:14s} " + " ".join(f"{s_}:{v[list(s).index(s_)]:.4f}/{h[list(s).index(s_)]:.4f}" for s_ in pk if s_ in list(s)))
open(f"{OUT}/curves_report.txt", "w").write("\n".join(lines))
print("\n".join(lines))
