"""Group every parsed training segment into run families and print a markdown table (CPU, read-only)."""
import os
import re

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
r = pd.read_csv(f"{OUT}/runs.csv").drop_duplicates(["variant", "start"])
r = r.sort_values("seg").drop_duplicates(["src", "variant"], keep="last")  # last segment per log (relaunches)

RULES = [
    (r"^v12_all_|^v12_allprew_", "F5: Muon lr2e-3 WSD mix0.3 cvnd/prew 10ep"),
    (r"^v13_", "v13 hedges seed23 cvnd 10ep (lr x mix)"),
    (r"^v14_s9[2-6]$", "F6: Muon lr1e-3 mix0.1 cvnd 10ep"),
    (r"^v14_mix1_s91$", "v14_mix1_s91: lr2e-3 mix0.1 prew 10ep"),
    (r"^v15_cd_", "F5cd: 1-ep cooldown continuation (lr2e-3, mix0.1)"),
    (r"^v16_pre2_e3", "v16: cvnd_pre2 (clip bug) 3ep AdamW cos"),
    (r"^v17_pre2s_", "F7: Muon lr2e-3 mix0.1 pre2s 10ep"),
    (r"^v18_pre2s_e3", "v18: cvnd_pre2s 3ep AdamW cos"),
    (r"^v19_pre2s_m0", "v19: pre2s lr1e-3 mix0 10ep (stopped at 5k)"),
    (r"^v20_pre2s_e2", "F8: Muon lr1e-3 mix0.1 pre2s 2ep"),
    (r"^v21_cd_", "F6cd: 1-ep cooldown continuation (lr1e-3)"),
    (r"^v22_pre2s_lr1", "F9: Muon lr1e-3 mix0.1 pre2s 10ep"),
    (r"^v23_gru", "F10: GRU, F7 recipe (running)"),
    (r"^v24_din", "F11: DIN, F7 recipe (running)"),
    (r"^v26_d192|^smoke_d192", "F12: d192/6h, F7 recipe (queued)"),
    (r"^smoke_", "smoke tests"),
    (r"^sx_", "solvers 2ep harness (F8 recipe) arms"),
    (r"^v12_muwsd_e10|^v12_rg_e10|^v12_prew3", "round-4 component probes (killed early / 3ep)"),
    (r"^v11_", "v11: AdamW cos 10-20ep probes (gate/d256/deep/GRU; mostly killed)"),
    (r"^v10_pre", "v10: train-row pre_* (leaky, rejected)"),
    (r"^v9_", "v9: AdamW cos cvnd 5/8/10ep (+GRU/DIN/deep 5ep)"),
    (r"^v8_", "v8: AdamW cos cvnd 2/3/5ep (+GRU/DIN/deep)"),
    (r"^v[3-7]_", "v3-v7: AdamW cos 2ep, cv/cvix/cvhn/cvraw inputs, GRU/DIN"),
    (r"^v2_", "v2: AdamW cos 2ep loss-norm/K9/intent/idcg"),
    (r"^s2_|^s2rand", "stage-2 continuations (old recipe)"),
    (r"^optim_|^base_s2|^feat_base", "nnx optim 2ep arms (Muon/WSD/KD/lr2)"),
    (r"^attn_", "nnx attn 2ep arms (FoX/QK/MoE/retr/gate)"),
    (r"^feat_", "nnx feat 2ep arms (PLE/prop/sid/TabM/ts)"),
    (r".*", "round-1 2ep AdamW (base/nopos/aux/sim/heads/blocks/d256)"),
]


def grp(v):
    for pat, lab in RULES:
        if re.search(pat, str(v)):
            return lab
    return "other"


r["group"] = r.variant.map(grp)
rows = []
for g, d in r.groupby("group", sort=False):
    c = d[d.complete == True]  # noqa: E712
    def uniq(col):
        vals = sorted({str(x) for x in d[col].dropna()})
        return "/".join(vals)
    sel = c.best_step / c.total_steps
    rows.append(dict(group=g, segs=len(d), complete=len(c), epochs=uniq("epochs"), lr=uniq("lr"),
                     muon=uniq("muon"), sched=uniq("sched"), mix=uniq("cooldown_mix"), retr=uniq("retr_m"),
                     gate=uniq("attn_gate"), arch=uniq("arch"), d=uniq("d"), heads=uniq("heads"), blocks=uniq("blocks"),
                     hist=uniq("hist_layers"), xt=uniq("xt"),
                     val=f"{c.best_val.mean():.5f}+-{c.best_val.std(ddof=1):.5f}" if len(c) > 1 else (f"{c.best_val.mean():.5f}" if len(c) else "-"),
                     hold=f"{c.sel_hold.mean():.5f}" if c.sel_hold.notna().any() else "-",
                     sel_frac=f"{sel.median():.2f} [{sel.min():.2f}-{sel.max():.2f}]" if len(c) else "-"))
T = pd.DataFrame(rows)
def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


print(md(T))
T.to_csv(f"{OUT}/groups.csv", index=False)
print(len(r), "segments after dedup;", int((r.complete == True).sum()), "complete")  # noqa: E712
