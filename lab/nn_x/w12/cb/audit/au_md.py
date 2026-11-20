"""audit: markdown fragments for the report from out/level.json (no computation)."""
import json, numpy as np
from pathlib import Path
OUT = Path("<repo>/nn_x/w12/cb/audit/out"); rows = json.load(open(OUT / "level.json"))
f = lambda r, k, fm="{:.5f}": fm.format(r[k]) if k in r else "n/a"
pm = lambda r, sp, w: f"{r[f'{sp}_d{w}']:+.5f} ± {r[f'{sp}_s{w}']:.5f}" if f"{sp}_d{w}" in r else "n/a"
L = ["| model | val | holdout | gap val (± SE) | gap holdout (± SE) | rho val / holdout / test | law val: record / refit | law holdout: record / refit |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['name']} | {f(r,'val_ndcg')} | {f(r,'hold_ndcg')} | {r['val_gap']:.5f} ± {r['val_gap_se']:.5f} | " + (f"{r['hold_gap']:.5f} ± {r['hold_gap_se']:.5f}" if 'hold_gap' in r else "n/a") +
             f" | {f(r,'val_rho','{:.4f}')} / {f(r,'hold_rho','{:.4f}')} / {f(r,'test_rho','{:.4f}')} | {f(r,'val_law','{:+.5f}')} / {f(r,'val_law2','{:+.5f}')} | {f(r,'hold_law','{:+.5f}')} / {f(r,'hold_law2','{:+.5f}')} |")
L += ["", "| model | w 0.05: val | w 0.05: holdout | **w 0.1: val** | **w 0.1: holdout** | w 0.2: val | w 0.2: holdout | w 0.5 (equal mix): val / holdout | rule at 0.1: z val, z holdout, pooled |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "hold_d0.1" in r:
        zv, zh = r['val_d0.1'] / r['val_s0.1'], r['hold_d0.1'] / r['hold_s0.1']; rule = f"{zv:+.2f}, {zh:+.2f}, {(zv+zh)/np.sqrt(2):+.2f} FAIL"
    else: rule = f"{r['val_d0.1']/r['val_s0.1']:+.2f}, n/a, not computable"
    L.append(f"| {r['name'].split(' (')[0]} | {pm(r,'val',0.05)} | {pm(r,'hold',0.05)} | **{pm(r,'val',0.1)}** | **{pm(r,'hold',0.1)}** | {pm(r,'val',0.2)} | {pm(r,'hold',0.2)} | {f(r,'val_d0.5','{:+.5f}')} / {f(r,'hold_d0.5','{:+.5f}')} | {rule} |")
open(OUT / "level.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
