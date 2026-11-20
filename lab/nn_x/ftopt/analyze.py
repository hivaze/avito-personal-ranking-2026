"""Post-hoc TUNE-only diagnostics per trial (member-aligned by name): single-member mix deltas, FT-alone member NDCG.
Usage: python analyze.py   -> analyze_tune.csv (CONFIRM is not touched here)."""
import csv
import json

import numpy as np

from ftlib import FT_DIR, SET_A, Val, load_base_ft, load_to

v = Val()
rows = [json.loads(l) for l in open(FT_DIR / "trials.jsonl")]
to = {m: load_to(m) for m in SET_A}
base = {m: load_base_ft(m) for m in SET_A}
bmix = {m: v.nd(0.5 * v.z(base[m]) + 0.5 * v.z(to[m])) for m in SET_A}
bft = {m: v.nd(base[m]) for m in SET_A}
out = []
for r in sorted(rows, key=lambda r: r["trial"]):
    t = r["trial"]
    rec = dict(trial=t, value=round(r["value"], 6))
    for m in SET_A:
        f = np.load(FT_DIR / "out" / f"t{t}" / f"nn_{m}_t{t}_val_oof.npy").astype(np.float64)
        rec[f"mix_{m}"] = round(v.stat(v.nd(0.5 * v.z(f) + 0.5 * v.z(to[m])) - bmix[m], v.tune)[0], 6)
        rec[f"ft_{m}"] = round(v.stat(v.nd(f) - bft[m], v.tune)[0], 6)
    out.append(rec)
    print(" ".join(f"{k}={val}" if isinstance(val, int) else f"{val:+.5f}" for k, val in rec.items()))
with open(FT_DIR / "analyze_tune.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print("cols: trial value | mix r101 ft r101 | mix r103 ft r103 | mix gru ft gru | mix lr1 ft lr1 (TUNE, vs n2cpv3)")
