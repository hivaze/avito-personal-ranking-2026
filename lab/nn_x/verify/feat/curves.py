"""Per-eval deltas of feat runs vs the shared baseline logs (kill rule, design 1.6).  $PY curves.py feat_ple [feat_prop ...]"""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BL = Path("<repo>/nnx/optim/logs")
def curve(p):
    if not p.exists(): return {}
    t = p.read_text().split("==== ")[-1]
    return {int(a): (float(b), float(c)) for a, b, c in re.findall(r"step (\d+)/\d+ loss \S+ val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)", t)}
for exp in sys.argv[1:]:
    for s in (23, 24, 25):
        c, b = curve(HERE / "logs" / f"{exp}_s{s}.log"), curve(BL / f"base_s{s}.log")
        if not c: continue
        row = []
        for st in sorted(c):
            if st in b:
                row.append(f"{st//1000}k:{(c[st][0]-b[st][0])*1e4:+.0f}/{(c[st][1]-b[st][1])*1e4:+.0f}")
            else:
                row.append(f"{st//1000}k:{c[st][0]:.4f}/{c[st][1]:.4f}(nobase)")
        print(f"{exp}_s{s} (dval/dhold x1e-4): " + " ".join(row))
