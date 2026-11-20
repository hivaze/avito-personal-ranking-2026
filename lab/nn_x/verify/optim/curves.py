"""Per-eval curve comparison against the baseline log at the same step (kill rule, design doc 1.6).
Usage: python curves.py optim_muon_s23 [base_s23]"""
import re, sys
from pathlib import Path
L = Path(__file__).resolve().parent / "logs"
def parse(v):
    out = {}
    for line in (L / f"{v}.log").read_text().splitlines():
        m = re.match(r"step (\d+)/\d+ .*val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)", line)
        if m:
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out
t = sys.argv[1]
b = sys.argv[2] if len(sys.argv) > 2 else "base_s" + t.rsplit("_s", 1)[1]
T, B = parse(t), parse(b)
print(f"{t} vs {b}")
for s in sorted(T):
    if s in B:
        print(f"step {s:6d}  val {T[s][0]:.5f} ({T[s][0] - B[s][0]:+.5f})  hold {T[s][1]:.5f} ({T[s][1] - B[s][1]:+.5f})")
    else:
        print(f"step {s:6d}  val {T[s][0]:.5f}  hold {T[s][1]:.5f}")
