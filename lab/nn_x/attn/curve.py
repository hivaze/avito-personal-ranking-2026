"""Compare per-eval val/holdout curves of treatment logs against the shared baseline logs (kill rule, sec 1.6).
  python curve.py attn_gate [--base_dir <repo>/nnx/optim/logs] [--seeds 23,24]"""
import argparse, re
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("variant"); p.add_argument("--base_dir", default="<repo>/nnx/optim/logs")
p.add_argument("--base", default="base"); p.add_argument("--seeds", default="23,24")
p.add_argument("--dir", default="<repo>/nn_x/attn/logs")
a = p.parse_args()
pat = re.compile(r"^step (\d+)/\d+ loss ([\d.]+) val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)")
def read(f):
    d = {}
    if not Path(f).exists():
        return d
    for line in open(f):
        m = pat.match(line)
        if m:
            d[int(m[1])] = (float(m[2]), float(m[3]), float(m[4]))
    return d
for s in a.seeds.split(","):
    B, T = read(f"{a.base_dir}/{a.base}_s{s}.log"), read(f"{a.dir}/{a.variant}_s{s}.log")
    print(f"seed {s}: step  loss(b/t)  val b -> t (d)  hold b -> t (d)")
    for st in sorted(T):
        if st in B:
            (lb, vb, hb), (lt, vt, ht) = B[st], T[st]
            print(f"  {st:6d} {lb:.4f}/{lt:.4f}  {vb:.5f} -> {vt:.5f} ({vt-vb:+.5f})  {hb:.5f} -> {ht:.5f} ({ht-hb:+.5f})")
        else:
            lt, vt, ht = T[st]
            print(f"  {st:6d} -/{lt:.4f}  val {vt:.5f}  hold {ht:.5f} (no base yet)")
