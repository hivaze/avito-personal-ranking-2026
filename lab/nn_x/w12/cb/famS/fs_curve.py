"""Train-side selection curve (click NDCG of the 13,527 selection feeds; no val label) of famS arms.
usage: python fs_curve.py <tag> [<tag> ...] [--at 100,200,...] [--dir preds]"""
import sys, json
from pathlib import Path
H = Path("<repo>/nn_x/w12/cb/famS")
a = sys.argv[1:]; at = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500, 3000]; d = "preds"
if "--at" in a:
    i = a.index("--at"); at = [int(v) for v in a[i + 1].split(",")]; a = a[:i] + a[i + 2:]
if "--dir" in a:
    i = a.index("--dir"); d = a[i + 1]; a = a[:i] + a[i + 2:]
for t in a:
    c = json.load(open(H / d / f"{t}.curve.json"))
    v = {it: m.get("validation:NDCG:type=Base") for it, s, m in c if m}
    sec = {it: s for it, s, m in c}
    if not v:
        print(t, "no metrics yet"); continue
    b = max(v, key=v.get); last = max(v)
    print(f"{t}: {len(c)} iters, {sec[last] / 60:.1f} min ({sec[last] / max(last, 1):.2f} s/iter) | best sel NDCG {v[b]:.5f} at iteration {b} | " +
          " ".join(f"{n}:{v[n]:.5f}" for n in at if n in v) + f" | last {last}:{v[last]:.5f}")
