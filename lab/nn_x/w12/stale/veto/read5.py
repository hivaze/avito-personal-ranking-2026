"""veto read 5: systematic screen of every stored 10-epoch family in nn_x/w5/preds against its same-seed reference, for the pattern
"val >= 0 (or mildly negative) with pmix-private >= +1 SE while the holdout is <= 0". Information; rules of PREDECLARED.txt.
Output: logs/read5.txt + out/read5.json
"""
import json
import re
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
W = ROOT / "nn_x/w5/preds"
fam = {}
for p in sorted(W.glob("nn_w5_e10_*_trainonly_val.npy")):
    m = re.match(r"nn_(w5_e10_(.+)_s(\d+))_trainonly_val\.npy", p.name)
    if m and "@" not in m.group(1):
        fam.setdefault(m.group(2), []).append(int(m.group(3)))
HARD = {"ap1", "bs4", "bs4lr4", "e15", "e20", "e30", "pseudo", "pseudo_xpre2", "trunc", "trunc2", "xf0", "xf1", "xf2", "xf3", "xf4"}
D192 = {a for a in fam if "w192" in a or "w224" in a or "w256" in a or a.startswith(("kw", "kx", "u20", "kdmix"))}
R = []
print("arm | n | ref | val | pmix-priv | coldpos | warmpos | holdout | holdout >= Sep 13 | flag")
for a, seeds in sorted(fam.items()):
    if a in ("base", "kd05", "kd05w192"): continue
    ref = "base" if a in HARD else ("kd05w192" if a in D192 else "kd05")
    sd = [s for s in seeds if find(f"w5_e10_{ref}_s{s}", "val") is not None and find(f"w5_e10_{a}_s{s}", "hold") is not None]
    if not sd: continue
    A = [f"w5_e10_{a}_s{s}" for s in sd]; B = [f"w5_e10_{ref}_s{s}" for s in sd]
    try:
        r = read(c, f"{a} vs {ref}", A, B, kinds=("val", "hold"))
    except FileNotFoundError:
        continue
    v, pp, h = r["val"], r["pmix_priv"], r.get("hold", (float("nan"), 1))
    flag = ""
    if pp[0] >= pp[1] and v[0] >= -v[1] and h[0] <= 0: flag = "PATTERN (pmix-priv >= +1 SE, val >= -1 SE, holdout <= 0)"
    elif pp[0] >= pp[1] and v[0] >= 0: flag = "pmix-priv >= +1 SE, val >= 0, holdout > 0"
    r["flag"] = flag; R.append(r)
    print(f"{a} | {len(sd)} | {ref} | {f(v)} | {f(pp)} | {f(r['coldpos'])} | {f(r['warmpos'])} | {f(h)} | {f(r.get('hold_late>=Sep13'))} | {flag}")
json.dump(R, open(HERE / "out/read5.json", "w"), indent=1, default=float)
