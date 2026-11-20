"""staleft (DECLARATION 4b): the W8P20 form on fresh val AND on stale-val, W8 fine-tune branch = 1-seed STALE-FT (tags given) against the 1-seed P = 0
control; popularity-light branch (abl_e10_pop1kd05t25_s951-954) = its own 1-seed P = 0 run in both arms. Needs the 4 pop-light P = 0 runs of jobs_w16ext.txt.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/stale/staleft/p20_stale_read.py [p10 p05]"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import numpy as np
import pilot_read as R
AB = R.ROOT / "nn_x/abl/preds"
W8 = [f"w5_e10_{a}_s{s}" for s in range(951, 955) for a in ("kd05w192", "kd05t25w192")]
P1 = [f"abl_e10_pop1kd05t25_s{s}" for s in range(951, 955)]
zm = lambda fs: np.mean([R.zv(np.load(f)) for f in fs], 0)  # noqa: E731
f = lambda m, t, k: R.PR / f"nn_{m}_sft_{t}_{k}.npy"  # noqa: E731
miss = [m for m in P1 if not f(m, "p0", "val_oof_stale").exists()]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
if miss:
    P("popularity-light P = 0 runs missing:", miss); sys.exit(0)
to = {"fresh": (zm([R.W5 / f"nn_{m}_trainonly_val.npy" for m in W8]), zm([AB / f"nn_{m}_trainonly_val.npy" for m in P1])),
      "stale": (zm([f(m, "p0", "val_e0_stale") for m in W8]), zm([f(m, "p0", "val_e0_stale") for m in P1]))}
key = {"fresh": "val_oof", "stale": "val_oof_stale"}
ftp = {v: zm([f(m, "p0", key[v]) for m in P1]) for v in key}
def final(tag, v, p20=True):
    a = zm([f(m, tag, key[v]) for m in W8])
    if not p20:
        return R.nd(0.5 * R.zv(a) + 0.5 * R.zv(to[v][0]))
    return R.nd(0.5 * R.zv(0.8 * R.zv(a) + 0.2 * R.zv(ftp[v])) + 0.5 * R.zv(0.8 * R.zv(to[v][0]) + 0.2 * R.zv(to[v][1])))
P("W8P20 form, 14,423 val feeds; columns: all | M>=50 | M<50 | pmix-public | pmix-private")
c = {v: final("p0", v) for v in key}; c8 = {v: final("p0", v, False) for v in key}
P(f"control levels: W8P20 form fresh {c['fresh'].mean():.5f} stale {c['stale'].mean():.5f} | W8 form fresh {c8['fresh'].mean():.5f} stale {c8['stale'].mean():.5f}")
P(f"P20 admixture worth with the control branches: fresh {R.row(c['fresh'] - c8['fresh'])}")
P(f"P20 admixture worth with the control branches: stale {R.row(c['stale'] - c8['stale'])}")
for t in (sys.argv[1:] or ["p10", "p05"]):
    if not all(f(m, t, "val_oof_stale").exists() for m in W8):
        continue
    d = {v: final(t, v) - c[v] for v in key}
    for v in key:
        P(f"{t} W8P20 form, FINAL {v:5s} delta vs control: {R.row(d[v])}")
    P(f"{t} W8P20 form, FINAL mix read:              {R.row(0.5 * d['fresh'] + 0.5 * d['stale'])}")
    a, b = R.ms(d["fresh"]), R.ms(d["stale"])
    P(f"{t} RULE on the W8P20 form: fresh {a[0]:+.5f} (z {a[0] / a[1]:+.2f}), stale-val {b[0]:+.5f} (z {b[0] / b[1]:+.2f}) -> {'PASS' if a[0] >= 0 and b[0] >= 2 * b[1] else 'FAIL'}")
open(R.HERE / "out/p20_stale_read.txt", "w").write("\n".join(out) + "\n")
