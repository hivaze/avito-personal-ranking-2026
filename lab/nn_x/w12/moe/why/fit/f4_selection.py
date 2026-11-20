"""why/fit F4b (CPU, read-only): how much of each arm's screen delta is the holdout-based checkpoint choice?
Per arm: 4-pool (rank-z mean) val / holdout delta vs the reference with the selected checkpoints (= the screen read) and with @last, and
the paired per-feed SE of the difference between the two deltas. Output: why/fit/f4_selection.txt"""
import json
import sys

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import Meta, ndcg_fast, se  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

R = Path("<repo>")
m = Meta()
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def base(v):
    return R / "lab/nn_x/w12/moe/out" if ("moe" in v or "b3ff1" in v) else R / "nn_x/w5"


def z(v, suf, sp):
    p = base(v) / "preds" / f"nn_{v}{suf}_trainonly_val.npy" if sp == "val" else base(v) / "preds/holdout" / f"nn_{v}{suf}_holdout.npy"
    return m.z(sp, np.load(p).astype(np.float32))


def best(v):
    return json.load(open(base(v) / "logs" / f"{v}.json"))["best_step"]


S4 = range(951, 955)
P("selected steps: ref " + str([best(f"w5_e10_kd05w192_s{s}") for s in S4]))
for arm in ("kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1"):
    P(f"== {arm}: selected steps {[best(f'w5_e10_{arm}_s{s}') for s in S4]}")
    for sp, y, o, M in (("val", m.yv, m.ov, m.Mv), ("hold", m.yh, m.oh, m.Mh)):
        d = {}
        for suf in ("", "@last"):
            na = ndcg_fast(np.mean([z(f"w5_e10_{arm}_s{s}", suf, sp) for s in S4], 0), y, o, M)
            nr = ndcg_fast(np.mean([z(f"w5_e10_kd05w192_s{s}", suf, sp) for s in S4], 0), y, o, M)
            d[suf] = na - nr
        dd = d["@last"] - d[""]
        P(f"   {sp:4s}: 4-pool delta selected {d[''].mean():+.5f}+-{se(d['']):.5f} | @last {d['@last'].mean():+.5f}+-{se(d['@last']):.5f} | "
          f"@last - selected {dd.mean():+.5f}+-{se(dd):.5f}")
open("<repo>/lab/nn_x/w12/moe/why/fit/f4_selection.txt", "w").write("\n".join(lines) + "\n")
