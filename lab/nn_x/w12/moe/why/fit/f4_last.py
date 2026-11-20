"""why/fit F4 (CPU, read-only): does checkpoint choice (holdout-selected vs @last) or the end of the schedule change the arm deltas?
Member and 4-pool deltas (rank-z mean, paired per feed) on val and holdout, selected vs @last, each against the reference at the same
choice. Output: why/fit/f4_last.txt"""
import sys

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import Meta, ndcg_fast, se  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

R = Path("<repo>")
m = Meta()
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def paths(v, suf):
    B = R / "lab/nn_x/w12/moe/out" if ("moe" in v or "b3ff1" in v) else R / "nn_x/w5"
    return B / "preds" / f"nn_{v}{suf}_trainonly_val.npy", B / "preds/holdout" / f"nn_{v}{suf}_holdout.npy"


def load(v, suf):
    pv, ph = paths(v, suf)
    return np.load(pv).astype(np.float32), np.load(ph).astype(np.float32)


S4 = range(951, 955)
for arm in ("kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1"):
    P(f"== {arm}")
    for suf in ("", "@last"):
        mem = {"val": [], "hold": []}
        zp = {"a": {"val": [], "hold": []}, "r": {"val": [], "hold": []}}
        for s in S4:
            av, ah = load(f"w5_e10_{arm}_s{s}", suf)
            rv, rh = load(f"w5_e10_kd05w192_s{s}", suf)
            mem["val"].append(ndcg_fast(av, m.yv, m.ov, m.Mv) - ndcg_fast(rv, m.yv, m.ov, m.Mv))
            mem["hold"].append(ndcg_fast(ah, m.yh, m.oh, m.Mh) - ndcg_fast(rh, m.yh, m.oh, m.Mh))
            zp["a"]["val"].append(m.z("val", av)); zp["a"]["hold"].append(m.z("hold", ah))
            zp["r"]["val"].append(m.z("val", rv)); zp["r"]["hold"].append(m.z("hold", rh))
        out = []
        for sp, y, o, M in (("val", m.yv, m.ov, m.Mv), ("hold", m.yh, m.oh, m.Mh)):
            d = np.mean(mem[sp], 0)
            dp = ndcg_fast(np.mean(zp["a"][sp], 0), y, o, M) - ndcg_fast(np.mean(zp["r"][sp], 0), y, o, M)
            out.append(f"{sp}: member {d.mean():+.5f}+-{se(d):.5f} 4-pool {dp.mean():+.5f}+-{se(dp):.5f}")
        P(f"   {'selected' if suf == '' else '@last   '} (ref at the same choice): " + " | ".join(out))
open("<repo>/lab/nn_x/w12/moe/why/fit/f4_last.txt", "w").write("\n".join(lines) + "\n")
