"""Cold/warm split of the P4 POP-TILT delta (w=-0.02) on val: by all-cold-positive feeds and by feed cold share tercile.
Also the same split for a binary cold offset g + c*cold (the closed 'cold offset' lever) for comparison of the signature."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, ndcg, fsum
dv = load("val"); se = lambda x: x.std() / np.sqrt(len(x))
x = fz(dv, dv["xt_seq_cv_pop"]); g = dv["g"]; base = ndcg(dv, g)
cold = dv["cold"].astype(np.float64); y = dv["y"].astype(np.float64)
npos = fsum(dv, y); ncpos = fsum(dv, y * cold); cshare = fsum(dv, cold) / dv["M"]
has = npos > 0
acp = has & (ncpos == npos); warmpos = has & (ncpos == 0)
print("corr(z pop, cold) %.3f" % np.corrcoef(x, cold)[0, 1])
def split(name, s):
    d = ndcg(dv, s) - base
    q = np.quantile(cshare, [1/3, 2/3]); t = np.digitize(cshare, q)
    print(f"{name:22s} all {d.mean():+.5f}+-{se(d):.5f} | all-cold-pos ({acp.mean():.3f}) {d[acp].mean():+.5f}+-{se(d[acp]):.5f} | "
          f"all-warm-pos ({warmpos.mean():.3f}) {d[warmpos].mean():+.5f}+-{se(d[warmpos]):.5f} | cshare terciles "
          + " ".join(f"{d[t==k].mean():+.5f}+-{se(d[t==k]):.5f}" for k in range(3)))
split("POP-TILT w-0.02", g - 0.02 * x)
split("POP-TILT w-0.03", g - 0.03 * x)
for c in [0.01, 0.02, 0.05]:
    split(f"cold offset +{c}", g + c * fz(dv, cold))
