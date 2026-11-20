"""Re-weight s_ft vs s_to inside the head (top K by score); the head set stays fixed."""
import hk, numpy as np

c = hk.Ctx()
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float); s = c.score; r = c.col("rank")


def head_reweight(K, w):
    inh = r <= K
    new = s.copy()
    new[inh] = 100.0 + (w * a[inh] + (1 - w) * b[inh])
    return new + 1e-9 * s


for K in [2, 3, 5, 10, 20, 40, 300]:
    line = []
    for w in [0.0, 0.2, 0.3, 0.4, 0.6]:
        m, se, _ = c.delta(head_reweight(K, w))
        line.append(f"w={w}:{m:+.5f}({m/se:+.1f})")
    print("K", K, " ".join(line))
