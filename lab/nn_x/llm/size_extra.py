"""Information-only size-curve extras (reports/llm_large.md R5; not decision rules):
(1) paired val / holdout differences between LLM sizes at 2 epochs, pooled over seeds 23/24 (same subsets);
(2) mean within-feed z-score correlation rho (compare.rho) between all train-only arms on val."""
import numpy as np
import compare as C
from pathlib import Path
root = C.HERE
val, ho = C.load_ctx(root)
S = {"val": val, "hold": ho}
arms = {"B23": ("llmcmp_base_s23", "B"), "B24": ("llmcmp_base_s24", "B"), "R23": ("llm06rand_L1_s23", "L"),
        "L06_23": ("llm06_L1_s23", "L"), "L06_24": ("llm06_L1_s24", "L"), "L06e4_23": ("llm06_L1_e4_s23", "L"),
        "L06e4_24": ("llm06_L1_e4_s24", "L"), "Q4_23": ("llmq4_L1_s23", "L"), "Q4_24": ("llmq4_L1_s24", "L"),
        "Q9_23": ("llmq9_L1_s23", "L"), "Q9_24": ("llmq9_L1_s24", "L")}
sc, nd, z = {}, {}, {}
for k, (n, kind) in arms.items():
    pv, ph = C.paths(root, n, kind)
    for sp, p in (("val", pv), ("hold", ph)):
        sc[k, sp] = np.load(p)
        nd[k, sp] = C.pf(sc[k, sp], S[sp]["y"], S[sp]["off"], S[sp]["M"])
        z[k, sp] = C.D.feed_z(sc[k, sp], S[sp]["M"])
print("(1) paired differences between sizes (pooled over seeds 23/24, same subsets), mean +- SE")
for a, b in (("Q4", "L06"), ("Q9", "L06"), ("Q9", "Q4"), ("L06e4", "L06")):
    out = []
    for sp in ("val", "hold"):
        d = np.mean([nd[f"{a}_{s}", sp] - nd[f"{b}_{s}", sp] for s in (23, 24)], 0)
        out.append(f"{sp} {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}")
    print(f"  {a} - {b}: " + "; ".join(out))
print("(2) rho on val (mean within-feed z product)")
keys = list(arms)
print("          " + " ".join(f"{k:>8}" for k in keys))
for i in keys:
    print(f"{i:>9} " + " ".join(f"{C.rho(z[i, 'val'], z[j, 'val'], val['M']):8.4f}" for j in keys))
