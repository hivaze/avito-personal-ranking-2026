"""Directional near-duplicate penalty: same category follower x near-twin indicators."""
import fx, kit, numpy as np
df = kit.load(); c = fx.Ctx(df); s = c.base
cr = df["cat_rank_in_feed"].to_numpy().astype(float)
cmx = np.nan_to_num(df["cand_maxcos"].to_numpy().astype(float), nan=0.0)
nd95 = df["n_cand_dup95"].to_numpy().astype(float)
M = df["M"].to_numpy().astype(float); Mf = M[c.off]
nvar = 0
def run(nm, x, grid):
    global nvar
    nvar += len(grid) - 1
    mean, se, z, ch, D = c.crossfit(lambda w: s - w * x, list(grid))
    segtxt = " | ".join(f"{lab}:" + ",".join(f"{d*1e4:+.1f}" for d in D[:, m].mean(1)[1:])
                        for lab, m in [("M<30", Mf < 30), ("30-100", (Mf >= 30) & (Mf < 100)), ("100+", Mf >= 100)])
    print(f"{nm:28s} ins " + " ".join(f"{d*1e4:+5.1f}" for d in D.mean(1)[1:]) + f" CF {mean*1e4:+.2f} se {se*1e4:.2f} z {z:+.2f} ch {ch}")
    print("        ", segtxt, flush=True)
g = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
run("[cr>=2]*[nd95>=1]", ((cr >= 2) & (nd95 >= 1)).astype(float), g)
run("[cr>=2]*cmx^4", (cr >= 2) * cmx ** 4, g)
run("[cr>=2]*[cmx<0.9]", ((cr >= 2) & (cmx < 0.9)).astype(float), g)
run("[nd95>=1]", (nd95 >= 1).astype(float), g)
print("n variants", nvar, "feeds by M seg", (Mf < 30).sum(), ((Mf >= 30) & (Mf < 100)).sum(), (Mf >= 100).sum())
