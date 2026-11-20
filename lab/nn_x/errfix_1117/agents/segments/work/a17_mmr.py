"""Within-category saturation (MMR-like) family. new = score - w * g(cat_rank_in_feed) [* segment]; cross-fitted."""
import fx, kit, numpy as np, polars as pl
df = kit.load(); c = fx.Ctx(df); s = c.base; nd = c.base_nd
cr = df["cat_rank_in_feed"].to_numpy().astype(float)
cn = df["cat_n_in_feed"].to_numpy().astype(float)
M = df["M"].to_numpy().astype(float)
Mf = M[c.off]
rk = df["rank"].to_numpy().astype(float)
nvar = 0
def run(nm, x, grid=(0.0, 0.02, 0.035, 0.05, 0.075, 0.1)):
    global nvar
    nvar += len(grid) - 1
    mean, se, z, ch, D = c.crossfit(lambda w: s - w * x, list(grid))
    segtxt = " | ".join(f"{lab}:" + ",".join(f"{d*1e4:+.1f}" for d in D[:, m].mean(1)[1:])
                        for lab, m in [("M<30", Mf < 30), ("30-100", (Mf >= 30) & (Mf < 100)), ("100+", Mf >= 100)])
    print(f"{nm:26s} ins " + " ".join(f"{d*1e4:+5.1f}" for d in D.mean(1)[1:]) + f" CF {mean*1e4:+.2f} se {se*1e4:.2f} z {z:+.2f} ch {ch}")
    print("        ", segtxt)
    return D
for p in (0.3, 0.5, 0.7, 1.0):
    run(f"(cr-1)^{p}", (cr - 1) ** p)
run("sqrt(cr) M>=30", np.sqrt(cr) * (M >= 30))
run("sqrt(cr-1) M>=30", np.sqrt(cr - 1) * (M >= 30))
run("sqrt(cr-1)*sqrt(M/100)", np.sqrt(cr - 1) * np.sqrt(M / 100))
run("log(cr) M>=30", np.log(cr) * (M >= 30))
# relative to rank: share of higher-ranked items that are the same category
run("(cr-1)/rank", (cr - 1) / rk, grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5))
run("sqrt(cr-1)/sqrt(cn)", np.sqrt(cr - 1) / np.sqrt(cn))
