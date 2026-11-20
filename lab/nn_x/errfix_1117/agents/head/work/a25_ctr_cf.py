import hk, numpy as np
c = hk.Ctx(); s = c.score; r = c.col("rank")
f5 = c.col("f5").astype(float); xp = c.col("xp_imp_c50").astype(float)
zf = (f5 - f5.mean()) / f5.std()
for nm, K in [("full", 300), ("head10", 10), ("head30", 30)]:
    grid = [(t, w) for t in [3.0, 3.9] for w in [0.0, 0.03, 0.06, 0.1, 0.15]]
    m, se, ch, oof, D = c.crossfit(lambda p: s + p[1] * zf * (xp >= p[0]) * (r <= K), grid)
    print(f"{nm} f5|exposed cf={m:+.5f} se={se:.5f} z={m/se if se else 0:+.2f} chosen={ch}")
    print("   ", " ".join(f"{p}:{D[p].mean()*1e4:+.1f}" for p in grid))
