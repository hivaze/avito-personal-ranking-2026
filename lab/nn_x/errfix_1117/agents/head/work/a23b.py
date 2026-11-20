import hk, numpy as np
c = hk.Ctx(); s = c.score; r = c.col("rank")
for nm in ["it_hist_et1_luo", "it_hist_all_luo"]:
    x = c.col(nm).astype(float)
    grid = [(t, w) for t in [250, 300, 400, 500] for w in [0.0, 0.2, 0.3, 0.5, 0.8]]
    m, se, ch, oof, D = c.crossfit(lambda p: s + p[1] * (x >= p[0]), grid)
    print(f"{nm} viral boost full cf={m:+.5f} se={se:.5f} z={m/se:+.2f} chosen={ch}")
    print("   ", " ".join(f"{p}:{D[p].mean()*1e4:+.1f}" for p in grid))
    # smooth: w * clip(log(x/T),0,..)
    grid2 = [(t, w) for t in [100, 200, 300] for w in [0.0, 0.1, 0.2, 0.4]]
    m, se, ch, oof, D = c.crossfit(lambda p: s + p[1] * np.maximum(np.log((x + 1) / p[0]), 0), grid2)
    print(f"{nm} hinge-log full cf={m:+.5f} se={se:.5f} z={m/se:+.2f} chosen={ch}")
    print("   ", " ".join(f"{p}:{D[p].mean()*1e4:+.1f}" for p in grid2))
