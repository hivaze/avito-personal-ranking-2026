import hk, numpy as np
c = hk.Ctx(); s = c.score; x = c.col("it_hist_et1_luo").astype(float); r = c.col("rank")
grid = [(t, w) for t in [200, 300, 400] for w in [0.0, 0.05, 0.1, 0.2, 0.3]]
m, se, ch, oof, D = c.crossfit(lambda p: s + p[1] * (x >= p[0]), grid, verbose=True)
print(f"viral boost full cf={m:+.5f} se={se:.5f} z={m/se:+.2f} chosen={ch}")
m, se, ch, oof, D = c.crossfit(lambda p: s + p[1] * (x >= p[0]) * (r <= 30), grid)
print(f"viral boost head30 cf={m:+.5f} se={se:.5f} z={m/se:+.2f} chosen={ch}")
