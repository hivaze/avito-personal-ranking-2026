"""Focused users (few categories in the newest 32 events): boost similarity to history.
Small feeds (M <= 30): boost the category's share of the feed. Cross-fitted."""
import hk, numpy as np, itertools

c = hk.Ctx(); r = c.col("rank"); s = c.score
nhc = c.col("n_hist_cats32"); M = c.col("M")
def z(x):
    x = x.astype(float); return np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x), nan=0.0)
feats = {"hmax": z(c.col("hmax")), "h5": z(c.col("h5")), "hmean5": z(c.col("hmean5")), "f1": z(c.col("f1")),
         "ucos": z(c.col("ucos")), "cat_actions": z(np.log1p(c.col("cat_actions"))), "cat_n": z(np.log1p(c.col("cat_n_in_feed")))}
total = 0
print("== focused users: nhc32 <= T, score + w*z(x) [full list]")
for nm, xz in feats.items():
    grid = [(T, w) for T in [1, 2, 3] for w in [0.0, 0.05, 0.1, 0.2, 0.3]]
    def mk(p):
        T, w = p
        return s + w * xz * (nhc <= T)
    m, se, ch, oof, D = c.crossfit(mk, grid); total += len(grid)
    bi = max(grid, key=lambda p: D[p].mean())
    print(f"  {nm:12s} cf={m:+.5f} se={se:.5f} z={m/se if se>0 else 0:+.2f} chosen={ch}  in-sample best {bi} {D[bi].mean():+.5f}")
print("== small feeds: M <= T, score + w*z(x) [full list]")
for nm in ["cat_n", "hmax", "f1", "hmean5"]:
    xz = feats[nm]
    grid = [(T, w) for T in [20, 30, 50] for w in [0.0, 0.05, 0.1, 0.2, 0.3]]
    def mk(p):
        T, w = p
        return s + w * xz * (M <= T)
    m, se, ch, oof, D = c.crossfit(mk, grid); total += len(grid)
    bi = max(grid, key=lambda p: D[p].mean())
    print(f"  {nm:12s} cf={m:+.5f} se={se:.5f} z={m/se if se>0 else 0:+.2f} chosen={ch}  in-sample best {bi} {D[bi].mean():+.5f}")
print("variants", total)
