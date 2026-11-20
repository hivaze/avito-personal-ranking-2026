"""Head demotion of items whose category is not recent (cat_newest_idx 5-31, 32+, absent)."""
import numpy as np
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
rk, cni, ur = c("rank"), c("cat_newest_idx"), c("cat_user_rank")
ws = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
for K in (3, 5, 10):
    C.crossfit(ws, lambda w: s - w * ((rk <= K) & (cni >= 5) & (cni < 32)), label=f"head K={K} demote idx5-31")
    C.crossfit(ws, lambda w: s - w * ((rk <= K) & (cni >= 5)), label=f"head K={K} demote idx>=5 or absent... (cni>=5)")
    C.crossfit(ws, lambda w: s - w * ((rk <= K) & ((cni >= 5) | (ur == 0))), label=f"head K={K} demote not-recent incl absent")
print("variants", C.n_tried)
