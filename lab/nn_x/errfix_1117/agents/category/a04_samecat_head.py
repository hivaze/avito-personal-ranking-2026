"""Rule family: boost same-category-as-newest items inside the model's head."""
import numpy as np
import cu

C = cu.Ctx()
s, rk = C.s, C.col("rank")
same = C.col("same_cat_newest").astype(float)
nhc = C.col("n_hist_cats32")
ws = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
for K in (2, 3, 5, 10, 300):
    grid = ws
    C.crossfit(grid, lambda w: s + w * same * (rk <= K), label=f"same_cat_newest head K={K}")
for h in (2, 3, 4):
    for K in (3, 5, 300):
        C.crossfit(ws, lambda w: s + w * same * (rk <= K) * (nhc <= h), label=f"same x nhc<={h} K={K}")
# demote non-same in head when user is diverse
for K in (3, 5):
    C.crossfit(ws, lambda w: s - w * (1 - same) * (rk <= K) * (nhc >= 5), label=f"-nonsame x nhc>=5 K={K}")
print("variants tried", C.n_tried)
