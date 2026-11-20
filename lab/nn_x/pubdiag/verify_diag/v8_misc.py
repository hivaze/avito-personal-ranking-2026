"""7-mix F7x-F7b on holdout by weekend; F7x-F7 TO member gain by M>=50 / M<50 on val; pool-level val-TO vs holdout gaps."""
from v_lib import *
ftr, Mtr, otr = feed_meta("train")
pr = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mtr < 50] = 0.0
cand = np.where((pr > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
tsh = ftr["timestamp"].to_numpy()[hold]; dow = ((tsh + 10800) // 86400 + 3) % 7; wk = dow >= 5
d = pf("holdout", pool_half(["F7x"], "holdout", "to")) - pf("holdout", pool_half(["F7b"], "holdout", "to"))
print(f"7-mix F7x-F7b holdout: weekday {d[~wk].mean():+.5f}+-{se_iid(d[~wk]):.5f} weekend {d[wk].mean():+.5f}+-{se_iid(d[wk]):.5f} all {d.mean():+.5f}")
fv, Mv, ov = meta("val"); g = groups_val()
dm = np.mean([pf("val", np.load(member_file(a, "val", "to"))) - pf("val", np.load(member_file(b, "val", "to"))) for a, b in zip(FAM["F7x"], FAM["F7"])], 0)
big = Mv >= 50
print(f"F7x-F7 member TO val: M>=50 {dm[big].mean():+.5f}  M<50 {dm[~big].mean():+.5f}")
# pool-level val-TO-half vs holdout gap vs Cd
for k in ["PF7x", "G3", "M2"]:
    dv = pf("val", pool_half(POOLS[k], "val", "to")) - pf("val", pool_half(POOLS["Cd"], "val", "to"))
    dh = pf("holdout", pool_half(POOLS[k], "holdout", "to")) - pf("holdout", pool_half(POOLS["Cd"], "holdout", "to"))
    print(f"{k}-Cd TO: val {dv.mean():+.5f} (M>=50 {dv[big].mean():+.5f}) holdout {dh.mean():+.5f}  gap(valM>=50 - ho) {dv[big].mean()-dh.mean():+.5f}")
