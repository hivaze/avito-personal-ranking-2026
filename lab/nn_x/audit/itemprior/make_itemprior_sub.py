"""Write preds/ens_G3_itemprior_{val,test}.npy and subs/cand_G3_itemprior.parquet for an ACCEPTED prior (plan.txt).
Usage: python make_itemprior_sub.py <source: V|U>   (settings from summary.json: full-val OOF argmax)
val = z(G3) + beta * x_OOF (prior from the 4 other FT folds; same-user rows never in the source);
test = z(G3) + beta * x_test (prior from all val rows, leave-user-out of the test feed's user)."""
import sys, json
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub

OUT = "<repo>/nn_x/audit/itemprior"; P = "<repo>/preds"
src = sys.argv[1]
S = json.load(open(f"{OUT}/summary.json"))
fv, Mv, ov = feed_meta("val"); ft, Mt, ot = feed_meta("test")
zv = feed_rank_z(ov, Mv.astype(np.int64), np.load(f"{P}/ens_G3_final_val.npy").astype(np.float32))[:, 1].astype(np.float64)
zt = feed_rank_z(ot, Mt.astype(np.int64), np.load(f"{P}/ens_G3_final_test.npy").astype(np.float32))[:, 1].astype(np.float64)
xv_V = np.load(f"{OUT}/x_val_oof.npy").astype(np.float64); xt_V = np.load(f"{OUT}/x_test_V.npy").astype(np.float64)
if src == "V":
    a, b = S["V"]["full_best"]
    assert a == S["V"]["full_best"][0]
    sv, st = zv + b * xv_V, zt + b * xt_V
    print(f"V: a {a} beta {b}")
elif src == "U":
    b1, b2 = S["U"]["full_best"]
    xv_T = np.load(f"{OUT}/x_val_T.npy").astype(np.float64); xt_T = np.load(f"{OUT}/x_test_T.npy").astype(np.float64)
    sv, st = zv + b1 * xv_V + b2 * xv_T, zt + b1 * xt_V + b2 * xt_T
    print(f"U: b1 {b1} b2 {b2}")
else:
    raise SystemExit("source V or U")
assert np.isfinite(sv).all() and np.isfinite(st).all()
np.save(f"{P}/ens_G3_itemprior_val.npy", sv.astype(np.float32)); np.save(f"{P}/ens_G3_itemprior_test.npy", st.astype(np.float32))
print(make_sub(st.astype(np.float32), "<repo>/subs/cand_G3_itemprior.parquet"))
