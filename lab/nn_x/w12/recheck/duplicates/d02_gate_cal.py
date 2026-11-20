"""GATE-CAL (skeptic 'duplicates', new check): calibrate the residual gate with REAL positive controls.
The project's main closing instrument is 'nested residual on top of the pool' (270 linear reads, CCB-1113, DATA_PIPELINE, AUTOFE_1114,
forensics_skeptic).  PRETRAIN_PLAN_1113 l.312 notes it is 'not validated' (pre2 fails it at -0.00018, yet is worth ~+0.001 as an NN input).
Here: for an input block B that the ranker really uses, take the stored 10-epoch members trained WITHOUT B (abl_e10_xdrop_<B>) and their
with-B twins (w5_e10_base, same seeds).  TRUE pool-level NN-input value = NDCG(pool twins) - NDCG(pool ablated).
RECOVERED = what the gate finds when B's columns are offered as a residual on top of the ablated pool:
  (i) linear nested, best single column (old protocol);  (ii) monotone LightGBM residual of forensics_skeptic/hx.py (block + per-feed z).
Recovered / true = the gate's transfer factor for a real NN input.   Usage: d02_gate_cal.py val|hold pop|pre2
CPU only, 12 threads, read-only on predictions and features."""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/recheck/forensics_skeptic")
from hx import *
which, blk = sys.argv[1], sys.argv[2]
A = R / "nn_x/abl/preds"
BLK = dict(pop=("xdrop_pop", (951, 952, 953, 954), "seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo".split(",")),
           pre2=("xdrop_pre2", (951, 952), "pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32".split(",")))
arm, seeds, cols = BLK[blk]
D = load_val() if which == "val" else load_hold(); y, M, off = D["y"], D["M"], D["off"]
def pool(files): return zfeed(np.mean([zfeed(np.load(f), off, M) for f in files], 0), off, M)
if which == "val":
    abl = pool([A / f"nn_abl_e10_{arm}_s{s}_trainonly_val.npy" for s in seeds]); twin = pool([W / f"nn_w5_e10_base_s{s}_trainonly_val.npy" for s in seeds])
else:
    abl = pool([A / f"holdout/nn_abl_e10_{arm}_s{s}_holdout.npy" for s in seeds]); twin = pool([W / f"holdout/nn_w5_e10_base_s{s}_holdout.npy" for s in seeds])
e_abl = ndcg_per_feed(abl, y, off, M); e_twin = ndcg_per_feed(twin, y, off, M); true = e_twin - e_abl
print(f"== GATE-CAL {which} / block {blk} ({arm}, seeds {seeds}): ablated pool {e_abl.mean():.5f}, with-input twin pool {e_twin.mean():.5f}")
print(f"   TRUE pool-level NN-input value of the block: {fmt(true)} (z {true.mean() / se(true):+.1f})", flush=True)
D["base"] = abl
src = FEATS / ("val" if which == "val" else "train"); X = []; names = []
for n in cols:
    p = src / f"{n}.npy"
    if not p.exists(): print("   (column not in the feature store, skipped:", n, ")"); continue
    x = np.load(p, mmap_mode="r"); x = np.nan_to_num(np.asarray(x if D["rows"] is None else x[D["rows"]], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X.append(x); names.append(n); X.append(zfeed(x, off, M).astype(np.float32)); names.append(n + "_fz")
# (i) linear nested, per column (per-feed z of the column), weight chosen on the other user half
best = None
for n, x in zip(names, X):
    if not n.endswith("_fz"): continue
    d, wc = nested_lin(D, x.astype(np.float64)); print(f"   linear nested {n:22s} {fmt(d)}  w {wc}")
    if best is None or d.mean() > best[1].mean(): best = (n, d)
print(f"   (i) LINEAR NESTED best single column (post hoc pick, optimistic): {best[0]} {fmt(best[1])} = {best[1].mean() / true.mean():.2f} of true", flush=True)
# (ii) monotone GBDT residual, block + context
ctx = ctx_cols(D); Xm = np.stack([ctx[k] for k in ctx] + X, 1).astype(np.float32); nm = list(ctx) + names
d0, _, _ = resid_cv(D, np.stack([ctx[k] for k in ctx], 1).astype(np.float32), list(ctx), f"{blk}-control(no signal)")
d1, det, _ = resid_cv(D, Xm, nm, f"{blk}-block")
print(f"   (ii) MONOTONE GBDT RESIDUAL: control {fmt(d0)} | block {fmt(d1)} = {d1.mean() / true.mean():.2f} of true ({fmt(true)})")
