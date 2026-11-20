"""Per-family single NDCG (holdout to; val to/ft/mix) and holdout/val correlations of family z-means (context)."""
import os, sys, json
sys.argv = ["x", "P11", "none"]
src = open("<repo>/nn_x/stack/hfit/hfit.py").read().split("t0 = time.time()")[0]
exec(compile(src, "<repo>/nn_x/stack/hfit/hfit.py", "exec"))
for i, f in enumerate(FL):
    e = np.zeros(F); e[i] = 1
    h = hold_ndcg_pf(e).mean()
    vto = ndcg_pf(Xv_to[:, i].copy(), yv, offv, Mv).mean(); vft = ndcg_pf(Xv_ft[:, i].copy(), yv, offv, Mv).mean()
    vm = ndcg_pf(val_score(e), yv, offv, Mv).mean()
    print(f"{f:6s} n={int(NF[i])}  hold_to {h:.5f}  val_to {vto:.5f}  val_ft {vft:.5f}  val_mix {vm:.5f}", flush=True)
fidv = np.repeat(np.arange(len(Mv)), Mv)
def corr(X, fid, n):
    # mean per-feed pearson approximated by global pearson of within-feed z (already z-mean) 
    return np.corrcoef(X.T)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("holdout corr (to)\n", corr(Xh.astype(np.float64), None, None))
print("val corr (to)\n", corr(Xv_to, None, None))
