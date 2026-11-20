"""staleft_review: P=1.0 final-file delta (fresh, stale-val) for EVERY member subset of size k of the 8 W8 members: is the pooled-W8 number
(+0.00054 stale) a pool-size effect or a draw? CPU only. final = 0.5 z(FT pool) + 0.5 z(TO pool), pools over the same subset."""
import os, sys, itertools
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft_review")
import numpy as np
import rederive as R
V, P = R.V, R.P
R.lines.clear()
W8 = R.W8
Z = {}
for i, m in enumerate(W8):
    Z[i] = {"p0f": V.z(R.ld(m, "p0", "val_oof")), "p0s": V.z(R.ld(m, "p0", "val_oof_stale")), "p1f": V.z(R.ld(m, "p10", "val_oof")), "p1s": V.z(R.ld(m, "p10", "val_oof_stale")),
            "p5f": V.z(R.ld(m, "p05", "val_oof")), "p5s": V.z(R.ld(m, "p05", "val_oof_stale")),
            "tof": V.z(np.load(R.W5 / f"nn_{m}_trainonly_val.npy")), "tos": V.z(R.ld(m, "p0", "val_e0_stale"))}
def pool(S, k):
    return np.mean([Z[i][k] for i in S], 0)
def deltas(S):
    tf, ts = pool(S, "tof"), pool(S, "tos")
    c_f = V.ndcg(R.fin(pool(S, "p0f"), tf)); c_s = V.ndcg(R.fin(pool(S, "p0s"), ts))
    out = {}
    for t in ("p1", "p5"):
        out[t] = (V.ndcg(R.fin(pool(S, t + "f"), tf)) - c_f, V.ndcg(R.fin(pool(S, t + "s"), ts)) - c_s)
    return out
P("P=1.0 / P=0.5 final-file delta vs the P=0 control by pool size k (all subsets of the 8 W8 members); mean over subsets, SD over subsets, range; per-feed SE of one subset is 0.00025-0.00028")
for k in (1, 2, 4, 6, 7, 8):
    subs = list(itertools.combinations(range(8), k))
    res = [deltas(S) for S in subs]
    for t, lab in (("p1", "P=1.0"), ("p5", "P=0.5")):
        f = np.array([r[t][0].mean() for r in res]); s = np.array([r[t][1].mean() for r in res])
        sp = np.array([R.ms(r[t][1], R.wpriv)[0] for r in res])
        z2 = np.mean([r[t][1].mean() >= 2 * r[t][1].std(ddof=1) / np.sqrt(V.n) and r[t][0].mean() >= 0 for r in res])
        P(f"k={k} ({len(subs):2d} subsets) {lab}: fresh mean {f.mean():+.5f} SD {f.std():.5f} [{f.min():+.5f}, {f.max():+.5f}] | stale mean {s.mean():+.5f} SD {s.std():.5f} [{s.min():+.5f}, {s.max():+.5f}] "
          f"| stale pmix-priv mean {sp.mean():+.5f} | share of subsets passing the lead's rule {z2:.2f} | share with stale delta > 0: {np.mean(s > 0):.2f}")
(R.OUT / "poolsize.txt").write_text("\n".join(R.lines) + "\n")
