"""P2b: where the forward f5 echo (LEAK CLASS) carries its information: positive rate and model rank of val rows by the sign of
f5(next later appearance in another user's feed) - f5(now) and by the time to that appearance."""
import sys, json; sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *
NAMES = json.load(open(f"{OUT}/out/p2_names.json"))
v = Split("val", hist=False); s = v.best(); y = v.y
X = np.load(f"{OUT}/out/p2_val.npy"); col = {n: X[:, j].astype(np.float64) for j, n in enumerate(NAMES)}
f5 = np.load(f"{FLAT}/val/cand_f5.npy").astype(np.float64)
d = col["f5_next"] - f5; dt = col["dt_next"]
# within-feed rank of the base score (0 = top)
o = np.lexsort((-s, v.fid)); rk = np.empty(v.N); rk[o] = np.arange(v.N) - np.repeat(v.off, v.M)
rel = rk / np.maximum(v.M[v.fid] - 1, 1)
print(f"all rows: pos rate {y.mean():.4f}")
for lab, m0 in [("no later appearance", ~np.isfinite(d)), ("later, f5 up", d > 1e-6), ("later, f5 equal", np.abs(d) <= 1e-6), ("later, f5 down", d < -1e-6)]:
    print(f"{lab:22s} rows {m0.mean():.4f} pos rate {y[m0].mean():.4f} mean rel rank {rel[m0].mean():.3f}")
for lo, hi in [(0, 6), (6, 24), (24, 72), (72, 1e9)]:
    m0 = (dt >= lo) & (dt < hi)
    for lab, mm in [("up", d > 1e-6), ("equal", np.abs(d) <= 1e-6), ("down", d < -1e-6)]:
        q = m0 & mm
        if q.sum():
            print(f"  dt {lo:>4}-{hi:<6} {lab:6s} rows {q.sum():7d} pos rate {y[q].mean():.4f}  positives {int(y[q].sum()):5d}  median rel rank of positives {np.median(rel[q & (y>0)]) if (q & (y>0)).any() else np.nan:.3f}")
# small-exposure items: few later impressions before the next appearance can only come from this user
for lab, m0 in [("CE+CL<=2 (rare item), up", (d > 1e-6) & (col["CE"] + col["CL"] <= 2) & (col["n_tr"] == 0)),
                ("n_tr==0 & up", (d > 1e-6) & (col["n_tr"] == 0)), ("n_tr>0 & up", (d > 1e-6) & (col["n_tr"] > 0))]:
    print(f"{lab:28s} rows {m0.sum():7d} pos rate {y[m0].mean():.4f}")
# oracle-style overlay size: lift 'up & dt<24h' rows to the top, keep order otherwise
for thr in (6, 24, 72):
    flag = (d > 1e-6) & (dt < thr)
    sc = s + 100 * flag
    print(f"overlay up & dt<{thr}h: flagged rows {flag.sum()} pos rate {y[flag].mean():.4f}  dNDCG {(v.ndcg(sc) - v.ndcg(s)).mean():+.5f}")
