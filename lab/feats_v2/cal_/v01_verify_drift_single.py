"""VERIFIER: drift (val / test-public / test-private / test-weekday / test-weekend) and single-feature NDCG for cal_ features
and the base feature `hour`. Single-feature NDCG uses src/metrics.py (pessimistic ties) with a random tie-break jitter
(mean of 3 draws), both directions."""
import sys
sys.path.insert(0, "<repo>/feats_v2/cal_")
from calcommon import *
from metrics import ndcg

names = sys.argv[1].split(",")
Fv = feeds("val"); Mv = Fv["M"].to_numpy().astype(np.int64); offv = Fv["cand_off"].to_numpy().astype(np.int64)
yv = cand("val", "target", mmap=False).astype(np.float64)
Ft = feeds("test"); fid_t = cand("test", "fidx", mmap=False)
ts_t = Ft["timestamp"].to_numpy()
pub = (ts_t < PUBLIC_END_TS)[fid_t]
we_t = time_parts(ts_t)["weekend"][fid_t].astype(bool)


def psi(a, b, edges):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    pa = np.histogram(a, edges)[0] / max(len(a), 1); pb = np.histogram(b, edges)[0] / max(len(b), 1)
    pa = np.clip(pa, 1e-4, None); pb = np.clip(pb, 1e-4, None)
    return float(((pa - pb) * np.log(pa / pb)).sum())


rng = np.random.default_rng(0)
jit = [rng.random(len(yv)) for _ in range(3)]
rand = np.mean([ndcg(j, yv, offv, Mv) for j in jit])
print("random NDCG (pessimistic metric, random scores)", round(rand, 4))
rows = []
for n in names:
    v = np.load(f"{FEATS}/val/{n}.npy").astype(np.float64)
    t = np.load(f"{FEATS}/test/{n}.npy").astype(np.float64)
    vv = v[~np.isnan(v)]
    q = np.unique(np.quantile(vv, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, q[1:-1], np.inf] if len(q) > 2 else np.array([-np.inf, np.median(vv) - 1e-9, np.median(vv) + 1e-9, np.inf])
    # single-feature NDCG: scale-free tie-break -> rank-transform then add jitter < 1 rank step
    sc = {}
    for sgn, nm in [(1, "pos"), (-1, "neg")]:
        x = np.where(np.isnan(v), -np.inf, sgn * v)
        # ties must share a rank before jitter: use dense rank of unique values
        _, inv = np.unique(x, return_inverse=True)
        sc[nm] = float(np.mean([ndcg(inv + 0.5 * j, yv, offv, Mv) for j in jit]))
    qs = lambda a: [round(float(x), 4) for x in np.nanquantile(a, [0.05, 0.5, 0.95])]
    rows.append(dict(name=n, ndcg_pos=round(sc["pos"], 4), ndcg_neg=round(sc["neg"], 4),
                     psi_val_pub=round(psi(v, t[pub], edges), 3), psi_val_priv=round(psi(v, t[~pub], edges), 3),
                     psi_pub_priv=round(psi(t[pub], t[~pub], edges), 3),
                     psi_val_testwd=round(psi(v, t[~we_t], edges), 3), psi_testwd_testwe=round(psi(t[~we_t], t[we_t], edges), 3),
                     q_val=str(qs(v)), q_pub=str(qs(t[pub])), q_priv=str(qs(t[~pub]))))
df = pl.DataFrame(rows)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
print(df)
df.write_csv(f"{OUT}/verify/v01_drift_single.csv")
