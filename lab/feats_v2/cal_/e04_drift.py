"""Drift check for cal_ features: PSI (10 bins from val quantiles) and quantiles for val vs test-public vs test-private
(and train random sample vs val). Also per-day means."""
from calcommon import *
import glob, sys

names = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{FEATS}/val/cal_*.npy"))
Ft = feeds("test"); fid_t = cand("test", "fidx", mmap=False)
pub = (Ft["timestamp"].to_numpy() < PUBLIC_END_TS)[fid_t]
tr_rows = np.sort(np.random.default_rng(0).choice(len(cand("train", "fidx")), 3_000_000, replace=False))


def psi(a, b, edges):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    pa = np.histogram(a, edges)[0] / max(len(a), 1); pb = np.histogram(b, edges)[0] / max(len(b), 1)
    pa = np.clip(pa, 1e-4, None); pb = np.clip(pb, 1e-4, None)
    return float(((pa - pb) * np.log(pa / pb)).sum())


rows = []
for n in names:
    v = np.load(f"{FEATS}/val/{n}.npy").astype(np.float64)
    t = np.load(f"{FEATS}/test/{n}.npy").astype(np.float64)
    tr = np.asarray(np.load(f"{FEATS}/train/{n}.npy", mmap_mode="r")[tr_rows], np.float64)
    vv = v[~np.isnan(v)]
    q = np.unique(np.quantile(vv, np.linspace(0, 1, 11)))
    edges = np.r_[-np.inf, q[1:-1], np.inf] if len(q) > 2 else np.array([-np.inf, np.median(vv) - 1e-9, np.median(vv) + 1e-9, np.inf])
    qs = lambda a: [round(float(x), 4) for x in np.nanquantile(a, [0.05, 0.5, 0.95])] if np.isfinite(a).any() else None
    rows.append(dict(name=n, psi_val_pub=psi(v, t[pub], edges), psi_val_priv=psi(v, t[~pub], edges), psi_pub_priv=psi(t[pub], t[~pub], edges),
                     psi_train_val=psi(tr, v, edges), nan_val=float(np.isnan(v).mean()), nan_pub=float(np.isnan(t[pub]).mean()),
                     nan_priv=float(np.isnan(t[~pub]).mean()), q_val=str(qs(v)), q_pub=str(qs(t[pub])), q_priv=str(qs(t[~pub])), q_train=str(qs(tr))))
df = pl.DataFrame(rows)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(40)
print(df)
df.write_csv(f"{OUT}/e04_drift.csv")
