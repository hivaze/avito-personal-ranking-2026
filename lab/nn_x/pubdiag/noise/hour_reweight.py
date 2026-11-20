"""Part 6 (extra): expected public deltas when val is reweighted to the public mix of (M bin x cold-share bin x hour bin)
and of (hour bin) alone. Label-free test descriptors only (eda/errors/segments)."""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from common import *  # noqa
from metrics import ndcg_per_feed
from segdefs import seg_defs
O = Path("<repo>/nn_x/pubdiag/noise")
d = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
te = pl.read_parquet("<repo>/eda/errors/segments/test_segments.parquet")
fv, Mv, ov = feed_meta("val"); ft, Mt, _ = feed_meta("test"); y = np.load(FLAT / "val/cand_target.npy")
tf = te["fidx"].to_numpy(); pub = ft["timestamp"].to_numpy()[tf] < 1790497920
S, St = seg_defs(d), seg_defs(te, test=True)
nd = {k: ndcg_per_feed(np.load(O / f"cache_{k}_val.npy"), y, ov, Mv) for k in ["Cd", "PF7x", "G3", "M2", "M2c"]}


def w_for(keys):
    kv = np.array(["|".join(t) for t in zip(*[S[k].astype(str) for k in keys])])
    kt = np.array(["|".join(t) for t in zip(*[St[k].astype(str) for k in keys])])[pub]
    w = np.zeros(len(kv))
    for k in np.unique(kv):
        w[kv == k] = (kt == k).mean() / (kv == k).mean()
    miss = 1 - np.isin(kt, kv).mean()
    return w / w.mean(), miss


for keys in [["hour"], ["M", "cold_share"], ["M", "cold_share", "hour"], ["M", "cold_share", "user"], ["M", "cold_share", "hour", "user"]]:
    w, miss = w_for(keys)
    neff = w.sum() ** 2 / (w ** 2).sum()
    out = []
    for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "M2c")]:
        dd = nd[a] - nd[b]; m = np.average(dd, weights=w)
        se = np.sqrt(np.average((dd - m) ** 2, weights=w) / neff)
        out.append(f"{a}-{b} {m:+.5f}±{se:.5f}")
    print(f"{'x'.join(keys):24s} n_eff {neff:6.0f} public mass w/o val support {miss:.3f} | " + "  ".join(out))
print("hour mix val vs public:", dict(zip(*np.unique(S['hour'], return_counts=True))), dict(zip(*np.unique(St['hour'][pub], return_counts=True))))
print("user mix val vs public:", dict(zip(*np.unique(S['user'], return_counts=True))), dict(zip(*np.unique(St['user'][pub], return_counts=True))))
