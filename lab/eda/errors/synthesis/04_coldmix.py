"""nn_final3: does the val-label stage still trade cold-positive feeds for warm-positive ones? Test-mix (M x cold) reweighting."""
import sys; sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from common import *
from segdefs import seg_defs

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn

d_all = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
te = pl.read_parquet("<repo>/eda/errors/segments/test_segments.parquet")
keep = ~d_all["leak"].to_numpy(); d = d_all.filter(pl.col("leak").not_())
S = seg_defs(d); St = seg_defs(te, test=True); N = len(d)
key_v = np.char.add(S["M"].astype(str), S["cold_share"].astype(str)); key_t = np.char.add(St["M"].astype(str), St["cold_share"].astype(str))
w = np.zeros(N)
for k in np.unique(key_v):
    w[key_v == k] = (key_t == k).mean() / (key_v == k).mean()
w /= w.mean()
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
P = Path("<repo>/preds"); Q = Path("<repo>/eda/errors/synthesis/out")
per = {n: ndcg_per_feed(np.load(pp(f)).astype(np.float32), y, off, M)[keep] for n, f in
       [("f3_tr", "nn_final3_trainonly_val.npy"), ("f3_oof", "nn_final3_val_oof.npy"), ("f2_tr", "nn_final2_trainonly_val.npy"), ("f2_oof", "nn_final2_val_oof.npy")]}
r = np.load(Q / "qt_base.npz"); per["stk3"] = r["per"].mean(0)[keep]; per["blend3"] = r["perb"].mean(0)[keep]
cold_all = d["pos_cold_all"].to_numpy().astype(bool); cold_any = d["pos_cold_any"].to_numpy().astype(bool)
warm = ~cold_any
assert len(per["f3_tr"]) == N
def se(x, ww=None):
    return x.std() / np.sqrt(len(x)) if ww is None else np.sqrt(np.average((x - np.average(x, weights=ww)) ** 2, weights=ww) / (ww.sum() ** 2 / (ww ** 2).sum()))
print(f"feeds {N}: all-cold-positive {cold_all.mean():.3f} (test-mix weight {w[cold_all].sum() / N:.3f}), warm-positive {warm.mean():.3f}")
for a, b in [("f3_oof", "f3_tr"), ("stk3", "f3_tr"), ("blend3", "f3_tr"), ("f2_oof", "f2_tr"), ("f3_tr", "f2_tr")]:
    dd = per[a] - per[b]
    print(f"{a:7s}-{b:6s} val {dd.mean():+.5f} ± {se(dd):.5f} | test-mix {np.average(dd, weights=w):+.5f} ± {se(dd, w):.5f} | "
          f"all-cold-pos {dd[cold_all].mean():+.5f} ± {se(dd[cold_all]):.5f} | warm-pos {dd[warm].mean():+.5f} ± {se(dd[warm]):.5f}")
for k in per:
    print(f"  {k:7s} val {per[k].mean():.5f} test-mix {np.average(per[k], weights=w):.5f}")
