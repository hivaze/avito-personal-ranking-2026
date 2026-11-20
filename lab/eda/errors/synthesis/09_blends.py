"""Equal-weight z-mean candidates on the nn_final3 generation (no fitted weights), leak-free val + test-mix reweighting."""
import sys; sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from common import *
from stack_val import feed_rank_z
from segdefs import seg_defs

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn

d_all = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
te = pl.read_parquet("<repo>/eda/errors/segments/test_segments.parquet")
kf = ~d_all["leak"].to_numpy(); S = seg_defs(d_all); St = seg_defs(te, test=True)
kv = np.char.add(S["M"].astype(str), S["cold_share"].astype(str)); kt = np.char.add(St["M"].astype(str), St["cold_share"].astype(str))
w = np.zeros(len(kv))
for k in np.unique(kv[kf]): w[(kv == k) & kf] = (kt == k).mean() / (kv[kf] == k).mean()
w = w[kf] / w[kf].mean()
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
P = Path("<repo>/preds"); Q = Path("<repo>/eda/errors/synthesis/out")
z = lambda a: feed_rank_z(off, M, a.astype(np.float32))[:, 1]
Z = {"f3_tr": z(np.load(pp("nn_final3_trainonly_val.npy"))), "f3_oof": z(np.load(pp("nn_final3_val_oof.npy"))),
     "f2_oof": z(np.load(pp("nn_final2_val_oof.npy"))), "stk3": z(np.load(Q / "qt_base_oof.npy"))}
C = {"f3_tr": Z["f3_tr"], "f3_oof": Z["f3_oof"], "f3_oof+f3_tr": (Z["f3_oof"] + Z["f3_tr"]) / 2,
     "f3_oof+stk3 (v5-style)": (Z["f3_oof"] + Z["stk3"]) / 2, "f3_oof+f3_tr+stk3": (Z["f3_oof"] + Z["f3_tr"] + Z["stk3"]) / 3,
     "f3_oof+f3_tr+f2_oof": (Z["f3_oof"] + Z["f3_tr"] + Z["f2_oof"]) / 3}
per = {k: ndcg_per_feed(v, y, off, M)[kf] for k, v in C.items()}
ref = per["f3_tr"]
def wse(x):
    m = np.average(x, weights=w); return np.sqrt(np.average((x - m) ** 2, weights=w) / (w.sum() ** 2 / (w ** 2).sum()))
for k, p in per.items():
    d = p - ref
    print(f"{k:24s} val {p.mean():.5f} test-mix {np.average(p, weights=w):.5f} | vs f3_tr {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f} "
          f"(test-mix {np.average(d, weights=w):+.5f} ± {wse(d):.5f}) | vs f3_oof {(p - per['f3_oof']).mean():+.5f} ± {(p - per['f3_oof']).std() / np.sqrt(len(d)):.5f}")
