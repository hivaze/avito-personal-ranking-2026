import sys
sys.path.insert(0, "<repo>/src")
from common import *
S = Path("/tmp/scratchpad/oof"); C = Path("<repo>/data/eval_cache")
tr_items = np.unique(np.load(FLAT / "train/cand_item_id.npy"))
def feed_unseen(split):
    f, M, off = feed_meta(split); it = np.load(FLAT / split / "cand_item_id.npy")
    un = ~np.isin(it, tr_items); fi = np.repeat(np.arange(len(M)), M)
    return np.bincount(fi, weights=un, minlength=len(M)) / M, M, f["timestamp"].to_numpy()
uv, Mv, _ = feed_unseen("val"); ut, Mt, tt = feed_unseen("test")
priv = tt >= 1790497920; pub = ~priv
print("unseen row share: val %.3f pub %.3f priv %.3f" % ((uv * Mv).sum() / Mv.sum(), (ut[pub] * Mt[pub]).sum() / Mt[pub].sum(), (ut[priv] * Mt[priv]).sum() / Mt[priv].sum()))
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
pb = np.mean([ndcg_per_feed(np.load(C / f"base_oof_s{s}.npy"), y, off, M) for s in range(5)], 0)
pc = np.mean([ndcg_per_feed(np.load(S / f"k24fix_{o}_s{s}.npy"), y, off, M) for o in ["alpha", "shuf1", "shuf2"] for s in range(5)], 0)
d = pc - pb
ub = np.quantile(uv, np.linspace(0, 1, 6)[1:-1]); mb = np.quantile(Mv, [.25, .5, .75])
cell = lambda u, m: np.searchsorted(ub, u) * 4 + np.searchsorted(mb, m)
cv = cell(uv, Mv)
for nm, mask in [("public", pub), ("private", priv), ("all test", np.ones(len(ut), bool))]:
    ct = np.bincount(cell(ut[mask], Mt[mask]), minlength=20) / mask.sum(); cvv = np.bincount(cv, minlength=20) / len(cv)
    w = (ct / np.maximum(cvv, 1e-9))[cv]; w /= w.mean()
    dm = (w * d).mean(); se = np.sqrt(((w * d - dm) ** 2).mean() / len(d))
    print(f"{nm:8s} reweighted (unseen-share x M) mean delta {dm:+.5f} se {se:.5f}  (unweighted {d.mean():+.5f})")
for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
    m = (uv >= lo) & (uv < hi)
    print(f"unseen share [{lo:.1f},{hi:.1f}): val feeds {m.sum():5d} ({m.mean():.2f}), priv feeds share {((ut[priv]>=lo)&(ut[priv]<hi)).mean():.2f}, delta {d[m].mean():+.5f} se {d[m].std()/np.sqrt(m.sum()):.5f}")
