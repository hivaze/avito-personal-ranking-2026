"""Predict public/private LB from val per-feed NDCG by reweighting val feeds to the test feed-size (M) mix.
Public = test feeds with timestamp < 1790497920 (first ~30% by time), private = the rest."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from metrics import ndcg_per_feed

BOUND = 1790497920
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])


def weights(kind):
    fv, Mv, _ = feed_meta("val"); ft, Mt, _ = feed_meta("test")
    tt = ft["timestamp"].to_numpy()
    mask = tt < BOUND if kind == "public" else tt >= BOUND
    bv = np.searchsorted(EDGES, Mv, side="right"); bt = np.searchsorted(EDGES, Mt[mask], side="right")
    pv = np.bincount(bv, minlength=len(EDGES) + 1) / len(bv); pt = np.bincount(bt, minlength=len(EDGES) + 1) / len(bt)
    w = (pt / np.maximum(pv, 1e-12))[bv]
    return w / w.mean()


def predict(per_feed):
    return {k: float((weights(k) * per_feed).mean()) for k in ["public", "private"]}


if __name__ == "__main__":
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
    for p in sys.argv[1:]:
        per = ndcg_per_feed(np.load(p), y, off, M)
        r = predict(per)
        print(f"{p}: val {per.mean():.5f}  pred public {r['public']:.5f}  pred private {r['private']:.5f}")
