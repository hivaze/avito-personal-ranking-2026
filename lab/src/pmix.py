"""Private-mix (and public-mix) reweighting of val per-feed NDCG by feed size M x feed cold share.

Why: private differs from val mainly in item staleness (feed cold share 0.29 val -> 0.41 private) and feed size.
calib.py reweights by M only. This module reweights val feeds so that the joint (M bin x cold-share bin) distribution
matches the public / private test feeds (covariate shift on label-free feed descriptors; no test labels used).
Bins = eda/errors/segments/segdefs.py (M: 6 bins, cold_share: <15/15-30/30-45/>=45%; the err_segments.md test-mix).

Role in decisions (CURRENT_STATUS Log 2026-11-12 12:50, pre-declared before any pmix number of a candidate was read):
  report-only + veto. A candidate/recipe whose private-mix delta vs its reference is < -1 SE is not adopted even if
  its val rule passes; pmix never promotes anything on its own.

CLI:  .venv/bin/python src/pmix.py A B        (A, B = val score .npy files, or comma lists = member-equal z-mean)
"""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from common import *  # noqa: F401,F403
from metrics import ndcg_per_feed
from segdefs import seg_defs

BOUND = 1790497920  # calib.py: public = test feeds with timestamp < BOUND
_W = None


def weights():
    """{'public': w, 'private': w} over the 14,423 val feeds (mean 1), plus diagnostics."""
    global _W
    if _W is not None:
        return _W
    d = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
    te = pl.read_parquet("<repo>/eda/errors/segments/test_segments.parquet")
    fv, Mv, _ = feed_meta("val"); ft, Mt, _ = feed_meta("test")
    assert len(d) == len(Mv) and (d["fidx"].to_numpy() == np.arange(len(Mv))).all() and (d["M"].to_numpy() == Mv).all()
    tf = te["fidx"].to_numpy(); assert (te["M"].to_numpy() == Mt[tf]).all()
    ts = ft["timestamp"].to_numpy()[tf]
    S, St = seg_defs(d), seg_defs(te, test=True)
    kv = np.char.add(np.char.add(S["M"].astype(str), "|"), S["cold_share"].astype(str))
    kt = np.char.add(np.char.add(St["M"].astype(str), "|"), St["cold_share"].astype(str))
    out = {}
    for name, m in [("public", ts < BOUND), ("private", ts >= BOUND)]:
        w = np.zeros(len(kv))
        for k in np.unique(kv):
            w[kv == k] = (kt[m] == k).mean() / (kv == k).mean()
        assert np.isin(np.unique(kt[m]), np.unique(kv)).all(), "test cell without val support"
        out[name] = w / w.mean()
    out["cs_val"] = d["cold_share"].to_numpy().mean()
    out["cs_pub"] = te["cold_share"].to_numpy()[ts < BOUND].mean(); out["cs_priv"] = te["cold_share"].to_numpy()[ts >= BOUND].mean()
    _W = out
    return out


def wmean_se(x, w=None):
    if w is None:
        return x.mean(), x.std() / np.sqrt(len(x))
    m = np.average(x, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((x - m) ** 2, weights=w) / neff)


def zmean(paths):
    f, M, off = feed_meta("val")
    fid = np.repeat(np.arange(len(M)), M)
    acc = None
    for p in paths:
        s = np.load(p).astype(np.float64)
        mu = np.bincount(fid, s, len(M)) / M; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 1e-12))
        z = (s - mu[fid]) / sd[fid]
        acc = z if acc is None else acc + z
    return acc / len(paths)


def per_feed(spec):
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
    paths = spec.split(",") if isinstance(spec, str) else list(spec)
    s = np.load(paths[0]) if len(paths) == 1 else zmean(paths)
    return ndcg_per_feed(s.astype(np.float32), y, off, M)


def compare(pa, pb, label=""):
    """pa, pb: per-feed NDCG arrays (A = arm, B = reference). Returns dict and prints one line."""
    W = weights(); d = pa - pb
    r = {"val": wmean_se(d), "public": wmean_se(d, W["public"]), "private": wmean_se(d, W["private"]),
         "absA": (pa.mean(), np.average(pa, weights=W["public"]), np.average(pa, weights=W["private"]))}
    print(f"{label:34s} val {r['val'][0]:+.5f}±{r['val'][1]:.5f} (z {r['val'][0] / r['val'][1]:+.2f}) | "
          f"pub-mix {r['public'][0]:+.5f}±{r['public'][1]:.5f} | priv-mix {r['private'][0]:+.5f}±{r['private'][1]:.5f} "
          f"(z {r['private'][0] / r['private'][1]:+.2f})")
    return r


if __name__ == "__main__":
    W = weights()
    print(f"mean feed cold share: val {W['cs_val']:.3f} public {W['cs_pub']:.3f} private {W['cs_priv']:.3f}; "
          f"n_eff public {W['public'].sum() ** 2 / (W['public'] ** 2).sum():.0f} private {W['private'].sum() ** 2 / (W['private'] ** 2).sum():.0f}")
    a, b = sys.argv[1], sys.argv[2]
    pa, pb = per_feed(a), per_feed(b)
    print(f"A val {pa.mean():.5f} pub-mix {np.average(pa, weights=W['public']):.5f} priv-mix {np.average(pa, weights=W['private']):.5f}")
    print(f"B val {pb.mean():.5f} pub-mix {np.average(pb, weights=W['public']):.5f} priv-mix {np.average(pb, weights=W['private']):.5f}")
    compare(pa, pb, "A - B")
