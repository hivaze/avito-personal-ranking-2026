# The overlay that was sent to the public board: in each feed whose user has a later feed, the
# candidates found in those later histories are pushed above all others; other scores are kept.
# How it was run (its own command line): `later_overlay.py eval BASE_VAL.npy` for the paired val
# delta, `later_overlay.py apply BASE_TEST.npy OUT.npy` for the test scores; 2026-11-12, on the
# best hard-label pool (code name G3).
# What it produced: subs/cand_G3_later_overlay.parquet, val NDCG 0.40486 against 0.40032 for the
# base pool. The public board gave 0.392819 against 0.392820: the leak is neutralised on test.
# The same user's later snapshots are banned from every input of the solution.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/src/later_overlay.py
"""Same-user LATER-feed overlay (test-time information; permitted by the competition FAQ: "no restrictions on the
solution or additional data"; user authorised 2026-11-12 20:5x UTC).

Mechanism (reports/eda_history_leak.md): a user's history is a strict snapshot at feed time, so the items the user
interacted with in feed A appear in the history of the same user's LATER feeds. For a feed F of user u (u != 0), every
candidate of F that occurs in the union of the histories of u's later feeds (same split, or a later split) is pushed to the
top of F's ranking (ties among pushed items keep the base order). Everything else keeps the base score.

Coverage is small by construction (later feeds exist only for ~0.7% of public and ~0.3% of private feeds).
CLI:  later_overlay.py eval  BASE_VAL.npy            -> paired val delta of the overlay on top of BASE (uses val + test later feeds)
      later_overlay.py apply BASE_TEST.npy OUT.npy    -> overlaid test scores (uses later test feeds)
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: F401,F403
from metrics import ndcg_per_feed


def later_sets(split, extra_splits=()):
    """For each feed of `split`: the set of items in the histories of the same user's later feeds (within split and in
    extra_splits, which must be later in time). Returns dict fidx -> set, only for feeds that have later feeds."""
    f, M, off = feed_meta(split)
    uid, ts = f["user_id"].to_numpy(), f["timestamp"].to_numpy()
    srcs = [(split, f)] + [(s, feed_meta(s)[0]) for s in extra_splits]
    H = {}
    for s, fs in srcs:
        hf = np.load(FLAT / f"{s}/hist_fidx.npy"); hi = np.load(FLAT / f"{s}/hist_item_id.npy")
        H[s] = (fs["user_id"].to_numpy(), fs["timestamp"].to_numpy(), hf, hi)
    # user -> list of (ts, split, fidx)
    by_user = {}
    for s, (u_s, t_s, _, _) in H.items():
        for i in np.nonzero(u_s != 0)[0]:
            by_user.setdefault(int(u_s[i]), []).append((int(t_s[i]), s, int(i)))
    hist_items = {}
    def items_of(s, i):
        if (s, i) not in hist_items:
            hf, hi = H[s][2], H[s][3]
            lo, hi_ = np.searchsorted(hf, i, "left"), np.searchsorted(hf, i, "right")
            hist_items[(s, i)] = set(hi[lo:hi_].tolist())
        return hist_items[(s, i)]
    out = {}
    for i in np.nonzero(uid != 0)[0]:
        later = [(t, s, j) for (t, s, j) in by_user.get(int(uid[i]), []) if t > ts[i] and not (s == split and j == i)]
        if not later:
            continue
        acc = set()
        for _, s, j in later:
            acc |= items_of(s, j)
        if acc:
            out[int(i)] = acc
    return out


def overlay(split, base, sets):
    f, M, off = feed_meta(split); cand = np.load(FLAT / f"{split}/cand_item_id.npy")
    s = np.asarray(base, np.float64).copy(); n_flag = 0; n_feeds = 0
    for i, items in sets.items():
        a, b = off[i], off[i] + M[i]
        flag = np.fromiter((c in items for c in cand[a:b].tolist()), bool, b - a)
        if flag.any():
            n_feeds += 1; n_flag += int(flag.sum())
            s[a:b] = s[a:b] + flag * (s[a:b].max() - s[a:b].min() + 1.0) * 10.0  # pushed above every unflagged item
    return s.astype(np.float32), n_feeds, n_flag


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "eval":
        f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
        base = np.load(sys.argv[2]); sets = later_sets("val", ("test",))
        new, nf, nfl = overlay("val", base, sets)
        p0, p1 = ndcg_per_feed(base.astype(np.float32), y, off, M), ndcg_per_feed(new, y, off, M)
        d = p1 - p0; aff = np.zeros(len(M), bool); aff[list(sets)] = True
        print(f"val: feeds with later same-user feeds {len(sets)} ({len(sets)/len(M):.4f}); feeds with flagged candidates {nf}; flagged candidates {nfl}")
        print(f"val NDCG {p0.mean():.5f} -> {p1.mean():.5f}  delta {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}")
        print(f"affected feeds: {p0[aff].mean():.4f} -> {p1[aff].mean():.4f} (n {aff.sum()}); unaffected unchanged: {np.allclose(p0[~aff], p1[~aff])}")
        # precision of the flag
        cand = np.load(FLAT / "val/cand_item_id.npy"); tp = fp = 0
        for i, items in sets.items():
            a, b = off[i], off[i] + M[i]
            for c, t in zip(cand[a:b].tolist(), y[a:b].tolist()):
                if c in items:
                    tp += t > 0; fp += t == 0
        print(f"flag precision on val: {tp}/{tp+fp} = {tp/max(tp+fp,1):.3f}")
    elif mode == "apply":
        base = np.load(sys.argv[2]); sets = later_sets("test")
        new, nf, nfl = overlay("test", base, sets)
        f, M, off = feed_meta("test"); ts = f["timestamp"].to_numpy(); pub = ts < 1790497920
        aff = np.zeros(len(M), bool); aff[list(sets)] = True
        print(f"test: feeds with later same-user feeds {len(sets)}; public {int((aff & pub).sum())}/{int(pub.sum())}, private {int((aff & ~pub).sum())}/{int((~pub).sum())}; feeds with flagged candidates {nf}; flagged {nfl}")
        np.save(sys.argv[3], new); print("saved", sys.argv[3])
