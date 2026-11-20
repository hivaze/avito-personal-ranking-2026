"""web_shift probe s1 (label-free): how the item-occurrence count grows with the listing's exposure length E before the
dense cutoff, on val / test rows and on a train-row sample. Reads only; writes nothing but stdout.
E (hours) = cutoff - posting time (feat_new/hz_common.expo_hours); val/test cutoff = 2026-09-20 00:00 MSK; for a train row
the as-built count sees every dense snapshot of the data, so its E is taken against the same cutoff (end of the dense data).
"""
import sys
import numpy as np
sys.path.insert(0, "<repo>/feat_new")
import hz_common as H

BINS = np.array([-1e9, 0, 24, 48, 96, 168, 240, 336, 480, 1e9])
LAB = ["E<0 (after cutoff)", "0-1 d", "1-2 d", "2-4 d", "4-7 d", "7-10 d", "10-14 d", "14-20 d", ">20 d / capped"]

def table(split, sel=None, name=None):
    f = H.feeds(split)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    cut = np.full(len(ts), H.CUT_VT, np.int64)
    E = H.expo_hours(split, cut, ts)
    c = np.load(H.FEATS / split / "it_hist_all_luo.npy", mmap_mode="r")
    fi = np.load(H.FLAT / split / "cand_fidx.npy", mmap_mode="r")
    if sel is not None:
        keep = sel[np.asarray(fi)]
        E, c = E[keep], np.asarray(c)[keep]
    else:
        c = np.asarray(c)
    b = np.digitize(E, BINS) - 1
    print(f"\n== {name or split}: rows {len(E):,}")
    print(f"{'E bucket':22s} {'share':>7s} {'zero share':>10s} {'mean cnt':>9s} {'mean cnt | >0':>13s} {'mean log1p':>10s}")
    out = []
    for k, l in enumerate(LAB):
        m = b == k
        if m.sum() == 0:
            continue
        x = c[m].astype(np.float64)
        out.append((l, m.mean(), (x <= 0).mean(), x.mean(), x[x > 0].mean() if (x > 0).any() else 0, np.log1p(np.maximum(x, 0)).mean()))
        print(f"{l:22s} {m.mean():7.3f} {(x <= 0).mean():10.3f} {x.mean():9.2f} {out[-1][4]:13.2f} {out[-1][5]:10.3f}")
    return out

table("val")
ft = H.feeds("test"); tts = ft["timestamp"].to_numpy().astype(np.int64)
table("test", tts < 1790497920, "test public")
table("test", tts >= 1790497920, "test private")
# train: a 2% feed sample, all rows; E against the end of the dense data (as-built counts are not causal)
ftr = H.feeds("train")
rng = np.random.default_rng(7)
sel = rng.random(len(ftr)) < 0.02
table("train", sel, "train 2% feed sample (E to the end of the dense data)")
hold = np.zeros(len(ftr), bool); hold[H.holdout_feeds()] = True
table("train", hold, "train holdout feeds (random-like, M>=50)")
