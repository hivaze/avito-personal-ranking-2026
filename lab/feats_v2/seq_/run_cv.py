"""Combined driver for item co-vis (feat_covis) and item->cluster co-vis (feat_icv): one index build per source.
  val/test: one index (all-train source + causal val/test histories) for both splits.
  train:    one index per train day d (source strictly before d), queries = feeds of day d.
Usage: python run_cv.py <splits> [threads] [which=cv,icv]
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import train_days, source_fidx, day_of
import feat_covis as FC
import feat_icv as FI


def main():
    sp = sys.argv[1].split(",")
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["cv", "icv"]
    vt = [s for s in sp if s != "train"]
    if vt:
        parts = [("train", source_fidx(None), False)] + [(s, np.arange(len(feeds(s))), True) for s in ("val", "test")]
        idx = Index_(parts, which)
        for split in vt:
            n = len(arr(split, "cand_fidx")); sel = np.arange(len(feeds(split)))
            outs = alloc(n, which)
            query(idx, split, sel, outs)
            write(split, outs)
            log(split, "done")
        del idx
    if "train" in sp:
        n = len(arr("train", "cand_fidx"))
        outs = alloc(n, which)
        f = feeds("train")
        d = day_of(f["timestamp"].to_numpy())
        days = train_days()
        di = np.searchsorted(days, d, side="right") - 1
        for i, c in enumerate(days):
            sel = np.nonzero(di == i)[0]
            src = source_fidx(c)
            if len(src) == 0:
                log("train day", c, "empty source", len(sel))
                if "icv" in outs:
                    # no source: icv undefined (NaN like a feed without known history items), tot = 0
                    rows = np.concatenate([np.arange(o, o + m) for o, m in
                                           zip(f["cand_off"].to_numpy()[sel], f["M"].to_numpy()[sel])])
                    outs["icv"][rows, :3] = np.nan
                continue
            idx = Index_([("train", src, False)], which)
            query(idx, "train", sel, outs)
            log("train day", c, "queries", len(sel))
            del idx
        write("train", outs)


class Index_:
    def __init__(self, parts, which):
        self.idx = FC.Index(parts)
        if "icv" in which:
            self.c1 = self.idx.cluster_codes(1024)
            self.c2 = self.idx.cluster_codes(4096)


def alloc(n, which):
    o = {}
    if "cv" in which:
        o["cv"] = np.zeros((n, len(FC.NAMES)), np.float32)
    if "icv" in which:
        o["icv"] = np.zeros((n, len(FI.NAMES)), np.float32)
    return o


def query(I, split, sel, outs):
    if "cv" in outs:
        FC.run_queries(I.idx, split, sel, outs["cv"])
    if "icv" in outs:
        FI.run_queries(I.idx, I.c1, I.c2, split, sel, outs["icv"])


def write(split, outs):
    if "cv" in outs:
        for i, nm in enumerate(FC.NAMES):
            save(split, nm, outs["cv"][:, i])
    if "icv" in outs:
        for i, nm in enumerate(FI.NAMES):
            save(split, nm, outs["icv"][:, i])


if __name__ == "__main__":
    main()
