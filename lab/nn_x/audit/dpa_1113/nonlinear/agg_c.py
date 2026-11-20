"""Aggregate (c): per config and level, per-seed OOF deltas vs G3, the seed-mean of the paired (level - control) per-feed
difference with its SE, the seed-averaged-score version (mean of the 3 seeds' scores), slices, pmix, holdout."""
import sys, glob, re
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, ndcg, wmean_se, pmix_w, D  # noqa

dv = load("val"); dh = load("hold")
bv = ndcg(dv, dv["g"]); bh = ndcg(dh, dh["g"])
PW = pmix_w()["private"]
M = dv["M"]
se = lambda x: x.std() / np.sqrt(len(x))


def report(kind, cfg, levels=(0, 1, 2), seeds=(1, 2, 3)):
    print(f"=== {kind} {cfg} ===")
    S = {}
    for L in levels:
        for s in seeds:
            fv = f"{D}/{kind}_oof_L{L}_s{s}_{cfg}.npy"; fh = f"{D}/{kind}_hold_L{L}_s{s}_{cfg}.npy"
            try:
                S[(L, s)] = (np.load(fv), np.load(fh))
            except FileNotFoundError:
                pass
    for L in levels:
        ss = [s for s in seeds if (L, s) in S]
        if not ss:
            continue
        dvs = [ndcg(dv, S[(L, s)][0]) - bv for s in ss]
        dhs = [ndcg(dh, S[(L, s)][1]) - bh for s in ss]
        mv = np.mean(dvs, 0); mh = np.mean(dhs, 0)
        avg_v = ndcg(dv, np.mean([S[(L, s)][0] for s in ss], 0)) - bv
        avg_h = ndcg(dh, np.mean([S[(L, s)][1] for s in ss], 0)) - bh
        pm, ps = wmean_se(mv, PW)
        print(f"  L{L} seeds {ss}: per-seed val {[round(x.mean(), 5) for x in dvs]} | mean {mv.mean():+.5f}+-{se(mv):.5f} | "
              f"seed-avg score {avg_v.mean():+.5f}+-{se(avg_v):.5f} | pmix {pm:+.5f}+-{ps:.5f} | M<50 {mv[M<50].mean():+.5f} M>=50 {mv[M>=50].mean():+.5f} | "
              f"HOLD per-seed {[round(x.mean(), 5) for x in dhs]} seed-avg {avg_h.mean():+.5f}+-{se(avg_h):.5f}")
        if L > 0 and all((0, s) in S for s in ss):
            pv = np.mean([ndcg(dv, S[(L, s)][0]) - ndcg(dv, S[(0, s)][0]) for s in ss], 0)
            ph = np.mean([ndcg(dh, S[(L, s)][1]) - ndcg(dh, S[(0, s)][1]) for s in ss], 0)
            print(f"     L{L} - L0 (paired, seed-mean): val {pv.mean():+.5f}+-{se(pv):.5f} (z {pv.mean()/se(pv):+.2f}) | hold {ph.mean():+.5f}+-{se(ph):.5f} (z {ph.mean()/se(ph):+.2f})")


for cfg in sys.argv[1:]:
    kind, c = cfg.split(":")
    report(kind, c)
