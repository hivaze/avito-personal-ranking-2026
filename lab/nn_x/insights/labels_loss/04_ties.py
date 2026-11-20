"""(1c) Ties under pessimistic tie-breaking in the CURRENT members and pools (val + test).
Counts exact within-feed equal scores (all pairs, adjacent after sort) and pos-neg ties on val; NDCG pessimistic vs
optimistic on val. Also the final mix 0.5 z(ft) + 0.5 z(to) for each pool, cast to float32 as in make_sub."""
import sys, os
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

OUT = "<repo>/nn_x/insights/labels_loss/out"
PR = "<repo>/preds"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

def zfeed(s, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    s = s.astype(np.float64)
    mu = np.bincount(fid, weights=s, minlength=len(M)) / M
    d = s - np.repeat(mu, M)
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=len(M)) / M)
    return d / (np.repeat(sd, M) + 1e-9)

def ties(s, off, M, y=None):
    fid = np.repeat(np.arange(len(M)), M)
    o = np.lexsort((s, fid))
    ss, ff = s[o], fid[o]
    eq = (ss[1:] == ss[:-1]) & (ff[1:] == ff[:-1])
    feeds_with = len(np.unique(ff[1:][eq]))
    res = dict(tied_adjacent=int(eq.sum()), feeds_with_tie=feeds_with)
    if y is not None:
        yy = y[o]
        pn = eq & (yy[1:] != yy[:-1])
        res["posneg_adjacent_ties"] = int(pn.sum())
        pess = ndcg_per_feed(s, y, off, M).mean()
        # optimistic: positives first among ties -> break ties with +eps*y
        opt = ndcg_per_feed(s.astype(np.float64) + 1e-7 * np.abs(s).max() * 0 + (y * 1e-12), y, off, M).mean()
        res["pess"], res["opt_minus_pess"] = float(pess), float(opt - pess)
    return res

for split in ["val", "test"]:
    f, M, off = feed_meta(split)
    y = np.asarray(np.load(FLAT / f"{split}/cand_target.npy")).astype(np.float64) if split == "val" else None
    names = [f"nn_r6_cd_m02_r10{i}_trainonly" for i in (1, 2)] + ["nn_v28_lr1_m02_s411_trainonly",
             "nn_v23_gru_pre2s_s501_trainonly", "nn_r6_cd_m02_r101_n2cpv3"] + \
            [f"ens_{t}_{h}" for t in ["G3", "PF7x", "M2", "Cd"] for h in ["ft", "to"]]
    for n in names:
        fn = f"{PR}/{n}_{split}.npy" if not n.endswith("n2cpv3") else f"{PR}/{n}_{'val_oof' if split=='val' else 'test'}.npy"
        if not os.path.exists(fn):
            P("missing", fn); continue
        s = np.load(fn)
        P(f"{split} {n}: dtype {s.dtype} " + str(ties(s, off, M, y)))
    for t in ["G3", "PF7x", "M2", "Cd"]:
        a = np.load(f"{PR}/ens_{t}_ft_{split}.npy"); b = np.load(f"{PR}/ens_{t}_to_{split}.npy")
        mix = (0.5 * zfeed(a, off, M) + 0.5 * zfeed(b, off, M)).astype(np.float32)
        P(f"{split} mix {t} (float32): " + str(ties(mix, off, M, y)))
open(f"{OUT}/04_ties.txt", "w").write("\n".join(lines) + "\n")
