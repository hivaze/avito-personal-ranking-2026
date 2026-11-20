"""Step 2 (DESCRIPTIVE, in-sample val labels, a sanity read; nothing is chosen from it - DECLARATION.txt section B).

Within-feed AUC (mean over feeds with >= 1 positive and >= 1 negative in the scored subset, ties = 0.5) of every form for the
click label, overall / among cold / among warm candidates; yardsticks xp_imp_c50, it_hist_all_luo, the pool score b.
All within-feed monotone forms of one count (raw, log1p, rank, z) share one AUC by construction; the indicators differ.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s2_auc.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import FEATS, OUT, WIN, forms
from nested import Val

V = Val()
print(f"SANITY pool score b: plain val NDCG {V.base_ndcg.mean():.5f} (build_r4 read-out of FD16P20: 0.40271); "
      f"feeds {V.nf}, rows {V.N}, positives {int(V.y.sum())}, feeds with >= 1 positive {(V.npos > 0).sum()}, base rate {V.y.mean():.5f}")
assert abs(V.base_ndcg.mean() - 0.40271) < 2e-5
cm = np.load(f"{OUT}/cnt_main.npy")[:V.N].astype(np.float64); cz = np.load(f"{OUT}/cnt_nzsrc.npy")[:V.N].astype(np.float64)


def feed_auc(x, sub=None):
    """mean per-feed AUC of x for y within the subset rows (bool mask or None); returns (auc, n_feeds, pair-weighted auc)."""
    rows = np.arange(V.N) if sub is None else np.flatnonzero(sub)
    fid = V.fid[rows]; y = V.y[rows]
    m = np.bincount(fid, minlength=V.nf).astype(np.int64)
    keep = m > 0
    Mk = m[keep]; offk = np.concatenate([[0], np.cumsum(Mk)[:-1]]).astype(np.int64)
    r, _ = forms(offk, Mk, x[rows])
    fk = np.repeat(np.arange(len(Mk)), Mk)
    rank0 = r * (Mk[fk] - 1)  # average 0-based rank; single-row feeds are dropped below
    P = np.bincount(fk, y, len(Mk)); Ng = Mk - P
    ok = (P > 0) & (Ng > 0)
    sr = np.bincount(fk, rank0 * y, len(Mk))
    auc = (sr[ok] - P[ok] * (P[ok] - 1) / 2) / (P[ok] * Ng[ok])
    return float(auc.mean()), int(ok.sum()), float(np.average(auc, weights=P[ok] * Ng[ok])), float(auc.std() / np.sqrt(ok.sum()))


res = {}
print("\nWithin-feed AUC for the click label (mean over feeds; SE; number of feeds; pair-weighted AUC in brackets)")
print(f"{'feature':34s} {'ALL candidates':>34s} {'COLD candidates':>34s} {'WARM candidates':>34s}")


def line(name, x):
    out = []
    for sub in (None, V.cold, ~V.cold):
        a, n, aw, se = feed_auc(x, sub)
        out.append((a, se, n, aw))
    res[name] = out
    print(f"{name:34s} " + " ".join(f"{a:.4f}+-{se:.4f} n={n:5d} [{aw:.4f}]" for a, se, n, aw in out), flush=True)


line("YARD pool score b", V.b)
line("YARD xp_imp_c50 (train exposure)", V.xp)
line("YARD it_hist_all_luo", np.load(f"{FEATS}/val/it_hist_all_luo.npy").astype(np.float64))
line("YARD cold flag", V.cold.astype(np.float64))
for tag, c in (("main", cm), ("nzsrc", cz)):
    for w, wn in enumerate(WIN):
        n = c[:, w]
        line(f"{tag} n_{wn} (raw=log1p=rank=z)", n)
        line(f"{tag} i_{wn} = 1[n>=1]", (n >= 1).astype(np.float64))
        line(f"{tag} cs_{wn} = cold & shown", ((n >= 1) & V.cold).astype(np.float64))

# click rate by count bucket (in-sample descriptive), cold / warm, with the mean pool z of the bucket
print("\nClick rate (percent) by count bucket [main]; in brackets: share of rows (percent), mean pool z of the bucket")
bk = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10 ** 9, "5+")]
res["rate"] = {}
for w, wn in enumerate(WIN):
    for sub, sm in (("cold", V.cold), ("warm", ~V.cold)):
        s = f"{wn:4s} {sub:5s}"
        for lo, hi, lab in bk:
            m = sm & (cm[:, w] >= lo) & (cm[:, w] <= hi)
            res["rate"][f"{wn}|{sub}|{lab}"] = [float(V.y[m].mean()) if m.any() else None, float(m.sum() / sm.sum()), float(V.b[m].mean()) if m.any() else None]
            s += f" | n={lab}: {100 * V.y[m].mean():5.2f} [{100 * m.sum() / sm.sum():5.1f}, z {V.b[m].mean():+.3f}]"
        print(s)
# residual view (still in sample, descriptive): click rate relative to the pool model's expectation softmax(beta b), beta = 1
p = np.exp(V.b); p = p / np.bincount(V.fid, p, V.nf)[V.fid] * V.npos[V.fid]
print("\nObserved / expected clicks (expected = npos_feed * softmax_feed(b), beta 1) by count bucket [main]")
res["oe"] = {}
for w, wn in enumerate(WIN):
    for sub, sm in (("cold", V.cold), ("warm", ~V.cold)):
        s = f"{wn:4s} {sub:5s}"
        for lo, hi, lab in bk:
            m = sm & (cm[:, w] >= lo) & (cm[:, w] <= hi)
            oe = V.y[m].sum() / p[m].sum()
            res["oe"][f"{wn}|{sub}|{lab}"] = [float(oe), int(V.y[m].sum())]
            s += f" | n={lab}: O/E {oe:5.3f} (clicks {int(V.y[m].sum())})"
        print(s)
json.dump(res, open(f"{OUT}/s2_auc.json", "w"), indent=1)
print("DONE")
