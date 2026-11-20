"""Descriptive only (round T1117 is decided by perm_read.py): did each arm move the part of the error it targeted?

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/gap1117/mechanism_check.py

Validation, 16-pool train-only (same pools as perm_read.py). For each pool, on its own order:
  within-category quality = NDCG after moving the categories that hold a positive to the top, keeping the pool's order
                            inside them (only the order inside categories still counts);
  category quality        = NDCG after sorting each category's items perfectly inside the pool's category slots
                            (only the placement of categories still counts);
  plain NDCG by feed size.
Categories come from the error-analysis tables (lab/nn_x/errfix_1117), checked row by row against the validation labels.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import Meta, ndcg_fast, paths  # noqa: E402

ROOT = Path("<repo>")
E = ROOT / "lab/nn_x/errfix_1117"
m = Meta()
t = pl.concat([pl.read_parquet(E / "data/disc.parquet", columns=["fidx", "row", "cat", "label"]),
               pl.read_parquet(E / "private/confirm.parquet", columns=["fidx", "row", "cat", "label"])]).sort(["fidx", "row"])
assert t.height == len(m.yv) and (t["label"].to_numpy() == m.yv).all()
cat = t["cat"].to_numpy()
disc = 1.0 / np.log2(np.arange(2, 400))


def oracles(s):
    a, b = np.empty(len(m.Mv)), np.empty(len(m.Mv))
    for k, (o, M) in enumerate(zip(m.ov, m.Mv)):
        yy, cc = m.yv[o:o + M], cat[o:o + M]
        order = np.lexsort((yy, -s[o:o + M]))                  # pessimistic ties, as the metric
        ideal = disc[:int(yy.sum())].sum()
        if ideal == 0:
            a[k] = b[k] = np.nan; continue
        pc = set(cc[yy == 1].tolist())
        first = np.array([c in pc for c in cc[order]])
        o1 = order[np.argsort(~first, kind="stable")]
        a[k] = (yy[o1] * disc[:M]).sum() / ideal
        o2 = order.copy()
        for c in np.unique(cc):
            sl = np.nonzero(cc[order] == c)[0]; it = order[sl]
            o2[sl] = it[np.argsort(-yy[it], kind="stable")]
        b[k] = (yy[o2] * disc[:M]).sum() / ideal
    return a, b


def pool(arm):
    vs = [f"w5_{st}cd_{arm}_s{x}" for st in "kl" for x in range(951, 959)]
    return np.mean([m.z("val", np.load(paths(v)[0])) for v in vs], 0)


arms = ["ref", "ref2", "wcl05", "wcl10", "ccl05", "fsw25t", "fsw50t", "kw03", "lr4", "e4", "hx"]
res = {}
for a in arms:
    s = pool(a)
    nd = ndcg_fast(s, m.yv, m.ov, m.Mv)
    w, c = oracles(s)
    res[a] = (nd, w, c)
    print(f"done {a}", flush=True)
nd0, w0, c0 = res["ref"]
big = m.Mv >= 101
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
print("\nvalidation, 16-pool vs ref, mean delta (SE)")
print(f"{'arm':8s} {'NDCG all':>17s} {'NDCG M>=101':>17s} {'NDCG M<=100':>17s} {'within-cat order':>17s} {'category choice':>17s}")
for a in arms[1:]:
    nd, w, c = res[a]
    cols = [nd - nd0, (nd - nd0)[big], (nd - nd0)[~big], w - w0, c - c0]
    print(f"{a:8s} " + " ".join(f"{np.nanmean(d):+.5f} ({se(d[~np.isnan(d)]):.5f})" for d in cols))
print(f"\nlevels for ref: NDCG {nd0.mean():.4f}, within-category-only {np.nanmean(w0):.4f}, category-only {np.nanmean(c0):.4f}")
