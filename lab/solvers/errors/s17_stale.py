"""Stale-val (private-like popularity coverage, review_shift 'both') check of the pools, TO halves only (CPU replays)."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import os

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
CF = f"{SCR}/out/cf"
v12 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
v15 = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
v13 = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr2_mix1"]
avail = [m for m in v12 + v15 + v13 if os.path.exists(f"{CF}/cf_{m}_both.npy")]
print("available:", len(avail))
Z = {(m, s): fz(np.load(f"{CF}/cf_{m}_{s}.npy"), fid, M) for m in avail for s in ("base", "both")}
nd1 = {}
for m in avail:
    a = ndcg_per_feed(Z[(m, "base")], y, off, M); b = ndcg_per_feed(Z[(m, "both")], y, off, M)
    print(f"  {m:20s} base {a.mean():.5f} both {b.mean():.5f}  stale cost {(b - a).mean():+.5f}")
pools = {"S0": v12, "S1": v12 + v15, "S2": v12 + v13, "S3": v12 + v15 + v13, "v15 only": v15, "v13 only": v13}
res = {}
for k, mem in pools.items():
    if not all(m in avail for m in mem):
        continue
    for s in ("base", "both"):
        res[(k, s)] = ndcg_per_feed(np.mean([Z[(m, s)] for m in mem], 0), y, off, M)
for k in pools:
    if (k, "base") not in res:
        continue
    db = res[(k, "base")] - res[("S0", "base")]
    ds = res[(k, "both")] - res[("S0", "both")]
    dd = ds - db
    se = lambda x: x.std() / np.sqrt(n)  # noqa: E731
    print(f"{k:9s} TO-half base {res[(k, 'base')].mean():.5f} both {res[(k, 'both')].mean():.5f} | vs S0: base {db.mean():+.5f}+-{se(db):.5f}  "
          f"both {ds.mean():+.5f}+-{se(ds):.5f}  diff-in-diff {dd.mean():+.5f}+-{se(dd):.5f}")
