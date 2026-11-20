"""veto (resume run): PREDECLARED ADDENDUM R1.5: every member with a CPU stale-val file both in my preds/ and in the witness agent's cache
(read-only) is compared. Output: logs/crosscheck.txt"""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); WIT = ROOT / "nn_x/w12/stale/witness/cache/staleval"
rows = []
for p in sorted((HERE / "preds").glob("cpu_*_staleval.npy")):
    v = p.name[4:-13]; q = WIT / f"nn_{v}_staleval.npy"
    if not q.exists():
        continue
    a, b = np.load(p).astype(np.float64), np.load(q).astype(np.float64)
    na, nb_ = c.nv(c.z("val", a)), c.nv(c.z("val", b)); d = na - nb_
    rows.append((v, float(np.abs(a - b).max()), float(na.mean()), float(nb_.mean()), float(d.mean()), float(d.std() / np.sqrt(len(d))), float((d != 0).mean())))
    print(f"{v:32s} max |logit diff| {rows[-1][1]:.2e} | NDCG mine {rows[-1][2]:.5f} witness {rows[-1][3]:.5f} | diff {rows[-1][4]:+.6f} +- {rows[-1][5]:.6f} | feeds with a different NDCG {rows[-1][6]:.4f}")
print(f"\nmembers in both sources: {len(rows)}; largest |logit diff| {max(r[1] for r in rows):.2e}; largest |NDCG diff| {max(abs(r[4]) for r in rows):.6f}")
