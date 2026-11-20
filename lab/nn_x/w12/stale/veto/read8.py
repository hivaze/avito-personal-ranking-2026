"""veto read 8 (resume run): COVERAGE_MAP row 301 re-read. Every stored family F as a 0.2 admixture to the W16 train-only pool:
  NDCG(0.8 z(W16) + 0.2 z(F pool)) - NDCG(z(W16)), selected checkpoints, paired per feed.
Rule: PREDECLARED.txt ADDENDUM R1 (written 03:39:52Z, before this file). Families are discovered from the file names (no outcome enters the list).
Output: logs/read8.txt (stdout) + out/read8.json
"""
import json
import re
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
W16 = rng("w5_e10_kd05w192_s", 951, 958) + rng("w5_e10_kd05t25w192_s", 951, 958)

# ---- family discovery: nn_<arm>_{s|r}<n>_trainonly_val.npy without '@', 10-epoch members and re-cools; holdout file required
fam = {}
pat = re.compile(r"^nn_(.+)_([sr])(\d+)_trainonly_val\.npy$")
for d, keep in ((ROOT / "nn_x/w5/preds", lambda a: True), (ROOT / "nn_x/abl/preds", lambda a: a.startswith("abl_e10_"))):
    for p in sorted(d.glob("nn_*_trainonly_val.npy")):
        m = pat.match(p.name)
        if not m or "@" in p.name or not keep(m.group(1)):
            continue
        v = f"{m.group(1)}_{m.group(2)}{m.group(3)}"
        if find(v, "hold") is not None:
            fam.setdefault(m.group(1), []).append(v)
fam["F10 (G3 part, GRU)"] = rng("v23_gru_pre2s_s", 501, 506)
fam["F7x (G3 part)"] = rng("r6_cd_m02_r", 101, 107)
fam["F7x70 (G3 part)"] = rng("r6_cd_m02_p70k_r", 101, 107)
fam = {k: v for k, v in fam.items() if len(v) >= 2 and all(find(x, "val") is not None and find(x, "hold") is not None for x in v)}
STALE = lambda k: ("pop" in k and "pseudo" not in k) or "hzcd" in k          # PREDECLARED item 2 (a), (b), (c): popularity-light views and hz
INW16 = lambda k: k in ("w5_e10_kd05w192", "w5_e10_kd05t25w192")
print(f"families: {len(fam)}; staleness family: {[k for k in fam if STALE(k)]}")


def zpool(names, kind):
    sp = "hold" if kind == "hold" else "val"
    acc = None
    for v in names:
        z = c.z(sp, np.load(find(v, kind)))
        acc = z.astype(np.float64) if acc is None else acc + z
    return c.z(sp, acc / len(names))


zw = {k: zpool(W16, k) for k in ("val", "hold")}
bw = {"val": c.nv(zw["val"]), "hold": c.nh(zw["hold"])}
print(f"W16 train-only pool: val {bw['val'].mean():.5f}, holdout {bw['hold'].mean():.5f}")
R = []
for k in sorted(fam):
    names = fam[k]
    dv = c.nv(0.8 * zw["val"] + 0.2 * zpool(names, "val")) - bw["val"]
    dh = c.nh(0.8 * zw["hold"] + 0.2 * zpool(names, "hold")) - bw["hold"]
    r = {"family": k, "n": len(names), "stale": bool(STALE(k)), "inW16": bool(INW16(k)), "val": ms(dv), "pmix_pub": ms(dv, c.wpub), "pmix_priv": ms(dv, c.wpriv),
         "hold": ms(dh), "hold_late": ms(dh[c.hl["late>=Sep13"]]), **{s: ms(dv[m]) for s, m in c.sl.items()}}
    zv, zp, zh = (r[x][0] / r[x][1] for x in ("val", "pmix_priv", "hold"))
    r["pooled_z"] = (zv + zh) / np.sqrt(2)
    r["flag"] = bool(r["val"][0] >= 0 and zp >= 1 and r["hold"][0] < 0)
    R.append(r)
    print(f"{k:34s} n {r['n']:2d} | val {f(r['val'])} | M>=50 {r['M>=50'][0]:+.5f} M<50 {r['M<50'][0]:+.5f} | coldpos {r['coldpos'][0]:+.5f} warmpos {r['warmpos'][0]:+.5f} | "
          f"pmix-priv {f(r['pmix_priv'])} | holdout {f(r['hold'])} late {r['hold_late'][0]:+.5f} | pooled z {r['pooled_z']:+.2f} | "
          f"{'STALE-FAMILY ' if r['stale'] else ''}{'inW16 ' if r['inW16'] else ''}{'FLAG' if r['flag'] else ''}")
    sys.stdout.flush()
json.dump(R, open(HERE / "out/read8.json", "w"), indent=1, default=float)
nf = [r for r in R if r["flag"]]
print(f"\nflags: {len(nf)} of {len(R)} families (expected by chance about {0.08 * len(R):.1f}); flagged: {[(r['family'], r['n']) for r in nf]}")
print("flagged outside the staleness family:", [r["family"] for r in nf if not r["stale"]])
print("two-witness passes (val >= 0, holdout >= 0, pooled z >= 2) among non-staleness families:",
      [r["family"] for r in R if not r["stale"] and not r["inW16"] and r["val"][0] >= 0 and r["hold"][0] >= 0 and r["pooled_z"] >= 2])
