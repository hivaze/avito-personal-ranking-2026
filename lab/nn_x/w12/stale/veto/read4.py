"""veto read 4: (a) stored-prediction extras for the reverse check; (b) CPU fp32 stale-val reads (INFORMATION ONLY: my validity bar of
PREDECLARED.txt item 1 was missed by 0.00008 on the first member, see the report). CPU stale files: my preds/cpu_<v>_staleval.npy, and,
read-only, the witness agent's nn_x/w12/stale/witness/cache/staleval/nn_<v>_staleval.npy (same scenario, same CPU fp32 path).
Output: logs/read4.txt + out/read4.json
"""
import json
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
R = []
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W16 = W192 + T192; W8 = W192[:4] + T192[:4]
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954)
WIT = ROOT / "nn_x/w12/stale/witness/cache/staleval"


def go(label, A, B, **kw):
    try:
        r = read(c, label, A, B, **kw)
    except FileNotFoundError as e:
        print(f"== {label}: MISSING {e}"); return None
    show(r); R.append(r); return r


print("##### A1. teacher with popularity-light members (teacher_mean25 = 11 + 6 GRU + 8 pop1) vs teacher_mean11, wide students, 8 seeds")
go("kd05t25w192 x8 vs kd05w192 x8", T192, W192)
go("kd05t25 x4 vs kd05 x4 (d128)", rng("w5_e10_kd05t25_s", 951, 954), rng("w5_e10_kd05_s", 951, 954))
go("kd08t25 x3 vs kd08 x3 (d128, same kd_w 0.8)", rng("w5_e10_kd08t25_s", 951, 953), rng("w5_e10_kd08_s", 951, 953))
go("POOLED: all t25 students (kd05t25 x4 + kd05t25w192 x8) vs their t11 twins (kd05 x4 + kd05w192 x8)", rng("w5_e10_kd05t25_s", 951, 954) + T192, rng("w5_e10_kd05_s", 951, 954) + W192)
print("##### A2. final-form W16 with the TO half taken at @last (FT half unchanged): 0.5 z(FT) + 0.5 z(TO@last) vs the built file")
ft = c.z("val", np.load(ROOT / "preds/ens_W16kd_ft_val.npy")); to = c.z("val", np.load(ROOT / "preds/ens_W16kd_to_val.npy"))
tl = c.pool([v + "@last" for v in W16], "val"); ts = c.pool(W16, "val")
print(f"   check: stored TO pool vs my TO pool of selected members: max |z diff| {float(np.abs(c.z('val', ts) - to).max()):.4f}")
d = c.nv(0.5 * ft + 0.5 * c.z("val", tl)) - c.nv(0.5 * ft + 0.5 * to)
r = {"label": "W16 final with TO@last minus built W16", "val": ms(d), "pmix_pub": ms(d, c.wpub), "pmix_priv": ms(d, c.wpriv),
     "M<50": ms(d[c.sl["M<50"]]), "M>=50": ms(d[c.sl["M>=50"]]), "coldpos": ms(d[c.sl["coldpos"]]), "warmpos": ms(d[c.sl["warmpos"]])}
print("   " + " | ".join(f"{k} {f(v)}" for k, v in r.items() if k != "label")); R.append(r)


def cst(v):
    """CPU stale-val z of a member (mine first, else the witness cache)."""
    p = HERE / "preds" / f"cpu_{v}_staleval.npy"
    if not p.exists(): p = WIT / f"nn_{v}_staleval.npy"
    if not p.exists(): raise FileNotFoundError(f"CPU stale file of {v}")
    return c.z("val", np.load(p))


def cpool(spec):
    if spec and isinstance(spec[0], tuple):
        acc = 0.0
        for w, names in spec: acc = acc + w * c.z("val", np.mean([cst(v) for v in names], 0))
        return np.asarray(acc, np.float32)
    return np.mean([cst(v) for v in spec], 0).astype(np.float32)


def cgo(label, A, B):
    try:
        a, b = c.nv(cpool(A)), c.nv(cpool(B))
    except FileNotFoundError as e:
        print(f"== [CPU stale] {label}: MISSING {e}"); return None
    d = a - b
    r = {"label": "[CPU stale] " + label, "cstale": ms(d), "levelA": float(a.mean()), "levelB": float(b.mean()), "cstale_priv": ms(d, c.wpriv),
         **{"cstale_" + k: ms(d[m]) for k, m in c.sl.items()}}
    print(f"== [CPU stale] {label}\n   stale-val {f(r['cstale'])} [A {r['levelA']:.5f} B {r['levelB']:.5f}] | priv-weighted {f(r['cstale_priv'])} | "
          f"M>=50 {f(r['cstale_M>=50'])} | M<50 {f(r['cstale_M<50'])} | coldpos {f(r['cstale_coldpos'])} | warmpos {f(r['cstale_warmpos'])}")
    R.append(r); return r


print("##### B0. cross-check of the two CPU sources on one member (mine vs the witness cache), if both exist")
v = "w5_e10_kd05w192_s951"; pm, pw = HERE / "preds" / f"cpu_{v}_staleval.npy", WIT / f"nn_{v}_staleval.npy"
if pm.exists() and pw.exists():
    a, b = np.load(pm), np.load(pw); print(f"   {v}: max |diff| {float(np.abs(a - b).max()):.2e}, NDCG {c.nv(c.z('val', a)).mean():.5f} vs {c.nv(c.z('val', b)).mean():.5f}")
else:
    print(f"   not available (mine {pm.exists()}, witness {pw.exists()})")
print("##### B1. checkpoint selection on CPU stale-val: S - L for the W16 members whose selected step != last step")
D9 = [W192[i] for i in (0, 2, 4, 5, 7)] + [T192[i] for i in (0, 2, 3, 4)]
have = [v for v in D9 if (HERE / "preds" / f"cpu_{v}@last_staleval.npy").exists() and ((HERE / "preds" / f"cpu_{v}_staleval.npy").exists() or (WIT / f"nn_{v}_staleval.npy").exists())]
print(f"   members with both files: {len(have)} of 9: {have}")
if have:
    cgo(f"S - L, pool of {len(have)} members", have, [v + "@last" for v in have])
    dm = np.mean([c.nv(cst(v)) - c.nv(cst(v + '@last')) for v in have], 0)
    print(f"   member-level mean S - L: {f(ms(dm))} | coldpos {f(ms(dm[c.sl['coldpos']]))} | warmpos {f(ms(dm[c.sl['warmpos']]))} | priv-weighted {f(ms(dm, c.wpriv))}")
    R.append({"label": f"[CPU stale] S - L, member-level mean over {len(have)} runs", "cstale": ms(dm), "levelA": float(np.mean([c.nv(cst(v)).mean() for v in have])),
              "levelB": float(np.mean([c.nv(cst(v + '@last')).mean() for v in have])), "cstale_priv": ms(dm, c.wpriv), **{"cstale_" + k: ms(dm[m]) for k, m in c.sl.items()}})
    go(f"(fresh val, stored files) S - L, the same {len(have)} members", have, have, sufA="", sufB="@last", kinds=("val", "hold"))
print("##### B2. popularity-light wide students on CPU stale-val")
cgo("pop1kd w192 x4 vs kd05w192 x4 (family level)", P1KDW, W192[:4])
cgo("pop1kd d128 x4 vs kd05t25 d128 x4 (family level)", P1KD, rng("w5_e10_kd05t25_s", 951, 954))
cgo("0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8", [(0.8, W8), (0.2, P1KD)], W8)
cgo("0.8 z(W16) + 0.2 z(pop1kd d128 x4) vs W16", [(0.8, W16), (0.2, P1KD)], W16)
cgo("0.8 z(W16) + 0.2 z(pop1kd w192 x4) vs W16", [(0.8, W16), (0.2, P1KDW)], W16)
cgo("0.8 z(W8) + 0.2 z(pop1kd w192 x4) vs W8", [(0.8, W8), (0.2, P1KDW)], W8)
for w in (0.1, 0.3, 0.5):
    cgo(f"{1 - w:.1f} z(W8) + {w:.1f} z(pop1kd w192 x4) vs W8", [(1 - w, W8), (w, P1KDW)], W8)
print("##### B3. width and KD weight on CPU stale-val (witness cache members)")
cgo("W8 (8 wide) vs 8 narrow (kd05 + kd05t25)", W8, rng("w5_e10_kd05_s", 951, 954) + rng("w5_e10_kd05t25_s", 951, 954))
cgo("kd05w192 x4 vs kd05 x4", W192[:4], rng("w5_e10_kd05_s", 951, 954))
cgo("kd05 x4 vs hard base x4", rng("w5_e10_kd05_s", 951, 954), rng("w5_e10_base_s", 951, 954))
cgo("W16 vs W8", W16, W8)
print("##### B4. level under staleness (CPU stale minus stored fresh val; numerics differ by about +0.0004 fp32 vs bf16, so read the DIFFERENCE between families)")
for lab, names in (("kd05w192 x4", W192[:4]), ("pop1kd w192 x4", P1KDW), ("pop1kd d128 x4", P1KD), ("W8", W8)):
    try:
        s = c.nv(cpool(names)); fr = c.nv(c.pool(names, "val"))
        print(f"   {lab}: stale {s.mean():.5f} fresh(stored) {fr.mean():.5f} drop {f(ms(s - fr))}")
        R.append({"label": f"LEVEL {lab}", "stale": float(s.mean()), "fresh": float(fr.mean()), "drop": ms(s - fr)})
    except FileNotFoundError as e:
        print(f"   {lab}: MISSING {e}")
json.dump(R, open(HERE / "out/read4.json", "w"), indent=1, default=float)
