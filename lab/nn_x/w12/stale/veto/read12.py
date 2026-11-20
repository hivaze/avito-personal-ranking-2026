"""veto read 12 (resume run): CPU stale-val (information only, PREDECLARED item 1 / R1.5) of the forms the lead declared at 03:35Z and of the
hard-label members the witness cache holds. One CPU source per member (mine first, else the witness cache). Output: logs/read12.txt + out/read12.json"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); R = []; WIT = ROOT / "nn_x/w12/stale/witness/cache/staleval"
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W8 = W192[:4] + T192[:4]; W16 = W192 + T192
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954); K5 = rng("w5_e10_kd05_s", 951, 954); K5T = rng("w5_e10_kd05t25_s", 951, 954)
def cst(v):
    p = HERE / "preds" / f"cpu_{v}_staleval.npy"
    if not p.exists(): p = WIT / f"nn_{v}_staleval.npy"
    if not p.exists(): raise FileNotFoundError(v)
    return c.z("val", np.load(p))
def pool(spec, src):
    g = cst if src == "stale" else (lambda v: c.zfile(v, "val"))
    if isinstance(spec[0], tuple):
        return np.asarray(sum(w * c.z("val", np.mean([g(v) for v in ns], 0)) for w, ns in spec), np.float32)
    return np.mean([g(v) for v in spec], 0).astype(np.float32)
def go(label, A, B):
    try:
        ds = c.nv(pool(A, "stale")) - c.nv(pool(B, "stale")); dv = c.nv(pool(A, "val")) - c.nv(pool(B, "val"))
    except FileNotFoundError as e:
        print(f"== {label}: MISSING {e}"); return
    r = {"label": label, "val": ms(dv), "pmix_priv": ms(dv, c.wpriv), "cstale": ms(ds), "cstale_priv": ms(ds, c.wpriv), **{"cstale_" + k: ms(ds[m]) for k, m in c.sl.items()},
         "d_stale_minus_val": ms(ds - dv)}
    print(f"== {label}\n   fresh val {f(r['val'])} | pmix-priv {f(r['pmix_priv'])} || CPU stale-val {f(r['cstale'])} | priv-weighted {f(r['cstale_priv'])} | M>=50 {f(r['cstale_M>=50'])} | "
          f"M<50 {f(r['cstale_M<50'])} | coldpos {f(r['cstale_coldpos'])} | warmpos {f(r['cstale_warmpos'])} || stale minus fresh {f(r['d_stale_minus_val'])}")
    R.append(r); sys.stdout.flush()
print("##### the declared W16P20x8 form and its weight neighbours (train-only level)")
for w in (0.1, 0.2, 0.3, 0.5):
    go(f"{1 - w:.1f} z(W16) + {w:.1f} z(pop1 students x8 = d128 x4 + w192 x4) vs W16", [(1 - w, W16), (w, P1KD + P1KDW)], W16)
go("0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 (the scored W8P20 form)", [(0.8, W8), (0.2, P1KD)], W8)
print("##### teacher axis and width under private-like inputs")
go("teacher_mean25 students x12 vs teacher_mean11 twins x12", K5T + T192, K5 + W192)
go("kd05t25w192 x8 vs kd05w192 x8", T192, W192)
go("kd05t25 x4 vs kd05 x4 (d128)", K5T, K5)
go("T8+pop: 0.8 z(kd05t25w192 x8) + 0.2 z(pop1 students x8) vs 0.8 z(W16) + 0.2 z(pop1 students x8)", [(0.8, T192), (0.2, P1KD + P1KDW)], [(0.8, W16), (0.2, P1KD + P1KDW)])
go("W8 vs 8 narrow students", W8, K5 + K5T)
go("W16 vs W8", W16, W8)
print("##### control: same-view admixture under private-like inputs")
go("0.8 z(W16) + 0.2 z(narrow kd05t25 x4) vs W16", [(0.8, W16), (0.2, K5T)], W16)
go("0.8 z(W16) + 0.2 z(hard base x4) vs W16", [(0.8, W16), (0.2, rng("w5_e10_base_s", 951, 954))], W16)
json.dump(R, open(HERE / "out/read12.json", "w"), indent=1, default=float)
