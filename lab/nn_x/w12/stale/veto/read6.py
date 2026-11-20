"""veto read 6: free pool variants inside the 16 fine-tuned wide students: the 8 teacher_mean25 students (T8) against W16 / the 8
teacher_mean11 students (K8); train-only and final form (0.5 z(FT pool) + 0.5 z(TO pool)). Information; stored predictions only."""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); R = []
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
K8 = rng("w5_e10_kd05w192_s", 951, 958); T8 = rng("w5_e10_kd05t25w192_s", 951, 958); W16 = K8 + T8
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954)
for lab, A, B in (("T8 vs W16 (TO)", T8, W16), ("T8 vs K8 (TO)", T8, K8), ("K8 vs W16 (TO)", K8, W16),
                  ("0.8 z(T8) + 0.2 z(pop1kd x8) vs W16 (TO)", [(0.8, T8), (0.2, P1KD + P1KDW)], W16),
                  ("0.8 z(T8) + 0.2 z(pop1kd x8) vs 0.8 z(W16) + 0.2 z(pop1kd x8) (TO)", [(0.8, T8), (0.2, P1KD + P1KDW)], [(0.8, W16), (0.2, P1KD + P1KDW)])):
    r = read(c, lab, A, B, kinds=("val", "hold")); show(r); R.append(r)
# final form where FT out-of-fold files exist
def ftz(v):
    for d in DIRS:
        for suf in ("n2cpv3s3", "n2cpv3"):
            p = d / f"nn_{v}_{suf}_val_oof.npy"
            if p.exists(): return c.z("val", np.load(p))
    raise FileNotFoundError(v)
try:
    fin = lambda names: 0.5 * c.z("val", np.mean([ftz(v) for v in names], 0)) + 0.5 * c.z("val", c.pool(names, "val"))
    built = 0.5 * c.z("val", np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * c.z("val", np.load(ROOT / "preds/ens_W16kd_to_val.npy"))
    print(f"check: my final W16 vs the built file: NDCG {c.nv(fin(W16)).mean():.5f} vs {c.nv(built).mean():.5f}")
    for lab, A, B in (("T8 final vs W16 final", T8, W16), ("T8 final vs K8 final", T8, K8)):
        d = c.nv(fin(A)) - c.nv(fin(B))
        r = {"label": lab, "val": ms(d), "pmix_pub": ms(d, c.wpub), "pmix_priv": ms(d, c.wpriv), **{k: ms(d[m]) for k, m in c.sl.items()}}
        print(f"== {lab}: val {f(r['val'])} | M>=50 {f(r['M>=50'])} | M<50 {f(r['M<50'])} | coldpos {f(r['coldpos'])} | warmpos {f(r['warmpos'])} | pmix-pub {f(r['pmix_pub'])} | pmix-priv {f(r['pmix_priv'])}")
        R.append(r)
except FileNotFoundError as e:
    print("final form: FT file missing for", e)
json.dump(R, open(HERE / "out/read6.json", "w"), indent=1, default=float)
