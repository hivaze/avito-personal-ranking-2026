"""pop1 POST-HOC information read (NOT pre-declared, no gate; written 2026-11-13 ~07:10 UTC after the pre-declared readout).
Question: the pop-free family's role is diversity next to pop-ful members. Does pop1 move back toward the pop-ful base?
2-epoch screen, LAST checkpoints, seeds 23/24, 2-member rz-mean pools of base / xdrop_pop / pop1, on val and on the 30K
train holdout:
  - Pearson correlation of the within-feed rank-z pool scores (pop1 vs base, xdrop_pop vs base, pop1 vs xdrop_pop);
  - 2-family blend: rz-mean(base pool, X pool) for X = xdrop_pop / pop1. NDCG, plus the paired per-feed delta
    (pop1-blend - xdrop_pop-blend) with SE = sd / sqrt(n_feeds).
CPU only, read-only on predictions.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/pop1/pop1_diversity.py
"""
import json
import sys

sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *  # noqa: E402,F401,F403  (Split, ROOT, np)

PA = ROOT / "nn_x/abl/preds"
OUTD = ROOT / "nn_x/abl/pop1/out"
ARMS = ("base", "xdrop_pop", "pop1")
res, lines = {}, []


def P(s):
    print(s, flush=True)
    lines.append(s)


P("== pop1 POST-HOC diversity read (not pre-declared; information only): 2-epoch @last pools, seeds 23/24")
for kind in ("val", "hold"):
    S = Split(kind)
    f = (lambda a, s: PA / f"nn_abl_{a}_s{s}@last_trainonly_val.npy") if kind == "val" else \
        (lambda a, s: PA / f"holdout/nn_abl_{a}_s{s}@last_holdout.npy")
    raw = {a: [np.load(f(a, s)).astype(np.float64) for s in (23, 24)] for a in ARMS}
    for a in ARMS:
        assert all(len(x) == S.N for x in raw[a]), (kind, a)
    pool = {a: S.rz(np.mean([S.rz(x) for x in raw[a]], 0)) for a in ARMS}
    nd = {a: S.nd(pool[a]) for a in ARMS}
    single = {f"{a}_s{s}": float(S.nd(raw[a][i]).mean()) for a in ARMS for i, s in enumerate((23, 24))}
    corr = {f"{a}~{b}": float(np.corrcoef(pool[a], pool[b])[0, 1]) for a, b in
            (("pop1", "base"), ("xdrop_pop", "base"), ("pop1", "xdrop_pop"))}
    blend = {x: S.nd(S.rz((pool["base"] + pool[x]) / 2)) for x in ("xdrop_pop", "pop1")}
    d = blend["pop1"] - blend["xdrop_pop"]
    r = dict(pool_ndcg={a: float(v.mean()) for a, v in nd.items()}, single_ndcg=single, corr=corr,
             blend_ndcg={x: float(v.mean()) for x, v in blend.items()}, blend_delta=float(d.mean()),
             blend_delta_se=float(d.std(ddof=1) / np.sqrt(len(d))), blend_vs_base_pool={
                 x: [float((blend[x] - nd["base"]).mean()), float((blend[x] - nd["base"]).std(ddof=1) / np.sqrt(len(d)))]
                 for x in blend})
    res[kind] = r
    P(f"\n  [{kind}] pool NDCG " + "  ".join(f"{a} {v:.5f}" for a, v in r["pool_ndcg"].items())
      + " | singles " + "  ".join(f"{k} {v:.5f}" for k, v in single.items()))
    P(f"  [{kind}] corr(rank-z pools): " + "  ".join(f"{k} {v:.4f}" for k, v in corr.items()))
    P(f"  [{kind}] blend rz-mean(base pool, X pool): X=xdrop_pop {r['blend_ndcg']['xdrop_pop']:.5f} "
      f"(vs base pool {r['blend_vs_base_pool']['xdrop_pop'][0]:+.5f} +- {r['blend_vs_base_pool']['xdrop_pop'][1]:.5f})  "
      f"X=pop1 {r['blend_ndcg']['pop1']:.5f} (vs base pool {r['blend_vs_base_pool']['pop1'][0]:+.5f} +- "
      f"{r['blend_vs_base_pool']['pop1'][1]:.5f});  pop1-blend - xdrop_pop-blend {r['blend_delta']:+.5f} +- {r['blend_delta_se']:.5f}")
json.dump(res, open(OUTD / "pop1_diversity.json", "w"), indent=1)
open(OUTD / "pop1_diversity.log", "w").write("\n".join(lines) + "\n")
print(f"wrote {OUTD}/pop1_diversity.{{log,json}}")
