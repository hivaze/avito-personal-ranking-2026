"""sanity of the residual GBM objective: noise features -> ~0; damaged base + the damaging feature -> recovery"""
import numpy as np, lightgbm as lgb, sys
from lib import *
from lib import _rk, _z
import importlib.util
load_stats()
V = DS("val"); hA = V.half == 0
src = open(OUT / "blocks.py").read()
i = src.index("def softmax_obj"); j = src.index("ROUNDS =")
exec(src[i:j])
P["num_threads"] = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(0)
z5 = _z(V.off, V.M, V.get("f5"))
for nm, zb, X in (("noise8", V.zb, rng.standard_normal((V.n, 8)).astype(np.float32)),
                  ("damaged base (-0.1 z(f5)), features f5, z_feed(f5)", (V.zb - 0.1 * z5).astype(np.float32), np.column_stack([V.get("f5"), z5]))):
    base = V.ndcg(zb)
    Xm = np.column_stack([zb, _rk(V.off, V.M, zb), X])
    r = np.repeat(hA, V.M)
    for n in (25, 100, 200):
        m = lgb.train(P, lgb.Dataset(Xm[r], label=V.y[r], group=V.M[hA], init_score=zb[r].astype(np.float64)), num_boost_round=n)
        p = m.predict(Xm, num_threads=P["num_threads"])
        d = (V.ndcg(zb + p) - base)
        print(nm, "rounds", n, "pred sd", float(p.std()), "delta fit-half", round(float(d[hA].mean()), 5), "other half", round(float(d[~hA].mean()), 5), "±", round(float(d[~hA].std() / np.sqrt((~hA).sum())), 5), "| undamaged-base gap", round(float((V.ndcg(V.zb) - base).mean()), 5), flush=True)
