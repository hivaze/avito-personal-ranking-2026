"""cb26: existing GBDT / stacker val preds vs the current G3 pool (eq z-add at w in {0.05,0.1,0.2}). CPU, numpy only."""
import sys, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, ens_feats, zf, P
from metrics import ndcg_per_feed
from cbrule import pse
fv, Mv, ov, yv = meta("val"); z = lambda x: zf(ov, Mv, x)
ev = ens_feats("G3", "val"); pool = z(0.5 * z(ev["ft"]) + 0.5 * z(ev["to"]))
nv = lambda s: ndcg_per_feed(s, yv, ov, Mv); b = nv(pool)
fid = np.repeat(np.arange(len(Mv)), Mv)
print(f"G3 pool val {b.mean():.5f}")
for k in ["cb_tr300k_luo", "cb_tr500k_pw", "cb_tr300k_none", "xgb_rand50", "xgb_x300_luo", "xgb_tr1m_ndcg", "lgb_tr300k_feat", "stk2_res4_clean", "stk_valonly_es"]:
    f = P / f"{k}_val.npy"
    if not f.exists():
        continue
    x = z(np.load(f)); s1 = nv(x).mean(); rho = (np.bincount(fid, x * pool) / Mv).mean()
    out = []
    for w in (0.05, 0.1, 0.2):
        m, s = pse(nv(pool + w * x) - b); out.append(f"w{w:.2f} {m:+.5f}+-{s:.5f}")
    print(f"{k:18s} single {s1:.5f} rho {rho:.3f} law {0.0186*(1-rho)-0.378*(b.mean()-s1):+.4f} | " + " | ".join(out))
