"""Candidate-level AUC per input group, train random-like (Sep 9+) vs val (M>=50) and vs private (M>=50):
which NN input family carries the train->val/test shift?"""
import json, time
from common_av import *
from adv_lib import *
import cand_adv as C

t0 = time.time()
fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test"); tt = ft["timestamp"].to_numpy()
pri = np.where(tt >= BOUND)[0]
ftr = meta("train")[0]
dtr = (ftr["timestamp"].to_numpy().astype(np.int64) + MSK) // 86400
tr = C.train_random_like()
tr9 = tr[dtr[tr] >= 20705]
tr9 = np.sort(np.random.default_rng(1).choice(tr9, min(30000, len(tr9)), replace=False))
blocks = {"train9": C.load_block("train", tr9, False, 16), "val50": C.load_block("val", np.arange(len(fv))[Mv >= 50], False, 14),
          "private50": C.load_block("test", pri[Mt[pri] >= 50], False, 15)}
GROUPS = {"f1-5+category": ["f1", "f2", "f3", "f4", "f5", "category"], "M,H": ["M", "H"],
          "cvnd popularity (it_hist/seq_cv)": [n for n in C.CVND if n.startswith(("it_hist", "seq_cv"))],
          "cvnd context (ctx_*)": [n for n in C.CVND if n.startswith("ctx_")],
          "prew (5)": C.PREW, "pre2s (5)": C.PRE2S, "embedding proxies": C.EMBP}
out = {}
for comp, (A, B) in {"train9_vs_val50": ("train9", "val50"), "train9_vs_private50": ("train9", "private50"),
                     "val50_vs_private50": ("val50", "private50")}.items():
    ca, ua, fa = blocks[A]; cb, ub, fb = blocks[B]
    y = np.r_[np.zeros(len(ua)), np.ones(len(ub))].astype(np.int8)
    g = groups_of(np.r_[ua, ub], np.r_[fa, fb + 10 ** 8])
    out[comp] = {}
    for gn, fl in GROUPS.items():
        X = np.stack([np.r_[np.asarray(ca[n], np.float32), np.asarray(cb[n], np.float32)] for n in fl], 1)
        res = run_adv(X, y, g, fl, P_CAND, 200, cat=["category"] if "category" in fl else None, shap_rows=5000)
        out[comp][gn] = res["auc"]
        print(f"[{time.time() - t0:6.0f}s] {comp} | {gn} | AUC {res['auc']:.4f}", flush=True)
json.dump(out, open(f"{OUT}/cand_groups.json", "w"), indent=1)
