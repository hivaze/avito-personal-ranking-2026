"""Stability diagnostics for P11-W1-TOonly (plan.txt ADDENDUM 3)."""
import os, sys, json
H_ = "<repo>/nn_x/stack/hfit"
sys.argv = ["x", "P11", "none"]
src = open(f"{H_}/hfit.py").read().split("t0 = time.time()")[0]
exec(compile(src, f"{H_}/hfit.py", "exec"))
R = json.load(open(f"{H_}/logs/P11_W1-W2-W3-W4-W5.json")); w = np.array(R["variants"]["W2"]["w"])
ME_v = ndcg_pf(val_score(W_ME), yv, offv, Mv); zft = fz(offv, Mv, Xv_ft @ W_ME); ME_h = hold_ndcg_pf(W_ME)
def vto(wt): return ndcg_pf(0.5 * zft + 0.5 * fz(offv, Mv, Xv_to @ wt), yv, offv, Mv)
def rep(name, wt):
    v = vto(wt); d, s = paired(v, ME_v); dc, sc = paired(v, ME_v, SL["coldpos"]); dw, sw = paired(v, ME_v, SL["warmpos"])
    print(f"{name:22s} w={np.round(wt, 3).tolist()}  val vs ME {d:+.5f}+-{s:.5f} ({d / s:+.1f})  coldpos {dc:+.5f} warmpos {dw:+.5f}", flush=True)
    return d
print("families", FL)
rep("full", w)
fo2 = group_folds(uh, 2, 11); allf = np.arange(nh)
for k in range(2):
    rep(f"half{k}", fit(allf[fo2 != k], "simplex", 0.0, W_ME))
uu = np.unique(uh); rng = np.random.default_rng(5); ws = []
feeds_by_user = {}
order = np.argsort(uh, kind="stable"); us, st = np.unique(uh[order], return_index=True); en = np.r_[st[1:], len(order)]
for b in range(30):
    pick = rng.integers(0, len(us), len(us))
    feeds = np.concatenate([order[st[i]:en[i]] for i in pick])
    ws.append(fit(feeds, "simplex", 0.0, W_ME))
ws = np.array(ws); print("bootstrap weight mean", np.round(ws.mean(0), 3).tolist(), "sd", np.round(ws.std(0), 3).tolist())
rep("bagged30", ws.mean(0) / ws.mean(0).sum())
for i, f in enumerate(FL):
    if w[i] == 0: continue
    w2 = w.copy(); w2[i] = 0; w2 /= w2.sum()
    d = rep(f"drop {f}", w2)
    dh, sh = paired(hold_ndcg_pf(w2), hold_ndcg_pf(w))
    print(f"{'':22s} holdout(in-sample) drop-{f} vs full {dh:+.5f}+-{sh:.5f}")
