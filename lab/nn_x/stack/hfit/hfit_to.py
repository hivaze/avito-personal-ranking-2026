"""Exploratory TOonly variant (plan.txt ADDENDUM 2): holdout weights applied to the to-half only; ft half = ME."""
import os, sys, json
H = "<repo>/nn_x/stack/hfit"
for pool in sys.argv[1:]:
    sys.argv = ["x", pool, "none"]
    src = open(f"{H}/hfit.py").read().split("t0 = time.time()")[0]
    exec(compile(src, f"{H}/hfit.py", "exec"))
    R = json.load(open(f"{H}/logs/{pool}_W1-W2-W3-W4-W5.json"))
    ME_v = ndcg_pf(val_score(W_ME), yv, offv, Mv)
    zft = fz(offv, Mv, Xv_ft @ W_ME)
    P(f"POOL {pool} ME val {ME_v.mean():.5f}")
    out = {}
    for var, r in R["variants"].items():
        w = np.array(r["w"])
        v = ndcg_pf(0.5 * zft + 0.5 * fz(offv, Mv, Xv_to @ w), yv, offv, Mv)
        d, s = paired(v, ME_v); sl = {k: paired(v, ME_v, m) for k, m in SL.items()}
        acc = (d > 2 * s) and (r["hold_oos"] >= -r["hold_oos_se"]) and all(a >= -2 * b for a, b in sl.values())
        P(f"  {var}-TOonly: val {v.mean():.5f} vs ME {d:+.5f}+-{s:.5f} ({d / s:+.1f} SE) | hold OOS {r['hold_oos']:+.5f}+-{r['hold_oos_se']:.5f}"
          f" | " + " ".join(f"{k} {a:+.5f}+-{b:.5f}" for k, (a, b) in sl.items()) + f" | ACCEPT={acc}")
        out[var] = {"val": v.mean(), "d": d, "se": s, "slices": {k: list(x) for k, x in sl.items()}, "accept": bool(acc)}
    json.dump(out, open(f"{H}/logs/{pool}_TOonly.json", "w"), indent=1, default=float)
open(f"{H}/logs/TOonly.txt", "w").write("\n".join(LOG) + "\n")
