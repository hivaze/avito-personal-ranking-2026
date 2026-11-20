"""decide5-style proxies for the holdout-fitted weights (context; the acceptance rule is in plan.txt):
V = final val delta vs ME, Vto = train-only val delta vs ME, H = 2-fold OOS holdout delta vs ME, D = Vto - H, S = min(V, V-D).
Also the re-cooled share (F7x + F7x70 weight) of each weighting.
  CUDA_VISIBLE_DEVICES="" python nn_x/stack/hfit/evalS.py P6 [P3 P11]"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "12")
pools = sys.argv[1:]
H = "<repo>/nn_x/stack/hfit"
for pool in pools:
    sys.argv = ["x", pool, "none"]
    src = open(f"{H}/hfit.py").read().split("t0 = time.time()")[0]
    exec(compile(src, f"{H}/hfit.py", "exec"))
    js = [f for f in sorted(os.listdir(f"{H}/logs")) if f.startswith(pool + "_") and f.endswith(".json")]
    ME_v = ndcg_pf(val_score(W_ME), yv, offv, Mv)
    ME_to = ndcg_pf(fz(offv, Mv, Xv_to @ W_ME), yv, offv, Mv)
    rec = [i for i, f in enumerate(FL) if f in ("F7x", "F7x70")]
    print(f"POOL {pool} ME val {ME_v.mean():.5f} ME val_to {ME_to.mean():.5f}  ME recooled share {W_ME[rec].sum():.3f}")
    for j in js:
        R = json.load(open(f"{H}/logs/{j}"))
        for var, r in R["variants"].items():
            w = np.array(r["w"])
            vto = ndcg_pf(fz(offv, Mv, Xv_to @ w), yv, offv, Mv)
            dV, sV = r["dME"], r["sME"]; dto, sto = paired(vto, ME_to); dH = r["hold_oos"]
            D = dto - dH; S = min(dV, dV - D)
            print(f"  {var}: recooled share {w[rec].sum():.3f}  V {dV:+.5f}+-{sV:.5f}  Vto {dto:+.5f}+-{sto:.5f}  "
                  f"H(oos) {dH:+.5f}+-{r['hold_oos_se']:.5f}  D {D:+.5f}  S {S:+.5f}  accept(rule)={r['accept']}")
