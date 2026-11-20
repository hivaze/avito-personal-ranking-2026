"""Optuna search over per-feed LOSS WEIGHTING families in the CPU GBDT proxy (listwise softmax CE, proxy_lib_ll),
trained on 60K uniformly drawn train feeds (production + ~5% random-like, holdout excluded), 300 rounds, fixed seed
and data across trials (paired). Weight of feed f (normalised to mean 1):
    w_f = (M_f/50)^alpha * (K_f/IDCG(K_f))^beta * (1 + eta * p_random_f) * S_f^gamma
S_f = detached NDCG rank-elasticity of the positives at the current scores (ce_rw-like; refreshed every 20 rounds).
TPE (optuna) maximises val NDCG on half A of val (users split by hash); the chosen setting is then scored on half B
(nested), paired per feed against the baseline (all exponents 0). Seed noise: baseline re-run with 2 other seeds.
Writes out/08_optuna_weights.txt (+ trials csv)."""
import sys, time, json
import numpy as np, optuna
from proxy_lib_ll import *
sys.path.insert(0, "<repo>/src")
from common import feed_meta

OUT = "<repo>/nn_x/insights/labels_loss/out"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
X, y, M = load("mix"); p = np.load(f"{D}/mix_p.npy")
rng = np.random.default_rng(7)
fe = np.sort(rng.choice(len(M), 60000, replace=False))
Xs, ys, Ms, _ = subset(X, y, M, fe); ps = p[fe]
Xv, yv, Mv = load("val"); Xv = np.asarray(Xv); ov = offs(Mv)
fv, _, _ = feed_meta("val")
uid = fv["user_id"].to_numpy().astype(np.int64)
halfA = ((uid * 2654435761) % 1000 < 500) & (uid != 0) | ((uid == 0) & (np.arange(len(uid)) % 2 == 0))
P(f"train rows {len(ys)}, feeds {len(Ms)}, random-like share {(ps>0.5).mean():.3f}; val half A {halfA.sum()} / B {(~halfA).sum()}")
IDCG = np.concatenate([[1.0], np.cumsum(1.0 / np.log2(np.arange(2, 700)))])
Ks = np.add.reduceat(ys.astype(np.float64), offs(Ms))

def make_dyn(gamma):
    if gamma == 0: return None
    cache = {"it": 0, "w": None}
    def dyn(s, obj):
        if cache["w"] is None or cache["it"] % 20 == 0:
            o = obj.o; fid = obj.fid
            order = np.lexsort((-obj.y, -s, fid))  # optimistic within ties is irrelevant (continuous scores)
            rank = np.empty(len(s), np.int64); rank[order] = np.arange(len(s)) - np.repeat(o, obj.M)
            r = rank + 1.0
            # elasticity of 1/log2(1+r) w.r.t. log r:  r / ((1+r) ln2 log2(1+r)^2)
            el = r / ((1 + r) * np.log(2) * np.log2(1 + r) ** 2)
            Sf = np.bincount(fid, weights=el * obj.y, minlength=len(obj.M)) / IDCG[np.minimum(Ks, 698).astype(int)]
            cache["w"] = np.maximum(Sf, 1e-4) ** gamma
        cache["it"] += 1
        return cache["w"]
    return dyn

def weights(alpha, beta, eta):
    return (Ms / 50.0) ** alpha * (Ks / IDCG[np.minimum(Ks, 698).astype(int)]) ** beta * (1 + eta * ps)

def run(alpha, beta, eta, gamma, seed=11):
    bst = train(Xs, ys, Ms, w=weights(alpha, beta, eta), dyn=make_dyn(gamma), rounds=300, seed=seed)
    return ndcg_per_feed(bst.predict(Xv, raw_score=True), yv, ov, Mv)

t0 = time.time()
base = run(0, 0, 0, 0)
P(f"baseline val {base.mean():.5f} (A {base[halfA].mean():.5f}, B {base[~halfA].mean():.5f}) [{time.time()-t0:.0f}s]")
noise = []
for sd in [12, 13]:
    b2 = run(0, 0, 0, 0, seed=sd); d = b2 - base; noise.append(d.mean())
    P(f"baseline seed {sd}: val {b2.mean():.5f}, delta vs seed 11 {d.mean():+.5f} (SE {d.std()/np.sqrt(len(d)):.5f})")

store = {}
def objective(trial):
    a = trial.suggest_float("alpha", -0.6, 0.6); b = trial.suggest_float("beta", -1.0, 2.0)
    e = trial.suggest_float("eta", 0.0, 4.0); g = trial.suggest_float("gamma", -1.0, 1.0)
    t = time.time(); nd = run(a, b, e, g); store[trial.number] = nd
    d = nd - base
    P(f"trial {trial.number}: a {a:+.2f} b {b:+.2f} e {e:.2f} g {g:+.2f} | val {nd.mean():.5f} dA {d[halfA].mean():+.5f} "
      f"dB {d[~halfA].mean():+.5f} dAll {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f}) [{time.time()-t:.0f}s]")
    return nd[halfA].mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
for fixed in [dict(alpha=0, beta=0, eta=0, gamma=0), dict(alpha=-0.17, beta=0, eta=0, gamma=0),
              dict(alpha=0, beta=1.0, eta=0, gamma=0), dict(alpha=0, beta=0, eta=0, gamma=0.5)]:
    study.enqueue_trial(fixed)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS)
best = study.best_trial
d = store[best.number] - base
P(f"\nBEST on half A: trial {best.number} {json.dumps({k: round(v, 3) for k, v in best.params.items()})}")
P(f"nested estimate (half B, paired vs baseline): {d[~halfA].mean():+.5f} (SE {d[~halfA].std()/np.sqrt((~halfA).sum()):.5f}); "
  f"half A (selection, optimistic) {d[halfA].mean():+.5f}")
# reverse direction: select on B among the same trials, score on A (trial set was guided by A -> mildly optimistic)
bB = max(store, key=lambda k: store[k][~halfA].mean())
dB = store[bB] - base
P(f"reverse: best on half B = trial {bB} {json.dumps({k: round(v, 3) for k, v in study.trials[bB].params.items()})}; "
  f"scored on half A {dB[halfA].mean():+.5f} (SE {dB[halfA].std()/np.sqrt(halfA.sum()):.5f})")
short = Mv < 50
P(f"best-A trial on M<50: {d[short].mean():+.5f}, M>=50: {d[~short].mean():+.5f}")
P(f"baseline seed-to-seed |delta| {np.mean(np.abs(noise)):.5f}")
import csv
with open(f"{OUT}/08_optuna_trials.csv", "w") as fh:
    w = csv.writer(fh); w.writerow(["trial", "alpha", "beta", "eta", "gamma", "val", "dA", "dB"])
    for t in study.trials:
        nd = store[t.number]; dd = nd - base
        w.writerow([t.number] + [round(t.params[k], 4) for k in ["alpha", "beta", "eta", "gamma"]] +
                   [round(nd.mean(), 6), round(dd[halfA].mean(), 6), round(dd[~halfA].mean(), 6)])
open(f"{OUT}/08_optuna_weights.txt", "w").write("\n".join(lines) + "\n")
