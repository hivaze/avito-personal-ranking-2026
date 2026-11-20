"""A4: what could make the FT half useful again, judged at POOL level on existing preds (CPU only, no NN training).
(1) infinite-member limit a_inf of the TO, FT and final branches (fit NDCG(k) = a_inf - b/k over random k-member subsets):
    separates FT's mean-function (bias) effect from its member-level noise reduction;
(2) FT on fewer members: G3 TO branch over all 20 members + FT branch over k members (k = 1..20, random subsets);
(3) blend weight w_ft for several pools: in-sample curve + nested over the FT user folds;
(4) per-family (FT, TO) branch weights with Optuna TPE, nested over the FT user folds (outer fold held out, weights fitted
    on the 4 others), for G3+NEW;
(5) trunk-input test: member FT gain of F5cd members with prew in the trunk (allprew) vs without (all);
Writes a4_*.csv; stdout -> a4_ft_alternatives.log"""
import optuna
from lib import *  # noqa

optuna.logging.set_verbosity(optuna.logging.WARNING)
add_v30()
vm = meta("val")
fold = user_folds()
rng = np.random.default_rng(1)


def zs(ms, br, split="val"):
    return [member_z(m, split, br).astype(np.float64) for m in ms]


# ---------- (1) a_inf ----------
print("(1) infinite-member limit: NDCG(k) = a_inf - b/k over random k-member subsets (val)")
rows = []
for name, ms in [("G3", POOLS["G3"]), ("F7x", FAM["F7x"]), ("F7b", FAM["F7b"]), ("F7", FAM["F7"]), ("F6", FAM["F6"]),
                 ("F5cd", FAM["F5cd"]), ("V28l", FAM["V28l"]), ("F12", FAM["F12"])]:
    ZT, ZF = zs(ms, "to"), zs(ms, "ft")
    n = len(ms)
    ks = sorted(set([1, 2, 3, 4, 5, 7, 10, 14, 20]) & set(range(1, n + 1)) | {n})
    pts = {"to": [], "ft": [], "fin": []}
    for k in ks:
        R = 1 if k == n else (8 if n > 7 else 6)
        for _ in range(R):
            idx = rng.choice(n, k, replace=False)
            t = np.mean([ZT[i] for i in idx], 0); f = np.mean([ZF[i] for i in idx], 0)
            pts["to"].append((k, pf(t).mean())); pts["ft"].append((k, pf(f).mean()))
            pts["fin"].append((k, pf(0.5 * zfeed(t, "val") + 0.5 * zfeed(f, "val")).mean()))
    r = dict(pool=name, n=n)
    for br, p in pts.items():
        p = np.array(p)
        X = np.c_[np.ones(len(p)), -1 / p[:, 0]]
        (a, b), *_ = np.linalg.lstsq(X, p[:, 1], rcond=None)
        r[f"a_inf_{br}"] = a; r[f"b_{br}"] = b; r[f"k1_{br}"] = p[p[:, 0] == 1, 1].mean(); r[f"kn_{br}"] = p[p[:, 0] == n, 1].mean()
    rows.append(r)
    print(f"  {name:5s} n={n:2d} | TO: k1 {r['k1_to']:.5f} kn {r['kn_to']:.5f} a_inf {r['a_inf_to']:.5f} b {r['b_to']:.5f} | "
          f"FT: k1 {r['k1_ft']:.5f} kn {r['kn_ft']:.5f} a_inf {r['a_inf_ft']:.5f} b {r['b_ft']:.5f} | "
          f"FIN: k1 {r['k1_fin']:.5f} kn {r['kn_fin']:.5f} a_inf {r['a_inf_fin']:.5f} b {r['b_fin']:.5f} | "
          f"a_inf FT-TO {r['a_inf_ft'] - r['a_inf_to']:+.5f} FIN-TO {r['a_inf_fin'] - r['a_inf_to']:+.5f}", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a4_ainf.csv")

# ---------- (2) FT on fewer members ----------
print("\n(2) G3: TO branch of all 20 + FT branch of k random members (10 draws each) vs full G3 and vs TO-only")
ms = POOLS["G3"]; ZT, ZF = zs(ms, "to"), zs(ms, "ft")
zto = zfeed(np.mean(ZT, 0), "val"); nto = pf(zto); nG3 = pf(0.5 * zto + 0.5 * zfeed(np.mean(ZF, 0), "val"))
rows = []
for k in [1, 2, 3, 5, 7, 10, 14, 20]:
    ds = []
    for _ in range(1 if k == 20 else 10):
        idx = rng.choice(20, k, replace=False)
        ds.append(pf(0.5 * zto + 0.5 * zfeed(np.mean([ZF[i] for i in idx], 0), "val")))
    d = np.mean(ds, 0)
    r = dict(k=k, vs_G3=pse(d - nG3)[0], vs_G3_se=pse(d - nG3)[1], vs_TO=pse(d - nto)[0], vs_TO_se=pse(d - nto)[1],
             draw_sd=np.std([x.mean() for x in ds]))
    rows.append(r)
    print(f"  k={k:2d}: vs G3 {r['vs_G3']:+.5f}+-{r['vs_G3_se']:.5f} | vs TO-only {r['vs_TO']:+.5f}+-{r['vs_TO_se']:.5f} | "
          f"sd over draws {r['draw_sd']:.5f}", flush=True)
# FT of one family only
for f in ["F10", "F7x", "F7x70"]:
    d = pf(0.5 * zto + 0.5 * zfeed(branch(FAM[f], "val", "ft"), "val"))
    print(f"  FT branch = {f} only: vs G3 {fmt(d - nG3)} | vs TO-only {fmt(d - nto)}")
pl.DataFrame(rows).write_csv(OUT / "a4_ft_fewer.csv")

# ---------- (3) w_ft curves ----------
print("\n(3) w_ft: final = w z(FT) + (1-w) z(TO); in-sample and nested (grid chosen on 4 user folds)")
grid = np.round(np.arange(0, 1.01, 0.1), 2)
rows = []
pools3 = {"G3": POOLS["G3"], "PF7x": POOLS["PF7x"], "M2": POOLS["M2"], "Cd": POOLS["Cd"],
          "G3+NEW": POOLS["G3"] + sum((FAM[f] for f in NEW), []), "NEW": sum((FAM[f] for f in NEW), [])}
for name, ms in pools3.items():
    zt_ = zfeed(branch(ms, "val", "to"), "val"); zf_ = zfeed(branch(ms, "val", "ft"), "val")
    nds = np.stack([pf(w * zf_ + (1 - w) * zt_) for w in grid])
    base = nds[list(grid).index(0.5)]
    nest = np.zeros(vm["n"]); picks = []
    for k in range(5):
        j = int(np.argmax(nds[:, fold != k].mean(1))); picks.append(grid[j]); nest[fold == k] = nds[j, fold == k]
    curve = " ".join(f"{w:.1f}:{(nds[i].mean() - base.mean()) * 1e4:+.1f}" for i, w in enumerate(grid))
    r = dict(pool=name, w_best=grid[int(np.argmax(nds.mean(1)))], nested=pse(nest - base)[0], nested_se=pse(nest - base)[1],
             to_only=pse(nds[0] - base)[0], to_only_se=pse(nds[0] - base)[1], picks=",".join(map(str, picks)))
    rows.append(r)
    print(f"  {name:7s} curve (x1e4 vs w=0.5): {curve} | nested vs 0.5 {r['nested']:+.5f}+-{r['nested_se']:.5f} picks {r['picks']}", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a4_wft.csv")

# ---------- (4) Optuna per-family branch weights, nested ----------
print("\n(4) Optuna TPE per-family (FT, TO) weights for G3+NEW, nested over the 5 FT user folds (60 trials per outer fold)")
fams4 = ["F10", "F7x", "F7x70"] + list(NEW)
FZ = {f: (zfeed(branch(FAM[f], "val", "ft"), "val"), zfeed(branch(FAM[f], "val", "to"), "val")) for f in fams4}
nm = {f: len(FAM[f]) for f in fams4}
fid = vm["fid"]


def score(params):
    sf = sum(params[f"wf_{f}"] * nm[f] * FZ[f][0] for f in fams4)
    st = sum(params[f"wt_{f}"] * nm[f] * FZ[f][1] for f in fams4)
    return params["w_ft"] * zfeed(sf, "val") + (1 - params["w_ft"]) * zfeed(st, "val")


eq = {**{f"wf_{f}": 1.0 for f in fams4}, **{f"wt_{f}": 1.0 for f in fams4}, "w_ft": 0.5}
nd_eq = pf(score(eq)); nd_g3 = pf(final(POOLS["G3"], "val"))
nest = np.zeros(vm["n"]); fitted = []
for k in range(5):
    trf = fold[np.arange(vm["n"])] != k

    def obj(trial):
        p = {**{f"wf_{f}": trial.suggest_float(f"wf_{f}", 0.0, 2.0) for f in fams4},
             **{f"wt_{f}": trial.suggest_float(f"wt_{f}", 0.0, 2.0) for f in fams4},
             "w_ft": trial.suggest_float("w_ft", 0.0, 1.0)}
        return pf(score(p))[trf].mean()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
    st.enqueue_trial(eq)
    st.optimize(obj, n_trials=60)
    bp = st.best_params
    fitted.append(bp)
    nest[~trf] = pf(score(bp))[~trf]
    print(f"  outer fold {k}: inner best {st.best_value:.5f} (eq {pf(score(eq))[trf].mean():.5f}) -> held-out "
          f"{nest[~trf].mean():.5f} vs eq {nd_eq[~trf].mean():.5f}; w_ft {bp['w_ft']:.2f}", flush=True)
print(f"  NESTED Optuna weights vs equal G3+NEW: {fmt(nest - nd_eq)} | vs G3: {fmt(nest - nd_g3)} | equal G3+NEW vs G3 {fmt(nd_eq - nd_g3)}")
pl.DataFrame([dict(fold=k, **p) for k, p in enumerate(fitted)]).write_csv(OUT / "a4_optuna_weights.csv")

# ---------- (5) trunk pre-encoder inputs vs FT gain ----------
print("\n(5) member FT gain (final - TO, val) by trunk input set")
groups = {"F5cd all (cvnd, no prew)": FAM["F5cd"][:3], "F5cd allprew (cvnd_prew)": FAM["F5cd"][3:],
          "F5 v12_all (plateau)": ["v12_all_e10", "v12_all_s81", "v12_all_s82"],
          "F5 v12_allprew (plateau)": ["v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"],
          "F6 (cvnd)": FAM["F6"], "F7 (cvnd_pre2s)": FAM["F7"], "F7p (pre2s plateau)": FAM_EXTRA["F7p"]}
rows = []
for g, ms in groups.items():
    d = np.zeros(vm["n"]); dft = np.zeros(vm["n"])
    for m in ms:
        t = member_z(m, "val", "to").astype(np.float64); f = member_z(m, "val", "ft")
        a = pf(t)
        d += (pf(0.5 * t + 0.5 * f) - a) / len(ms); dft += (pf(f) - a) / len(ms)
    rows.append(dict(group=g, n=len(ms), fin_gain=pse(d)[0], fin_se=pse(d)[1], ft_gain=pse(dft)[0], ft_se=pse(dft)[1]))
    print(f"  {g:28s} n={len(ms)} final-TO {fmt(d)} | FT-TO {fmt(dft)}")
pl.DataFrame(rows).write_csv(OUT / "a4_trunk_inputs.csv")
