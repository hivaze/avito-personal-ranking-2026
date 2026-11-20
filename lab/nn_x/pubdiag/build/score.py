"""Score the pre-declared candidate pools (rule.json, declared 2026-11-12T14:42:30Z) against C0 = PF7x.
CPU only, read-only on preds/. Uses the per-member z cache of nn_x/pubdiag/overfit/cache (verified below against
preds/ens_PF7x_*). Output: score_out.txt (stdout) + score_out.json."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys, json
sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import numpy as np
from lib import Split, FAM, user_groups, P, H, ROOT  # noqa: E402
import pmix  # noqa: E402

OUT = ROOT / "nn_x/pubdiag/build"
C = ROOT / "nn_x/pubdiag/overfit/cache"
RULE = json.load(open(OUT / "rule.json"))
MEM = [str(m) for m in np.load(C / "members.npy")]
IDX = {m: i for i, m in enumerate(MEM)}
Zv_to = np.load(C / "Z_val_to.npy", mmap_mode="r"); Zv_ft = np.load(C / "Z_val_ft.npy", mmap_mode="r")
Zh = np.load(C / "Z_holdout_to.npy", mmap_mode="r")
V, HO = Split("val"), Split("holdout")
RECOOL = set(FAM["F7x"] + FAM["F7x70"] + FAM["F7b"])


def weights_for(fams, dedup=False):
    w = {}
    for f in fams:
        for m in FAM[f]:
            w[m] = 1.0
    if dedup:  # one member slot per F7 trunk: F7x / F7x70 of the same root share it
        for r in range(101, 108):
            a, b = f"r6_cd_m02_r{r}", f"r6_cd_m02_p70k_r{r}"
            if a in w and b in w:
                w[a] = w[b] = 0.5
    return w


def branch(Z, w):
    s = np.zeros(Z.shape[1], np.float64)
    for m, x in w.items():
        s += x * Z[IDX[m]]
    return (s / sum(w.values())).astype(np.float32)


def pool_scores(w):
    to, ft = branch(Zv_to, w), branch(Zv_ft, w)
    fin = 0.5 * V.z(ft) + 0.5 * V.z(to)
    return dict(val=V.nd(fin), val_to=V.nd(to), hold=HO.nd(branch(Zh, w)), to=to, ft=ft)


def cl_se(d, g):
    """user-cluster SE of mean(d)"""
    n = len(d); s = np.bincount(g, d - d.mean())
    return np.sqrt((s ** 2).sum()) / n


def wse(d, w):
    w = np.asarray(w, np.float64); m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)


# ---- verification against the saved pool arrays (spec identity) ----
w0 = weights_for(RULE["candidates"]["C0"]["families"])
for half, Z in (("to", Zv_to), ("ft", Zv_ft)):
    ref = np.load(P / f"ens_PF7x_{half}_val.npy")
    print(f"verify PF7x {half} val vs preds/ens_PF7x_{half}_val.npy: max abs diff {np.abs(branch(Z, w0) - ref).max():.2e}")

# ---- C5: holdout-only gates (a)+(b) ----
fam_all = ["F5cd", "F6", "F7", "F9", "F10", "F12", "F7x", "F7x70", "F7b"]
hs = {}
for f in fam_all:
    single = np.mean([HO.nd(Zh[IDX[m]]).mean() for m in FAM[f]])
    mix6 = HO.nd(branch(Zh, {m: 1.0 for m in FAM[f][:6]})).mean()
    hs[f] = (single, mix6)
b1, b2 = max(v[0] for v in hs.values()), max(v[1] for v in hs.values())
print("\nC5 holdout gates (single TO holdout / 6-member holdout mix; best %.5f / %.5f):" % (b1, b2))
hg = []
for f in fam_all:
    ok = hs[f][0] >= b1 - 0.002 and hs[f][1] >= b2 - 0.002
    print(f"  {f:6s} {hs[f][0]:.5f} / {hs[f][1]:.5f}  {'PASS' if ok else 'fail'}{'  (excluded by rule)' if f in ('F7', 'F7b') else ''}")
    if ok and f not in ("F7", "F7b"):
        hg.append(f)
print("  HG families:", hg)

cands = {}
for k, c in RULE["candidates"].items():
    fams = hg if k == "C5" else c["families"]
    cands[k] = (c["name"], fams, c["weighting"].startswith("root-dedup"))
if set(hg) == set(cands["C3"][1]) or set(hg) == set(cands["C0"][1]):
    print("C5 identical to C3/C0 -> dropped (rule)"); cands.pop("C5")
ctx = {"Cd": ["F5cd", "F6", "F7"], "Pall": ["F5cd", "F6", "F7", "F9", "F10", "F12"]}

# ---- slices ----
fv_ts, Mv = V.ts, V.M
tue = (fv_ts + 10800) // 86400 < 20720
cold = np.nan_to_num(np.load(ROOT / "nn_x/r6/feat_cache/val/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
npos = np.bincount(V.fid, V.y, V.n); ncold = np.bincount(V.fid, V.y * cold, V.n)
coldpos = (npos > 0) & (ncold == npos); warmpos = (npos > 0) & (ncold == 0)
vsl = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue), ("coldpos", coldpos), ("warmpos", warmpos)]
hdow = (((HO.ts + 10800) // 86400) + 3) % 7  # Mon=0
hsl = [("h_weekday", hdow < 5), ("h_weekend", hdow >= 5)]
gv, gh = user_groups(V.users), user_groups(HO.users)
Wp = np.load(ROOT / "nn_x/pubdiag/shift/p2_adv_weights.npz")["private_intrinsic+hour"]
PM = pmix.weights()["private"]

R = {}
for k, (name, fams, dd) in list(cands.items()) + [(c, (c, f, False)) for c, f in ctx.items()]:
    w = weights_for(fams, dd)
    s = pool_scores(w)
    share = sum(x for m, x in w.items() if m in RECOOL) / sum(w.values())
    R[k] = dict(name=name, fams=fams, dedup=dd, n=len(w), slots=sum(w.values()), recool=share, s=s)
    print(f"{k} {name}: n={len(w)} slots={sum(w.values()):.0f} recool={share:.3f} val {s['val'].mean():.5f} "
          f"val_to {s['val_to'].mean():.5f} holdout {s['hold'].mean():.5f}", flush=True)

# ---- CatBoost info row (never selectable) ----
cbs = {}
for nm in ["cb_tr300k_luo", "cb_tr300k_none", "cb_tr500k_pw"]:
    x = np.load(P / f"{nm}_val.npy"); cbs[nm] = V.nd(x).mean()
print("\nCatBoost single val:", {k: round(v, 5) for k, v in cbs.items()})
cb_best = max(cbs, key=cbs.get)
zcb = V.z(np.load(P / f"{cb_best}_val.npy"))

r0 = R["C0"]["s"]
rows = []
print("\n== deltas vs C0 = PF7x (paired per feed; val SE user-clustered) ==")
hdr = f"{'cand':14s} {'V':>17s} {'Vp(priv)':>17s} {'Vto':>17s} {'H':>17s} {'D=Vto-H':>17s} {'E0':>8s} {'E1':>8s} {'Ehalf':>8s} {'S':>8s} {'pmix':>17s}"
print(hdr)
for k, r in R.items():
    s = r["s"]
    dv = s["val"] - r0["val"]; dto = s["val_to"] - r0["val_to"]; dh = s["hold"] - r0["hold"]
    V_ = (dv.mean(), cl_se(dv, gv)); Vp = wse(dv, Wp); Vto = (dto.mean(), cl_se(dto, gv)); H_ = (dh.mean(), cl_se(dh, gh))
    D = (Vto[0] - H_[0], np.hypot(Vto[1], H_[1]))
    E0, E1, Eh = Vp[0], Vp[0] - D[0], Vp[0] - 0.5 * D[0]
    pm = pmix.wmean_se(dv, PM)
    sl = {nm: (dv[m].mean(), cl_se(dv[m], user_groups(V.users[m]))) for nm, m in vsl}
    sl.update({nm: (dh[m].mean(), cl_se(dh[m], user_groups(HO.users[m]))) for nm, m in hsl})
    elig = {}
    if k == "C0":
        elig = {"ref": True}
    elif k in ctx:
        elig = {"context": True}
    else:
        elig["e1_H>=-1SE"] = H_[0] >= -H_[1]
        elig["e2_Vp>=-1SE"] = Vp[0] >= -Vp[1]
        elig["e3_slices>=-2SE"] = all(m >= -2 * e for m, e in sl.values())
        elig["e4_D<=2SE"] = D[0] <= 2 * D[1]
        elig["e5_pmix>=-1SE"] = pm[0] >= -pm[1]
    row = dict(key=k, name=r["name"], families=r["fams"], dedup=r["dedup"], n=r["n"], slots=r["slots"], recool=r["recool"],
               val=float(s["val"].mean()), val_to=float(s["val_to"].mean()), hold=float(s["hold"].mean()),
               V=V_, Vp=Vp, Vto=Vto, H=H_, D=D, E0=E0, E1=E1, Ehalf=Eh, S=min(E0, E1), pmix=pm, slices=sl, elig=elig)
    rows.append(row)
    f = lambda t: f"{t[0]:+.5f}±{t[1]:.5f}"
    print(f"{k + ' ' + r['name']:14s} {f(V_):>17s} {f(Vp):>17s} {f(Vto):>17s} {f(H_):>17s} {f(D):>17s} {E0:+.5f} {E1:+.5f} {Eh:+.5f} {min(E0, E1):+.5f} {f(pm):>17s}")
print("\nslices (delta vs C0, z):")
for row in rows:
    print(f"  {row['key']:4s} " + "  ".join(f"{nm} {m:+.5f}({m / e if e > 0 else 0:+.1f})" for nm, (m, e) in row["slices"].items()))
print("\neligibility:")
for row in rows:
    print(f"  {row['key']:4s} {row['name']:6s} " + " ".join(f"{a}={'Y' if b else 'N'}" for a, b in row["elig"].items()))

# ---- selection (rule.json "selection") ----
el = [r for r in rows if r["key"] in RULE["candidates"] and all(r["elig"].values())]
el.sort(key=lambda r: -r["S"])
print("\nranking of eligible candidates by S = min(E0, E1):")
for r in el:
    print(f"  {r['key']} {r['name']:6s} S {r['S']:+.5f}  E0 {r['E0']:+.5f}  E1 {r['E1']:+.5f}  Ehalf {r['Ehalf']:+.5f}  recool {r['recool']:.3f}")
pick = el[0]
if len(el) > 1 and el[0]["S"] - el[1]["S"] < 0.00010:
    tied = [r for r in el if el[0]["S"] - r["S"] < 0.00010]
    tied.sort(key=lambda r: (r["recool"], -len(r["families"])))
    pick = tied[0]
    print("  tie (< 0.00010) among", [r["key"] for r in tied], "-> lower re-cooled share:", pick["key"])
e0best = max(el, key=lambda r: r["E0"]); e1best = max(el, key=lambda r: r["E1"])
others = [r for r in el if r["key"] != pick["key"]]
if others:
    hedge = max(others, key=lambda r: r["E0"]) if pick["key"] != e0best["key"] else max(others, key=lambda r: r["E1"])
else:
    hedge = None
print(f"\nSAFEST PICK (rule): {pick['key']} {pick['name']} (S {pick['S']:+.5f}); replaces PF7x: {pick['S'] > 0 and pick['key'] != 'C0'}")
print(f"HEDGE (rule): {hedge['key'] + ' ' + hedge['name'] if hedge else None}")

# ---- CatBoost info row: robust pick + 0.05 z(CatBoost) on val (val only; no holdout preds) ----
wp = weights_for(pick["families"], pick["dedup"]); sp = pool_scores(wp)
base = 0.5 * V.z(sp["ft"]) + 0.5 * V.z(sp["to"])
d = V.nd(V.z(base) + 0.05 * zcb) - V.nd(base)
print(f"\nINFO ONLY: {pick['name']} + 0.05*z({cb_best}) val delta {d.mean():+.5f} ± {cl_se(d, gv):.5f}")

json.dump(dict(rule_declared=RULE["declared_utc"], hg=hg, holdout_gates=hs, rows=rows, pick=pick["key"],
               hedge=hedge["key"] if hedge else None, catboost_single=cbs, catboost_info_delta=[d.mean(), cl_se(d, gv)]),
          open(OUT / "score_out.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
