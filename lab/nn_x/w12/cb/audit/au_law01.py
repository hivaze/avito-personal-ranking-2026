"""audit 1b: what (rho, gap) does the member rule need? The record's law is for the EQUAL mix. Here the same form is fitted for the
member rule's weight 0.1: delta(0.9 z(W16) + 0.1 z(X) vs W16) = a (1 - rho) - b gap, on every partner on disk: the tree models of
the level table + single NN members that are not in W16 (hard-label families, d128 students). Light: 2 threads."""
from au_common import *
light(2)
from pc_common import groups, paths
m = Meta(); P = {sp: w16_pool(m, sp) for sp in ("val", "hold")}; nd = {sp: ndcg_fast(P[sp], m.y[sp], m.off[sp], m.M[sp]) for sp in ("val", "hold")}
rows = []
def add(name, kind, zv, zh):
    r = dict(name=name, kind=kind)
    for sp, z in (("val", zv), ("hold", zh)):
        if z is None: continue
        e = ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp]); r[f"{sp}_gap"] = float((nd[sp] - e).mean()); r[f"{sp}_rho"] = corr(z, P[sp])
        for w in (0.05, 0.1, 0.2, 0.5):
            d = ndcg_fast((1 - w) * P[sp] + w * z, m.y[sp], m.off[sp], m.M[sp]) - nd[sp]; r[f"{sp}_d{w}"] = float(d.mean()); r[f"{sp}_s{w}"] = se(d)
    rows.append(r)
for name, pp in TREES.items():
    add(name, "tree", m.z("val", np.load(pp["val"])), m.z("hold", np.load(pp["hold"])) if "hold" in pp else None)
for f in sorted(OUT.glob("KL*_r*_val.npy")):
    add(f.name[:-8], "tree", m.z("val", np.load(f)), m.z("hold", np.load(str(f).replace("_val.npy", "_trH.npy"))))
g = groups()
for grp, take in (("base", 6), ("F10", 6), ("F7x", 7), ("F7x70", 3), ("pop1", 4), ("kd05", 4), ("kd10", 2), ("xf0", 3), ("kx05", 2)):
    for v in g[grp][:take]:
        pv, ph, _ = paths(v)
        if pv.exists() and ph.exists(): add(v, "nn:" + grp, m.z("val", np.load(pv)), m.z("hold", np.load(ph)))
json.dump(rows, open(OUT / "law01.json", "w"), indent=1)
print(f"{len(rows)} partners ({sum(r['kind']=='tree' for r in rows)} tree, {sum(r['kind']!='tree' for r in rows)} single NN members not in W16)")
for kind in sorted({r["kind"] for r in rows if r["kind"] != "tree"}):
    rr = [r for r in rows if r["kind"] == kind]
    print(f"  {kind:10s} n {len(rr)}: val gap {np.mean([r['val_gap'] for r in rr]):.5f} rho {np.mean([r['val_rho'] for r in rr]):.4f} d0.1 {np.mean([r['val_d0.1'] for r in rr]):+.5f} | hold gap {np.mean([r['hold_gap'] for r in rr]):.5f} rho {np.mean([r['hold_rho'] for r in rr]):.4f} d0.1 {np.mean([r['hold_d0.1'] for r in rr]):+.5f}")
for w in (0.05, 0.1, 0.2, 0.5):
    for sp in ("val", "hold", "both"):
        X, Y = [], []
        for r in rows:
            for s in (("val", "hold") if sp == "both" else (sp,)):
                if f"{s}_d{w}" in r: X.append([1 - r[f"{s}_rho"], -r[f"{s}_gap"]]); Y.append(r[f"{s}_d{w}"])
        X, Y = np.array(X), np.array(Y); co, *_ = np.linalg.lstsq(X, Y, rcond=None); pred = X @ co
        r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        s = f"w {w:4.2f} {sp:5s} (n {len(Y):3d}): delta = {co[0]:.5f} (1 - rho) - {co[1]:.5f} gap   R2 {r2:.3f}  rms resid {np.sqrt(np.mean((Y - pred)**2)):.5f}"
        if w in (0.1, 0.5): s += "  | break-even gap at rho 0.84 / 0.90 / 0.95: " + " / ".join(f"{co[0]*(1-x)/co[1]:.4f}" for x in (0.84, 0.90, 0.95))
        if w == 0.1: s += "  | gap needed for delta >= +0.0005 at rho 0.84 / 0.90: " + " / ".join(f"{(co[0]*(1-x)-0.0005)/co[1]:.4f}" for x in (0.84, 0.90))
        print(s)
