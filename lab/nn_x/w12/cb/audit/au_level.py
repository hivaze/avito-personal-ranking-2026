"""audit 1: level table of every tree model on disk vs the W16 train-only pool."""
from au_common import *
m = Meta()
P = {sp: w16_pool(m, sp) for sp in ("val", "hold", "test")}
nd = {sp: ndcg_fast(P[sp], m.y[sp], m.off[sp], m.M[sp]) for sp in ("val", "hold")}
print(f"W16 train-only pool: val {nd['val'].mean():.5f} (n {len(nd['val'])})  holdout {nd['hold'].mean():.5f} (n {len(nd['hold'])})")
Z = {}
for name, pp in TREES.items():
    Z[name] = {sp: m.z(sp, np.load(p)) for sp, p in pp.items()}
key = lambda s: [k for k in Z if k.startswith(s)][0]
Z["BUNDLE B2_hard = z-mean(cb26_e3, w25_A2c)"] = {sp: (Z[key("cb26_e3")][sp] + Z[key("w25_A2c")][sp]) / 2 for sp in ("val", "hold", "test")}
Z["BUNDLE B3_mixed = z-mean(cb26_e3, w25_A2c, lgb_K_134118)"] = {sp: (Z[key("cb26_e3")][sp] + Z[key("w25_A2c")][sp] + Z[key("lgb_K_xm_134118")][sp]) / 3 for sp in ("val", "hold")}
rows = []
for name, z in Z.items():
    r = dict(name=name)
    for sp in ("val", "hold"):
        if sp not in z: continue
        e = ndcg_fast(z[sp], m.y[sp], m.off[sp], m.M[sp]); r[f"{sp}_ndcg"] = e.mean(); g = nd[sp] - e
        r[f"{sp}_gap"] = g.mean(); r[f"{sp}_gap_se"] = se(g)
        r[f"{sp}_rho"] = corr(z[sp], P[sp]); r[f"{sp}_rhoS"] = corr(rank_pct(m, sp, z[sp]), rank_pct(m, sp, P[sp]))
        r[f"{sp}_law"] = 0.0186 * (1 - r[f"{sp}_rho"]) - 0.378 * abs(r[f"{sp}_gap"]); r[f"{sp}_law2"] = 0.0103 * (1 - r[f"{sp}_rho"]) - 0.304 * abs(r[f"{sp}_gap"])
        for w in (0.05, 0.1, 0.2, 0.5):
            d = ndcg_fast((1 - w) * P[sp] + w * z[sp], m.y[sp], m.off[sp], m.M[sp]) - nd[sp]
            r[f"{sp}_d{w}"] = d.mean(); r[f"{sp}_s{w}"] = se(d)
    if "test" in z: r["test_rho"] = corr(z["test"], P["test"])
    rows.append(r)
json.dump(rows, open(OUT / "level.json", "w"), indent=1)
g = lambda r, k, f="{:.5f}": (f.format(r[k]) if k in r else "   -   ")
print("\n== A. level, rho, law ==")
print(f"{'model':58s} {'val':>8s} {'hold':>8s} | {'gap_v':>16s} {'gap_h':>16s} | rho_v rhoS_v rho_h rho_t | law_v(rec) law_v(refit) law_h(rec) law_h(refit)")
for r in rows:
    gv = f"{r['val_gap']:.5f}+-{r['val_gap_se']:.5f}"; gh = f"{r['hold_gap']:.5f}+-{r['hold_gap_se']:.5f}" if 'hold_gap' in r else "       -        "
    print(f"{r['name'][:58]:58s} {g(r,'val_ndcg'):>8s} {g(r,'hold_ndcg'):>8s} | {gv:>16s} {gh:>16s} | {g(r,'val_rho','{:.4f}')} {g(r,'val_rhoS','{:.4f}')} {g(r,'hold_rho','{:.4f}')} {g(r,'test_rho','{:.4f}')} | {g(r,'val_law','{:+.5f}')} {g(r,'val_law2','{:+.5f}')} {g(r,'hold_law','{:+.5f}')} {g(r,'hold_law2','{:+.5f}')}")
print("\n== B. measured blend (1-w) z(W16) + w z(tree) minus z(W16), paired per feed ==")
for r in rows:
    s = f"{r['name'][:58]:58s}"
    for w in (0.05, 0.1, 0.2, 0.5):
        s += f" | w{w}: val {r[f'val_d{w}']:+.5f}+-{r[f'val_s{w}']:.5f}"
        s += f" hold {r[f'hold_d{w}']:+.5f}+-{r[f'hold_s{w}']:.5f}" if f"hold_d{w}" in r else " hold    -    "
    print(s)
print("\n== C. member rule at w 0.1: val delta >= 0 AND holdout delta >= 0 AND (z_val + z_hold)/sqrt(2) >= 2 ==")
for r in rows:
    if "hold_d0.1" not in r: print(f"{r['name'][:58]:58s} val z {r['val_d0.1']/r['val_s0.1']:+.2f}; no holdout predictions -> rule not computable"); continue
    zv, zh = r['val_d0.1'] / r['val_s0.1'], r['hold_d0.1'] / r['hold_s0.1']; zc = (zv + zh) / np.sqrt(2)
    ok = r['val_d0.1'] >= 0 and r['hold_d0.1'] >= 0 and zc >= 2
    print(f"{r['name'][:58]:58s} z_val {zv:+.2f} z_hold {zh:+.2f} pooled {zc:+.2f} -> {'PASS' if ok else 'FAIL'}")
