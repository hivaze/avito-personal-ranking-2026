"""A3: noise structure of the curves.
 (1) within-run eval-to-eval jitter as a function of lr (iterate noise; the val set is fixed, so jitter is model noise):
     sd of successive differences / sqrt(2), on the 1000-step re-cool grid and the 2500-step grid;
 (2) between-seed spread at fixed steps and its persistence (does a seed's plateau level predict its final level?);
 (3) selection luck decomposition of the re-cool val excess: luck = value at the selected eval minus a smooth trend
     (cubic in sqrt(k) over the cooldown evals), for re-cools and for the F7 originals they are compared with.
Prints -> a3_noise.txt; writes a3_jitter.csv, a3_luck.csv."""
import numpy as np
import pandas as pd

from fams import FAM10, OUT, RECOOL, T_COOL, curve, fmt, load, mse, orig_of, plateau, root_of

C, R = load()

# ---------- (1) jitter vs lr
rows = []
lr_bins = [(1.5e-3, 2.1e-3), (8e-4, 1.5e-3), (4e-4, 8e-4), (2e-4, 4e-4), (1e-4, 2e-4), (0, 1e-4)]
groups = {"re-cool m02 (F7x,F7x70,cs813)": RECOOL["m02 (cs811) =F7x"][0] + RECOOL["m02 p70k (cs812) =F7x70"][0]
          + RECOOL["m02 (cs813)"][0],
          "re-cool c0 (mix.1)": RECOOL["c0 (mix.1, cs811)"][0] + RECOOL["c0 (mix.1, cs813)"][0],
          "re-cool m0": RECOOL["m0 (no pool)"][0]}
for g, runs in groups.items():
    for run in runs:
        c = curve(C, run)
        dv, dh = np.diff(c.val.values), np.diff(c.hold.values)
        lr = c.lr.values[1:]
        for lo, hi in lr_bins:
            w = (lr > lo) & (lr <= hi)
            if w.sum() >= 2:
                rows.append(dict(group=g, run=run, lr_bin=f"{lo:.0e}-{hi:.0e}", n=int(w.sum()),
                                 jit_v=np.sqrt(np.mean(dv[w] ** 2) / 2), jit_h=np.sqrt(np.mean(dh[w] ** 2) / 2)))
# 2500-grid 10-epoch runs: plateau (lr 2e-3 or 1e-3, steps 40k-77.5k) and cooldown
for fam, runs in FAM10.items():
    for run in runs:
        c = curve(C, run)
        if c.step.max() < 77500:
            continue
        for name, w in [("plateau 40-77.5k", (c.step >= 40000) & (c.step <= 77500)), ("cooldown", c.step > T_COOL)]:
            x = c[w]
            if len(x) < 3:
                continue
            dv, dh = np.diff(x.val.values), np.diff(x.hold.values)
            rows.append(dict(group=fam, run=run, lr_bin=name, n=len(dv), jit_v=np.sqrt(np.mean(dv ** 2) / 2),
                             jit_h=np.sqrt(np.mean(dh ** 2) / 2)))
J = pd.DataFrame(rows)
J.to_csv(OUT / "a3_jitter.csv", index=False)
print("== A3.1 Within-run eval-to-eval jitter (rms successive difference / sqrt 2), val | hold, by lr bin")
print("   NB on the 2500 grid the plateau value includes the slow trend; the re-cool grid is 1000 steps.")
for g in list(groups) + [f for f in FAM10 if f in J.group.values]:
    j = J[J.group == g]
    print(f"{g:32s} " + "  ".join(f"[{b}] {j[j.lr_bin == b].jit_v.mean():.5f}|{j[j.lr_bin == b].jit_h.mean():.5f}"
                                   for b in j.lr_bin.unique()))

# ---------- (2) between-seed spread and persistence
print("\n== A3.2 Between-seed spread (sd over seeds) of val at fixed steps, and persistence across phases")
pers = []
for fam, runs in FAM10.items():
    cs = {r: curve(C, r) for r in runs if r in R.index and bool(R.loc[r, "complete"])}
    cs = {r: c for r, c in cs.items() if c.step.max() == 98350}
    if len(cs) < 3:
        continue
    get = lambda c, lo, hi, col="val": c.loc[(c.step >= lo) & (c.step <= hi), col].mean()
    X = pd.DataFrame({r: dict(e20=get(c, 17500, 22500), e40=get(c, 37500, 42500), P=get(c, 67500, 77500),
                              Ph=get(c, 67500, 77500, "hold"), sel=c.val.values[int(np.argmax(c.hold.values))],
                              hsel=c.hold.values.max(), last=c.val.values[-1]) for r, c in cs.items()}).T
    sds = X.std(ddof=1)
    print(f"{fam:26s} n{len(X)} sd: v20k {sds.e20:.5f} v40k {sds.e40:.5f} P {sds.P:.5f} sel {sds.sel:.5f} last {sds.last:.5f} "
          f"| P_hold {sds.Ph:.5f} sel_hold {sds.hsel:.5f}")
    Z = (X - X.mean()) / X.std(ddof=1)
    Z["fam"] = fam
    pers.append(Z)
Zp = pd.concat(pers)
print("   pooled within-family correlations (standardized per family, n runs = %d):" % len(Zp))
for a, b in [("e20", "P"), ("e40", "P"), ("P", "sel"), ("e40", "sel"), ("P", "last"), ("Ph", "hsel"), ("P", "hsel"),
             ("sel", "hsel")]:
    r = np.corrcoef(Zp[a].astype(float), Zp[b].astype(float))[0, 1]
    print(f"   corr({a}, {b}) = {r:+.2f}")

# ---------- (3) luck decomposition
print("\n== A3.3 Selection luck: value at the holdout-selected eval minus a smooth cooldown trend (cubic in sqrt(k));"
      " re-cool minus its F7 original, root-paired")


def trend_at(k, y, kq, deg=3):
    u = np.sqrt(np.asarray(k, float) / 19670.0)
    uq = np.sqrt(np.asarray(kq, float) / 19670.0)
    deg = min(deg, len(k) - 2)
    return np.polyval(np.polyfit(u, y, deg), uq)


def luck(run, orig):
    c = curve(C, run)
    if orig:
        cc = c[c.step > T_COOL]
        k = (cc.step - T_COOL).values
    else:
        cc, k = c, c.step.values.astype(float)
    sel = int(np.argmax(c.hold.values))
    ks = c.step.values[sel] - (T_COOL if orig else 0)
    if orig and c.step.values[sel] <= T_COOL:   # selected in the stable phase (not in F7)
        return dict(k_sel=ks, lv=np.nan, lh=np.nan, tv=np.nan, th=np.nan)
    tv, th = trend_at(k, cc.val.values, [ks])[0], trend_at(k, cc.hold.values, [ks])[0]
    return dict(k_sel=ks, lv=c.val.values[sel] - tv, lh=c.hold.values[sel] - th, tv=tv, th=th,
                v_sel=c.val.values[sel], h_sel=c.hold.values[sel])


L = []
for arm in ["c0 (mix.1, cs811)", "c0 (mix.1, cs813)", "m02 (cs811) =F7x", "m02 (cs813)", "m02 p70k (cs812) =F7x70",
            "m0 (no pool)"]:
    for run in RECOOL[arm][0]:
        a, o = luck(run, False), luck(orig_of(run), True)
        L.append(dict(arm=arm, run=run, root=root_of(run), **{f"a_{k}": v for k, v in a.items()},
                      **{f"o_{k}": v for k, v in o.items()}))
L = pd.DataFrame(L)
L.to_csv(OUT / "a3_luck.csv", index=False)
print(f"{'arm':26s} n  {'sel gap':>17s} = {'trend gap @sel':>17s} + {'luck gap':>17s} | luck_v arm/orig, luck_h arm/orig")
for arm in L.arm.unique():
    d = L[L.arm == arm]
    sel_gap = (d.a_v_sel - d.o_v_sel) - (d.a_h_sel - d.o_h_sel)
    tr_gap = (d.a_tv - d.o_tv) - (d.a_th - d.o_th)
    lk_gap = (d.a_lv - d.o_lv) - (d.a_lh - d.o_lh)
    print(f"{arm:26s} {len(d)}  {fmt(*mse(sel_gap)[:2]):>17s} = {fmt(*mse(tr_gap)[:2]):>17s} + {fmt(*mse(lk_gap)[:2]):>17s} | "
          f"{d.a_lv.mean():+.5f}/{d.o_lv.mean():+.5f}, {d.a_lh.mean():+.5f}/{d.o_lh.mean():+.5f}")
o7 = L[L.arm == "c0 (mix.1, cs811)"]
print(f"   F7 originals (7 roots): holdout luck at selection {fmt(*mse(o7.o_lh)[:2])}, val residual at the same eval "
      f"{fmt(*mse(o7.o_lv)[:2])}; selected k {o7.o_k_sel.mean():.0f}")

# ---------- (4) snapshot vs plateau
print("\n== A3.4 The @77500 / @70000 snapshot evals vs the run's plateau mean P (re-cools start from the snapshot)")
for snap in (70000, 77500):
    dv, dh = [], []
    for r in range(101, 108):
        o = curve(C, f"v17_pre2s_s{r}")
        Pv, Ph, _ = plateau(o)
        x = o[o.step == snap]
        dv.append(x.val.iloc[0] - Pv)
        dh.append(x.hold.iloc[0] - Ph)
    print(f"   @{snap}: val-P {fmt(*mse(dv)[:2])}  hold-P {fmt(*mse(dh)[:2])}  gap {fmt(*mse(np.array(dv) - np.array(dh))[:2])}")
pg = []
for fam in ["F7 pre2s lr2 mix.1", "F9 pre2s lr1 mix.1", "F10 GRU lr2 mix.1", "F12 d192 lr2 mix.1", "a4u e10 lr2 mix.1",
            "cset e10 lr2 mix.1", "F13l pre2s lr1 m02", "F13g GRU lr2 m02", "d192 lr2 m02 (v29)", "F7 lr2 m02 scratch (v30)"]:
    for r in FAM10[fam]:
        if r in R.index:
            c = curve(C, r)
            if c.step.max() >= 77500:
                Pv, Ph, _ = plateau(c)
                pg.append(dict(fam=fam, run=r, gap=Pv - Ph))
pg = pd.DataFrame(pg)
print("   between-seed sd of the plateau val-minus-hold gap P_val - P_hold (what an unpaired comparison must beat): " +
      ", ".join(f"{f.split()[0]} {g.gap.std(ddof=1):.5f}" for f, g in pg.groupby("fam", sort=False) if len(g) > 2))
