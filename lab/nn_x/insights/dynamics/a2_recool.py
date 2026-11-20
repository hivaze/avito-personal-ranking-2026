"""A2: re-cools (r6 CD harness) vs the original F7 cooldowns, at matched cooldown position.

Lr schedules are identical functions of the cooldown step k (re-cool: k = step; original: k = step - 78680), so the
re-cool curve is interpolated onto the original eval grid (k = 1320, 3820, ..., 18820, 19670) and compared root-paired.
Reference level for the 'gain' view: the original run's plateau mean P (67.5k-77.5k).
Also: arm-vs-arm curves inside the harness (same roots, same data seed 811), the first-eval transient, the snapshot
start, the selection-luck decomposition of the re-cool val excess, and pool-loss memorisation signatures.
Writes a2_*.csv and prints (-> a2_recool.txt)."""
import numpy as np
import pandas as pd

from fams import K_COOL, OUT, RECOOL, SNAP, T_COOL, curve, fmt, load, ma3, mse, orig_of, plateau, root_of

C, R = load()
KO = np.array([1320, 3820, 6320, 8820, 11320, 13820, 16320, 18820, 19670])  # original cooldown eval positions


def recool_k(c, snap, n_ep):
    """Map a re-cool's steps to the original-cooldown position k. Plain 2-epoch re-cools from @77500/@70000 have
    exactly the original cooldown schedule (cooldown_frac 1, 19670 steps). Longer ones are mapped by lr fraction."""
    tot = c.step.max()
    if tot == K_COOL:
        return c.step.values.astype(float)
    return c.step.values / tot * K_COOL  # e3 / cf4: same 1-sqrt shape stretched; s2c2 handled separately


rows = []
for arm, (runs, snap) in RECOOL.items():
    for run in runs:
        if run not in R.index or not R.loc[run, "complete"]:
            continue
        c = curve(C, run)
        o = curve(C, orig_of(run))
        Pv, Ph, _ = plateau(o)
        sv = float(o.loc[o.step == snap, "val"].iloc[0]) if (o.step == snap).any() else np.nan
        sh = float(o.loc[o.step == snap, "hold"].iloc[0]) if (o.step == snap).any() else np.nan
        if arm.startswith("s2c2"):  # 2 stable epochs at lr 2e-3 then a 2-epoch cooldown: cooldown = last half
            cc = c[c.step > 19670].copy()
            k = (cc.step - 19670).values.astype(float)
        else:
            cc = c.copy()
            k = recool_k(c, snap, None)
        oc = o[o.step > T_COOL]
        ko, ov, oh = (oc.step - T_COOL).values, oc.val.values, oc.hold.values
        iv, ih = np.interp(ko, k, cc.val.values), np.interp(ko, k, cc.hold.values)
        sel = int(np.argmax(c.hold.values))
        osel = int(np.argmax(o.hold.values))
        rows.append(dict(arm=arm, run=run, root=root_of(run), Pv=Pv, Ph=Ph, snap_v=sv, snap_h=sh,
                         first_k=float(k[0]), first_dv=cc.val.values[0] - sv, first_dh=cc.hold.values[0] - sh,
                         o_first_dv=ov[0] - float(o.loc[o.step == 77500, "val"].iloc[0]),
                         o_first_dh=oh[0] - float(o.loc[o.step == 77500, "hold"].iloc[0]),
                         **{f"dv_{kk}": iv[i] - ov[i] for i, kk in enumerate(ko)},
                         **{f"dh_{kk}": ih[i] - oh[i] for i, kk in enumerate(ko)},
                         **{f"gv_{kk}": iv[i] - Pv for i, kk in enumerate(ko)},
                         **{f"gh_{kk}": ih[i] - Ph for i, kk in enumerate(ko)},
                         traj_dv=np.mean(iv - ov), traj_dh=np.mean(ih - oh),
                         sel_step=int(c.step.values[sel]), v_sel=c.val.values[sel], h_sel=c.hold.values[sel],
                         o_sel_step=int(o.step.values[osel]), o_v_sel=o.val.values[osel], o_h_sel=o.hold.values[osel],
                         v_last=c.val.values[-1], h_last=c.hold.values[-1],
                         peak_ma3_v=ma3(cc.val.values).max(), peak_ma3_h=ma3(cc.hold.values).max(),
                         o_peak_ma3_v=ma3(ov).max(), o_peak_ma3_h=ma3(oh).max(),
                         peak_ma3_k=float(k[int(np.argmax(ma3(cc.val.values)))])))
D = pd.DataFrame(rows)
D.to_csv(OUT / "a2_recool_runs.csv", index=False)

print("== A2.1 Re-cool minus ORIGINAL F7 cooldown at matched cooldown position k (root-paired; mean+-SE over roots)")
print("   traj = mean over the 9 original eval positions (selection-free); sel = holdout-selected checkpoints (as used)")
print(f"{'arm':28s} n  {'traj dval':>17s} {'traj dhold':>17s} {'traj gap(v-h)':>17s} | {'sel dval':>17s} "
      f"{'sel dhold':>17s} {'sel gap':>17s} | sel step (orig k)")
for arm in RECOOL:
    d = D[D.arm == arm]
    if not len(d):
        continue
    print(f"{arm:28s} {len(d)}  {fmt(*mse(d.traj_dv)[:2]):>17s} {fmt(*mse(d.traj_dh)[:2]):>17s} "
          f"{fmt(*mse(d.traj_dv - d.traj_dh)[:2]):>17s} | {fmt(*mse(d.v_sel - d.o_v_sel)[:2]):>17s} "
          f"{fmt(*mse(d.h_sel - d.o_h_sel)[:2]):>17s} {fmt(*mse((d.v_sel - d.o_v_sel) - (d.h_sel - d.o_h_sel))[:2]):>17s} | "
          f"{d.sel_step.mean():6.0f} ({(d.o_sel_step - T_COOL).mean():6.0f})")

print("\n== A2.2 Re-cool minus original by position k: dval | dhold (mean over roots)")
print(f"{'arm':28s} " + " ".join(f"{k:>15d}" for k in KO))
for arm in RECOOL:
    d = D[D.arm == arm]
    if not len(d):
        continue
    print(f"{arm:28s} " + " ".join(f"{d[f'dv_{k}'].mean():+.4f}|{d[f'dh_{k}'].mean():+.4f}" for k in KO))
print("   SE of one cell (c0 cs811, 7 roots) val/hold: " +
      " ".join(f"{mse(D[D.arm == 'c0 (mix.1, cs811)'][f'dv_{k}'])[1]:.4f}/{mse(D[D.arm == 'c0 (mix.1, cs811)'][f'dh_{k}'])[1]:.4f}" for k in KO))

print("\n== A2.3 Cooldown gain vs the ORIGINAL plateau mean P (same roots), by position k: (val-P) | (hold-P)")
print(f"{'arm':28s} " + " ".join(f"{k:>15d}" for k in KO))
for arm in RECOOL:
    d = D[D.arm == arm]
    if not len(d):
        continue
    print(f"{arm:28s} " + " ".join(f"{d[f'gv_{k}'].mean():+.4f}|{d[f'gh_{k}'].mean():+.4f}" for k in KO))
o_rows = []
for r in range(101, 108):
    o = curve(C, f"v17_pre2s_s{r}")
    Pv, Ph, _ = plateau(o)
    oc = o[o.step > T_COOL]
    o_rows.append(dict(**{f"gv_{k}": v - Pv for k, v in zip(KO, oc.val)}, **{f"gh_{k}": h - Ph for k, h in zip(KO, oc.hold)}))
O = pd.DataFrame(o_rows)
print(f"{'ORIGINAL F7 (mix.1)':28s} " + " ".join(f"{O[f'gv_{k}'].mean():+.4f}|{O[f'gh_{k}'].mean():+.4f}" for k in KO))

print("\n== A2.4 First-eval transient: change from the @snapshot eval (original log) to the first re-cool eval (k=1000),"
      " vs the original's change 77.5k -> 80k (k=1320)")
for arm in RECOOL:
    d = D[D.arm == arm]
    if not len(d) or arm.startswith(("e3", "cf4", "s2c2")):
        continue
    print(f"{arm:28s} n{len(d)} re-cool dval {fmt(*mse(d.first_dv)[:2])} dhold {fmt(*mse(d.first_dh)[:2])} | original "
          f"dval {fmt(*mse(d.o_first_dv)[:2])} dhold {fmt(*mse(d.o_first_dh)[:2])} | diff gap(v-h) "
          f"{fmt(*mse((d.first_dv - d.first_dh) - (d.o_first_dv - d.o_first_dh))[:2])}")

print("\n== A2.5 Peak (3-eval moving average) and decline inside the harness")
for arm in RECOOL:
    d = D[D.arm == arm]
    if not len(d):
        continue
    print(f"{arm:28s} n{len(d)} peak_ma3 val-P {fmt(*mse(d.peak_ma3_v - d.Pv)[:2])} hold-P {fmt(*mse(d.peak_ma3_h - d.Ph)[:2])} "
          f"at k {d.peak_ma3_k.mean():6.0f} | last-peak val {fmt(*mse(d.v_last - d.peak_ma3_v)[:2])} hold "
          f"{fmt(*mse(d.h_last - d.peak_ma3_h)[:2])} | sel val-P {fmt(*mse(d.v_sel - d.Pv)[:2])} hold-P {fmt(*mse(d.h_sel - d.Ph)[:2])}")
print(f"{'ORIGINAL F7 (mix.1)':28s}     peak_ma3 val-P {fmt(*mse(D[D.arm == 'c0 (mix.1, cs811)'].o_peak_ma3_v - D[D.arm == 'c0 (mix.1, cs811)'].Pv)[:2])} "
      f"hold-P {fmt(*mse(D[D.arm == 'c0 (mix.1, cs811)'].o_peak_ma3_h - D[D.arm == 'c0 (mix.1, cs811)'].Ph)[:2])}")

print("\n== A2.6 Arm vs arm inside the harness (same snapshot, same data seed 811 => data-order paired), "
      "on the re-cool's own 1000-step grid: dval | dhold by k")
pairs = [("m02 (cs811) =F7x", "c0 (mix.1, cs811)"), ("m0 (no pool)", "c0 (mix.1, cs811)"),
         ("m02 (cs811) =F7x", "m0 (no pool)"), ("c0 (mix.1, cs813)", "c0 (mix.1, cs811)"),
         ("m02 (cs813)", "m02 (cs811) =F7x"), ("w10 (mix.1)", "c0 (mix.1, cs811)"), ("d0 (mix.1)", "c0 (mix.1, cs811)"),
         ("d20 (mix.1)", "c0 (mix.1, cs811)"), ("aa5 (mix.1)", "c0 (mix.1, cs811)"), ("pp02 (mix.1)", "c0 (mix.1, cs811)"),
         ("aug (mix.1)", "c0 (mix.1, cs811)")]
grid = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 19670]
arm_rows = []
for a, b in pairs:
    ra, rb = RECOOL[a][0], RECOOL[b][0]
    out_v, out_h = [], []
    for kk in grid:
        dv, dh = [], []
        for x in ra:
            r = root_of(x)
            y = [z for z in rb if root_of(z) == r]
            if not y or x not in R.index or y[0] not in R.index:
                continue
            ca, cb = curve(C, x), curve(C, y[0])
            va, vb = ca[ca.step == kk], cb[cb.step == kk]
            if len(va) and len(vb):
                dv.append(va.val.iloc[0] - vb.val.iloc[0])
                dh.append(va.hold.iloc[0] - vb.hold.iloc[0])
        mv, sv, n = mse(dv)
        mh, sh, _ = mse(dh)
        out_v.append(f"{mv:+.4f}")
        out_h.append(f"{mh:+.4f}")
        arm_rows.append(dict(a=a, b=b, k=kk, dv=mv, se_v=sv, dh=mh, se_h=sh, n=n))
    print(f"{a} - {b} (n{n}):\n   k    " + " ".join(f"{k:>7d}" for k in grid) + "\n   val  " + " ".join(f"{x:>7s}" for x in out_v)
          + "\n   hold " + " ".join(f"{x:>7s}" for x in out_h))
pd.DataFrame(arm_rows).to_csv(OUT / "a2_arm_pairs.csv", index=False)

print("\n== A2.7 Pool-batch training loss minus regular-batch loss (cd_pool - cd_reg) by cooldown position: memorisation"
      " signature (lower = pool fitted better than fresh regular batches; aux weight differs by p_random, so read the"
      " TREND and the between-arm difference at equal k)")
segs = [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 19670)]


def loss_gap(runs, kmap):
    out = []
    for lo, hi in segs:
        vals = []
        for run in runs:
            if run not in R.index:
                continue
            c = curve(C, run)
            k = kmap(c)
            w = (k > lo) & (k <= hi) & c.cd_pool.notna()
            if w.sum():
                vals.append(np.nanmean(c.cd_pool[w] - c.cd_reg[w]))
        out.append(mse(vals)[:2])
    return out


from fams import FAM10
groups = [("ORIGINAL F7 mix.1 (scratch)", FAM10["F7 pre2s lr2 mix.1"], lambda c: (c.step - T_COOL).values),
          ("F9 lr1 mix.1 (scratch)", FAM10["F9 pre2s lr1 mix.1"], lambda c: (c.step - T_COOL).values),
          ("F10 GRU mix.1 (scratch)", FAM10["F10 GRU lr2 mix.1"], lambda c: (c.step - T_COOL).values),
          ("F12 d192 mix.1 (scratch)", FAM10["F12 d192 lr2 mix.1"], lambda c: (c.step - T_COOL).values),
          ("F5 mix.3 (scratch)", FAM10["F5 cvnd lr2 mix.3"], lambda c: (c.step - T_COOL).values),
          ("F7 m02 scratch (v30)", FAM10["F7 lr2 m02 scratch (v30)"], lambda c: (c.step - T_COOL).values),
          ("F13l lr1 m02 (scratch)", FAM10["F13l pre2s lr1 m02"], lambda c: (c.step - T_COOL).values),
          ("F13g GRU m02 (scratch)", FAM10["F13g GRU lr2 m02"], lambda c: (c.step - T_COOL).values),
          ("d192 m02 (scratch)", FAM10["d192 lr2 m02 (v29)"], lambda c: (c.step - T_COOL).values),
          ("c0 re-cool mix.1", RECOOL["c0 (mix.1, cs811)"][0], lambda c: c.step.values),
          ("m02 re-cool", RECOOL["m02 (cs811) =F7x"][0], lambda c: c.step.values),
          ("pp02 re-cool (pool +47%)", RECOOL["pp02 (mix.1)"][0], lambda c: c.step.values)]
print(f"{'group':30s} " + " ".join(f"{f'k {lo // 1000}-{hi // 1000}k':>17s}" for lo, hi in segs))
for name, runs, km in groups:
    g = loss_gap(runs, km)
    print(f"{name:30s} " + " ".join(f"{fmt(m, s, d=3):>17s}" for m, s in g))
