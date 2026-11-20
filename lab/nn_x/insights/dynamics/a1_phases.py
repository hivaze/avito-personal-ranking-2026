"""A1: where the gain happens (stable vs cooldown), peak-vs-final declines, selected position, per 10-epoch family.
Also the cooldown curve shape per family (val/hold minus the run's own plateau mean) and the stable-phase slope.
Writes a1_runs.csv, a1_shape.csv and prints the tables (-> a1_phases.txt)."""
import numpy as np
import pandas as pd

from fams import FAM10, OUT, PAIRS_LR, PAIRS_MIX, T_COOL, curve, fmt, load, ma3, mse, plateau

C, R = load()
rows, shape = [], []
for fam, runs in FAM10.items():
    for run in runs:
        if run not in R.index:
            continue
        c = curve(C, run)
        if c.step.max() < 77500:
            continue
        Pv, Ph, _ = plateau(c)
        e = lambda s: c.loc[(c.step >= s - 2500) & (c.step <= s + 2500), "val"].mean()
        eh = lambda s: c.loc[(c.step >= s - 2500) & (c.step <= s + 2500), "hold"].mean()
        st = c[(c.step >= 40000) & (c.step <= 77500)]
        slope_v = np.polyfit(st.step / 1e4, st.val, 1)[0]   # val per 10k steps in the late stable phase
        slope_h = np.polyfit(st.step / 1e4, st.hold, 1)[0]
        cd = c[c.step > T_COOL].reset_index(drop=True)
        complete = bool(R.loc[run, "complete"]) and c.step.max() == 98350
        r = dict(fam=fam, run=run, complete=complete, n_cd=len(cd), v2500=c.val.iloc[0], h2500=c.hold.iloc[0],
                 v10k=e(10000), v20k=e(20000), v40k=e(40000), h10k=eh(10000), h40k=eh(40000), Pv=Pv, Ph=Ph,
                 slope_v=slope_v, slope_h=slope_h)
        if len(cd):
            mv, mh = ma3(cd.val), ma3(cd.hold)
            r.update(cd_peak_raw=cd.val.max(), cd_peak_step=int(cd.step[cd.val.idxmax()]),
                     cd_peak_ma3=mv.max(), cd_peak_ma3_step=int(cd.step[int(np.argmax(mv))]),
                     cdh_peak_ma3=mh.max(), cdh_peak_ma3_step=int(cd.step[int(np.argmax(mh))]),
                     v_last=cd.val.iloc[-1], h_last=cd.hold.iloc[-1], last_step=int(cd.step.iloc[-1]),
                     v80k=cd.val.iloc[0], h80k=cd.hold.iloc[0])
        if complete:
            k = int(np.argmax(c.hold.values))
            r.update(sel_step=int(c.step[k]), v_sel=c.val[k], h_sel=c.hold[k], vbest_step=int(c.step[c.val.idxmax()]),
                     v87500=float(c.loc[c.step == 87500, "val"].iloc[0]), h87500=float(c.loc[c.step == 87500, "hold"].iloc[0]))
        rows.append(r)
        for _, x in cd.iterrows():
            shape.append(dict(fam=fam, run=run, step=int(x.step), k=int(x.step) - T_COOL, lr=x.lr, dv=x.val - Pv,
                              dh=x.hold - Ph, cd_reg=x.cd_reg, cd_pool=x.cd_pool))
D = pd.DataFrame(rows)
S = pd.DataFrame(shape)
D.to_csv(OUT / "a1_runs.csv", index=False)
S.to_csv(OUT / "a1_shape.csv", index=False)

print("== A1.1 Phase decomposition per family (mean over seeds; +-SE). P = plateau mean of the 5 evals 67.5k-77.5k.")
print("   stable gain = P - v(10k, 3-eval mean); cd gain = cooldown peak (3-eval moving average) - P; "
      "final-P = last eval - P; decline = last - ma3 peak; sel-P = holdout-selected val - P")
hdr = f"{'family':26s} n  {'v10k':>7s} {'P_val':>7s} {'stable gain':>17s} {'slope/10k (40-77.5k)':>20s} " \
      f"{'cd gain(ma3 peak)':>18s} {'final-P':>17s} {'decline':>17s} {'sel-P':>17s} {'peak step':>9s}"
print(hdr)
for fam in FAM10:
    d = D[(D.fam == fam) & (D.n_cd > 0)]
    if not len(d):
        continue
    dc = d[d.complete]
    out = f"{fam:26s} {len(d)}  {d.v10k.mean():.5f} {d.Pv.mean():.5f} " \
          f"{fmt(*mse(d.Pv - d.v10k)[:2]):>17s} {fmt(*mse(d.slope_v)[:2]):>20s} " \
          f"{fmt(*mse(d.cd_peak_ma3 - d.Pv)[:2]):>18s} "
    out += f"{fmt(*mse(dc.v_last - dc.Pv)[:2]):>17s} {fmt(*mse(dc.v_last - dc.cd_peak_ma3)[:2]):>17s} " \
           f"{fmt(*mse(dc.v_sel - dc.Pv)[:2]):>17s} {d.cd_peak_ma3_step.mean():9.0f}" if len(dc) else " (incomplete)"
    print(out)
print("\n   same on the HOLDOUT")
for fam in FAM10:
    d = D[(D.fam == fam) & (D.n_cd > 0)]
    if not len(d):
        continue
    dc = d[d.complete]
    out = f"{fam:26s} {len(d)}  {d.h10k.mean():.5f} {d.Ph.mean():.5f} {fmt(*mse(d.Ph - d.h10k)[:2]):>17s} " \
          f"{fmt(*mse(d.slope_h)[:2]):>20s} {fmt(*mse(d.cdh_peak_ma3 - d.Ph)[:2]):>18s} "
    out += f"{fmt(*mse(dc.h_last - dc.Ph)[:2]):>17s} {fmt(*mse(dc.h_last - dc.cdh_peak_ma3)[:2]):>17s} " \
           f"{fmt(*mse(dc.h_sel - dc.Ph)[:2]):>17s} {d.cdh_peak_ma3_step.mean():9.0f}" if len(dc) else " (incomplete)"
    print(out)

print("\n== A1.2 Share of the total gain (v10k -> holdout-selected val) earned in the cooldown, complete runs")
for fam in FAM10:
    dc = D[(D.fam == fam) & D.complete]
    if len(dc) < 1:
        continue
    tot = dc.v_sel - dc.v10k
    print(f"{fam:26s} n{len(dc)} total {tot.mean():+.5f}  cooldown part {(dc.v_sel - dc.Pv).mean():+.5f} "
          f"({100 * (dc.v_sel - dc.Pv).mean() / tot.mean():.0f}%)   holdout: total {(dc.h_sel - dc.h10k).mean():+.5f} "
          f"cooldown part {(dc.h_sel - dc.Ph).mean():+.5f} ({100 * (dc.h_sel - dc.Ph).mean() / (dc.h_sel - dc.h10k).mean():.0f}%)")

print("\n== A1.3 Cooldown curve shape: mean (val - P_val) | (hold - P_hold) at each cooldown eval, per family")
steps = sorted(S.step.unique())
print(f"{'family':26s} " + " ".join(f"{s:>13d}" for s in steps))
for fam in FAM10:
    s = S[S.fam == fam]
    if not len(s):
        continue
    g = s.groupby("step")
    print(f"{fam:26s} " + " ".join(f"{g.dv.mean().get(st, np.nan):+.4f}|{g.dh.mean().get(st, np.nan):+.4f}"
                                   if st in g.groups else f"{'':>13s}" for st in steps))

print("\n== A1.4 mix.1 -> m02 from scratch, matched family pairs: cooldown shape difference (m02 - mix.1) of (val-P)")
for a, b in PAIRS_MIX:
    sa, sb = S[S.fam == a].groupby("step"), S[S.fam == b].groupby("step")
    line = []
    for st in steps:
        if st in sa.groups and st in sb.groups:
            xa, xb = sa.get_group(st), sb.get_group(st)
            d = xb.dv.mean() - xa.dv.mean()
            se = np.sqrt(xa.dv.var(ddof=1) / len(xa) + xb.dv.var(ddof=1) / len(xb)) if len(xa) > 1 and len(xb) > 1 else np.nan
            line.append(f"{st}:{d:+.4f}+-{se:.4f}(n{len(xb)})")
    print(f"{b} - {a}:\n   " + "  ".join(line))
    da, db = D[(D.fam == a) & D.complete], D[(D.fam == b) & D.complete]
    if len(db):
        print(f"   selected step: mix.1 {da.sel_step.mean():.0f} (sd {da.sel_step.std():.0f}) vs m02 {db.sel_step.mean():.0f} "
              f"(sd {db.sel_step.std():.0f}); decline last-ma3peak val {(da.v_last - da.cd_peak_ma3).mean():+.5f} vs "
              f"{(db.v_last - db.cd_peak_ma3).mean():+.5f}; hold {(da.h_last - da.cdh_peak_ma3).mean():+.5f} vs "
              f"{(db.h_last - db.cdh_peak_ma3).mean():+.5f}; peak(ma3) step val {da.cd_peak_ma3_step.mean():.0f} vs "
              f"{db.cd_peak_ma3_step.mean():.0f}; P_val {da.Pv.mean():.5f} vs {db.Pv.mean():.5f}; "
              f"cd gain {(da.cd_peak_ma3 - da.Pv).mean():+.5f} vs {(db.cd_peak_ma3 - db.Pv).mean():+.5f}")

print("\n== A1.5 Selection: holdout-selected step vs val-best step, val at sel / last / 87500 (complete runs)")
for fam in FAM10:
    dc = D[(D.fam == fam) & D.complete]
    if not len(dc):
        continue
    print(f"{fam:26s} n{len(dc)} sel_step {dc.sel_step.mean():7.0f} vbest_step {dc.vbest_step.mean():7.0f} | "
          f"v_sel {dc.v_sel.mean():.5f} v_last {dc.v_last.mean():.5f} v87.5k {dc.v87500.mean():.5f} "
          f"v_ma3peak {dc.cd_peak_ma3.mean():.5f} | sel-last {fmt(*mse(dc.v_sel - dc.v_last)[:2])} "
          f"87.5k-sel {fmt(*mse(dc.v87500 - dc.v_sel)[:2])} | hold sel-last {fmt(*mse(dc.h_sel - dc.h_last)[:2])}")

print("\n== A1.6 lr 1e-3 vs 2e-3 over time (family-mean val | hold difference lr1 - lr2 by step bin)")
bins = [(2500, 7500), (10000, 20000), (22500, 40000), (42500, 60000), (62500, 77500), (80000, 87500), (90000, 98350)]
for a, b in PAIRS_LR:
    out = []
    for lo, hi in bins:
        ca = C[C.run.isin(FAM10[a]) & (C.step >= lo) & (C.step <= hi)].groupby("run")[["val", "hold"]].mean()
        cb = C[C.run.isin(FAM10[b]) & (C.step >= lo) & (C.step <= hi)].groupby("run")[["val", "hold"]].mean()
        if len(ca) and len(cb):
            dv = cb.val.mean() - ca.val.mean()
            se = np.sqrt(ca.val.var(ddof=1) / len(ca) + cb.val.var(ddof=1) / len(cb)) if len(ca) > 1 and len(cb) > 1 else np.nan
            dh = cb.hold.mean() - ca.hold.mean()
            seh = np.sqrt(ca.hold.var(ddof=1) / len(ca) + cb.hold.var(ddof=1) / len(cb)) if len(ca) > 1 and len(cb) > 1 else np.nan
            out.append(f"[{lo // 1000}-{hi // 1000}k] {dv:+.5f}+-{se:.5f} | {dh:+.5f}+-{seh:.5f}")
    print(f"{b} - {a}:\n   " + "\n   ".join(out))
    da, db = D[(D.fam == a) & D.complete], D[(D.fam == b) & D.complete]
    if len(da) and len(db):
        print(f"   selected: val {db.v_sel.mean() - da.v_sel.mean():+.5f} hold {db.h_sel.mean() - da.h_sel.mean():+.5f}; "
              f"ma3 cd peak val {db.cd_peak_ma3.mean() - da.cd_peak_ma3.mean():+.5f}; plateau P "
              f"{db.Pv.mean() - da.Pv.mean():+.5f}; cd gain {((db.cd_peak_ma3 - db.Pv).mean() - (da.cd_peak_ma3 - da.Pv).mean()):+.5f}; "
              f"decline {((db.v_last - db.cd_peak_ma3).mean() - (da.v_last - da.cd_peak_ma3).mean()):+.5f}")
