"""A6: the M-dependence of the FT half at pool level.
(1) per M bin: member-level FT gain, pool-level FT gain, ensemble gain (pool - mean member) of the TO and FT branches;
(2) M-split blend w_ft(M<50) in {0, .25, .5} with w_ft(M>=50) = 0.5: per family pool (replication count), nested over the
    FT user folds for the main pools, and leave-family-out (rule decided on the OTHER families, applied to the held-out one);
(3) test-side: size of the FT shift (rms of z_FT - z_TO) by public/private, M bin and luo cold share; val vs test.
Writes a6_*.csv; stdout -> a6_msplit.log"""
from lib import *  # noqa

v30 = add_v30()
print("v30 members complete:", v30)
vm, tm = meta("val"), meta("test")
Sv, St = segments("val"), segments("test")
fold = user_folds()
small_v = vm["M"][vm["fid"]] < 50
fams = ["F5cd", "F6", "F7", "F9", "F10", "F12", "F7b", "F7x", "F7x70"] + list(NEW)
pools = {f: FAM[f] for f in fams}
pools.update({"G3": POOLS["G3"], "PF7x": POOLS["PF7x"], "M2": POOLS["M2"], "Cd": POOLS["Cd"],
              "G3+NEW": POOLS["G3"] + sum((FAM[f] for f in NEW), [])})

# ---------- (1) per M bin decomposition ----------
print("(1) per M bin (x1e4): member FT gain | pool FT gain | ens gain TO | ens gain FT")
rows = []
for name in ["G3", "F7x", "F7", "F6", "V28l"]:
    ms = pools[name]
    mt = np.mean([pf(member_z(m, "val", "to")) for m in ms], 0)
    mf = np.mean([pf(member_z(m, "val", "ft")) for m in ms], 0)
    mfin = np.mean([pf(0.5 * member_z(m, "val", "to").astype(np.float64) + 0.5 * member_z(m, "val", "ft")) for m in ms], 0)
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    pt, pfz, pfin = pf(zt), pf(zf), pf(0.5 * zt + 0.5 * zf)
    for l in ["M<20", "M20-49", "M50-99", "M100-199", "M200+"]:
        i = Sv["M"] == l
        r = dict(pool=name, M=l, mem_ft_gain=(mfin - mt)[i].mean(), pool_ft_gain=(pfin - pt)[i].mean(),
                 pool_ft_gain_se=pse((pfin - pt)[i])[1], ens_to=(pt - mt)[i].mean(), ens_ft=(pfz - mf)[i].mean(),
                 ftonly_mem=(mf - mt)[i].mean(), ftonly_pool=(pfz - pt)[i].mean())
        rows.append(r)
        print(f"  {name:5s} {l:9s} member {r['mem_ft_gain'] * 1e4:+6.1f} | pool {r['pool_ft_gain'] * 1e4:+6.1f} +- {r['pool_ft_gain_se'] * 1e4:4.1f} | "
              f"ens TO {r['ens_to'] * 1e4:+6.1f} ens FT {r['ens_ft'] * 1e4:+6.1f} | FT-only member {r['ftonly_mem'] * 1e4:+6.1f} pool {r['ftonly_pool'] * 1e4:+6.1f}")
pl.DataFrame(rows).write_csv(OUT / "a6_mbin.csv")

# ---------- (2) M-split w_ft ----------
print("\n(2) M-split blend: w_ft = ws for M<50, 0.5 for M>=50 (delta vs standard 0.5/0.5, val)")
rows = []
for name, ms in pools.items():
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    base = pf(0.5 * zt + 0.5 * zf)
    r = dict(pool=name, n=len(ms))
    for ws in (0.0, 0.25):
        w = np.where(small_v, ws, 0.5)
        d = pf(w * zf + (1 - w) * zt) - base
        r[f"ws{ws}"] = pse(d)[0]; r[f"ws{ws}_se"] = pse(d)[1]
    # nested choice among {0, .25, .5}
    cand = {ws: pf(np.where(small_v, ws, 0.5) * zf + (1 - np.where(small_v, ws, 0.5)) * zt) for ws in (0.0, 0.25, 0.5)}
    nest = np.zeros(vm["n"]); picks = []
    for k in range(5):
        best = max(cand, key=lambda ws: cand[ws][fold != k].mean()); picks.append(best)
        nest[fold == k] = cand[best][fold == k]
    r["nested"] = pse(nest - base)[0]; r["nested_se"] = pse(nest - base)[1]; r["picks"] = ",".join(map(str, picks))
    rows.append(r)
    print(f"  {name:7s} n={len(ms):2d} ws=0: {r['ws0.0']:+.5f}+-{r['ws0.0_se']:.5f} | ws=0.25: {r['ws0.25']:+.5f}+-{r['ws0.25_se']:.5f} | "
          f"nested {r['nested']:+.5f}+-{r['nested_se']:.5f} (picks {r['picks']})", flush=True)
df = pl.DataFrame(rows); df.write_csv(OUT / "a6_msplit.csv")
fam_rows = df.filter(pl.col("pool").is_in(fams))
print(f"  families with ws=0 > standard: {(fam_rows['ws0.0'] > 0).sum()} of {len(fam_rows)}; mean {fam_rows['ws0.0'].mean():+.5f}")
# leave-family-out: rule (ws in {0,.25,.5}) chosen by the mean over the OTHER single families, applied to the held-out one
print("  leave-family-out (rule chosen on the other families' mean delta):")
for f in fams:
    oth = fam_rows.filter(pl.col("pool") != f)
    means = {0.0: oth["ws0.0"].mean(), 0.25: oth["ws0.25"].mean(), 0.5: 0.0}
    ws = max(means, key=means.get)
    own = fam_rows.filter(pl.col("pool") == f)
    val = 0.0 if ws == 0.5 else own[f"ws{ws}"][0]
    print(f"    held-out {f:6s}: rule ws={ws} -> {val:+.5f}")

# ---------- (3) test-side FT shift size ----------
print("\n(3) rms of the G3 FT shift (z_FT - z_TO, per-feed z) by segment, val | test")
sv = zfeed(branch(POOLS["G3"], "val", "ft"), "val") - zfeed(branch(POOLS["G3"], "val", "to"), "val")
st = zfeed(branch(POOLS["G3"], "test", "ft"), "test") - zfeed(branch(POOLS["G3"], "test", "to"), "test")
rows = []
for var in ["M", "luo_cold", "day", "user"]:
    for l in sorted(set(np.unique(Sv[var])) | set(np.unique(St[var]))):
        iv = (Sv[var] == l)[vm["fid"]]; it = (St[var] == l)[tm["fid"]]
        r = dict(var=var, seg=l, rms_val=np.sqrt(np.mean(sv[iv] ** 2)) if iv.any() else np.nan,
                 rms_test=np.sqrt(np.mean(st[it] ** 2)) if it.any() else np.nan)
        rows.append(r)
        print(f"  {var:9s} {l:10s} {r['rms_val']:.4f} | {r['rms_test']:.4f}")
pl.DataFrame(rows).write_csv(OUT / "a6_test_shift.csv")
