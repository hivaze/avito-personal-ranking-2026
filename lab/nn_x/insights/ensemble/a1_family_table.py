"""A1: per-family member / pool numbers for the TO, FT and final (0.5/0.5) branches; FT gain at member and pool
level; FT-vs-TO correlation; FT epoch curves from nn/logs/*_n2cpv3_ft.json.
Writes a1_family_table.csv, a1_ftcurves.csv and prints a text table."""
import json
from lib import *  # noqa

v30 = add_v30()
print("v30 complete members:", v30)
FAMS = dict(FAM); FAMS.update(FAM_EXTRA)
rows, curves = [], []
for fam, ms in FAMS.items():
    has_h = all((P / "holdout" / f"nn_{m}_holdout.npy").exists() for m in ms)
    # member level
    to_v, ft_v, fin_v, hd, gain_fin, gain_ft, cz = [], [], [], [], [], [], []
    d_fin = np.zeros(meta("val")["n"]); d_ft = np.zeros(meta("val")["n"])
    for m in ms:
        zt = member_z(m, "val", "to").astype(np.float64); zf = member_z(m, "val", "ft").astype(np.float64)
        a = pf(zt); b = pf(zf); c = pf(0.5 * zt + 0.5 * zf)
        to_v.append(a.mean()); ft_v.append(b.mean()); fin_v.append(c.mean())
        d_fin += (c - a) / len(ms); d_ft += (b - a) / len(ms)
        cz.append(np.corrcoef(zt, zf)[0, 1])
        if has_h:
            hd.append(pf(member_z(m, "holdout", "to"), "holdout").mean())
        js = ROOT / f"nn/logs/{m}_n2cpv3_ft.json"
        if not js.exists():
            js = ROOT / f"nn_x/r6/logs/{m}_n2cpv3_ft.json"
        if js.exists():
            d = json.load(open(js))
            curves.append(dict(fam=fam, member=m, **{f"ep{i}": v for i, v in enumerate(d["oof_by_epoch"])},
                               ema_final=d.get("oof_ndcg_ema_final"), snap=d.get("oof_ndcg_snapshot")))
    # pool level
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    pt, pfz, pfin = pf(zt), pf(zf), pf(0.5 * zt + 0.5 * zf)
    zt_te = zfeed(branch(ms, "test", "to"), "test"); zf_te = zfeed(branch(ms, "test", "ft"), "test")
    ph = pf(branch(ms, "holdout", "to"), "holdout").mean() if has_h else np.nan
    r = dict(fam=fam, n=len(ms),
             mem_to=np.mean(to_v), mem_ft=np.mean(ft_v), mem_fin=np.mean(fin_v), mem_hold=np.mean(hd) if hd else np.nan,
             mem_gain_fin=pse(d_fin)[0], mem_gain_fin_se=pse(d_fin)[1], mem_gain_ft=pse(d_ft)[0], mem_gain_ft_se=pse(d_ft)[1],
             mem_corr_ft_to=np.mean(cz),
             pool_to=pt.mean(), pool_ft=pfz.mean(), pool_fin=pfin.mean(), pool_hold=ph,
             pool_gain_fin=pse(pfin - pt)[0], pool_gain_fin_se=pse(pfin - pt)[1],
             pool_gain_ft=pse(pfz - pt)[0], pool_gain_ft_se=pse(pfz - pt)[1],
             pool_corr_ft_to_val=np.corrcoef(zt, zf)[0, 1], pool_corr_ft_to_test=np.corrcoef(zt_te, zf_te)[0, 1])
    rows.append(r)
    print(f"{fam:6s} n={len(ms)} member TO {r['mem_to']:.5f} FT {r['mem_ft']:.5f} fin {r['mem_fin']:.5f} hold {r['mem_hold']:.5f} | "
          f"mem gain fin {r['mem_gain_fin']:+.5f}+-{r['mem_gain_fin_se']:.5f} ft {r['mem_gain_ft']:+.5f} corr {r['mem_corr_ft_to']:.4f} | "
          f"pool TO {r['pool_to']:.5f} FT {r['pool_ft']:.5f} fin {r['pool_fin']:.5f} hold {ph:.5f} gain {r['pool_gain_fin']:+.5f}+-{r['pool_gain_fin_se']:.5f} "
          f"ft-only {r['pool_gain_ft']:+.5f} corr v/t {r['pool_corr_ft_to_val']:.4f}/{r['pool_corr_ft_to_test']:.4f}", flush=True)
for name, ms in list(POOLS.items()) + [("G3+NEW", POOLS["G3"] + sum((FAM[f] for f in NEW), [])),
                                       ("NEWonly", sum((FAM[f] for f in NEW), []))]:
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    pt, pfz, pfin = pf(zt), pf(zf), pf(0.5 * zt + 0.5 * zf)
    ph = pf(branch(ms, "holdout", "to"), "holdout").mean()
    zt_te = zfeed(branch(ms, "test", "to"), "test"); zf_te = zfeed(branch(ms, "test", "ft"), "test")
    r = dict(fam="POOL_" + name, n=len(ms), pool_to=pt.mean(), pool_ft=pfz.mean(), pool_fin=pfin.mean(), pool_hold=ph,
             pool_gain_fin=pse(pfin - pt)[0], pool_gain_fin_se=pse(pfin - pt)[1],
             pool_gain_ft=pse(pfz - pt)[0], pool_gain_ft_se=pse(pfz - pt)[1],
             pool_corr_ft_to_val=np.corrcoef(zt, zf)[0, 1], pool_corr_ft_to_test=np.corrcoef(zt_te, zf_te)[0, 1])
    rows.append(r)
    print(f"POOL {name:8s} n={len(ms)} TO {pt.mean():.5f} FT {pfz.mean():.5f} fin {pfin.mean():.5f} hold {ph:.5f} "
          f"gain fin {r['pool_gain_fin']:+.5f}+-{r['pool_gain_fin_se']:.5f} ft-only {r['pool_gain_ft']:+.5f}+-{r['pool_gain_ft_se']:.5f} "
          f"corr v/t {r['pool_corr_ft_to_val']:.4f}/{r['pool_corr_ft_to_test']:.4f}", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a1_family_table.csv")
c = pl.DataFrame(curves)
c.write_csv(OUT / "a1_ftcurves.csv")
print(c.group_by("fam", maintain_order=True).agg([pl.col(x).mean() for x in c.columns if x.startswith("ep") or x in ("ema_final", "snap")]))
