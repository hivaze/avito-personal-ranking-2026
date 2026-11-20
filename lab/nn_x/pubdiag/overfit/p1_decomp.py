"""Part 1: decompose each pool's val gain vs Cd into the train-only (to) and fine-tune (ft) halves; compare with holdout.
Writes p1_decomp.txt."""
from lib import *

C = OUT / "cache"
idx = {m: i for i, m in enumerate(MEMBERS)}
V, Hs = Split("val"), Split("holdout")
Zv = {h: np.load(C / f"Z_val_{h}.npy", mmap_mode="r") for h in ("to", "ft")}
Zh = np.load(C / "Z_holdout_to.npy", mmap_mode="r")
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))


def half_mean(Z, ms):
    return np.asarray(Z[[idx[m] for m in ms]], np.float64).mean(0)


R = {}
for p in POOLS:
    ms = pool_members(p)
    to, ft = V.z(half_mean(Zv["to"], ms)), V.z(half_mean(Zv["ft"], ms))
    # sanity: reproduce preds/ens_<p>_{ft,to}_val.npy when present
    for h, a in (("to", to), ("ft", ft)):
        f = P / f"ens_{p}_{h}_val.npy"
        if f.exists():
            ref = V.z(np.load(f))
            assert np.abs(ref - a).max() < 1e-3, (p, h, np.abs(ref - a).max())
    R[p] = dict(to=to, ft=ft, full=0.5 * to + 0.5 * ft, hold=Hs.z(half_mean(Zh, ms)), ms=ms)

nd = {p: {k: V.nd(R[p][k]) for k in ("to", "ft", "full")} for p in R}
ndh = {p: Hs.nd(R[p]["hold"]) for p in R}
P_("== Part 1: val gain vs Cd split into halves (paired per feed, n_val = %d, n_hold = %d)" % (V.n, Hs.n))
P_("pool   n  | val full          | val TO-half alone  | val FT-half alone  | holdout (TO members) | public (context)")
for p in R:
    s = f"{p:5s} {len(R[p]['ms']):2d} |"
    for k in ("full", "to", "ft"):
        d = nd[p][k] - nd["Cd"][k]
        s += f" {nd[p][k].mean():.5f} {d.mean():+.5f}+-{se(d):.5f} |"
    dh = ndh[p] - ndh["Cd"]
    s += f" {ndh[p].mean():.5f} {dh.mean():+.5f}+-{se(dh):.5f} |"
    s += f" {PUBLIC[p]:.5f} {PUBLIC[p] - PUBLIC['Cd']:+.5f}" if p in PUBLIC else ""
    P_(s)

P_()
P_("Shapley attribution of the FULL-pool val gain vs Cd to the two halves (swap one half at a time):")
for p in R:
    if p == "Cd":
        continue
    f = lambda to, ft: V.nd(0.5 * to + 0.5 * ft)
    base, full = nd["Cd"]["full"], nd[p]["full"]
    a_to = f(R[p]["to"], R["Cd"]["ft"]) - base  # swap in to-half only
    a_ft = f(R["Cd"]["to"], R[p]["ft"]) - base  # swap in ft-half only
    sh_to = 0.5 * (a_to + (full - f(R["Cd"]["to"], R[p]["ft"])))
    sh_ft = 0.5 * (a_ft + (full - f(R[p]["to"], R["Cd"]["ft"])))
    P_(f"  {p:5s} total {(full - base).mean():+.5f} = TO-half {sh_to.mean():+.5f}+-{se(sh_to):.5f}"
       f"  + FT-half {sh_ft.mean():+.5f}+-{se(sh_ft):.5f}")

P_()
P_("Per-family single-member means (val TO, val FT, FT-TO, holdout TO) and family-mix (z-mean) values:")
P_("fam     val_TO   val_FT   FT-TO     hold_TO | mixTO_val mixFT_val mix_full  mix_hold")
fam_rows = {}
for fname, ms in FAM.items():
    vto = np.mean([V.nd(np.asarray(Zv["to"][idx[m]])).mean() for m in ms])
    vft = np.mean([V.nd(np.asarray(Zv["ft"][idx[m]])).mean() for m in ms])
    hto = np.mean([Hs.nd(np.asarray(Zh[idx[m]])).mean() for m in ms])
    to, ft = V.z(half_mean(Zv["to"], ms)), V.z(half_mean(Zv["ft"], ms))
    mt, mf, mfu = V.nd(to).mean(), V.nd(ft).mean(), V.nd(0.5 * to + 0.5 * ft).mean()
    mh = Hs.nd(Hs.z(half_mean(Zh, ms))).mean()
    fam_rows[fname] = (vto, vft, hto, mt, mf, mfu, mh)
    P_(f"{fname:6s} {vto:.5f} {vft:.5f} {vft - vto:+.5f}  {hto:.5f} | {mt:.5f}   {mf:.5f}   {mfu:.5f}  {mh:.5f}")

P_()
P_("Val-to-holdout transfer of the TO-half gain vs Cd (ratio = holdout gain / val TO gain):")
for p in R:
    if p == "Cd":
        continue
    dv = (nd[p]["to"] - nd["Cd"]["to"]).mean(); dh = (ndh[p] - ndh["Cd"]).mean()
    P_(f"  {p:5s} val TO {dv:+.5f}  holdout {dh:+.5f}  ratio {dh / dv:+.2f}")

# slices of the TO-half and FT-half gain (M, Tue/Thu, cold share)
P_()
tue = (V.ts + 10800) // 86400 < 20720
cold = np.nan_to_num(np.load(ROOT / "nn_x/r6/feat_cache/val/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
ccs = np.bincount(V.fid, cold, V.n) / V.M  # feed cold share
npos = np.bincount(V.fid, V.y, V.n); ncp = np.bincount(V.fid, V.y * cold, V.n)
coldpos = (npos > 0) & (ncp == npos)
sl = [("M<50", V.M < 50), ("M>=50", V.M >= 50), ("Tue", tue), ("Thu", ~tue), ("cs<0.2", ccs < 0.2),
      ("cs0.2-0.4", (ccs >= 0.2) & (ccs < 0.4)), ("cs>=0.4", ccs >= 0.4), ("coldpos", coldpos), ("nopos", npos == 0)]
P_("Slices of the val gain vs Cd (mean +- SE; slice share in brackets):")
for p in ("PF7x", "G3", "M2"):
    for k in ("to", "ft", "full"):
        d = nd[p][k] - nd["Cd"][k]
        P_(f"  {p:5s} {k:4s} " + "  ".join(f"{n}[{m.mean():.2f}] {d[m].mean():+.5f}+-{se(d[m]):.5f}" for n, m in sl))
np.save(OUT / "p1_perfeed.npy", {p: {k: nd[p][k] for k in nd[p]} | {"hold": ndh[p]} for p in nd}, allow_pickle=True)
(OUT / "p1_decomp.txt").write_text("\n".join(lines) + "\n")
