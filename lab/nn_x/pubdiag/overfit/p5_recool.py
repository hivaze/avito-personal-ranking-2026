"""Part 5 (supports 1 and 3): split the F7x-vs-F7 gain into the re-cool effect (F7b c0 - F7, same roots, no m02) and the
m02 effect (F7x - F7b), on val and holdout, with slices; label-free check that the score changes look the same on test.
Writes p5_recool.txt."""
from lib import *

C = OUT / "cache"
idx = {m: i for i, m in enumerate(MEMBERS)}
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))
V, Hs, T = Split("val"), Split("holdout"), Split("test")
pub = T.ts < PUB_TS
Zv = {h: np.load(C / f"Z_val_{h}.npy", mmap_mode="r") for h in ("to", "ft")}
Zh = np.load(C / "Z_holdout_to.npy", mmap_mode="r")
Zt = np.load(C / "Z_test_to.npy", mmap_mode="r")


def mix(S, Z, ms):
    return S.z(np.asarray(Z[[idx[m] for m in ms]], np.float64).mean(0))


tue = (V.ts + 10800) // 86400 < 20720
coldv = np.nan_to_num(np.load(ROOT / "nn_x/r6/feat_cache/val/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
coldt = np.nan_to_num(np.load(ROOT / "nn_x/r6/feat_cache/test/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
npos = np.bincount(V.fid, V.y, V.n); ncp = np.bincount(V.fid, V.y * coldv, V.n)
coldpos = (npos > 0) & (ncp == npos); warmpos = (npos > 0) & (ncp == 0)
sl = [("M<50", V.M < 50), ("M>=50", V.M >= 50), ("Tue", tue), ("Thu", ~tue), ("coldpos", coldpos), ("warmpos", warmpos),
      ("mixedpos", ~coldpos & ~warmpos)]
P_(f"val slice shares: " + " ".join(f"{n} {m.mean():.3f}" for n, m in sl))
P_(f"cold-candidate share (it_hist_all_luo == 0): val {coldv.mean():.3f}, test public {coldt[np.repeat(pub, T.M)].mean():.3f},"
   f" test private {coldt[np.repeat(~pub, T.M)].mean():.3f}")

steps = [("re-cool c0 (F7b - F7)", "F7b", "F7"), ("m02 (F7x - F7b)", "F7x", "F7b"), ("F7x - F7 (total)", "F7x", "F7"),
         ("p70k (F7x70 - F7x)", "F7x70", "F7x"), ("M2 fams (F7x+F7x70) - F7", None, "F7")]
P_()
P_("== 7-member family mixes (TO = train-only; FT = n2cpv3 OOF), paired per feed")
for nm, a, b in steps:
    A = FAM[a] if a else FAM["F7x"] + FAM["F7x70"]
    out = []
    for lab, S, Z in (("val TO", V, Zv["to"]), ("val FT", V, Zv["ft"]), ("hold TO", Hs, Zh)):
        d = S.nd(mix(S, Z, A)) - S.nd(mix(S, Z, FAM[b]))
        out.append(f"{lab} {d.mean():+.5f}+-{se(d):.5f}")
        if lab == "val TO":
            dv = d
    P_(f"  {nm:28s} " + " | ".join(out))
    P_("      val TO slices: " + "  ".join(f"{n} {dv[m].mean():+.5f}+-{se(dv[m]):.5f}" for n, m in sl))
P_()
P_("== same-root single members (mean over 7 roots of the per-feed delta)")
for nm, a, b in steps[:4]:
    dv = np.mean([V.nd(np.asarray(Zv["to"][idx[x]])) - V.nd(np.asarray(Zv["to"][idx[y]])) for x, y in zip(FAM[a], FAM[b])], 0)
    dh = np.mean([Hs.nd(np.asarray(Zh[idx[x]])) - Hs.nd(np.asarray(Zh[idx[y]])) for x, y in zip(FAM[a], FAM[b])], 0)
    P_(f"  {nm:28s} val TO {dv.mean():+.5f}+-{se(dv):.5f} | hold TO {dh.mean():+.5f}+-{se(dh):.5f} | coldpos {dv[coldpos].mean():+.5f}"
       f" warmpos {dv[warmpos].mean():+.5f}")

P_()
P_("== label-free behaviour check: how much each change promotes COLD candidates (feed-equal mean of"
   " [mean z-change on cold cands - mean z-change on warm cands], feeds with both), val vs test")


def cold_promo(S, Z, A, B, cold, mask=None):
    d = mix(S, Z, A) - mix(S, Z, B)
    nc = np.bincount(S.fid, cold, S.n); nw = S.M - nc
    mc = np.bincount(S.fid, d * cold, S.n) / np.maximum(nc, 1); mw = np.bincount(S.fid, d * ~cold, S.n) / np.maximum(nw, 1)
    ok = (nc > 0) & (nw > 0)
    if mask is not None:
        ok &= mask
    x = (mc - mw)[ok]
    return x.mean(), se(x), ok.mean()


for nm, a, b in steps:
    A = FAM[a] if a else FAM["F7x"] + FAM["F7x70"]
    rv = cold_promo(V, Zv["to"], A, FAM[b], coldv)
    rp = cold_promo(T, Zt, A, FAM[b], coldt, pub)
    rq = cold_promo(T, Zt, A, FAM[b], coldt, ~pub)
    P_(f"  {nm:28s} val {rv[0]:+.4f}+-{rv[1]:.4f} | public {rp[0]:+.4f}+-{rp[1]:.4f} | private {rq[0]:+.4f}+-{rq[1]:.4f}")
for p in ("PF7x", "G3", "M2"):
    rv = cold_promo(V, Zv["to"], pool_members(p), pool_members("Cd"), coldv)
    rp = cold_promo(T, Zt, pool_members(p), pool_members("Cd"), coldt, pub)
    rq = cold_promo(T, Zt, pool_members(p), pool_members("Cd"), coldt, ~pub)
    P_(f"  pool {p:5s} - Cd (TO)           val {rv[0]:+.4f}+-{rv[1]:.4f} | public {rp[0]:+.4f}+-{rp[1]:.4f} | private {rq[0]:+.4f}+-{rq[1]:.4f}")
(OUT / "p5_recool.txt").write_text("\n".join(lines) + "\n")
