"""Follow-up F1-F3 on a4u (unique retrieval slots), DECLARATION block 2."""
from lib import *
R6, PR = ROOT / "nn_x/r6/preds", ROOT / "preds"
pool = lambda fs: np.mean([Z(np.load(f)) for f in fs], 0)
cov = pl.read_parquet(OUT / "feed_cov.parquet"); hd = cov["hall_distinct"].to_numpy().astype(np.float64); cold = np.load(OUT / "cand_cold.npy")
def line(lab, d):
    r = [lab, "%+.5f+-%.5f" % cse(d)[:2]]
    for m in (CAP, NEAR, Hv < 100):
        mu, se, _ = cse(d, m); r.append(f"{mu:+.5f}+-{se:.5f} (z {mu / se:+.2f})")
    di, si = cdiff(d, CAP, ~CAP); r.append(f"{di:+.5f} (z {di / si:+.2f})"); return r
rows = []
for suf, lab in (("trainonly_val", "train-only"), ("n2cpv3_val_oof", "fine-tuned OOF")):
    roots = [r for r in range(101, 108) if (R6 / f"nn_r6_cd_m02_a4u_r{r}_{suf}.npy").exists() and (R6 / f"nn_r6_cd_m02_r{r}_{suf}.npy").exists()]
    d = ev(pool([R6 / f"nn_r6_cd_m02_a4u_r{r}_{suf}.npy" for r in roots])) - ev(pool([R6 / f"nn_r6_cd_m02_r{r}_{suf}.npy" for r in roots]))
    rows.append(line(f"F1 a4u + m02 re-cool vs m02 re-cool, {lab}, roots {roots[0]}-{roots[-1]} ({len(roots)})", d))
    if suf == "trainonly_val": dF1 = d
a = [R6 / f"nn_r6_a4u_e10_s{s}_trainonly_val.npy" for s in range(101, 107)]; c = [PR / f"nn_v17_pre2s_s{s}_trainonly_val.npy" for s in range(101, 107)]
d10 = ev(pool(a)) - ev(pool(c)); rows.append(line("(seen) a4u 10 ep vs F7, 6 seeds, train-only", d10))
# per-seed members at H=256 (is the pool read carried by one seed?)
ms = [cse(ev(Z(np.load(x))) - ev(Z(np.load(y))), CAP)[0] for x, y in zip(a, c)]
print(tab(rows, ["read", "ALL", "H=256", "NEAR (100-255)", "H<100", "H=256 minus rest"]))
print("a4u 10 ep member-level deltas at H=256 by seed:", " ".join(f"{x:+.4f}" for x in ms), f"| mean {np.mean(ms):+.4f}, seeds > 0: {sum(x > 0 for x in ms)}/6")
# F2 dose-response by distinct-item share, H >= 100
m100 = Hv >= 100; q = np.quantile(hd[m100], [1 / 3, 2 / 3]); T = [("bottom tercile (most repeats)", m100 & (hd <= q[0])), ("middle", m100 & (hd > q[0]) & (hd <= q[1])), ("top tercile (fewest repeats)", m100 & (hd > q[1]))]
rows = []
for lab, d in (("a4u 10 ep (6 seeds)", d10), ("a4u + m02 re-cool (train-only)", dF1)):
    for nm, m in T: rows.append([lab, nm, int(m.sum()), f"{hd[m].mean():.3f}", "%+.5f+-%.5f" % cse(d, m)[:2]])
    di, si = cdiff(d, T[0][1], T[2][1]); rows.append([lab, "bottom minus top", "", "", f"{di:+.5f}+-{si:.5f} (z {di / si:+.2f})"])
    for nm, m in (("H=256 & bottom tercile", CAP & (hd <= q[0])), ("H=256 & top tercile", CAP & (hd > q[1]))): rows.append([lab, nm, int(m.sum()), f"{hd[m].mean():.3f}", "%+.5f+-%.5f" % cse(d, m)[:2]])
print("\n### F2 dose-response by the distinct-item share of the stored history (H >= 100)\n" + tab(rows, ["pair", "feeds", "n", "mean distinct share", "delta"]))
# F3 cold / warm trade at H=256
ncold = np.bincount(fid, weights=yv * cold, minlength=NF); allc, allw = ncold == NPOS, ncold == 0; rows = []
for lab, d in (("a4u 10 ep (6 seeds)", d10), ("a4u + m02 re-cool (train-only)", dF1)):
    for seg, sm in (("H=256", CAP), ("NEAR", NEAR), ("ALL", np.ones(NF, bool))):
        rows.append([lab, seg] + [f"{cse(d, sm & m)[0]:+.5f}+-{cse(d, sm & m)[1]:.5f} (n {int((sm & m).sum())})" for m in (allc, allw, ~allc & ~allw)])
print("\n### F3 the same deltas split by cold / warm positives\n" + tab(rows, ["pair", "segment", "all positives cold", "all positives warm", "mixed"]))
print("share of feeds with all-cold positives: H=256 %.3f, NEAR %.3f, REST %.3f" % (allc[CAP].mean(), allc[NEAR].mean(), allc[REST].mean()))
