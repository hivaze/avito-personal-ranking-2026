"""A2: disagreement structure of families / members on val and test.
(a) per-feed disagreement (1 - mean pairwise within-feed Pearson of z) among the family pools and among the 20 G3 members;
    NDCG, pool gain over mean member, oracle by disagreement decile (within-M-bin deciles to remove the M confound);
(b) who is right when families disagree: per-family NDCG relative to the all-family mean, by disagreement tercile and
    by segment;
(c) oracle headroom: per-feed best family (noise-calibrated by the per-feed best of same-family seeds), per-segment
    best family chosen in-sample and nested over the FT user folds;
(d) complementarity matrix: family-equal pair pools (final 0.5/0.5 on val; TO on holdout), pair gain over the mean and
    over the better single; marginal gain of adding each family to G3 (member-equal, val + holdout);
(e) diversity by segment on val and test (incl. public / private).
Writes a2_*.csv and a2_disagreement.log (stdout)."""
from lib import *  # noqa

add_v30()
FAMS = [f for f in FAM]
vm, tm, hm = meta("val"), meta("test"), meta("holdout")
Sv, St = segments("val"), segments("test")
fold = user_folds()
rng = np.random.default_rng(0)

# ---------- family pools ----------
Zv = {f: final(FAM[f], "val") for f in FAMS}
Zt = {f: final(FAM[f], "test") for f in FAMS}
Zh = {f: zfeed(branch(FAM[f], "holdout", "to"), "holdout") for f in FAMS}
NDv = {f: pf(Zv[f]) for f in FAMS}
NDh = {f: pf(Zh[f], "holdout") for f in FAMS}
G3v = final(POOLS["G3"], "val"); G3t = final(POOLS["G3"], "test")
ndG3 = pf(G3v)
print("G3 val", ndG3.mean())
print("family final val:", {f: round(NDv[f].mean(), 5) for f in FAMS})
print("family TO holdout:", {f: round(NDh[f].mean(), 5) for f in FAMS})


def feed_pair_corr(za, zb, split):
    m = meta(split)
    return np.bincount(m["fid"], za * zb, m["n"]) / m["M"]


def feed_avg_corr(zs, split):
    """mean pairwise within-feed Pearson of per-feed z arrays (population z => corr = mean(za*zb))"""
    m = meta(split); n = len(zs)
    s = np.sum(zs, 0)
    tot = np.bincount(m["fid"], s * s, m["n"]) / m["M"]
    self_ = sum(np.bincount(m["fid"], z * z, m["n"]) / m["M"] for z in zs)
    return (tot - self_) / (n * (n - 1))


# ---------- (a) disagreement ----------
fam_list = [f for f in FAMS]
dis_fam_v = 1 - feed_avg_corr([zfeed(Zv[f], "val") for f in fam_list], "val")
dis_fam_t = 1 - feed_avg_corr([zfeed(Zt[f], "test") for f in fam_list], "test")
g3m = POOLS["G3"]
dis_mem_v = 1 - feed_avg_corr([0.5 * member_z(m, "val", "to") + 0.5 * member_z(m, "val", "ft") for m in g3m], "val")
dis_mem_t = 1 - feed_avg_corr([0.5 * member_z(m, "test", "to") + 0.5 * member_z(m, "test", "ft") for m in g3m], "test")
mem_nd = np.mean([pf(0.5 * member_z(m, "val", "to") + 0.5 * member_z(m, "val", "ft")) for m in g3m], 0)
fam_mean_nd = np.mean([NDv[f] for f in fam_list], 0)
fam_max_nd = np.max([NDv[f] for f in fam_list], 0)
print(f"\n(a) disagreement: family-level mean val {dis_fam_v.mean():.4f} test {dis_fam_t.mean():.4f} | "
      f"G3-member-level val {dis_mem_v.mean():.4f} test {dis_mem_t.mean():.4f}")
print(f"    corr(dis_fam, dis_mem) val {np.corrcoef(dis_fam_v, dis_mem_v)[0, 1]:.3f}; corr(dis_mem, logM) val "
      f"{np.corrcoef(dis_mem_v, np.log(vm['M']))[0, 1]:.3f}")


def within_bin_decile(x, bins):
    q = np.zeros(len(x), np.int64)
    for b in np.unique(bins):
        i = np.where(bins == b)[0]
        r = np.argsort(np.argsort(x[i]))
        q[i] = (10 * r) // len(i)
    return q


rows = []
for name, dis in [("member(G3)", dis_mem_v), ("family", dis_fam_v)]:
    q = within_bin_decile(dis, Sv["M"])
    print(f"\n  disagreement deciles within M bins [{name}]: dec | n | dis | G3 ndcg | mean-member ndcg | G3-member gain | "
          f"fam-oracle - G3 | luo_cold | M | npos")
    for d in range(10):
        i = q == d
        g = ndG3[i] - mem_nd[i]
        r = dict(kind=name, dec=d, n=int(i.sum()), dis=dis[i].mean(), g3=ndG3[i].mean(), mem=mem_nd[i].mean(),
                 gain=g.mean(), gain_se=pse(g)[1], orc=(fam_max_nd[i] - ndG3[i]).mean(),
                 luo_cold=pl.read_parquet(ROOT / "nn_x/pubdiag/shift/desc_val.parquet")["luo_cold_share"].to_numpy()[i].mean(),
                 M=vm["M"][i].mean())
        rows.append(r)
        print(f"  {d} | {r['n']} | {r['dis']:.4f} | {r['g3']:.4f} | {r['mem']:.4f} | {r['gain']:+.5f}+-{r['gain_se']:.5f} | "
              f"{r['orc']:+.4f} | {r['luo_cold']:.3f} | {r['M']:.0f}")
pl.DataFrame(rows).write_csv(OUT / "a2_dis_deciles.csv")

# ---------- (b) who is right when families disagree ----------
print("\n(b) family NDCG minus all-family mean, by family-disagreement tercile (within M bins); val final pools")
qt = within_bin_decile(dis_fam_v, Sv["M"]) // 4  # 0..2 (deciles 0-3, 4-7, 8-9)
lab = {0: "low(d0-3)", 1: "mid(d4-7)", 2: "high(d8-9)"}
rows = []
hdr = "family | " + " | ".join(lab.values()) + " | win-share(high)"
print(hdr)
best_idx = np.argmax(np.stack([NDv[f] for f in fam_list]), 0)
tie = (np.stack([NDv[f] for f in fam_list]) == fam_max_nd).sum(0)
for j, f in enumerate(fam_list):
    d = NDv[f] - fam_mean_nd
    cells = []
    for t in range(3):
        i = qt == t
        cells.append(fmt(d[i]))
        rows.append(dict(fam=f, tercile=lab[t], d=d[i].mean(), se=pse(d[i])[1]))
    hi = qt == 2
    win = (((np.stack([NDv[g] for g in fam_list]) == fam_max_nd)[j]) / tie)[hi].mean()
    print(f"{f:6s} | " + " | ".join(cells) + f" | {win:.3f} (uniform {1 / len(fam_list):.3f})")
pl.DataFrame(rows).write_csv(OUT / "a2_whoisright_tercile.csv")

print("\n(b2) family NDCG minus all-family mean, by segment (val final pools); cells with |t|>=2 marked *")
rows = []
for var in ["M", "luo_cold", "user", "cat_in_hist", "n_cat", "day", "pos_cold(label)"]:
    labs = sorted(np.unique(Sv[var]))
    print(f"  [{var}] " + " | ".join(f"{l}(n={int((Sv[var] == l).sum())})" for l in labs))
    for f in fam_list:
        d = NDv[f] - fam_mean_nd
        cells = []
        for l in labs:
            i = Sv[var] == l
            mu, se = pse(d[i])
            cells.append(f"{mu:+.5f}{'*' if abs(mu) >= 2 * se else ' '}")
            rows.append(dict(var=var, seg=l, fam=f, d=mu, se=se))
        print(f"    {f:6s} " + " | ".join(cells))
pl.DataFrame(rows).write_csv(OUT / "a2_whoisright_segment.csv")

# ---------- (c) oracle headroom ----------
print("\n(c) oracle headroom (val, final pools)")
stackF = np.stack([NDv[f] for f in fam_list])
print(f"  G3 {ndG3.mean():.5f}; per-feed oracle over {len(fam_list)} family pools {stackF.max(0).mean():.5f} "
      f"(+{(stackF.max(0) - ndG3).mean():.5f}); mean family {fam_mean_nd.mean():.5f}")
# noise calibration: per-feed oracle over k seeds of ONE family vs over k different families (one member each)
for k in (3, 6):
    same, diff = [], []
    for f in ["F7x", "F7x70", "F7b", "F7", "F12", "V28l"]:
        if len(FAM[f]) < k:
            continue
        ms = FAM[f][:k]
        same.append(np.max([pf(0.5 * member_z(m, "val", "to") + 0.5 * member_z(m, "val", "ft")) for m in ms], 0).mean()
                    - np.mean([pf(0.5 * member_z(m, "val", "to") + 0.5 * member_z(m, "val", "ft")).mean() for m in ms]))
    for rep in range(4):
        fs = rng.choice([f for f in fam_list if len(FAM[f]) >= 1], k, replace=False)
        ms = [FAM[f][rng.integers(len(FAM[f]))] for f in fs]
        nds = [pf(0.5 * member_z(m, "val", "to") + 0.5 * member_z(m, "val", "ft")) for m in ms]
        diff.append(np.max(nds, 0).mean() - np.mean([x.mean() for x in nds]))
    print(f"  per-feed oracle gain over the mean of k={k} single members: same-family seeds {np.mean(same):+.5f} "
          f"(n fam {len(same)}), different families {np.mean(diff):+.5f} (4 random draws) -> structured share "
          f"{np.mean(diff) - np.mean(same):+.5f}")
# per-segment best family: in-sample and nested over the user folds
print("  per-segment family selection (choose the best family pool per segment cell), vs G3:")
rows = []
for var in ["M", "luo_cold", "user", "cat_in_hist", "n_cat", "H", "max_cat_share"]:
    cells = Sv[var]
    ins = np.zeros(vm["n"]); nest = np.zeros(vm["n"])
    for l in np.unique(cells):
        i = cells == l
        bestf = max(fam_list + ["G3"], key=lambda f: (ndG3 if f == "G3" else NDv[f])[i].mean())
        ins[i] = (ndG3 if bestf == "G3" else NDv[bestf])[i]
        for k in range(5):
            tr = i & (fold != k); te = i & (fold == k)
            bf = max(fam_list + ["G3"], key=lambda f: (ndG3 if f == "G3" else NDv[f])[tr].mean())
            nest[te] = (ndG3 if bf == "G3" else NDv[bf])[te]
    r = dict(var=var, insample=pse(ins - ndG3)[0], insample_se=pse(ins - ndG3)[1], nested=pse(nest - ndG3)[0],
             nested_se=pse(nest - ndG3)[1])
    rows.append(r)
    print(f"    {var:14s} in-sample {fmt(ins - ndG3)} | nested {fmt(nest - ndG3)}")
pl.DataFrame(rows).write_csv(OUT / "a2_segment_oracle.csv")

# ---------- (d) complementarity ----------
print("\n(d) complementarity: family-equal pair pools (val final; holdout TO). gain vs mean of the two | vs better single")
rows = []
nF = len(fam_list)
corr_v = np.zeros((nF, nF)); corr_t = np.zeros((nF, nF))
for a in range(nF):
    for b in range(nF):
        fa, fb = fam_list[a], fam_list[b]
        if b < a:
            corr_v[a, b] = corr_v[b, a]; corr_t[a, b] = corr_t[b, a]; continue
        za, zb = zfeed(Zv[fa], "val"), zfeed(Zv[fb], "val")
        corr_v[a, b] = np.mean(za * zb)
        ta, tb = zfeed(Zt[fa], "test"), zfeed(Zt[fb], "test")
        corr_t[a, b] = np.mean(ta * tb)
        if a == b:
            continue
        pv = pf(0.5 * za + 0.5 * zb)
        ph = pf(0.5 * Zh[fa] + 0.5 * Zh[fb], "holdout")
        mv = 0.5 * (NDv[fa] + NDv[fb]); mh = 0.5 * (NDh[fa] + NDh[fb])
        bv = NDv[fa] if NDv[fa].mean() >= NDv[fb].mean() else NDv[fb]
        bh = NDh[fa] if NDh[fa].mean() >= NDh[fb].mean() else NDh[fb]
        rows.append(dict(a=fa, b=fb, corr_val=corr_v[a, b], corr_test=corr_t[a, b], pair_val=pv.mean(), pair_hold=ph.mean(),
                         gain_mean_val=(pv - mv).mean(), gain_best_val=(pv - bv).mean(), gain_best_val_se=pse(pv - bv)[1],
                         gain_mean_hold=(ph - mh).mean(), gain_best_hold=(ph - bh).mean(), gain_best_hold_se=pse(ph - bh)[1]))
cm = pl.DataFrame(rows)
cm.write_csv(OUT / "a2_pairs.csv")
print("  candidate-level corr of family final pools (val upper / test lower triangle):")
print("        " + " ".join(f"{f:>6s}" for f in fam_list))
for a in range(nF):
    print(f"  {fam_list[a]:6s}" + " ".join(f"{(corr_v[a, b] if b > a else corr_t[a, b] if b < a else 1):6.4f}" for b in range(nF)))
print("  pair gain over the BETTER single (val final | holdout TO), x1e4; upper = val, lower = holdout:")
gb = {(r["a"], r["b"]): r for r in rows}
print("        " + " ".join(f"{f:>6s}" for f in fam_list))
for a in range(nF):
    cells = []
    for b in range(nF):
        if a == b:
            cells.append("     .")
        elif b > a:
            cells.append(f"{gb[(fam_list[a], fam_list[b])]['gain_best_val'] * 1e4:6.1f}")
        else:
            cells.append(f"{gb[(fam_list[b], fam_list[a])]['gain_best_hold'] * 1e4:6.1f}")
    print(f"  {fam_list[a]:6s}" + " ".join(cells))
# fit the strength-diversity law on the pairs: gain_best ~ alpha*(1-rho) - beta*gap
X = np.c_[1 - cm["corr_val"].to_numpy(), -np.abs(cm["pair_val"].to_numpy() * 0 + (
    np.array([abs(NDv[r['a']].mean() - NDv[r['b']].mean()) for r in rows])))]
y = cm["gain_best_val"].to_numpy()
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print(f"  strength-diversity law on val pairs: alpha {coef[0]:.4f} beta {coef[1]:.3f} R2 {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
print("\n  marginal: G3 + family X (member-equal; val final, holdout TO, test corr with G3):")
rows = []
G3h = pf(branch(POOLS["G3"], "holdout", "to"), "holdout")
for f in fam_list:
    if f in ("F10", "F7x", "F7x70"):
        continue
    ms = POOLS["G3"] + FAM[f]
    zv = final(ms, "val"); nv = pf(zv)
    nh = pf(branch(ms, "holdout", "to"), "holdout")
    zt = final(ms, "test")
    r = dict(fam=f, d_val=pse(nv - ndG3)[0], se_val=pse(nv - ndG3)[1], d_hold=pse(nh - G3h)[0], se_hold=pse(nh - G3h)[1],
             test_corr=np.corrcoef(zfeed(zt, "test"), zfeed(G3t, "test"))[0, 1])
    rows.append(r)
    print(f"    +{f:6s} val {fmt(nv - ndG3)} | hold {fmt(nh - G3h)} | test corr {r['test_corr']:.5f}")
pl.DataFrame(rows).write_csv(OUT / "a2_marginal_G3.csv")

# ---------- (e) diversity by segment, val and test ----------
print("\n(e) G3-member disagreement and family disagreement by segment (val | test); G3 gain over mean member (val)")
rows = []
for var in ["M", "luo_cold", "user", "cat_in_hist", "n_cat", "H", "day"]:
    labs = sorted(set(np.unique(Sv[var])) | set(np.unique(St[var])))
    for l in labs:
        iv = Sv[var] == l; it = St[var] == l
        g = ndG3[iv] - mem_nd[iv]
        r = dict(var=var, seg=l, n_val=int(iv.sum()), n_test=int(it.sum()),
                 dis_mem_val=dis_mem_v[iv].mean() if iv.any() else np.nan, dis_mem_test=dis_mem_t[it].mean() if it.any() else np.nan,
                 dis_fam_val=dis_fam_v[iv].mean() if iv.any() else np.nan, dis_fam_test=dis_fam_t[it].mean() if it.any() else np.nan,
                 g3_gain=g.mean() if iv.any() else np.nan, g3_gain_se=pse(g)[1] if iv.sum() > 1 else np.nan)
        rows.append(r)
        print(f"  {var:12s} {l:10s} n {r['n_val']:5d}/{r['n_test']:5d} | dis_mem {r['dis_mem_val']:.4f}/{r['dis_mem_test']:.4f} | "
              f"dis_fam {r['dis_fam_val']:.4f}/{r['dis_fam_test']:.4f} | G3-member {r['g3_gain']:+.5f}+-{r['g3_gain_se']:.5f}")
pl.DataFrame(rows).write_csv(OUT / "a2_diversity_segments.csv")
# M-adjusted val vs test disagreement (reweight test to val's M mix)
for nm, dv, dt in [("member", dis_mem_v, dis_mem_t), ("family", dis_fam_v, dis_fam_t)]:
    adj = 0.0
    for l in np.unique(Sv["M"]):
        adj += (Sv["M"] == l).mean() * dt[St["M"] == l].mean()
    pub = St["day"] == "public"
    print(f"  {nm} disagreement val {dv.mean():.4f} | test {dt.mean():.4f} | test M-adjusted to val mix {adj:.4f} | "
          f"public {dt[pub].mean():.4f} private {dt[~pub].mean():.4f}")
np.savez_compressed(CACHE / "a2_perfeed.npz", dis_fam_v=dis_fam_v, dis_fam_t=dis_fam_t, dis_mem_v=dis_mem_v, dis_mem_t=dis_mem_t,
                    ndG3=ndG3, mem_nd=mem_nd, **{f"nd_{f}": NDv[f] for f in fam_list})
