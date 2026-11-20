"""S5: stale-val (scenario 'both_pre2both' + xp_imp_c50 cut) pool deltas vs G3 from the GPU-3 inference probe (s3b), and
their transfer to public, next to val / val-TO / holdout / extrapolated-staleness proxies. Writes out/s5_stale_transfer.log.
"""
from lib import *  # noqa: F401,F403

LOG = open(OUT / "s5_stale_transfer.log", "w")
ST = OUT / "stale"


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def gls_k(P, SP, X, SX, it=60):
    k = 1.0
    for _ in range(it):
        Vi = np.linalg.inv(SP + k * k * SX)
        kn = (X @ Vi @ P) / (X @ Vi @ X)
        if abs(kn - k) < 1e-10:
            break
        k = kn
    Vi = np.linalg.inv(SP + k * k * SX); se = 1 / np.sqrt(X @ Vi @ X)
    def chi2(kk):
        r = P - kk * X
        return float(r @ np.linalg.inv(SP + kk * kk * SX) @ r)
    return k, se, chi2(k), chi2(1.0), chi2(0.0)


def loo(P, SP, X, SX):
    out = []
    for i in range(len(P)):
        m = np.arange(len(P)) != i
        k, se, *_ = gls_k(P[m], SP[np.ix_(m, m)], X[m], SX[np.ix_(m, m)])
        out.append((P[i] - k * X[i], np.sqrt(SP[i, i] + k * k * SX[i, i] + (X[i] * se) ** 2)))
    e = np.array(out)
    return float(np.sqrt((e[:, 0] ** 2).mean())), float((e[:, 0] ** 2 / e[:, 1] ** 2).sum())


def main():
    f, M, off = meta("val"); y = val_y()
    dd = day(f["timestamp"].to_numpy()); tue = dd <= 20719
    Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
    pools = ["G3"] + Ch

    def member_file(m, scen):
        if scen == "both_xp14" and not (ST / f"nn_{m}_both_xp14.npy").exists():
            scen = "both"
        return ST / f"nn_{m}_{scen}.npy"
    missing = [m for p in pools for m in MEMBERS[p] if not member_file(m, "both").exists()]
    if missing:
        pr("MISSING stale members:", sorted(set(missing)))
    pf = {}
    for scen in ("base", "both", "both_xp14"):
        for p in pools:
            ms = [m for m in MEMBERS[p] if member_file(m, scen).exists()]
            to = zmean([np.load(member_file(m, scen)) for m in ms], M)
            pf[(scen, p)] = ndcg_per_feed(to.astype(np.float32), y, off, M)
            if scen != "base":
                ft = zf(np.load(P / f"ens_{ENS[p]}_ft_val.npy"), M)
                fin = 0.5 * ft + 0.5 * zf(to, M)
                pf[(scen + "_final", p)] = ndcg_per_feed(fin.astype(np.float32), y, off, M)
            if scen == "base":
                ref = np.load(OUT / f"pfv_{p}_to.npy")
                pr(f"  base TO pool {p}: {pf[(scen, p)].mean():.5f} vs ens_to file {ref.mean():.5f} (members {len(ms)}/{len(MEMBERS[p])})")
    pr("\nlevels (TO half): pool | val-TO | stale both | stale both_xp14 | final(FT + stale TO) | public")
    for p in pools:
        pr(f"  {p:5s} {pf[('base', p)].mean():.5f} {pf[('both', p)].mean():.5f} {pf[('both_xp14', p)].mean():.5f} {pf[('both_final', p)].mean():.5f} {PUBLIC[p]:.6f}")

    # proxies on the H6 contrasts
    pv = {p: np.load(OUT / f"pfv_{p}.npy") for p in pools}
    pvto = {p: np.load(OUT / f"pfv_{p}_to.npy") for p in pools}
    ph = {p: np.load(OUT / f"pfh_{p}.npy") for p in pools}
    gap_pub = 7.0 * 6050 / 8562 + 8.0 * 2512 / 8562  # days from the last full train day (20715) to public feeds' day
    c = (gap_pub - 3.0) / 2.0  # Tue = 3 days, Thu = 5 days
    def vec(get, n):
        D = np.stack([get(p) for p in Ch], 1)
        X = D.mean(0); R = D - X
        return X, R.T @ R / n ** 2, D
    prox = {}
    prox["val"] = vec(lambda p: pv[p] - pv["G3"], len(M))[:2]
    prox["val_TO"] = vec(lambda p: pvto[p] - pvto["G3"], len(M))[:2]
    prox["holdout"] = vec(lambda p: ph[p] - ph["G3"], 30000)[:2]
    dh = pl.read_parquet(OUT / "desc_holdout.parquet")
    for lab, msk in [("hold_weekend", dh["dow"].to_numpy() >= 5), ("hold_late3d", dh["day"].to_numpy() >= 20713),
                     ("hold_luo0>=.3", dh["luo_zero"].to_numpy() >= 0.3)]:
        prox[lab] = vec(lambda p, msk=msk: (ph[p] - ph["G3"])[msk], int(msk.sum()))[:2]
    for s in ("both", "both_xp14", "both_final", "both_xp14_final"):
        prox["stale_" + s] = vec(lambda p, s=s: pf[(s, p)] - pf[(s, "G3")], len(M))[:2]
    # staleness extrapolation inside val: d_pub = d_Tue + (d_Thu - d_Tue) * c  (per-feed: weights on the two days)
    w = np.where(tue, -(c) / tue.sum() + 1.0 / tue.sum(), c / (~tue).sum()) * len(M)  # mean(w*d) = (1-c) d_Tue + c d_Thu
    prox["val_trend"] = vec(lambda p: (pv[p] - pv["G3"]) * w, len(M))[:2]
    prox["val_Thu"] = (np.array([(pv[p] - pv["G3"])[~tue].mean() for p in Ch]),
                       np.cov(np.stack([(pv[p] - pv["G3"])[~tue] for p in Ch], 1), rowvar=False, bias=True) / (~tue).sum())
    Pv = np.array([PUBLIC[p] - PUBLIC["G3"] for p in Ch])
    D = np.stack([pv[p] - pv["G3"] for p in Ch], 1)
    wpub = np.load(OUT / "w_val_pubmatch.npz")["w_pub"]
    mu = (wpub[:, None] * D).sum(0) / wpub.sum(); R = D - mu
    SP = (wpub[:, None] * R).T @ R / wpub.sum() / N_PUB
    pr(f"\npublic gap (days from 20715) {gap_pub:.2f}; trend coefficient c {c:.2f}")
    pr("\n== per-contrast deltas vs G3 (H6) ==")
    pr(f"  {'contrast':8s} {'public':>9s} " + " ".join(f"{k:>18s}" for k in prox))
    for i, p in enumerate(Ch):
        pr(f"  {p:8s} {Pv[i]:+9.5f} " + " ".join(f"{prox[k][0][i]:+9.5f}±{np.sqrt(prox[k][1][i, i]):.5f}" for k in prox))
    pr("\n  stale - base (diff-in-diff, TO half) per contrast: " + ", ".join(
        f"{p} {prox['stale_both'][0][i] - prox['val_TO'][0][i]:+.5f}" for i, p in enumerate(Ch)))
    for lab, cs in [("H6", Ch), ("H4 (pre-11-13)", ["Cd", "Pall", "PF7x", "M2"]), ("POP2 (Q1a, Q2a)", ["Q1a", "Q2a"])]:
        idx = [Ch.index(p) for p in cs]
        Ps, SPs = Pv[idx], SP[np.ix_(idx, idx)]
        pr(f"\n== transfer fits on {lab}; public = k * proxy (GLS, public noise + k^2 proxy noise) ==")
        pr(f"  {'proxy':18s} {'k':>7s} {'SE':>6s} {'chi2(k^)':>9s} {'chi2(k=1)':>9s} {'chi2(k=0)':>9s} {'gR2':>6s} {'LOOrmse':>8s} {'LOOz2':>6s}")
        for kname, (X, SX) in prox.items():
            Xs, SXs = X[idx], SX[np.ix_(idx, idx)]
            k, se, ch, c1, c0 = gls_k(Ps, SPs, Xs, SXs)
            lr, lz = loo(Ps, SPs, Xs, SXs) if len(cs) > 2 else (np.nan, np.nan)
            pr(f"  {kname:18s} {k:+7.2f} {se:6.2f} {ch:9.2f} {c1:9.2f} {c0:9.2f} {1 - ch / c0:6.2f} {lr:8.5f} {lz:6.2f}")
    pr(f"\n  null rmse of public deltas (predict 0), H6: {np.sqrt((Pv ** 2).mean()):.5f}; mean public SE {np.sqrt(np.diag(SP)).mean():.5f}")


if __name__ == "__main__":
    main()
