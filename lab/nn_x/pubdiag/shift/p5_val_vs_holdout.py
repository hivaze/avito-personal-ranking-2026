"""Part 5: model x population interaction. For train-only single models on the same F7 roots, compare the val delta
(post-period exploration traffic) with the holdout delta (in-period exploration-like train feeds) of each arm vs a
reference. If the per-feed SEs captured all the noise of 'transferring' a gain to another population, the
discrepancy (val delta - holdout delta)/SE would be ~N(0,1) for replicate-like arms."""
from shiftlib import *  # noqa: F401,F403

R6P = ROOT / "nn_x/r6/preds"
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy")
hold, fh, Mh, offh = holdout_index(); yh = np.load(P / "holdout/holdout_y.npy")


def load(name):
    """(val per-feed, holdout per-feed) of a train-only single model, or None"""
    for vd, hd in [(P, P / "holdout"), (R6P, R6P / "holdout")]:
        fvp, fhp = vd / f"nn_{name}_trainonly_val.npy", hd / f"nn_{name}_holdout.npy"
        if fvp.exists() and fhp.exists():
            return (ndcg_per_feed(np.load(fvp), yv, ov, Mv), ndcg_per_feed(np.load(fhp), yh, offh, Mh))
    return None


ARMS = {"F7": "v17_pre2s_s{r}", "F7@77.5k": "v17_pre2s_s{r}@77500", "c0": "r6_cd_c0_r{r}", "m02": "r6_cd_m02_r{r}",
        "m02p70k": "r6_cd_m02_p70k_r{r}", "m0": "r6_cd_m0_r{r}", "w10": "r6_cd_w10_r{r}", "aa5": "r6_cd_aa5_r{r}",
        "d0": "r6_cd_d0_r{r}", "d20": "r6_cd_d20_r{r}", "pp02": "r6_cd_pp02_r{r}", "aug": "r6_cd_aug_r{r}",
        "e3": "r6_cd_e3_r{r}"}
S = {}
for arm, pat in ARMS.items():
    for r in range(101, 108):
        x = load(pat.format(r=r))
        if x is not None:
            S[(arm, r)] = x
print("loaded", len(S), "single models")
for arm in ARMS:
    rs = [r for r in range(101, 108) if (arm, r) in S]
    if rs:
        print(f"{arm:9s} roots {rs[0]}-{rs[-1]} (n={len(rs)}) mean single val {np.mean([S[(arm, r)][0].mean() for r in rs]):.5f} "
              f"holdout {np.mean([S[(arm, r)][1].mean() for r in rs]):.5f}")

rows = []
print("\n== arm vs reference, same roots: mean over roots of paired single-model deltas; discrepancy = val - holdout ==")
for arm, ref in [("c0", "F7"), ("m02", "F7"), ("m02", "c0"), ("m02p70k", "m02"), ("m02p70k", "c0"), ("m0", "c0"),
                 ("m0", "m02"), ("w10", "c0"), ("aa5", "c0"), ("d0", "c0"), ("d20", "c0"), ("pp02", "c0"),
                 ("aug", "c0"), ("e3", "m02"), ("F7", "F7@77.5k"), ("c0", "F7@77.5k")]:
    rs = [r for r in range(101, 108) if (arm, r) in S and (ref, r) in S]
    if not rs:
        continue
    dv = np.mean([S[(arm, r)][0] - S[(ref, r)][0] for r in rs], 0)
    dh = np.mean([S[(arm, r)][1] - S[(ref, r)][1] for r in rs], 0)
    mv, sv = mse(dv); mh, sh = mse(dh)
    disc = mv - mh; sd = np.hypot(sv, sh)
    # per-root discrepancies (replicate scatter)
    pr = [(S[(arm, r)][0] - S[(ref, r)][0], S[(arm, r)][1] - S[(ref, r)][1]) for r in rs]
    per = [(mse(a)[0] - mse(b)[0], np.hypot(mse(a)[1], mse(b)[1])) for a, b in pr]
    rows.append(dict(arm=arm, ref=ref, n_roots=len(rs), val=mv, val_se=sv, hold=mh, hold_se=sh, disc=disc, disc_se=sd,
                     per_root_disc=";".join(f"{p[0]:+.4f}" for p in per)))
    print(f"{arm:8s}-{ref:8s} n={len(rs)} val {mv:+.5f}±{sv:.5f} | holdout {mh:+.5f}±{sh:.5f} | val-hold {disc:+.5f} "
          f"(z {disc / sd:+.1f}) | per-root val-hold: " + " ".join(f"{p[0]:+.4f}" for p in per))
pl.DataFrame(rows).write_csv(OUT / "p5_val_vs_holdout.csv")

# replicate-scatter test: per-root (val - holdout) discrepancies of c0 vs F7 and m02 vs c0 around their mean
print("\n== per-root scatter of (val delta - holdout delta): chi2/df around the arm mean (1 = per-feed SE explains it) ==")
for arm, ref in [("c0", "F7"), ("m02", "c0"), ("m02", "F7"), ("m0", "c0"), ("m02p70k", "c0")]:
    rs = [r for r in range(101, 108) if (arm, r) in S and (ref, r) in S]
    dd, ss = [], []
    for r in rs:
        a = S[(arm, r)][0] - S[(ref, r)][0]; b = S[(arm, r)][1] - S[(ref, r)][1]
        dd.append(a.mean() - b.mean()); ss.append(np.hypot(*[x.std() / np.sqrt(len(x)) for x in (a, b)]))
    dd, ss = np.array(dd), np.array(ss)
    m = np.average(dd, weights=1 / ss ** 2)
    chi = np.sum(((dd - m) / ss) ** 2)
    tau2 = max(0.0, (np.var(dd, ddof=1) - np.mean(ss ** 2)))
    print(f"{arm}-{ref}: n={len(rs)} mean disc {m:+.5f}, chi2/df {chi / (len(rs) - 1):.2f}, implied extra per-model SD "
          f"{np.sqrt(tau2):.5f} (per-feed SE of one disc ~{np.mean(ss):.5f})")
