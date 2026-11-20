# Noise model for public-board differences between pools of near-identical members. From per-feed
# val NDCG it derives the paired standard error a difference would have on the 8,562 public feeds,
# tests whether val differences carried over 1:1, and fits the val-to-public level line on 8 files.
# No test label enters; board scores are used only in the tests of parts 2 and 3.
# How it was run: CPU, 2026-11-12, after a val-best pool had scored below a val-worse one on the
# public board; the command line is not on record.
# What it produced: noise_out.json and the read-out in results/public_noise_model.txt (expected
# public paired SE 0.0003 to 0.0007 between pools; public = 0.0022 + 0.976 val over 8 files).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/pubdiag/noise/noise.py
"""Public-LB noise model for the pool contrasts Cd / PF7x / G3 / M2 (CPU only, no test labels, no LB fitting).

Writes nn_x/pubdiag/noise/noise_out.json + prints everything that REPORT.md quotes.

Parts
 1. per-feed val NDCG of each pool (0.5*z(ft)+0.5*z(to), per-feed z = stack_val.feed_rank_z[:,1]),
    paired val SE (iid and user-clustered; user_id 0 = singletons), expected PUBLIC paired SE under several
    noise models (iid / M-reweighted / M x cold-reweighted / label-free disagreement-cell model / + user-cluster deff).
 2. joint chi2 test of "val deltas transfer 1:1" for (PF7x,G3,M2)-Cd with Sigma = Sigma_val + Sigma_pub,
    transfer-factor profiles (total delta; m02 part only, using F7-substituted counterpart pools).
 3. 8-point val->public calibration, residuals, leave-one-out studentised residuals.
"""
import json, os, sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from common import *  # noqa
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix, calib
from scipy import stats

OUT = Path("<repo>/nn_x/pubdiag/noise")
P = Path("<repo>/preds")
BOUND = 1790497920

F5cd = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
F6 = ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)]
F7 = [f"v17_pre2s_s{i}" for i in range(101, 108)]
F9 = [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)]
F10 = [f"v23_gru_pre2s_s{i}" for i in range(501, 507)]
F12 = [f"v26_d192_pre2s_s{i}" for i in range(701, 708)]
F7x = [f"r6_cd_m02_r{i}" for i in range(101, 108)]
F7x70 = [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]
POOLS = {  # member lists (duplicates allowed = member weight 2 in the counterpart pools)
    "Cd": F5cd + F6 + F7,
    "PF7x": F5cd + F6 + F7x + F9 + F10 + F12,
    "G3": F10 + F7x + F7x70,
    "M2": F7x + F7x70,
    "Pall": F5cd + F6 + F7 + F9 + F10 + F12,          # PF7x with F7 in place of F7x
    "G3c": F10 + F7 + F7,                              # G3 with F7 in place of F7x and of F7x70
    "M2c": F7 + F7,                                    # M2 with F7 in place of F7x and F7x70 (= F7 pool)
}
COUNTER = {"PF7x": "Pall", "G3": "G3c", "M2": "M2c"}
PUB = {"Cd": 0.39208, "PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434}   # context only, never used to select
HOLD_REPORTED = {"Cd": 0.32319, "PF7x": 0.32360, "G3": 0.32379, "M2": 0.32361}


def zfeed(off, M, x):
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)


def pool_scores(split, name, f, M, off):
    """final per-candidate pool score on split; uses the saved ens arrays when they exist (exact builder output)."""
    ftn, ton = ("n2cpv3_val_oof", "trainonly_val") if split == "val" else ("n2cpv3_test", "trainonly_test")
    cache = OUT / f"cache_{name}_{split}.npy"
    if cache.exists():
        return np.load(cache)
    if (P / f"ens_{name}_ft_{split}.npy").exists():
        eft, eto = np.load(P / f"ens_{name}_ft_{split}.npy"), np.load(P / f"ens_{name}_to_{split}.npy")
    else:
        ms = POOLS[name]; u = sorted(set(ms)); cnt = {m: ms.count(m) for m in u}
        eft = sum(cnt[m] * zfeed(off, M, np.load(P / f"nn_{m}_{ftn}.npy")) for m in u) / len(ms)
        eto = sum(cnt[m] * zfeed(off, M, np.load(P / f"nn_{m}_{ton}.npy")) for m in u) / len(ms)
    s = 0.5 * zfeed(off, M, eft) + 0.5 * zfeed(off, M, eto)
    np.save(cache, s.astype(np.float32))
    return s.astype(np.float32)


def half_scores(name, half, f, M, off):
    """val score from one half only (ft or to) of a pool, for the FT/TO decomposition."""
    ftn = "n2cpv3_val_oof" if half == "ft" else "trainonly_val"
    if (P / f"ens_{name}_{half}_val.npy").exists():
        e = np.load(P / f"ens_{name}_{half}_val.npy")
    else:
        ms = POOLS[name]; u = sorted(set(ms)); cnt = {m: ms.count(m) for m in u}
        e = sum(cnt[m] * zfeed(off, M, np.load(P / f"nn_{m}_{ftn}.npy")) for m in u) / len(ms)
    return zfeed(off, M, e).astype(np.float32)


def groups_of(uid):
    g = uid.astype(np.int64).copy()
    z = g == 0
    g[z] = -(np.arange(z.sum()) + 1)       # uid 0 = pooled bucket -> singleton groups
    _, gi = np.unique(g, return_inverse=True)
    return gi


def cov_mean(D, gi=None, w=None):
    """covariance matrix of the (weighted) mean of the rows of D (n x K). gi: cluster ids -> cluster-robust."""
    n = len(D)
    w = np.ones(n) if w is None else w / w.mean()
    mu = (w[:, None] * D).sum(0) / w.sum()
    R = w[:, None] * (D - mu)
    if gi is not None:
        R = np.stack([np.bincount(gi, R[:, k]) for k in range(R.shape[1])], 1)
    return R.T @ R / w.sum() ** 2


def wcov(D, w=None):
    """per-feed (weighted) covariance of the rows of D; divide by n_pub to get the cov of a public mean."""
    w = np.ones(len(D)) if w is None else w
    mu = (w[:, None] * D).sum(0) / w.sum(); R = D - mu
    return (w[:, None] * R).T @ R / w.sum()


def cluster_m(gi):
    """m = sum n_g^2 / sum n_g (mean cluster size seen by a feed)."""
    c = np.bincount(gi)
    return (c ** 2).sum() / c.sum()


def main():
    np.set_printoptions(precision=6, suppress=True)
    R = {}
    fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy")
    ft, Mt, ot = feed_meta("test")
    tt = ft["timestamp"].to_numpy(); pubm = tt < BOUND
    n_val, n_pub = len(Mv), int(pubm.sum())
    gv = groups_of(fv["user_id"].to_numpy()); gp = groups_of(ft["user_id"].to_numpy()[pubm])
    print(f"n_val {n_val}  n_public {n_pub}  n_private {len(Mt) - n_pub}")
    print(f"cluster size m (sum n^2/sum n, uid0 singletons): val {cluster_m(gv):.3f}  public {cluster_m(gp):.3f}  "
          f"users val {gv.max() + 1} public {gp.max() + 1}")
    R["n"] = {"val": n_val, "public": n_pub, "m_val": cluster_m(gv), "m_pub": cluster_m(gp)}

    names = ["Cd", "PF7x", "G3", "M2", "Pall", "G3c", "M2c"]
    S = {k: pool_scores("val", k, fv, Mv, ov) for k in names}
    nd = {k: ndcg_per_feed(S[k], yv, ov, Mv) for k in names}
    print("\n== pool val NDCG (check vs CURRENT_STATUS: Cd .39901 PF7x .39974 G3 .40032 M2 .40094 Pall .39896)")
    for k in names:
        print(f"  {k:5s} {nd[k].mean():.5f}")
    R["val_ndcg"] = {k: float(nd[k].mean()) for k in names}

    Wpub_Mc = pmix.weights()["public"]; Wpub_M = calib.weights("public")
    print(f"n_eff of val under public weights: M-only {Wpub_M.sum() ** 2 / (Wpub_M ** 2).sum():.0f}, "
          f"MxCold {Wpub_Mc.sum() ** 2 / (Wpub_Mc ** 2).sum():.0f}")

    # ---------------- part 1: paired SEs
    pairs = [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x"), ("M2", "G3"),
             ("PF7x", "Pall"), ("G3", "G3c"), ("M2", "M2c")]
    # label-free disagreement per feed (val and public test) between the two pools' final scores
    St = {k: pool_scores("test", k, ft, Mt, ot) for k in ["Cd", "PF7x", "G3", "M2", "Pall", "G3c", "M2c"]}

    def disagree(a, b, M, off):
        fid = np.repeat(np.arange(len(M)), M)
        za, zb = zfeed(off, M, a), zfeed(off, M, b)
        return 1.0 - np.bincount(fid, za * zb, len(M)) / M   # 1 - within-feed Pearson

    Mbins = np.array([10, 25, 50, 100, 200])
    bv_M = np.searchsorted(Mbins, Mv, side="left"); bt_M = np.searchsorted(Mbins, Mt[pubm], side="left")
    print("\n== part 1: paired deltas and SEs (A-B).  val SE iid / user-cluster;  expected PUBLIC SE under models:\n"
          "   iid = sd_val(d)/sqrt(n_pub); Mw = M-reweighted var; MCw = M x cold reweighted var; DIS = E[d^2 | M bin x\n"
          "   disagreement decile] fitted on val, averaged over the public feeds' own label-free disagreement; +cl = x user deff")
    hdr = f"  {'pair':11s} {'val d':>9s} {'SEiid':>8s} {'SEcl':>8s} {'pubmix d':>9s} | {'iid':>8s} {'Mw':>8s} {'MCw':>8s} {'DIS':>8s} {'MCw+cl':>8s} | dis val/pub"
    print(hdr)
    R["pairs"] = {}
    for a, b in pairs:
        d = nd[a] - nd[b]
        se_iid = d.std() / np.sqrt(n_val)
        se_cl = np.sqrt(cov_mean(d[:, None], gv)[0, 0])
        deff_val = (se_cl / se_iid) ** 2
        icc = (deff_val - 1) / (R["n"]["m_val"] - 1)
        deff_pub = max(1 + (R["n"]["m_pub"] - 1) * icc, 0.0)
        mu_mc = np.average(d, weights=Wpub_Mc)
        v_iid = d.var(); v_M = np.average((d - np.average(d, weights=Wpub_M)) ** 2, weights=Wpub_M)
        v_Mc = np.average((d - mu_mc) ** 2, weights=Wpub_Mc)
        # disagreement-cell model
        dv = disagree(S[a], S[b], Mv, ov); dp = disagree(St[a], St[b], Mt, ot)[pubm]
        q = np.quantile(dv, np.linspace(0, 1, 11)[1:-1])
        cv = bv_M * 10 + np.searchsorted(q, dv); cp = bt_M * 10 + np.searchsorted(q, dp)
        e2 = np.bincount(cv, d ** 2, 60) / np.maximum(np.bincount(cv, minlength=60), 1)
        e1 = np.bincount(cv, d, 60) / np.maximum(np.bincount(cv, minlength=60), 1)
        assert np.isin(np.unique(cp), np.unique(cv)).all()
        v_dis = e2[cp].mean() - e1[cp].mean() ** 2
        r = {"val_d": float(d.mean()), "se_val_iid": float(se_iid), "se_val_cl": float(se_cl), "deff_val": float(deff_val),
             "icc": float(icc), "deff_pub": float(deff_pub), "pubmix_d": float(mu_mc),
             "se_pub_iid": float(np.sqrt(v_iid / n_pub)), "se_pub_Mw": float(np.sqrt(v_M / n_pub)),
             "se_pub_MCw": float(np.sqrt(v_Mc / n_pub)), "se_pub_DIS": float(np.sqrt(v_dis / n_pub)),
             "se_pub_MCw_cl": float(np.sqrt(v_Mc * deff_pub / n_pub)),
             "dis_val": float(dv.mean()), "dis_pub": float(dp.mean()), "frac_d0": float((d == 0).mean())}
        R["pairs"][f"{a}-{b}"] = r
        print(f"  {a + '-' + b:11s} {r['val_d']:+.5f} {se_iid:8.5f} {se_cl:8.5f} {mu_mc:+.5f}  | {r['se_pub_iid']:8.5f} "
              f"{r['se_pub_Mw']:8.5f} {r['se_pub_MCw']:8.5f} {r['se_pub_DIS']:8.5f} {r['se_pub_MCw_cl']:8.5f} | "
              f"{dv.mean():.4f}/{dp.mean():.4f}  deff_val {deff_val:.2f} deff_pub {deff_pub:.2f} d==0 {r['frac_d0']:.2f}")

    # FT / TO decomposition of the val deltas (test TO half = identical models; test FT half = fold mean)
    print("\n== val delta by half (score = z(ft) only / z(to) only):")
    R["halves"] = {}
    for half in ["ft", "to"]:
        H = {k: ndcg_per_feed(half_scores(k, half, fv, Mv, ov), yv, ov, Mv) for k in ["Cd", "PF7x", "G3", "M2"]}
        for a in ["PF7x", "G3", "M2"]:
            d = H[a] - H["Cd"]
            print(f"  {half} {a}-Cd {d.mean():+.5f} +- {d.std() / np.sqrt(n_val):.5f}")
            R["halves"][f"{half}:{a}-Cd"] = [float(d.mean()), float(d.std() / np.sqrt(n_val))]

    # ---------------- part 2: joint test
    A = ["PF7x", "G3", "M2"]
    D = np.stack([nd[a] - nd["Cd"] for a in A], 1)
    pub = np.array([PUB[a] - PUB["Cd"] for a in A])
    out2 = {}
    for model in ["MCw_cl", "MCw", "iid"]:
        for ref in ["val", "pubmix"]:
            if ref == "val":
                mu = D.mean(0); Sv = cov_mean(D, gv)
            else:
                mu = np.average(D, axis=0, weights=Wpub_Mc); Sv = cov_mean(D, gv, Wpub_Mc)
            w = Wpub_Mc if model.startswith("MCw") else None
            Sp = wcov(D, w) / n_pub                             # cov of mean over n_pub iid feeds
            if model == "MCw_cl":
                deffs = np.array([R["pairs"][f"{a}-Cd"]["deff_pub"] for a in A])
                Sp = Sp * np.sqrt(np.outer(deffs, deffs))
            Sig = Sv + Sp
            e = pub - mu
            chi2 = float(e @ np.linalg.solve(Sig, e)); p = float(stats.chi2.sf(chi2, 3))
            zs = e / np.sqrt(np.diag(Sig))
            # derived pairs
            L = {"M2-PF7x": [-1, 0, 1], "G3-PF7x": [-1, 1, 0], "M2-G3": [0, -1, 1]}
            zd = {k: float(np.dot(v, e) / np.sqrt(np.array(v) @ Sig @ np.array(v))) for k, v in L.items()}
            sed = {k: float(np.sqrt(np.array(v) @ Sig @ np.array(v))) for k, v in L.items()}
            # one-sided 'all under-transfer' summary: sum of standardized residual along (1,1,1)
            out2[f"{model}|{ref}"] = {"e": e.tolist(), "se": np.sqrt(np.diag(Sig)).tolist(), "z": zs.tolist(),
                                      "chi2": chi2, "p": p, "z_derived": zd, "se_derived": sed,
                                      "corr": (Sig / np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))).tolist()}
            print(f"\n== part 2 joint test  noise={model:6s} expected=({ref}) : chi2(3) {chi2:.2f}  p {p:.3f}")
            for i, a in enumerate(A):
                print(f"   {a}-Cd: pub {pub[i]:+.5f}  expected {mu[i]:+.5f}  resid {e[i]:+.5f}  SE {np.sqrt(Sig[i, i]):.5f}"
                      f" (val {np.sqrt(Sv[i, i]):.5f}, pub {np.sqrt(Sp[i, i]):.5f})  z {zs[i]:+.2f}")
            for k in L:
                print(f"   {k}: resid {np.dot(L[k], e):+.5f}  SE {sed[k]:.5f}  z {zd[k]:+.2f}")
    R["joint"] = out2

    # transfer factor (total delta): pub = b * delta_true, val = delta_true (+noise); profile chi2(b)
    Sv = cov_mean(D, gv)
    deffs = np.array([R["pairs"][f"{a}-Cd"]["deff_pub"] for a in A])
    Sp = wcov(D, Wpub_Mc) / n_pub * np.sqrt(np.outer(deffs, deffs))
    mu = D.mean(0)
    bs = np.linspace(-3, 4, 7001)

    def prof(fun):
        c = np.array([fun(b) for b in bs]); i = c.argmin()
        ci = bs[c <= c[i] + 3.841]
        return float(bs[i]), float(c[i]), (float(ci.min()), float(ci.max())), c

    def chi_total(b):
        e = pub - b * mu
        return e @ np.linalg.solve(Sp + b * b * Sv, e)
    bh, cmin, ci, c = prof(chi_total)
    print(f"\n== transfer factor, total val delta vs Cd (pub = b*val):  b = {bh:.2f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  "
          f"chi2 at b=1: {chi_total(1.0):.2f} (min {cmin:.2f}, 2 df gof p {stats.chi2.sf(cmin, 2):.2f}); "
          f"LR test b=1 p {stats.chi2.sf(chi_total(1.0) - cmin, 1):.3f}")
    R["transfer_total"] = {"b": bh, "ci": ci, "chi2_b1": float(chi_total(1.0)), "chi2_min": cmin}

    # m02 part: delta_k = n_k (counterpart - Cd, i.e. non-m02 content) + m_k (pool - counterpart, the m02 re-cool gain)
    Nn = np.stack([nd[COUNTER[a]] - nd["Cd"] for a in A], 1)
    Mm = np.stack([nd[a] - nd[COUNTER[a]] for a in A], 1)
    print("\n== m02 decomposition of the val deltas vs Cd (n = counterpart-Cd, m = pool-counterpart), user-cluster SE")
    for i, a in enumerate(A):
        sn = np.sqrt(cov_mean(Nn[:, [i]], gv)[0, 0]); sm = np.sqrt(cov_mean(Mm[:, [i]], gv)[0, 0])
        print(f"   {a}: total {D[:, i].mean():+.5f} = n {Nn[:, i].mean():+.5f} (+-{sn:.5f}) + m02 part {Mm[:, i].mean():+.5f} (+-{sm:.5f})"
              f"   [pub-mix: n {np.average(Nn[:, i], weights=Wpub_Mc):+.5f}, m {np.average(Mm[:, i], weights=Wpub_Mc):+.5f}]")
    R["decomp"] = {a: {"n": float(Nn[:, i].mean()), "m": float(Mm[:, i].mean())} for i, a in enumerate(A)}

    def chi_m02(b, a_=1.0):
        U = a_ * Nn + b * Mm
        e = pub - U.mean(0)
        return e @ np.linalg.solve(Sp + cov_mean(U, gv), e)
    bh2, cmin2, ci2, c2 = prof(chi_m02)
    print(f"== transfer factor of the m02 part only (non-m02 part transfers 1:1): b = {bh2:.2f}  95% CI [{ci2[0]:.2f}, {ci2[1]:.2f}]"
          f"  chi2 at b=1: {chi_m02(1.0):.2f}  LR p(b=1) {stats.chi2.sf(chi_m02(1.0) - cmin2, 1):.3f}; at b=0: {chi_m02(0.0):.2f}"
          f" LR p(b=0) {stats.chi2.sf(chi_m02(0.0) - cmin2, 1):.3f}")
    R["transfer_m02"] = {"b": bh2, "ci": ci2, "chi2_b1": float(chi_m02(1.0)), "chi2_b0": float(chi_m02(0.0)), "chi2_min": cmin2}
    # both factors free: profile b over a
    as_ = np.linspace(-2, 4, 241)
    cb = np.array([min(chi_m02(b, a_) for a_ in as_) for b in bs[::10]])
    i = cb.argmin(); ci3 = bs[::10][cb <= cb[i] + 3.841]
    print(f"== m02 factor with the non-m02 factor free too: b = {bs[::10][i]:.2f}  95% CI [{ci3.min():.2f}, {ci3.max():.2f}]")
    R["transfer_m02_free_a"] = {"b": float(bs[::10][i]), "ci": [float(ci3.min()), float(ci3.max())]}

    # holdout context: train-only pools on the train-period holdout
    Mh, oh = np.load(P / "holdout/holdout_meta.npy"); yh = np.load(P / "holdout/holdout_y.npy")
    Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)

    def hold_pool(ms):
        u = sorted(set(ms)); cnt = {m: ms.count(m) for m in u}
        e = sum(cnt[m] * zfeed(oh, Mh, np.load(P / f"holdout/nn_{m}_holdout.npy")) for m in u) / len(ms)
        return ndcg_per_feed(e.astype(np.float32), yh, oh, Mh)
    H = {k: hold_pool(POOLS[k]) for k in ["Cd", "PF7x", "G3", "M2"]}
    print("\n== holdout (train-only pools; context): " + "  ".join(f"{k} {H[k].mean():.5f}" for k in H))
    R["holdout"] = {}
    for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x")]:
        d = H[a] - H[b]; se = d.std() / np.sqrt(len(d))
        print(f"   {a}-{b}: holdout {d.mean():+.5f} +- {se:.5f} (iid)")
        R["holdout"][f"{a}-{b}"] = [float(d.mean()), float(se)]

    # ---------------- part 3: 8-point calibration (diagnostic only)
    subs = ["v3", "v4", "v5", "v10", "Cd", "PF7x", "G3", "M2"]
    val = np.array([0.3816, 0.3839, 0.3869, 0.39399, nd["Cd"].mean(), nd["PF7x"].mean(), nd["G3"].mean(), nd["M2"].mean()])
    cal = np.array([0.3736, 0.3759, 0.3788, 0.3859, 0.3909, 0.3917, 0.3923, 0.3929])
    pb = np.array([0.3739, 0.3772, 0.3802, 0.38786, PUB["Cd"], PUB["PF7x"], PUB["G3"], PUB["M2"]])
    R["calib"] = {}
    for xname, x in [("val", val), ("calib_pred", cal)]:
        X = np.stack([np.ones(8), x], 1)
        beta, *_ = np.linalg.lstsq(X, pb, rcond=None)
        res = pb - X @ beta
        s2 = res @ res / 6; XtXi = np.linalg.inv(X.T @ X); h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        se_b = np.sqrt(np.diag(s2 * XtXi))
        # leave-one-out prediction residual and externally studentised residual
        loo = res / (1 - h)
        s2_i = (s2 * 6 - res ** 2 / (1 - h)) / 5
        t_ext = res / np.sqrt(s2_i * (1 - h))
        print(f"\n== part 3 calibration public = a + c*{xname}: a {beta[0]:+.4f}  c {beta[1]:.3f} +- {se_b[1]:.3f}  "
              f"resid sd {np.sqrt(s2):.5f}")
        for i, sname in enumerate(subs):
            print(f"   {sname:5s} x {x[i]:.5f} pub {pb[i]:.5f} fit {X[i] @ beta:.5f} resid {res[i]:+.5f} LOO {loo[i]:+.5f} "
                  f"t_ext {t_ext[i]:+.2f} (p {2 * stats.t.sf(abs(t_ext[i]), 4):.2f})")
        R["calib"][xname] = {"a": float(beta[0]), "c": float(beta[1]), "se_c": float(se_b[1]), "sd": float(np.sqrt(s2)),
                             "res": res.tolist(), "loo": loo.tolist(), "t_ext": t_ext.tolist()}
        # slope 1 offset model
        off_ = (pb - x).mean(); r1 = pb - x - off_
        print(f"   slope-1 model: offset {off_:+.5f}; residuals " + " ".join(f"{s}:{r:+.5f}" for s, r in zip(subs, r1)))
        R["calib"][xname]["slope1_res"] = r1.tolist()
        # fit on the 5 pre-m02 points, predict the 3 new ones
        X5 = X[:5]; b5, *_ = np.linalg.lstsq(X5, pb[:5], rcond=None); r5 = pb[:5] - X5 @ b5
        s25 = r5 @ r5 / 3; XtXi5 = np.linalg.inv(X5.T @ X5)
        print(f"   fit on v3..Cd only: a {b5[0]:+.4f} c {b5[1]:.3f}, resid sd {np.sqrt(s25):.5f};  out-of-sample:")
        for i in range(5, 8):
            pr = X[i] @ b5; sp = np.sqrt(s25 * (1 + X[i] @ XtXi5 @ X[i]))
            print(f"     {subs[i]:5s} pred {pr:.5f} +- {sp:.5f}  actual {pb[i]:.5f}  miss {pb[i] - pr:+.5f}  t {(pb[i] - pr) / sp:+.2f}")
    json.dump(R, open(OUT / "noise_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
