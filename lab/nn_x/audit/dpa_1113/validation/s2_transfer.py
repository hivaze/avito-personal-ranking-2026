"""S2: public-matched val weights (exact entropy balancing, written to out/w_val_pubmatch.npz) and the transfer of
public deltas vs G3 on val / M-only / pmix(pub, priv) / pub-matched val / val TO-half / holdout deltas over the 8 scored
contrasts (Cd, Pall, PF7x, M2, S2, TOonly, Q1a, Q2a). GLS with the full covariance of public noise (shared G3 draw) and
proxy noise. Writes out/s2_transfer.log, out/s2_deltas.csv.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python <scratch>/dpa/validation/s2_transfer.py
"""
import sys
from lib import *  # noqa: F401,F403
sys.path.insert(0, "<repo>/eda/errors/segments")

LOG = open(OUT / "s2_transfer.log", "w")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ------------------------------------------------------------------ weights
def calibM_pub():
    EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
    fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test")
    pub = public_mask()
    bv = np.searchsorted(EDGES, Mv, side="right"); bt = np.searchsorted(EDGES, Mt[pub], side="right")
    pv = np.bincount(bv, minlength=10) / len(bv); pt = np.bincount(bt, minlength=10) / len(bt)
    w = (pt / pv)[bv]
    return w / w.mean()


def eb_weights(Xv, target, base=None, iters=100):
    """entropy balancing: w_i ∝ base_i exp(X_i·lam) with sum_i w_i X_i / sum w = target exactly (Newton on the dual)."""
    n, k = Xv.shape
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-12
    Z = (Xv - mu) / sd; t = (target - mu) / sd
    lam = np.zeros(k); b = np.ones(n) if base is None else base / base.mean()
    for _ in range(iters):
        e = Z @ lam; e -= e.max(); w = b * np.exp(e); w /= w.sum()
        m = w @ Z; g = m - t
        C = (Z - m).T @ ((Z - m) * w[:, None])
        step = np.linalg.solve(C + 1e-10 * np.eye(k), g)
        lam -= step
        if np.abs(g).max() < 1e-10:
            break
    e = Z @ lam; e -= e.max(); w = b * np.exp(e); w /= w.mean()
    return w, np.abs((w @ Z) / w.sum() - t).max()


def desc_mats():
    dv = pl.read_parquet(OUT / "desc_val.parquet"); dp = pl.read_parquet(OUT / "desc_public.parquet")
    dq = pl.read_parquet(OUT / "desc_private.parquet"); dh = pl.read_parquet(OUT / "desc_holdout.parquet")
    def feats(d):
        lm = np.log(d["M"].to_numpy()); c = d["cold_tte"].to_numpy(); z = d["luo_zero"].to_numpy()
        return np.c_[lm, lm ** 2, (d["M"].to_numpy() < 50), c, c ** 2, c * lm, z, z ** 2, d["uid0"].to_numpy(),
                     d["user_in_train_before"].to_numpy(), d["H"].to_numpy() / 256.0].astype(np.float64)
    names = ["logM", "logM^2", "M<50", "cold", "cold^2", "cold*logM", "luo0", "luo0^2", "uid0", "user_tr", "H/256"]
    return feats(dv), feats(dp), feats(dq), feats(dh), names


# ------------------------------------------------------------------ GLS transfer
def gls_k(P, SP, X, SX, it=50):
    k = 1.0
    for _ in range(it):
        V = SP + k * k * SX
        Vi = np.linalg.inv(V)
        k_new = (X @ Vi @ P) / (X @ Vi @ X)
        if abs(k_new - k) < 1e-9:
            break
        k = k_new
    V = SP + k * k * SX; Vi = np.linalg.inv(V)
    se = 1 / np.sqrt(X @ Vi @ X)
    def chi2(kk):
        Vk = SP + kk * kk * SX; r = P - kk * X
        return float(r @ np.linalg.inv(Vk) @ r)
    return k, se, chi2(k), chi2(1.0), chi2(0.0)


def loo(P, SP, X, SX):
    """leave-one-contrast-out: fit k on the rest, standardised prediction error of the left-out contrast."""
    out = []
    n = len(P)
    for i in range(n):
        m = np.arange(n) != i
        k, se, *_ = gls_k(P[m], SP[np.ix_(m, m)], X[m], SX[np.ix_(m, m)])
        pred = k * X[i]
        var = SP[i, i] + k * k * SX[i, i] + (X[i] * se) ** 2
        out.append((P[i] - pred, np.sqrt(var)))
    e = np.array(out)
    return float(np.sqrt((e[:, 0] ** 2).mean())), float((e[:, 0] ** 2 / e[:, 1] ** 2).sum())


def main():
    fv, Mv, offv = meta("val")
    # -------- weights
    Xv, Xp, Xq, Xh, names = desc_mats()
    w_eb, err = eb_weights(Xv, Xp.mean(0))
    w_eb_priv, err2 = eb_weights(Xv, Xq.mean(0))
    w_M = calibM_pub()
    import pmix
    W = pmix.weights()
    pr(f"EB public weights: max moment error {err:.1e}, n_eff {neff(w_eb):.0f} of {len(w_eb)}, max w {w_eb.max():.2f}")
    pr(f"EB private weights: max moment error {err2:.1e}, n_eff {neff(w_eb_priv):.0f}")
    pr(f"pmix public n_eff {neff(W['public']):.0f}; pmix private n_eff {neff(W['private']):.0f}; calib M-only n_eff {neff(w_M):.0f}")
    pr("moment match (val raw | val EB-pub | public | private):")
    for j, nme in enumerate(names):
        pr(f"   {nme:10s} {Xv[:, j].mean():8.4f} {np.average(Xv[:, j], weights=w_eb):8.4f} {Xp[:, j].mean():8.4f} {Xq[:, j].mean():8.4f}")
    np.savez(OUT / "w_val_pubmatch.npz", w_pub=w_eb, w_priv=w_eb_priv, names=np.array(names),
             target_pub=Xp.mean(0), target_priv=Xq.mean(0),
             formula=np.array("w_i = exp(sum_j lam_j z_ij) / mean, z = standardised [logM, logM^2, 1(M<50), cold, cold^2, cold*logM, "
                              "luo0, luo0^2, uid0, user_in_train_before, H/256] of val feed i; lam solves sum_i w_i z_i / sum_i w_i = "
                              "mean over public (resp. private) test feeds (entropy balancing, exact). cold = share of candidates with "
                              "tte_it_imp == 0; luo0 = share with it_hist_all_luo == 0."))
    # holdout EB to public (show the support problem)
    w_h, errh = eb_weights(Xh[:, [3, 4, 6, 7, 8, 9, 10]], Xp[:, [3, 4, 6, 7, 8, 9, 10]].mean(0))
    pr(f"holdout EB to public on (cold, cold^2, luo0, luo0^2, uid0, user_tr, H): moment err {errh:.1e}, n_eff {neff(w_h):.0f} of 30000, "
       f"max w {w_h.max():.1f}")
    np.save(OUT / "w_hold_pubmatch_noM.npy", w_h)

    # -------- per-feed NDCG
    pv = {p: perfeed_val(p) for p in POOLS}
    pto = {p: perfeed_val(p, "to") for p in POOLS if p not in ("S2",)}
    ph = {p: perfeed_holdout(p) for p in POOLS}
    pr("\nlevels: pool | val (recorded G3 0.40032) | val TO-half | holdout | public")
    for p in POOLS:
        pr(f"  {p:7s} {pv[p].mean():.5f} {pto[p].mean() if p in pto else float('nan'):.5f} "
           f"{ph[p].mean() if ph[p] is not None else float('nan'):.5f} {PUBLIC[p]:.6f}")

    weights = {"val": None, "calibM_pub": w_M, "pmix_pub": W["public"], "pmix_priv": W["private"], "EB_pub": w_eb, "EB_priv": w_eb_priv}
    C = CONTRASTS
    d_val = {p: pv[p] - pv["G3"] for p in C}
    rows = []
    proxies = {}
    for wn, w in weights.items():
        X = np.array([wmean_se(d_val[p], w)[0] for p in C])
        D = np.stack([d_val[p] for p in C], 1)
        ww = np.ones(len(Mv)) if w is None else w
        mu = (ww[:, None] * D).sum(0) / ww.sum(); R = D - mu
        SX = (ww[:, None] ** 2 * R).T @ R / ww.sum() ** 2
        proxies[wn] = (X, SX)
    # TO half (val) and holdout (TO members); TOonly holdout = 0 by construction; S2 has none
    Cto = [p for p in C if p != "S2"]
    D = np.stack([pto[p] - pto["G3"] for p in Cto], 1)
    X = D.mean(0); R = D - X; proxies["val_TO"] = (X, R.T @ R / len(D) ** 2)
    Ch = [p for p in C if p not in ("S2", "TOonly")]
    D = np.stack([ph[p] - ph["G3"] for p in Ch], 1)
    X = D.mean(0); R = D - X; proxies["holdout"] = (X, R.T @ R / len(D) ** 2)
    # public noise: per-feed delta covariance on val, EB-pub weighted, / N_PUB (public shares G3's draw)
    Dall = np.stack([d_val[p] for p in C], 1)
    mu = (w_eb[:, None] * Dall).sum(0) / w_eb.sum(); R = Dall - mu
    SP_all = (w_eb[:, None] * R).T @ R / w_eb.sum() / N_PUB
    P_all = np.array([PUBLIC[p] - PUBLIC["G3"] for p in C])

    pr("\n== per-contrast deltas vs G3 (± SE): public | val | calibM | pmix_pub | EB_pub | pmix_priv | val_TO | holdout ==")
    for i, p in enumerate(C):
        line = f"  {p:7s} pub {P_all[i]:+.5f} (±{np.sqrt(SP_all[i, i]):.5f})"
        for wn in ["val", "calibM_pub", "pmix_pub", "EB_pub", "pmix_priv"]:
            X, SX = proxies[wn]; line += f" | {wn} {X[i]:+.5f}±{np.sqrt(SX[i, i]):.5f}"
        if p in Cto:
            j = Cto.index(p); X, SX = proxies["val_TO"]; line += f" | val_TO {X[j]:+.5f}±{np.sqrt(SX[j, j]):.5f}"
        if p in Ch:
            j = Ch.index(p); X, SX = proxies["holdout"]; line += f" | hold {X[j]:+.5f}±{np.sqrt(SX[j, j]):.5f}"
        pr(line)
        rows.append(dict(contrast=p, public=P_all[i], se_pub=np.sqrt(SP_all[i, i]),
                         **{wn: proxies[wn][0][i] for wn in weights},
                         val_TO=proxies["val_TO"][0][Cto.index(p)] if p in Cto else np.nan,
                         holdout=proxies["holdout"][0][Ch.index(p)] if p in Ch else np.nan))
    pl.DataFrame(rows).write_csv(OUT / "s2_deltas.csv")

    def fit_set(label, cs):
        idx = [C.index(p) for p in cs]
        P = P_all[idx]; SP = SP_all[np.ix_(idx, idx)]
        pr(f"\n== transfer fits on {label} ({len(cs)} contrasts: {','.join(cs)}); public = k * proxy ==")
        pr(f"  {'proxy':11s} {'k':>7s} {'SE':>6s} {'chi2(k^)':>9s} {'chi2(k=1)':>9s} {'chi2(k=0)':>9s} {'gR2':>6s} {'LOO rmse':>9s} {'LOO z2':>7s} "
           f"{'resid Q1a':>9s} {'resid Q2a':>9s}")
        for wn, (X, SX) in proxies.items():
            src = Cto if wn == "val_TO" else (Ch if wn == "holdout" else C)
            if not all(p in src for p in cs):
                continue
            j = [src.index(p) for p in cs]
            Xs, SXs = X[j], SX[np.ix_(j, j)]
            k, se, c_hat, c1, c0 = gls_k(P, SP, Xs, SXs)
            lr, lz = loo(P, SP, Xs, SXs)
            res = {p: P[cs.index(p)] - k * Xs[cs.index(p)] for p in ("Q1a", "Q2a") if p in cs}
            pr(f"  {wn:11s} {k:+7.2f} {se:6.2f} {c_hat:9.2f} {c1:9.2f} {c0:9.2f} {1 - c_hat / c0:6.2f} {lr:9.5f} {lz:7.2f} "
               f"{res.get('Q1a', np.nan):+9.5f} {res.get('Q2a', np.nan):+9.5f}")
        # implied SE of a 1:1 / fitted read
    fit_set("ALL (val-family proxies)", C)
    fit_set("H6 (holdout-defined)", Ch)
    fit_set("NOPOP-6 (pre-11-13 pools, as in VAL_PUBLIC_1113)", [p for p in C if p not in ("Q1a", "Q2a")])
    fit_set("H4 (holdout-defined, pre-11-13)", [p for p in Ch if p not in ("Q1a", "Q2a")])

    # -------- slices of the two new contrasts
    pr("\n== Q1a / Q2a vs G3 by slice on val (uniform) and holdout ==")
    dvd = pl.read_parquet(OUT / "desc_val.parquet"); dhd = pl.read_parquet(OUT / "desc_holdout.parquet")
    for p in ("Q1a", "Q2a"):
        dv = d_val[p]; dh = ph[p] - ph["G3"]; dto = pto[p] - pto["G3"]; dft = perfeed_val(p, "ft") - perfeed_val("G3", "ft")
        pr(f"  {p}: val {fmt(*wmean_se(dv))}  TO-half {fmt(*wmean_se(dto))}  FT-half {fmt(*wmean_se(dft))}  holdout {fmt(*wmean_se(dh))}")
        for nm, m in [("M<50", dvd["M"].to_numpy() < 50), ("M>=50", dvd["M"].to_numpy() >= 50),
                      ("cold<.3", dvd["cold_tte"].to_numpy() < .3), ("cold.3-.45", (dvd["cold_tte"].to_numpy() >= .3) & (dvd["cold_tte"].to_numpy() < .45)),
                      ("cold>=.45", dvd["cold_tte"].to_numpy() >= .45), ("luo0>=.45", dvd["luo_zero"].to_numpy() >= .45),
                      ("Tue", dvd["day"].to_numpy() < 20719), ("Thu", dvd["day"].to_numpy() >= 20719)]:
            pr(f"     val {nm:11s} n {m.sum():5d} d {fmt(*wmean_se(dv[m]))}  TO {fmt(*wmean_se(dto[m]))}")
        hd = dhd["day"].to_numpy(); hc = dhd["cold_tte"].to_numpy(); hz = dhd["luo_zero"].to_numpy()
        for nm, m in [("day0 (cold=1)", hd == 20702), ("days 20703-10", (hd > 20702) & (hd < 20711)), ("days 20713-15", hd >= 20713),
                      ("luo0>=.3", hz >= .3), ("luo0<.3", hz < .3), ("weekend", dhd["dow"].to_numpy() >= 5)]:
            pr(f"     hold {nm:14s} n {m.sum():5d} d {fmt(*wmean_se(dh[m]))}")
        pr(f"     hold EB-to-public (no M) d {fmt(*wmean_se(dh, w_h))}")


def fmt(m, se):
    return f"{m:+.5f}±{se:.5f} (z {m / se:+.2f})"


if __name__ == "__main__":
    main()
