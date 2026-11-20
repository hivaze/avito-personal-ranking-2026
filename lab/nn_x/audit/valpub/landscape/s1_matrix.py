"""s1: the solution matrix. Every current candidate pool: members / families, re-cool and m02 shares, FT/TO half values,
val, M-reweighted public/private expectations (src/calib), M x cold pmix expectations (src/pmix), holdout, public (context),
deltas vs G3 with paired SEs, and the level calibration public = a + c * val over all scored files (LOO)."""
from lib import *  # noqa
import calib, pmix

f, M, off = meta("val"); y = val_labels()
pubm = public_mask(); n_pub = int(pubm.sum()); n_priv = int((~pubm).sum())
Wpub, Wpriv = calib.weights("public"), calib.weights("private")
PM = pmix.weights(); Wpm_pub, Wpm_priv = PM["public"], PM["private"]
hd = holdout()

names = SCORED + UNSCORED
nd = {n: perfeed_val(n) for n in names}
nd["v10"] = perfeed_val("v10"); nd["v5"] = perfeed_val("v5")
hold = {n: perfeed_holdout(n) for n in names}
g3 = nd["G3"]; g3h = hold["G3"]


def fams(ms):
    c = {}
    for m in ms:
        c[FAMILY_OF.get(m, "?")] = c.get(FAMILY_OF.get(m, "?"), 0) + 1
    return "+".join(f"{k}({v})" for k, v in c.items())


print("=" * 150)
print("(1) SOLUTION MATRIX.  val = mean per-feed NDCG (14,423 feeds); pubM/privM = val reweighted to the public/private M mix (calib);")
print("    pubMC/privMC = M x cold-share reweighting (pmix; level-miscalibrated by ~+0.004, read deltas only); hold = 30K train-period")
print("    holdout of the TRAIN-ONLY members (TO half only; S2/S1/PoP3 have none); ft-half / to-half = val NDCG of z(ft) / z(to) alone.")
print("=" * 150)
hdr = f"{'pool':8s} {'n':>3s} {'recool':>6s} {'m02':>5s} {'val':>8s} {'ft-half':>8s} {'to-half':>8s} {'pubM':>8s} {'privM':>8s} {'pubMC':>8s} {'privMC':>8s} {'hold':>8s} {'public':>9s}  families"
print(hdr)
rows = []
for n in names + ["v10", "v5"]:
    ms = MEMBERS.get(n, [])
    rc = np.mean([m in RECOOL for m in ms]) if ms else np.nan
    m02 = np.mean([m in M02 for m in ms]) if ms else np.nan
    v = nd[n].mean()
    try:
        fth = ndcg_per_feed(half_scores(n, "ft", "val").astype(np.float32), y, off, M).mean()
        toh = ndcg_per_feed(half_scores(n, "to", "val").astype(np.float32), y, off, M).mean()
    except KeyError:
        fth = toh = np.nan
    if n == "TOonly":
        toh = v
    pubM, privM = np.average(nd[n], weights=Wpub), np.average(nd[n], weights=Wpriv)
    pubMC, privMC = np.average(nd[n], weights=Wpm_pub), np.average(nd[n], weights=Wpm_priv)
    h = hold.get(n); hm = np.nan if h is None else h.mean()
    pub = PUBLIC.get(n, np.nan)
    rows.append(dict(pool=n, n=len(ms), recool=rc, m02=m02, val=v, ft_half=fth, to_half=toh, pubM=pubM, privM=privM,
                     pubMC=pubMC, privMC=privMC, hold=hm, public=pub, families=fams(ms)))
    print(f"{n:8s} {len(ms):3d} {rc:6.2f} {m02:5.2f} {v:8.5f} {fth:8.5f} {toh:8.5f} {pubM:8.5f} {privM:8.5f} {pubMC:8.5f} {privMC:8.5f} {hm:8.5f} {pub:9.5f}  {fams(ms)}")
pl.DataFrame(rows).write_csv(OUT / "s1_matrix.csv")

print("\n" + "=" * 150)
print("(1b) DELTAS vs G3 (paired per-feed).  val d ± SE_val; pubM/privM d (M mix) ; pubMC/privMC d (M x cold) ; hold d ± SE (TO members only);")
print("     public d = exact difference on the 8,562 public feeds; SE_pub = expected paired SE sqrt(Var_pubM(d)/8562); resid = public d - pubM d.")
print("=" * 150)
print(f"{'pool':8s} {'val d':>9s} {'SE_val':>8s} {'pubM d':>9s} {'privM d':>9s} {'pubMC d':>9s} {'privMC d':>9s} {'hold d':>9s} {'SE_h':>8s} {'public d':>9s} {'SE_pub':>8s} {'resid':>9s} {'z':>5s}")
rows = []
for n in names:
    if n == "G3":
        continue
    d = nd[n] - g3
    m_v, se_v = mse(d)
    m_pub, _ = mse(d, Wpub); m_priv, _ = mse(d, Wpriv)
    m_pmc, _ = mse(d, Wpm_pub); m_pmcp, _ = mse(d, Wpm_priv)
    v_pub = np.average((d - m_pub) ** 2, weights=Wpub); se_pub = np.sqrt(v_pub / n_pub)
    se_priv = np.sqrt(np.average((d - m_priv) ** 2, weights=Wpriv) / n_priv)
    if hold.get(n) is not None:
        hdl = hold[n] - g3h; m_h, se_h = mse(hdl)
    else:
        m_h = se_h = np.nan
    pd_ = PUBLIC[n] - PUBLIC["G3"] if n in PUBLIC else np.nan
    resid = pd_ - m_pub; z = resid / np.sqrt(se_pub ** 2 + se_v ** 2)
    rows.append(dict(pool=n, val_d=m_v, se_val=se_v, pubM_d=m_pub, privM_d=m_priv, pubMC_d=m_pmc, privMC_d=m_pmcp, hold_d=m_h, se_hold=se_h,
                     public_d=pd_, se_pub=se_pub, se_priv=se_priv, resid=resid, z=z))
    print(f"{n:8s} {m_v:+9.5f} {se_v:8.5f} {m_pub:+9.5f} {m_priv:+9.5f} {m_pmc:+9.5f} {m_pmcp:+9.5f} {m_h:+9.5f} {se_h:8.5f} {pd_:+9.5f} {se_pub:8.5f} {resid:+9.5f} {z:+5.1f}")
pl.DataFrame(rows).write_csv(OUT / "s1_deltas_vs_G3.csv")

# FT value inside each pool: final - to-half (what the val-fine-tuned half adds on val), and vs G3's
print("\n(1c) FT-half value on val inside each pool (final - to-half, paired ± SE) and the to-half delta vs G3's to-half:")
g3to = ndcg_per_feed(half_scores("G3", "to", "val").astype(np.float32), y, off, M)
for n in names:
    try:
        toh = ndcg_per_feed(half_scores(n, "to", "val").astype(np.float32), y, off, M)
    except KeyError:
        continue
    d = nd[n] - toh; dt = toh - g3to
    print(f"  {n:8s} FT adds {fmt(*mse(d))}   to-half vs G3 to-half {fmt(*mse(dt))}   to-half pubM {mse(dt, Wpub)[0]:+.5f}")

# level calibration public = a + c * x on all scored files, x = val and x = pubM (M-reweighted val), LOO
print("\n" + "=" * 150)
print("(1d) LEVEL CALIBRATION public = a + c * x over ALL scored files (11 points incl. Pall/S2/TOonly; old v3/v4 use recorded val only).")
print("=" * 150)
xs_val = {k: VAL_RECORDED[k] for k in PUBLIC}
xs_pubM = {}
for k in PUBLIC:
    if k in nd:
        xs_pubM[k] = float(np.average(nd[k], weights=Wpub))
for label, xs in [("val", xs_val), ("pubM (M-reweighted val; v3/v4 missing)", xs_pubM)]:
    ks = sorted(xs, key=lambda k: xs[k]); x = np.array([xs[k] for k in ks]); p = np.array([PUBLIC[k] for k in ks])
    X = np.column_stack([np.ones_like(x), x]); beta = np.linalg.lstsq(X, p, rcond=None)[0]
    res = p - X @ beta; sd = res.std(ddof=2)
    print(f"\n  x = {label}: a {beta[0]:+.4f}  c {beta[1]:.3f}  resid sd {sd:.5f}  (n {len(x)})")
    print(f"  {'file':8s} {'x':>8s} {'public':>8s} {'fit':>8s} {'resid':>8s} {'LOO':>8s} {'t_ext':>6s}")
    for i, k in enumerate(ks):
        m = np.ones(len(x), bool); m[i] = False
        b = np.linalg.lstsq(X[m], p[m], rcond=None)[0]; r = p[m] - X[m] @ b; s = r.std(ddof=2)
        loo = p[i] - X[i] @ b
        xm = x[m]; lev = 1 / m.sum() + (x[i] - xm.mean()) ** 2 / ((xm - xm.mean()) ** 2).sum()
        print(f"  {k:8s} {x[i]:8.5f} {p[i]:8.5f} {X[i] @ beta:8.5f} {res[i]:+8.5f} {loo:+8.5f} {loo / (s * np.sqrt(1 + lev)):+6.2f}")
    # modern-only fit (7 points)
    mm = np.array([k in SCORED for k in ks])
    if mm.sum() >= 3:
        b = np.linalg.lstsq(X[mm], p[mm], rcond=None)[0]; r = p[mm] - X[mm] @ b
        print(f"  modern 7 only: a {b[0]:+.4f} c {b[1]:.3f} ± {np.sqrt((r.var(ddof=2)) / ((x[mm] - x[mm].mean()) ** 2).sum()):.3f}  resid sd {r.std(ddof=2):.5f}; "
              f"slope-1 offset {np.mean(p[mm] - x[mm]):+.5f} (sd {np.std(p[mm] - x[mm], ddof=1):.5f})")
print("\nNOTE: with 7 modern points spanning 0.002 of val and a public paired SE of 0.0004-0.0007 per near-copy, the slope within the modern")
print("cluster is not identifiable (SE ~0.3-0.4); the 0.976 slope over 11 points is carried by the 0.02 spread of the old generations.")
