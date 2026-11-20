"""s3: for each SCORED contrast vs G3 (Cd, Pall, PF7x, M2, S2, TOonly): decomposition of the val delta by feed type, what public
'should' have shown under 1:1 transfer per feed type (label-free public composition), what it did show, and which feed types
could carry the shortfall (leave-one-slice-out transfer). Then PRE-SPECIFIED transfer models compared on the 6 scored points with
leave-one-out prediction error, and the within-val (Tue -> Thu) and holdout (weekday -> weekend) forward evidence.
Pre-specified hypotheses (written before the numbers were computed):
  H1 1:1          public d = pubM d (val d reweighted to the public M mix)
  H1c 1:1 MxCold  public d = pubMC d (M x cold-share reweighting)
  H1h 1:1 hour    public d = hour-post-stratified val d
  H2 shrink k     public d = k * pubM d                      (k fitted; LOO)
  H3 a + b x      public d = a + b * pubM d                  (c2_03's shrinkage model; LOO)
  H4 re-cool      public d = pubM d + beta * (recool share - 0.70)   (publicprobe's dose-response; LOO)
  H5 holdout      public d = holdout d (TO members; TOonly = 0 by construction; S2 = -0.00006 recorded OOS)
  H6 blend        public d = 0.5 pubM d + 0.5 holdout d
  H7 TO-half      public d = val d of the TO halves only (FT half contributes 0 forward); S2 excluded (no half)
  H0 null         public d = 0 for every pool (all near-copies are equal on public; only G3's own draw matters)
Fit quality = chi2 of (realised - predicted) / SE_tot, SE_tot^2 = SE_pub^2 + SE_val^2; LOO RMSE for fitted models."""
from lib import *  # noqa
import calib, pmix

f, M, off = meta("val"); y = val_labels()
ft, Mt, ot = meta("test"); pubm = public_mask(); n_pub = int(pubm.sum())
Wpub, Wpriv = calib.weights("public"), calib.weights("private")
PM = pmix.weights(); Wpm_pub, Wpm_priv = PM["public"], PM["private"]
dv = descriptors("val"); dt = descriptors("test")
g3pos = best_pos_pct(pool_scores("G3", "val"), y, M, off)
Sv = slicings(dv, g3pos); St = slicings(dt)
pub_t, priv_t = dt["public"], ~dt["public"]
hd = holdout()

CON = [n for n in SCORED if n != "G3"]
nd = {n: perfeed_val(n) for n in SCORED}; g3 = nd["G3"]
hold = {n: perfeed_holdout(n) for n in SCORED}; g3h = hold["G3"]
real = {n: PUBLIC[n] - PUBLIC["G3"] for n in CON}
share = {n: float(np.mean([m in RECOOL for m in MEMBERS[n]])) for n in CON}


def poststrat(d, var, target):
    """sum over slices of target-share(s) * mean_val(d | s)"""
    labs = np.unique(Sv[var]); out = 0.0
    tgt = St[var][target] if var in St else None
    for lab in labs:
        m = Sv[var] == lab
        w = (tgt == lab).mean() if tgt is not None else m.mean()
        out += w * d[m].mean()
    return out


print("=" * 150)
print("(3a) VAL DELTA vs G3 -> EXPECTED PUBLIC under 1:1 transfer per feed type (post-stratified by one label-free variable at a time)")
print("     vs REALISED public delta. resid = realised - expectation; SE_tot = sqrt(SE_pub^2 + SE_val^2).")
print("=" * 150)
VARS = ["M", "hour", "cold_share", "luo_cold", "user", "H", "top_cat", "n_cat", "cand_cat_in_hist"]
print(f"{'pool':7s} {'val d':>9s} {'pubM':>9s} {'pubMC':>9s} " + " ".join(f"{'ps:' + v:>12s}" for v in VARS) + f" {'realised':>9s} {'resid':>9s} {'SE_tot':>8s} {'z':>5s}")
stats = {}
for n in CON:
    d = nd[n] - g3
    m_v, se_v = mse(d); m_pub, _ = mse(d, Wpub); m_pmc, _ = mse(d, Wpm_pub)
    se_pub = np.sqrt(np.average((d - m_pub) ** 2, weights=Wpub) / n_pub); se_tot = np.hypot(se_pub, se_v)
    ps = {v: poststrat(d, v, pub_t) for v in VARS}
    resid = real[n] - m_pub
    stats[n] = dict(val=m_v, se_val=se_v, pubM=m_pub, pubMC=m_pmc, se_pub=se_pub, se_tot=se_tot, real=real[n], ps=ps,
                    privM=mse(d, Wpriv)[0], privMC=mse(d, Wpm_priv)[0], ps_priv={v: poststrat(d, v, priv_t) for v in VARS})
    print(f"{n:7s} {m_v:+9.5f} {m_pub:+9.5f} {m_pmc:+9.5f} " + " ".join(f"{ps[v]:+12.5f}" for v in VARS) + f" {real[n]:+9.5f} {resid:+9.5f} {se_tot:8.5f} {resid / se_tot:+5.1f}")
print("\nReading: if any single label-free composition variable explained the shortfall, its post-stratified column would sit near 'realised'.")

print("\n" + "=" * 150)
print("(3b) WHERE THE VAL EDGE SITS, in public-composition terms: contribution share_pub(s) * d_val(s) per slice (sums to the ps column),")
print("     and LEAVE-ONE-SLICE-OUT: expected public if slice s transferred at ZERO and the rest 1:1. '<-' marks slices whose zeroing")
print("     brings the expectation within 1 SE_tot of the realised value (candidate carriers of the shortfall; NOT a proof).")
print("=" * 150)
for var in ["hour", "cold_share", "M", "user", "luo_cold"]:
    labs = sorted(set(Sv[var]))
    print(f"\n  == {var} ==      share_pub | " + " | ".join(f"{n:>24s}" for n in CON))
    print(f"  {'realised':16s}          | " + " | ".join(f"{real[n]:+24.5f}" for n in CON))
    print(f"  {'1:1 expectation':16s}          | " + " | ".join(f"{stats[n]['ps'][var]:+24.5f}" for n in CON))
    for lab in labs:
        m = Sv[var] == lab; sp = (St[var][pub_t] == lab).mean()
        cells = []
        for n in CON:
            d = nd[n] - g3; contrib = sp * d[m].mean(); loo = stats[n]["ps"][var] - contrib
            mark = "<-" if abs(loo - real[n]) <= stats[n]["se_tot"] else "  "
            cells.append(f"c {contrib:+.5f} loo {loo:+.5f}{mark}")
        print(f"  {lab:16s} {sp:8.3f} | " + " | ".join(f"{c:>24s}" for c in cells))

print("\n" + "=" * 150)
print("(3c) LABEL-DEFINED feed types on val (not observable on test): where the val edge sits by number of positives, cold positives,")
print("     and the rank of the positive under G3. Composition of these types on public/private is unknown; the pos_cold mix shifts")
print("     with the cold share of candidates (val 0.427 cold>.3 -> public 0.735 -> private 0.836), so edges that sit in warm-positive")
print("     feeds are the ones most diluted on test.")
print("=" * 150)
for var in ["npos", "pos_cold", "g3_pos_rank", "day", "pos_cat_in_hist"]:
    labs = sorted(set(Sv[var]))
    print(f"\n  == {var} ==        n   val% | " + " | ".join(f"{n:>17s}" for n in CON))
    for lab in labs:
        m = Sv[var] == lab
        cells = [fmt(*mse((nd[n] - g3)[m])) for n in CON]
        print(f"  {lab:16s} {m.sum():5d} {m.mean():5.3f} | " + " | ".join(f"{c:>17s}" for c in cells))
    # contribution of each slice to the total val delta
    print(f"  {'contrib to d_val':16s}             | " + " | ".join(
        f"{' '.join(f'{(Sv[var] == lab).mean() * (nd[n] - g3)[Sv[var] == lab].mean():+.4f}' for lab in labs):>17s}"[:17] for n in CON))

# cold-positive mix counterfactual: reweight val feeds so that P(pos_cold class) matches an assumed test mix
print("\n  pos_cold counterfactual: val d reweighted so that the pos_cold class mix equals an assumed test mix")
print("  (mix estimated label-free in nn_x/pubdiag/shift p3: coldpos/mixed/warm val .237/.131/.632, public .270/.147/.584, private .291/.150/.559)")
mixes = {"val": (0.237, 0.131, 0.632), "public": (0.270, 0.147, 0.584), "private": (0.291, 0.150, 0.559)}
cls = Sv["pos_cold"]; order = ["pos_all_cold", "pos_some_cold", "pos_warm"]
for n in CON:
    d = nd[n] - g3; out = []
    for k, mix in mixes.items():
        w = np.zeros(len(d))
        for lab, tgt in zip(order, mix):
            m = cls == lab; w[m] = tgt / m.mean()
        w[~np.isin(cls, order)] = 0.0   # feeds without positives (NDCG 1 for all pools; d = 0)
        w = w / w.mean()
        out.append(f"{k} {mse(d, w)[0]:+.5f}")
    print(f"    {n:7s} " + " | ".join(out) + f" | realised {real[n]:+.5f}")

# ---------------------------------------------------------------- (3d) pre-specified transfer models, LOO
print("\n" + "=" * 150)
print("(3d) PRE-SPECIFIED TRANSFER MODELS on the 6 scored contrasts vs G3 (realised public deltas are exact; SE_tot per point).")
print("=" * 150)
x_pubM = np.array([stats[n]["pubM"] for n in CON]); x_pubMC = np.array([stats[n]["pubMC"] for n in CON])
x_hour = np.array([stats[n]["ps"]["hour"] for n in CON]); r = np.array([real[n] for n in CON])
se = np.array([stats[n]["se_tot"] for n in CON]); w = 1 / se ** 2
sh = np.array([share[n] for n in CON])
hold_d = np.array([(hold[n] - g3h).mean() if hold.get(n) is not None else (0.0 if n == "TOonly" else -0.00006) for n in CON])
g3to = ndcg_per_feed(half_scores("G3", "to", "val").astype(np.float32), y, off, M)
to_d = []
for n in CON:
    if n in ("Cd", "Pall", "PF7x", "M2"):
        to_d.append((ndcg_per_feed(half_scores(n, "to", "val").astype(np.float32), y, off, M) - g3to).mean())
    elif n == "TOonly":
        to_d.append(0.0)
    else:
        to_d.append(np.nan)
to_d = np.array(to_d)
print(f"{'pool':7s} {'realised':>9s} {'SE_tot':>8s} {'pubM':>9s} {'pubMC':>9s} {'ps:hour':>9s} {'holdout d':>9s} {'TO-half d':>9s} {'recool':>7s}")
for i, n in enumerate(CON):
    print(f"{n:7s} {r[i]:+9.5f} {se[i]:8.5f} {x_pubM[i]:+9.5f} {x_pubMC[i]:+9.5f} {x_hour[i]:+9.5f} {hold_d[i]:+9.5f} {to_d[i]:+9.5f} {sh[i]:7.2f}")


def chi2(pred, mask=None):
    m = np.ones(len(r), bool) if mask is None else mask
    return float((((r - pred) ** 2) * w)[m].sum()), int(m.sum())


def wls(X, yv, ww):
    A = X.T @ (ww[:, None] * X); b = np.linalg.solve(A, X.T @ (ww * yv)); return b, np.linalg.inv(A)


def loo_rmse(design_fn, mask=None):
    """design_fn(idx) -> X rows; LOO prediction of each point from the other points."""
    m = np.ones(len(r), bool) if mask is None else mask
    idx = np.where(m)[0]; errs = []
    for i in idx:
        tr = idx[idx != i]
        X = design_fn(tr); b, _ = wls(X, r[tr], w[tr])
        errs.append((r[i] - design_fn(np.array([i]))[0] @ b) ** 2)
    return float(np.sqrt(np.mean(errs)))


res = []
res.append(("H0 null (all = G3)", chi2(np.zeros(6)), None, "no fit"))
res.append(("H1 1:1 pubM", chi2(x_pubM), None, "no fit"))
res.append(("H1c 1:1 pubMC", chi2(x_pubMC), None, "no fit"))
res.append(("H1h 1:1 hour-ps", chi2(x_hour), None, "no fit"))
k = (w * x_pubM * r).sum() / (w * x_pubM ** 2).sum(); sek = 1 / np.sqrt((w * x_pubM ** 2).sum())
res.append((f"H2 k*pubM  k={k:+.2f}±{sek:.2f}", chi2(k * x_pubM), loo_rmse(lambda ix: x_pubM[ix][:, None]), "1 param"))
X3 = np.column_stack([np.ones(6), x_pubM]); b3, C3 = wls(X3, r, w)
res.append((f"H3 a+b*pubM a={b3[0]:+.5f}±{np.sqrt(C3[0, 0]):.5f} b={b3[1]:+.2f}±{np.sqrt(C3[1, 1]):.2f}", chi2(X3 @ b3),
            loo_rmse(lambda ix: np.column_stack([np.ones(len(ix)), x_pubM[ix]])), "2 params"))
xs = sh - 0.70; beta = (w * xs * (r - x_pubM)).sum() / (w * xs ** 2).sum(); seb = 1 / np.sqrt((w * xs ** 2).sum())
res.append((f"H4 pubM+beta*(share-.7) beta={beta:+.5f}±{seb:.5f}", chi2(x_pubM + beta * xs),
            float(np.sqrt(np.mean([(r[i] - x_pubM[i] - ((np.delete(w * xs * (r - x_pubM), i)).sum() / np.delete(w * xs ** 2, i).sum()) * xs[i]) ** 2 for i in range(6)]))), "1 param"))
res.append(("H5 holdout d", chi2(hold_d), None, "no fit"))
res.append(("H6 0.5 pubM + 0.5 holdout", chi2(0.5 * x_pubM + 0.5 * hold_d), None, "no fit"))
mto = ~np.isnan(to_d)
res.append(("H7 TO-half d (S2 excl.)", chi2(np.where(mto, to_d, 0), mto), None, "no fit"))
# G3-luck variant of H1: 1:1 plus a common offset (G3's own public draw)
X1 = np.column_stack([np.ones(6)]); b1, C1 = wls(X1, r - x_pubM, w)
res.append((f"H1+offset (1:1 + common c={b1[0]:+.5f}±{np.sqrt(C1[0, 0]):.5f})", chi2(x_pubM + b1[0]),
            loo_rmse(lambda ix: np.ones((len(ix), 1))) if False else float(np.sqrt(np.mean([(r[i] - x_pubM[i] - np.average(np.delete(r - x_pubM, i), weights=np.delete(w, i))) ** 2 for i in range(6)]))), "1 param"))
bh, Ch = wls(X1, r - hold_d, w)
res.append((f"H5+offset (holdout + c={bh[0]:+.5f}±{np.sqrt(Ch[0, 0]):.5f})", chi2(hold_d + bh[0]),
            float(np.sqrt(np.mean([(r[i] - hold_d[i] - np.average(np.delete(r - hold_d, i), weights=np.delete(w, i))) ** 2 for i in range(6)]))), "1 param"))
from scipy.stats import chi2 as chi2dist
print(f"\n{'model':70s} {'chi2':>6s} {'n':>2s} {'df':>3s} {'p':>6s} {'LOO rmse':>9s}")
for name, (c, n_), loo, par in res:
    npar = {"no fit": 0, "1 param": 1, "2 params": 2}[par]; df = n_ - npar
    print(f"{name:70s} {c:6.2f} {n_:2d} {df:3d} {1 - chi2dist.cdf(c, df):6.3f} {'' if loo is None else f'{loo:9.5f}'}")
print("\nReading guide: 6 points, SE_tot 0.0005-0.0007 each; a model is 'rejected' only at chi2 >> df. The pure-noise scale of the realised")
print("deltas is rms(real) = %.5f vs rms(SE_tot) = %.5f." % (np.sqrt(np.mean(r ** 2)), np.sqrt(np.mean(se ** 2))))
print("corr over the 6 points: (recool share, pubM d) = %.3f ; (holdout d, pubM d) = %.3f ; (realised, pubM d) = %.3f ; (realised, holdout d) = %.3f"
      % (np.corrcoef(sh, x_pubM)[0, 1], np.corrcoef(hold_d, x_pubM)[0, 1], np.corrcoef(r, x_pubM)[0, 1], np.corrcoef(r, hold_d)[0, 1]))
# Spearman rank agreement of orderings
from scipy.stats import spearmanr, kendalltau
allp = SCORED
vals = np.array([nd[n].mean() for n in allp]); pubs = np.array([PUBLIC[n] for n in allp])
hs = np.array([hold[n].mean() if hold.get(n) is not None else np.nan for n in allp])
mh = ~np.isnan(hs)
print(f"ordering of the 7 scored pools: Kendall tau(val, public) = {kendalltau(vals, pubs)[0]:+.2f}; tau(holdout, public) over {mh.sum()} = {kendalltau(hs[mh], pubs[mh])[0]:+.2f}; "
      f"tau(val, holdout) = {kendalltau(vals[mh], hs[mh])[0]:+.2f}")

# ---------------------------------------------------------------- (3e) forward evidence inside the labelled samples
print("\n" + "=" * 150)
print("(3e) FORWARD-IN-TIME evidence inside the labelled samples: val Tue -> Thu (2 days), holdout weekday -> weekend and by day,")
print("     for ALL pools vs G3 (± SE). A val edge that is a recency artefact should shrink from Tue to Thu; a weekend-specific")
print("     effect should show on the holdout's 9,632 weekend feeds (the only labelled weekend sample).")
print("=" * 150)
day = Sv["day"]; wk = shiftlib.weekday(hd["ts"]) >= 5; hhour = shiftlib.msk_hour(hd["ts"])
publike = wk & ((shiftlib.weekday(hd["ts"]) == 5) | (hhour < 12))
late = hhour >= 14
print(f"{'pool':8s} {'val Tue':>17s} {'val Thu':>17s} {'Thu-Tue':>17s} | {'hold wkday':>17s} {'hold wkend':>17s} {'we-wd':>17s} {'hold publike':>17s} {'hold h>=14':>17s} {'hold h<14':>17s}")
allnd = {n: perfeed_val(n) for n in SCORED + UNSCORED}; allh = {n: perfeed_holdout(n) for n in SCORED + UNSCORED}
for n in SCORED + UNSCORED:
    if n == "G3":
        continue
    d = allnd[n] - g3
    t, th = mse(d[day == "Tue"]), mse(d[day == "Thu"]); dd = (th[0] - t[0], np.hypot(t[1], th[1]))
    if allh.get(n) is not None:
        h = allh[n] - g3h; a, b = mse(h[~wk]), mse(h[wk]); c = (b[0] - a[0], np.hypot(a[1], b[1]))
        hp = mse(h[publike]); hl = mse(h[late]); he = mse(h[~late])
        hs_ = f"{fmt(*a):>17s} {fmt(*b):>17s} {fmt(*c):>17s} {fmt(*hp):>17s} {fmt(*hl):>17s} {fmt(*he):>17s}"
    else:
        hs_ = f"{'n/a':>17s} {'n/a':>17s} {'n/a':>17s} {'n/a':>17s} {'n/a':>17s} {'n/a':>17s}"
    print(f"{n:8s} {fmt(*t):>17s} {fmt(*th):>17s} {fmt(*dd):>17s} | {hs_}")
