"""Part 2/4 core: how do val readings relate to the exact public scores of the 8 pools with arrays (7 modern + v10)?
Pre-specified predictors (label-free reweightings / halves of val) are compared on (a) level calibration, (b) paired
deltas vs G3 (chi2 under 1:1 transfer, transfer slope, LOO), (c) rank concordance. Writes out/s04_calibration.log."""
from lib import *  # noqa: F401,F403
import io, itertools
import numpy.linalg as la

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


dv, dt = descriptors("val"), descriptors("test")
fv, Mv, offv = feed_meta("val"); ft, Mt, offt = feed_meta("test")
pub_t = dt["public"].to_numpy() == 1
n_pub, n_priv = int(pub_t.sum()), int((~pub_t).sum())
POOLS = MODERN + ["v10"]
PF = {p: per_feed(p) for p in POOLS}
say("val means: " + "  ".join(f"{p} {PF[p].mean():.5f}" for p in POOLS))

# ---------- label-free val -> public reweightings ----------
EDGES_M = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])          # calib.py bins
SEG_M = np.array([10, 25, 50, 100, 200]); SEG_C = np.array([.15, .3, .45])  # pmix bins


def keys(d, M):
    k = {}
    k["M9"] = np.searchsorted(EDGES_M, M, side="right")
    mseg = np.searchsorted(SEG_M, M, side="left"); cseg = np.searchsorted(SEG_C, d["cold_share"].to_numpy(), side="left")
    hb = d["hourb"].to_numpy()
    k["MxCold"] = mseg * 4 + cseg
    k["hour5"] = hb
    k["Mxhour"] = mseg * 5 + hb
    k["MxColdxhour"] = (mseg * 4 + cseg) * 5 + hb
    k["MxUser"] = mseg * 3 + np.where(d["uid0"].to_numpy() == 1, 2, d["train_user"].to_numpy())
    k["MxAge"] = mseg * 3 + np.minimum((d["age_lt7d_share"].to_numpy() // .2).astype(int), 2)
    return k


kv, kt = keys(dv, Mv), keys(dt, Mt)
W = {"raw": np.ones(len(Mv))}
for name in kv:
    for part, m in [("pub", pub_t), ("priv", ~pub_t)]:
        W[f"{name}->{part}"], drop = cell_weights(kv[name], kt[name], m)
        if drop > 0:
            say(f"  [{name}->{part}] test mass without val support: {drop:.4f}")
day = dv["day"].to_numpy(); leak = dv["leak"].to_numpy() == 1
W["Tue"] = (day == 0).astype(float) / (day == 0).mean(); W["Thu"] = (day == 2).astype(float) / (day == 2).mean()
W["noleak"] = (~leak).astype(float) / (~leak).mean()
PRED_LEVEL = ["raw", "Tue", "Thu", "noleak", "M9->pub", "MxCold->pub", "hour5->pub", "Mxhour->pub", "MxColdxhour->pub", "MxUser->pub", "MxAge->pub"]
say("n_eff of val under each weighting: " + "  ".join(f"{k} {W[k].sum() ** 2 / (W[k] ** 2).sum():.0f}" for k in PRED_LEVEL))

say("")
say("=" * 120)
say("A. LEVEL: public = a + c * pred(val) over the 8 arrays; LOO residuals; and slope-1 offset model. (3 old level-only points added where marked)")
say("=" * 120)
pubv = np.array([PUBLIC[p] for p in POOLS])


def fit_report(x, yv, labels, tag):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = la.lstsq(X, yv, rcond=None); r = yv - X @ b
    s2 = (r @ r) / (len(x) - 2); cov = la.inv(X.T @ X) * s2
    loo = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        bi, *_ = la.lstsq(X[m], yv[m], rcond=None); loo.append(yv[i] - X[i] @ bi)
    loo = np.array(loo)
    off = (yv - x).mean(); r1 = yv - x - off
    loo1 = np.array([yv[i] - x[i] - (yv[np.arange(len(x)) != i] - x[np.arange(len(x)) != i]).mean() for i in range(len(x))])
    say(f"  {tag:18s} c {b[1]:+.3f}±{np.sqrt(cov[1, 1]):.3f} a {b[0]:+.4f} resid sd {np.sqrt(s2):.5f} LOO rmse {np.sqrt((loo ** 2).mean()):.5f} | slope-1: offset {off:+.5f} resid sd {r1.std(ddof=1):.5f} LOO rmse {np.sqrt((loo1 ** 2).mean()):.5f}")
    say("  " + " " * 18 + "slope-1 resid: " + " ".join(f"{l}:{v:+.5f}" for l, v in zip(labels, r1)))
    return b, r1


for pred in PRED_LEVEL:
    x = np.array([np.average(PF[p], weights=W[pred]) for p in POOLS])
    fit_report(x, pubv, POOLS, pred)
say("  -- modern 7 only (v10 dropped):")
for pred in ["raw", "M9->pub", "MxCold->pub", "Mxhour->pub", "Thu"]:
    x = np.array([np.average(PF[p], weights=W[pred]) for p in MODERN])
    fit_report(x, np.array([PUBLIC[p] for p in MODERN]), MODERN, pred)
say("  -- raw val, with the 3 old level-only points (v3/v4/v5):")
x = np.r_[[PF[p].mean() for p in POOLS], [LEVEL_ONLY[k][0] for k in LEVEL_ONLY]]
yy = np.r_[pubv, [LEVEL_ONLY[k][1] for k in LEVEL_ONLY]]
fit_report(x, yy, POOLS + list(LEVEL_ONLY), "raw+old")
say("  Note: absolute public level carries a common offset (M-mix level, ~-0.007 for the slope-1 raw model); the within-generation")
say("  dispersion of the slope-1 residuals among the 7 modern near-copies (sd above) is what the delta analysis in B measures exactly.")

say("")
say("=" * 120)
say("B. PAIRED DELTAS vs G3 (6 modern pools). realised r = exact public delta; p = predicted delta from val under each weighting")
say("   SE = sqrt(Var_w(d)/n_val_eff + Var_w(d)/n_pub) (val sampling + public sampling; public is a fixed sample but val->public")
say("   transfer needs both). chi2 under 1:1 transfer (b=1) on 6 df; fitted slope b (pub = b * pred, no intercept) with SE;")
say("   intercept model pub = a + b * pred; LOO RMSE of the slope model vs b=1 vs b=0.")
say("=" * 120)
OTHERS = [p for p in MODERN if p != "G3"]
r_pub = np.array([PUBLIC[p] - PUBLIC["G3"] for p in OTHERS])
say("  realised public deltas vs G3: " + "  ".join(f"{p} {v:+.5f}" for p, v in zip(OTHERS, r_pub)))
D = {p: PF[p] - PF["G3"] for p in OTHERS}
PRED_DELTA = PRED_LEVEL + ["M9->priv", "MxCold->priv", "Mxhour->priv", "MxColdxhour->priv"]
res_table = {}
for pred in PRED_DELTA:
    w = W[pred]
    pv = np.array([np.average(D[p], weights=w) for p in OTHERS])
    neff = w.sum() ** 2 / (w ** 2).sum()
    var = np.array([np.average((D[p] - np.average(D[p], weights=w)) ** 2, weights=w) for p in OTHERS])
    se = np.sqrt(var / neff + var / n_pub)
    se_pub_only = np.sqrt(var / n_pub)
    resid = r_pub - pv
    chi2 = float(((resid / se) ** 2).sum())
    b = (pv * r_pub / se ** 2).sum() / (pv ** 2 / se ** 2).sum(); se_b = 1 / np.sqrt((pv ** 2 / se ** 2).sum())
    X = np.column_stack([np.ones(len(pv)), pv]); Wm = np.diag(1 / se ** 2)
    ab = la.solve(X.T @ Wm @ X, X.T @ Wm @ r_pub); cov_ab = la.inv(X.T @ Wm @ X)
    loo_b, loo_1, loo_0 = [], [], []
    for i in range(len(pv)):
        m = np.arange(len(pv)) != i
        bi = (pv[m] * r_pub[m] / se[m] ** 2).sum() / (pv[m] ** 2 / se[m] ** 2).sum()
        loo_b.append(r_pub[i] - bi * pv[i]); loo_1.append(r_pub[i] - pv[i]); loo_0.append(r_pub[i])
    rm = lambda v: np.sqrt(np.mean(np.square(v)))
    res_table[pred] = dict(pv=pv, se=se, resid=resid, chi2=chi2, b=b, se_b=se_b)
    say(f"  {pred:18s} chi2(b=1) {chi2:5.2f}/6df | b {b:+.2f}±{se_b:.2f} | a {ab[0]:+.5f}±{np.sqrt(cov_ab[0, 0]):.5f} b {ab[1]:+.2f}±{np.sqrt(cov_ab[1, 1]):.2f} | LOO rmse: slope {rm(loo_b):.5f} b=1 {rm(loo_1):.5f} b=0 {rm(loo_0):.5f} | mean SE {se.mean():.5f} (pub-only {se_pub_only.mean():.5f})")
    say("  " + " " * 18 + "pred/resid: " + " ".join(f"{p}:{v:+.5f}/{rv:+.5f}" for p, v, rv in zip(OTHERS, pv, resid)))

say("")
say("B2. Transfer by pool TYPE (pre-specified groups): val-selected (M2, S2) vs reference/hedge pools (Cd, PF7x, Pall, TOonly);")
say("    residual (realised - M9->pub prediction) per group. Also the re-cool-share regression and the d_val regression (critique2 N3), re-fitted.")
rt = res_table["M9->pub"]
for grp, members in [("val-selected", ["M2", "S2"]), ("not selected", ["Cd", "PF7x", "Pall", "TOonly"])]:
    idx = [OTHERS.index(p) for p in members]
    say(f"  {grp:14s}: mean resid {rt['resid'][idx].mean():+.5f} (members: " + ", ".join(f"{p} {rt['resid'][i]:+.5f}±{rt['se'][i]:.5f}" for p, i in zip(members, idx)) + ")")
share = np.array([RECOOL_SHARE[p] - 0.70 for p in OTHERS]); dval = rt["pv"]
for nm, xcol in [("re-cool share - 0.70", share), ("d_val (M9->pub)", dval)]:
    X = np.column_stack([np.ones(6), xcol]); Wm = np.diag(1 / rt["se"] ** 2)
    ab = la.solve(X.T @ Wm @ X, X.T @ Wm @ rt["resid"]); cov = la.inv(X.T @ Wm @ X); rr = rt["resid"] - X @ ab
    say(f"  resid ~ a + g * [{nm}]: a {ab[0]:+.5f}±{np.sqrt(cov[0, 0]):.5f} g {ab[1]:+.5f}±{np.sqrt(cov[1, 1]):.5f} chi2 {((rr / rt['se']) ** 2).sum():.2f}/4df")
say(f"  corr(re-cool share, d_val) over the 6 pools: {np.corrcoef(share, dval)[0, 1]:+.3f} (collinear: the two stories are not separable here)")

say("")
say("B3. TO-half vs FT-half deltas (pools with ft/to arrays: Cd, Pall, PF7x, M2 vs G3): which half's val delta is closer to the realised public delta?")
y = np.load(FLAT / "val/cand_target.npy")
g3to = ndcg_per_feed(np.load(P / "ens_G3_to_val.npy"), y, offv, Mv); g3ft = ndcg_per_feed(np.load(P / "ens_G3_ft_val.npy"), y, offv, Mv)
for p in ["Cd", "Pall", "PF7x", "M2"]:
    pto = ndcg_per_feed(np.load(P / f"ens_{p}_to_val.npy"), y, offv, Mv); pft = ndcg_per_feed(np.load(P / f"ens_{p}_ft_val.npy"), y, offv, Mv)
    w = W["M9->pub"]
    dto, sto = wmean_se(pto - g3to, w); dft, sft = wmean_se(pft - g3ft, w); dmix, smix = wmean_se(D[p], w)
    say(f"  {p:5s} realised {PUBLIC[p] - PUBLIC['G3']:+.5f} | val mix {dmix:+.5f}±{smix:.5f} | TO-half {dto:+.5f}±{sto:.5f} | FT-half {dft:+.5f}±{sft:.5f}")

say("")
say("=" * 120)
say("C. RANK CONCORDANCE of the 7 modern pools: Kendall tau between predicted ordering and public ordering; exact permutation p (7! orderings)")
say("=" * 120)


def kendall(a, b):
    s = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (len(a) * (len(a) - 1) / 2)


pub7 = np.array([PUBLIC[p] for p in MODERN])
taus_null = np.array([kendall(np.array(perm), pub7) for perm in itertools.permutations(range(7))])
for pred in ["raw", "Tue", "Thu", "noleak", "M9->pub", "MxCold->pub", "Mxhour->pub", "MxColdxhour->pub", "MxUser->pub"]:
    x = np.array([np.average(PF[p], weights=W[pred]) for p in MODERN])
    t = kendall(x, pub7)
    say(f"  {pred:18s} tau {t:+.3f}  P(tau >= obs | random order) {np.mean(taus_null >= t - 1e-9):.3f}   val order: {' > '.join(np.array(MODERN)[np.argsort(-x)])}")
say(f"  public order: {' > '.join(np.array(MODERN)[np.argsort(-pub7)])}")
say("  holdout order (train-only pools, CURRENT_STATUS): G3 .32379 > PF7x .32360 > M2 .32361 ~ Pall .32335 > Cd .32319")

say("")
say("=" * 120)
say("D. WHAT 7 POINTS CAN SUPPORT. Power: for a true 1:1 transfer, the realised-vs-predicted chi2 has mean 6; the slope b has SE ~")
say("   the value printed above; a shrinkage of val deltas by half (b = 0.5) is distinguishable from b = 1 only if SE_b < 0.25.")
say("   Winner's-curse arithmetic: a pool picked as the max of K near-copies on val carries an expected val-only excess of about")
say("   SE_val(d) * E[max of K standard normals]:")
sev = np.mean([D[p].std(ddof=1) / np.sqrt(len(Mv)) for p in OTHERS])
for K in [2, 5, 10, 20, 50]:
    z = np.random.default_rng(0).standard_normal((200000, K)).max(1).mean()
    say(f"   K={K:3d}: E[max] {z:.2f} -> expected excess {sev * z:+.5f} (val paired SE of a near-copy delta ~{sev:.5f})")

(OUT / "s04_calibration.log").write_text(buf.getvalue())
print("written", OUT / "s04_calibration.log")
