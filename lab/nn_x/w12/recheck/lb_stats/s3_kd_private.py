"""Step 3: (D) W8 vs S16 vs C3 on public: is noise enough?  (E) private forecast from M and weekday mix.
(G) FT-half value by staleness day.  2-component (M<50 / M>=50, Tue / Thu) transfer fits.  Writes out/s3_kd_private.log."""
from lib import *
import io
from scipy import stats
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


z = np.load(OUT / "nd_val.npz"); names = list(z["names"]); ND = dict(zip(names, z["nd"]))
zh = np.load(OUT / "nd_hold.npz"); HN = dict(zip(list(zh["names"]), zh["nd"]))
zz = np.load(OUT / "nd_val_halves.npz"); HFT = dict(zip(list(zz["names"]), zz["ft"])); HTO = dict(zip(list(zz["names"]), zz["to"]))
fv, Mv, ov = meta("val"); pub, tday, Mt = test_parts()
wpub = mix_weights(Mv, Mt[pub]); wpriv = mix_weights(Mv, Mt[~pub])
day = val_day(); tue = day <= 20719; thu = ~tue
f = lambda d: f"{d.mean():+.5f}+-{se(d):.5f}"
def wf(d, w):
    m, s = wmean_se(d, w); return f"{m:+.5f}+-{s:.5f}"

# holdout day-of-week
ftr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["timestamp", "M"])
Mtr = ftr["M"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hidx = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
assert (Mtr[hidx] == hold()["M"]).all()
hts = ftr["timestamp"].to_numpy().astype(np.int64)[hidx]; hday = (hts + MSK) // 86400; hdow = (hday + 3) % 7
Mh = hold()["M"]
DN = "Mon Tue Wed Thu Fri Sat Sun".split()

say("== D. W8 / S16 / C3 (and G3): paired per-feed deltas")
PAIRS = [("W8", "C3"), ("S16", "C3"), ("W8", "S16"), ("W8", "G3"), ("C3", "G3"), ("W8C5", "W8"), ("W8C5", "C3"), ("KD4", "C3"), ("K5", "C3")]
SEP = {}
for a, b in PAIRS:
    d = ND[a] - ND[b]
    sd_iid = d.std(ddof=1); mw, _ = wmean_se(d, wpub); sd_pubw = np.sqrt(np.average((d - mw) ** 2, weights=wpub)); mq, _ = wmean_se(d, wpriv); sd_privw = np.sqrt(np.average((d - mq) ** 2, weights=wpriv))
    SEP[(a, b)] = (sd_pubw / np.sqrt(NPUB), sd_privw / np.sqrt(NPRIV), d.mean(), se(d))
    line = (f" {a:5s}-{b:4s} val {f(d)} (z {d.mean() / se(d):+.2f}) | M<50 {f(d[Mv < 50])}  M>=50 {f(d[Mv >= 50])} | Tue {f(d[tue])}  Thu {f(d[thu])} | pubM {wf(d, wpub)}  privM {wf(d, wpriv)}"
            f" | per-feed sd {sd_iid:.4f}, share d==0 {(d == 0).mean():.2f} -> SE of a PUBLIC delta {sd_pubw / np.sqrt(NPUB):.5f}, PRIVATE {sd_privw / np.sqrt(NPRIV):.5f}")
    if a in HN and b in HN:
        dh = HN[a] - HN[b]; line += f" | holdout(TO) {f(dh)} (z {dh.mean() / se(dh):+.2f})"
    if a in PUBLIC and b in PUBLIC:
        dp = PUBLIC[a] - PUBLIC[b]; sp = sd_pubw / np.sqrt(NPUB)
        line += f" | PUBLIC {dp:+.6f} = {dp / sp:+.2f} SE_pub from 0; vs val: z {(dp - d.mean()) / np.sqrt(sp ** 2 + se(d) ** 2):+.2f}"
    say(line)

say("\n label-free check that the public/private noise of a contrast equals the val noise: mean (1 - within-feed corr) of the two files' scores")
TZ = np.load(OUT / "test_z.npy"); tn = names
ft_, Mt_, ot = meta("test")
def within_dis(za, zb, M):
    fid = np.repeat(np.arange(len(M)), M); return 1 - np.bincount(fid, za * zb, len(M)) / M       # both per-feed standardised (pop sd) -> 1 - corr
for a, b in [("W8", "C3"), ("W8", "S16"), ("S16", "C3"), ("C3", "G3")]:
    dt = within_dis(TZ[tn.index(a)].astype(np.float64), TZ[tn.index(b)].astype(np.float64), Mt_)
    va = zf("val", final_score(a, "val")); vb = zf("val", final_score(b, "val")); dv = within_dis(va, vb, Mv)
    say(f"  {a}-{b}: val {dv.mean():.5f} (under public M mix {np.average(dv, weights=wpub):.5f}) | public {dt[pub].mean():.5f} | private {dt[~pub].mean():.5f}  -> ratio public/val {dt[pub].mean() / np.average(dv, weights=wpub):.3f}, private/val {dt[~pub].mean() / np.average(dv, weights=wpriv):.3f}")

say("\n three-file test (C3, W8, S16): is the public ordering W8 > S16 > C3 distinguishable?")
A = np.stack([ND[n] for n in ("C3", "W8", "S16")]); wn = wpub / wpub.sum(); mA = A @ wn; R = (A - mA[:, None]) * np.sqrt(wn)[None, :]; SP = R @ R.T / NPUB
SV = np.cov(A) / NVAL
P_ = np.array([PUBLIC[n] for n in ("C3", "W8", "S16")]); V_ = A.mean(1)
rng = np.random.default_rng(1)
# posterior of the true public-population means given val (flat prior) then predictive public draw -> P(order)
D = np.array([[-1, 1, 0], [-1, 0, 1]], float)
mu = D @ V_; C = D @ (SV + SP) @ D.T
draws = rng.multivariate_normal(mu, C, 400000)
w8, s16 = draws[:, 0], draws[:, 1]
say(f"  predictive from val (1:1): P(public W8 > S16) = {(w8 > s16).mean():.3f}; P(W8-S16 >= observed +0.00041) = {((w8 - s16) >= PUBLIC['W8'] - PUBLIC['S16']).mean():.3f}; "
    f"P(W8-C3 >= +0.00103) = {(w8 >= PUBLIC['W8'] - PUBLIC['C3']).mean():.3f}; P(S16-C3 <= +0.00062) = {(s16 <= PUBLIC['S16'] - PUBLIC['C3']).mean():.3f}")
r = D @ (P_ - V_); say(f"  chi2(2) of public deltas vs val deltas at 1:1 = {r @ np.linalg.solve(C, r):.2f}; vs 'all three equal' = {(D @ P_) @ np.linalg.solve(D @ SP @ D.T, D @ P_):.2f}")

# ---------------------------------------------------------------- 2-component transfer
say("\n== B2. Two-component transfer: public_d = k1 * sh1 * val_d[part1] + k2 * sh2 * val_d[part2]  (EIV GLS, modern 14; shares = public shares)")
def eiv2(files, m1, sh1, lab):
    A = np.stack([ND[n] for n in files]); P_ = np.array([PUBLIC[n] for n in files]); n = len(files)
    wn = wpub / wpub.sum(); mA = A @ wn; R = (A - mA[:, None]) * np.sqrt(wn)[None, :]; SP = R @ R.T / NPUB
    X1 = A[:, m1].mean(1); X2 = A[:, ~m1].mean(1); S1 = np.cov(A[:, m1]) / m1.sum(); S2 = np.cov(A[:, ~m1]) / (~m1).sum()
    D = np.eye(n)[1:] - np.eye(n)[0]; g = np.linspace(-1.5, 3, 181); best = None; grid = np.zeros((len(g), len(g)))
    for i, k1 in enumerate(g):
        for j, k2 in enumerate(g):
            r = D @ (P_ - k1 * sh1 * X1 - k2 * (1 - sh1) * X2); C = D @ (SP + (k1 * sh1) ** 2 * S1 + (k2 * (1 - sh1)) ** 2 * S2) @ D.T
            grid[i, j] = 0.5 * (r @ np.linalg.solve(C, r) + np.linalg.slogdet(C)[1])
    i, j = np.unravel_index(grid.argmin(), grid.shape); mn = grid.min()
    p1 = grid.min(1); p2 = grid.min(0)
    ci = lambda pr: (g[pr <= mn + 0.5].min(), g[pr <= mn + 0.5].max())
    eq = np.array([grid[q, q] for q in range(len(g))]); lr = 2 * (eq.min() - mn)
    say(f"  {lab:34s} k1 {g[i]:+.2f} [1s {ci(p1)[0]:+.2f},{ci(p1)[1]:+.2f}]  k2 {g[j]:+.2f} [1s {ci(p2)[0]:+.2f},{ci(p2)[1]:+.2f}]  | common k {g[eq.argmin()]:+.2f}; LR test k1=k2: {lr:.2f} (p {1 - stats.chi2.cdf(lr, 1):.3f}); LR vs k1=k2=1: {2 * (grid[np.abs(g - 1).argmin(), np.abs(g - 1).argmin()] - mn):.2f} (2 df)")
shp = (Mt[pub] < 50).mean()
for title, files in [("modern 14", MODERN), ("unselected 11", [n for n in MODERN if n not in ("M2", "S2", "TOonly")]), ("G3-family 12", [n for n in MODERN if n not in ("W8", "S16")])]:
    eiv2(files, Mv < 50, shp, f"{title}: part1 = M<50 (share {shp:.2f})")
    eiv2(files, tue, 0.5, f"{title}: part1 = Tue (share 0.5)")

# ---------------------------------------------------------------- E. private forecast
say("\n== E. PRIVATE vs PUBLIC composition and forecast")
say(f" public {pub.sum()} feeds: " + ", ".join(f"{DN[(d + 3) % 7]} {((tday == d) & pub).sum()}" for d in np.unique(tday)) + f" | private {(~pub).sum()}: " + ", ".join(f"{DN[(d + 3) % 7]} {((tday == d) & ~pub).sum()}" for d in np.unique(tday)))
say(f" M<50 share: val {(Mv < 50).mean():.3f} public {(Mt[pub] < 50).mean():.3f} private {(Mt[~pub] < 50).mean():.3f}; mean M {Mv.mean():.1f} / {Mt[pub].mean():.1f} / {Mt[~pub].mean():.1f}")
say(" M-bin shares (val | public | private):")
for bname, lo, hi in [("5-9", 0, 10), ("10-19", 10, 20), ("20-29", 20, 30), ("30-49", 30, 50), ("50-74", 50, 75), ("75-99", 75, 100), ("100-149", 100, 150), ("150-199", 150, 200), ("200-249", 200, 250), ("250+", 250, 10 ** 6)]:
    sh = lambda M: ((M >= lo) & (M < hi)).mean()
    g3 = ND["G3"][(Mv >= lo) & (Mv < hi)]
    say(f"   M {bname:8s} {sh(Mv):.3f} | {sh(Mt[pub]):.3f} | {sh(Mt[~pub]):.3f}   G3 val NDCG in bin {g3.mean():.4f}  W8-C3 {f((ND['W8'] - ND['C3'])[(Mv >= lo) & (Mv < hi)])}")
say("\n level forecast (M-only reweighting, src/calib): file | val | pred public | pred private | priv-pub | actual public | forecast private = actual public + (priv-pub)")
for n in ("G3", "C3", "W8", "S16", "W8C5"):
    v = ND[n]; pp = (wpub * v).mean(); pq = (wpriv * v).mean()
    sdq = np.sqrt(np.average((v - pq) ** 2, weights=wpriv)); sdp = np.sqrt(np.average((v - pp) ** 2, weights=wpub))
    act = PUBLIC.get(n)
    say(f"   {n:5s} {v.mean():.5f} | {pp:.5f} | {pq:.5f} | {pq - pp:+.5f} | {act if act else float('nan'):.6f} | {(act + pq - pp) if act else float('nan'):.5f}"
        f"   [sampling SE of private-minus-public level, common to all files and teams: {np.sqrt(sdq ** 2 / NPRIV + sdp ** 2 / NPUB):.5f}]")
say(" day-level check of the level on val: G3 Tue vs Thu, raw and M-standardised to the val mix")
for n in ("G3", "W8"):
    v = ND[n]; wT = mix_weights(Mv[tue], Mv); wH = mix_weights(Mv[thu], Mv)
    say(f"   {n}: Tue {v[tue].mean():.5f} Thu {v[thu].mean():.5f} diff {v[thu].mean() - v[tue].mean():+.5f} +- {np.sqrt(se(v[tue]) ** 2 + se(v[thu]) ** 2):.5f}; M-standardised diff {(wH * v[thu]).mean() - (wT * v[tue]).mean():+.5f}")
say("\n holdout (30K, M>=50, train period, TRAIN-ONLY pools) by weekday: pool deltas")
wk = hdow < 5
say(f"   holdout feeds: weekday {wk.sum()}, weekend {(~wk).sum()}; by dow " + ", ".join(f"{DN[d]} {(hdow == d).sum()}" for d in range(7)))
for a, b in [("W8", "C3"), ("W8", "G3"), ("S16", "C3"), ("W8", "S16"), ("C3", "G3"), ("KD4", "C3")]:
    d = HN[a] - HN[b]
    say(f"   {a:4s}-{b:3s} all {f(d)} | weekday {f(d[wk])} | weekend {f(d[~wk])} | weekend-weekday {d[~wk].mean() - d[wk].mean():+.5f}+-{np.sqrt(se(d[wk]) ** 2 + se(d[~wk]) ** 2):.5f}"
        f" | Sat {f(d[hdow == 5])} Sun {f(d[hdow == 6])} Mon {f(d[hdow == 0])} Tue {f(d[hdow == 1])}")
    # private-like weights: Sun 4855, Mon 7578, Tue 7600 ; public-like: Sat 6050, Sun 2512
    def dayw(sh):
        w = np.zeros(len(d))
        for dd, s in sh.items():
            m = hdow == dd; w[m] = s / m.sum()
        return w
    wq = dayw({6: 4855, 0: 7578, 1: 7600}); wp = dayw({5: 6050, 6: 2512})
    mq, sq = wmean_se(d[wq > 0], wq[wq > 0]); mp, sp = wmean_se(d[wp > 0], wp[wp > 0])
    say(f"          day-reweighted holdout: public-like (Sat/Sun) {mp:+.5f}+-{sp:.5f}; private-like (Sun/Mon/Tue) {mq:+.5f}+-{sq:.5f}; private-like minus public-like {mq - mp:+.5f}+-{np.sqrt(sq ** 2 + sp ** 2):.5f}")

say("\n combined estimate of the TRUE population delta and predictive private delta (val and public = two independent samples; flat prior; no selection correction)")
def comb(a, b, shrink_val=1.0, extra=None):
    d = ND[a] - ND[b]; vm, vs = d.mean() * shrink_val, se(d); spub, spriv = SEP[(a, b)][0], SEP[(a, b)][1]
    est = [(vm, vs)]
    if a in PUBLIC and b in PUBLIC: est.append((PUBLIC[a] - PUBLIC[b], spub))
    w = np.array([1 / s ** 2 for _, s in est]); m = (w * np.array([e for e, _ in est])).sum() / w.sum(); s = 1 / np.sqrt(w.sum())
    pred_s = np.sqrt(s ** 2 + spriv ** 2)
    return m, s, pred_s, 1 - stats.norm.cdf(0, m, s), 1 - stats.norm.cdf(0, m, pred_s)
for a, b in [("W8", "C3"), ("S16", "C3"), ("W8", "S16"), ("W8", "G3"), ("C3", "G3")]:
    for lab, sh in (("val 1:1", 1.0), ("val shrunk x0.4", 0.4)):
        m, s, ps, p1, p2 = comb(a, b, sh)
        say(f"   {a:4s}-{b:3s} [{lab:15s}] true delta {m:+.5f}+-{s:.5f} (P>0 {p1:.2f}); realised PRIVATE delta {m:+.5f}+-{ps:.5f} -> P({a} > {b} on private) = {p2:.2f}")

# ---------------------------------------------------------------- G. FT half by staleness
say("\n== G. FT half vs TO half on val by day (Tue = 3 days after the last train day, Thu = 5; public 7-8, private 8-10)")
for n in ("G3", "C3", "W8", "S16", "KD4", "Cd", "Pall"):
    ft, to, fin = HFT[n], HTO[n], ND[n]
    d1 = ft - to; d2 = fin - to
    say(f"   {n:4s} FT-TO all {f(d1)} | Tue {f(d1[tue])} Thu {f(d1[thu])} | Thu-Tue {d1[thu].mean() - d1[tue].mean():+.5f}+-{np.sqrt(se(d1[tue]) ** 2 + se(d1[thu]) ** 2):.5f}"
        f" || final-TO all {f(d2)} | Tue {f(d2[tue])} Thu {f(d2[thu])} | M<50 {f(d2[Mv < 50])} M>=50 {f(d2[Mv >= 50])}")
say(f"   public reading of the FT half on G3: G3 - TOonly = {PUBLIC['G3'] - PUBLIC['TOonly']:+.6f} (val {f(ND['G3'] - ND['TOonly'])}); SE_pub {np.sqrt(np.average(((ND['G3'] - ND['TOonly']) - np.average(ND['G3'] - ND['TOonly'], weights=wpub)) ** 2, weights=wpub)) / np.sqrt(NPUB):.5f}")
# w_ft curve on val for W8 with day split
ftz = half_score("W8", "ft", "val"); toz = half_score("W8", "to", "val"); base = ND["W8"]
say("   W8 w_ft curve on val (delta vs w_ft 0.5): w | all | Tue | Thu")
for w in (0.0, 0.25, 0.4, 0.6, 0.75, 1.0):
    d = nd_val(w * ftz + (1 - w) * toz) - base
    say(f"     w_ft {w:.2f}: {f(d)} | {f(d[tue])} | {f(d[thu])}  | sd {d.std():.4f} -> SE_pub {d.std() / np.sqrt(NPUB):.5f}")
d = nd_val(0.75 * ftz + 0.25 * toz) - nd_val(0.25 * ftz + 0.75 * toz)
say(f"   probe pair W8 w_ft 0.75 vs 0.25: val {f(d)}; per-feed sd {d.std():.4f} -> SE_pub {d.std() / np.sqrt(NPUB):.5f}")
d = nd_val(ftz) - nd_val(toz)
say(f"   probe pair W8 FT-only vs TO-only: val {f(d)}; per-feed sd {d.std():.4f} -> SE_pub {d.std() / np.sqrt(NPUB):.5f}")
(OUT / "s3_kd_private.log").write_text(buf.getvalue())
