"""Step 4: (F) SE of a cross-team public/private difference as a function of score correlation (proxies = our own diverse models).
(H) pool-size curve of the wide KD students (train-only branch, val + holdout) -> forecast for W16 / W32.
(B3) transfer blocks split by lineage (val-touched components vs holdout-selected components).  Writes out/s4_crossteam_pool.log."""
from lib import *
import io, itertools
from scipy import stats
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


z = np.load(OUT / "nd_val.npz"); names = list(z["names"]); ND = dict(zip(names, z["nd"]))
zh = np.load(OUT / "nd_hold.npz"); HN = dict(zip(list(zh["names"]), zh["nd"]))
fv, Mv, ov = meta("val"); pub, tday, Mt = test_parts()
wpub = mix_weights(Mv, Mt[pub]); wpriv = mix_weights(Mv, Mt[~pub])
f = lambda d: f"{d.mean():+.5f}+-{se(d):.5f}"

# ---------------------------------------------------------------- F
say("== F. CROSS-TEAM NOISE: per-feed NDCG delta sd vs score correlation (val; proxies for 'another strong team' = our own diverse models)")
ref = zf("val", final_score("W8", "val")); nref = ND["W8"]
prox = {}
def add(name, score):
    zs = zf("val", score); nd = nd_val(zs); prox[name] = (np.corrcoef(zs, ref)[0, 1], nd)
for n in ("C3", "G3", "Cd", "v10", "v5"):
    add(n, final_score(n, "val"))
for lab, path in [("single hard-label member (w5 base s951, TO)", W5 / "nn_w5_e10_base_s951_trainonly_val.npy"),
                  ("single KD wide student (TO)", W5 / "nn_w5_e10_kd05w192_s951_trainonly_val.npy"),
                  ("CatBoost 300k (cb_tr300k_none)", P / "cb_tr300k_none_val.npy"), ("LightGBM 300k (lgb_tr300k_none)", P / "lgb_tr300k_none_val.npy"),
                  ("pretrained encoder zero-shot (pre2_final)", P / "pre2_final_val.npy")]:
    if path.exists():
        add(lab, np.load(path))
cb = zf("val", np.load(P / "cb_tr300k_none_val.npy"))
for w in (0.3, 0.5):
    add(f"blend {1 - w:.1f} z(W8) + {w:.1f} z(CatBoost)", (1 - w) * ref + w * cb)
sdl = nref.std()
say(f" per-feed NDCG sd of W8 on val {sdl:.4f}; feeds with NDCG==1: {(nref == 1).mean():.3f}")
say(f" {'proxy':48s} {'score corr':>10s} {'val NDCG':>9s} {'delta vs W8':>12s} {'sd(d)':>7s} {'corr(nd)':>8s} {'SE pub':>8s} {'SE priv':>8s}")
xs, ys = [], []
for n, (c, nd) in prox.items():
    d = nd - nref; mw = np.average(d, weights=wpub); sdw = np.sqrt(np.average((d - mw) ** 2, weights=wpub))
    say(f" {n:48s} {c:10.4f} {nd.mean():9.5f} {d.mean():+12.5f} {sdw:7.4f} {np.corrcoef(nd, nref)[0, 1]:8.3f} {sdw / np.sqrt(NPUB):8.5f} {sdw / np.sqrt(NPRIV):8.5f}")
    xs.append(np.sqrt(1 - c)); ys.append(sdw)
xs, ys = np.array(xs), np.array(ys); b = (xs @ ys) / (xs @ xs)
say(f" fit sd(d) = {b:.3f} * sqrt(1 - score corr)  (max rel. error {np.abs(ys - b * xs).max() / ys.mean():.2f})")
gap1 = 0.398256 - PUBLIC["W8"]; gap2 = 0.395115 - PUBLIC["W8"]; gap3 = PUBLIC["W8"] - 0.393778
say(f" gaps on public: #1 - W8 = {gap1:+.6f}; #2 - W8 = {gap2:+.6f}; W8 - (old #3) = {gap3:+.6f}")
say(f" {'score corr':>10s} {'sd(d)':>7s} {'SE public':>10s} {'SE private':>10s} {'z(gap #1)':>9s} {'P(#1 true lead<=0)':>18s} {'P(lead<0.002)':>13s} {'P(we beat #2 on private)':>24s} {'P(#3 passes us on private)':>26s}")
for c in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
    s = b * np.sqrt(1 - c); sp, sq = s / np.sqrt(NPUB), s / np.sqrt(NPRIV)
    # flat prior on the true lead; private realised gap ~ N(public gap, sp^2 + sq^2)
    tot = np.sqrt(sp ** 2 + sq ** 2)
    say(f" {c:10.2f} {s:7.4f} {sp:10.5f} {sq:10.5f} {gap1 / sp:9.2f} {stats.norm.cdf(0, gap1, sp):18.4f} {stats.norm.cdf(0.002, gap1, sp):13.3f} {stats.norm.cdf(0, gap2, tot):24.3f} {stats.norm.cdf(0, gap3, tot):26.3f}")
# winner's curse on the public max: simulate a leaderboard top
say(" winner's curse of the public #1 (simulation): K contenders with true scores ~ N(0, tau), public noise sd s_team (per-team idiosyncratic part = SE_public/sqrt(2));")
say("   reported: E[public lead of #1 over the team observed 0.00377 below it  -  true lead], conditional on an observed lead in [0.0033, 0.0043]")
rng = np.random.default_rng(7)
for c in (0.85, 0.90, 0.95):
    s_team = b * np.sqrt(1 - c) / np.sqrt(NPUB) / np.sqrt(2)
    for K, tau in ((5, 0.001), (5, 0.002), (20, 0.001), (20, 0.002)):
        T = rng.normal(0, tau, (400000, K)); O = T + rng.normal(0, s_team, T.shape)
        i1 = O.argmax(1); r = np.arange(len(T))
        # 'us' = a random other contender
        j = (i1 + rng.integers(1, K, len(T))) % K
        lead_obs = O[r, i1] - O[r, j]; lead_true = T[r, i1] - T[r, j]
        m = (lead_obs > 0.0033) & (lead_obs < 0.0043)
        if m.sum() > 500:
            say(f"   corr {c:.2f} K {K:2d} tau {tau:.3f}: n {m.sum():6d}  mean true lead {lead_true[m].mean():+.5f} (observed 0.00377; shrink {lead_true[m].mean() / 0.00377:.2f})  P(true lead <= 0.001) {(lead_true[m] <= 0.001).mean():.3f}  P(true lead <= 0) {(lead_true[m] <= 0).mean():.4f}")

# ---------------------------------------------------------------- B3 lineage blocks
say("\n== B3. TRANSFER BY LINEAGE (EIV GLS on uniform val)")
def eiv(files, lab, pred="val"):
    A = np.stack([ND[n] for n in files]); P_ = np.array([PUBLIC[n] for n in files]); n = len(files)
    wn = wpub / wpub.sum(); mA = A @ wn; R = (A - mA[:, None]) * np.sqrt(wn)[None, :]; SP = R @ R.T / NPUB
    if pred == "val":
        X_ = A.mean(1); SX = np.cov(A) / NVAL
    else:
        B = np.stack([HN[n] for n in files]); X_ = B.mean(1); SX = np.cov(B) / B.shape[1]
    D = np.eye(n)[1:] - np.eye(n)[0]; grid = np.linspace(-3, 4, 1401); nll = []
    for k in grid:
        r = D @ (P_ - k * X_); C = D @ (SP + k * k * SX) @ D.T; c2 = r @ np.linalg.solve(C, r); nll.append((0.5 * (c2 + np.linalg.slogdet(C)[1]), c2))
    nll = np.array(nll); i = nll[:, 0].argmin(); ins = grid[nll[:, 0] <= nll[i, 0] + 0.5]; in2 = grid[nll[:, 0] <= nll[i, 0] + 1.92]
    i0, i1 = np.abs(grid).argmin(), np.abs(grid - 1).argmin()
    say(f"  {lab:74s} [{pred:7s}] k {grid[i]:+.2f} [1s {ins.min():+.2f},{ins.max():+.2f}] [95% {in2.min():+.2f},{in2.max():+.2f}]  chi2 k=0 {nll[i0, 1]:5.1f} k=1 {nll[i1, 1]:5.1f} /{n - 1}  LR p(k=1) {1 - stats.chi2.cdf(2 * (nll[i1, 0] - nll[i, 0]), 1):.3f} p(k=0) {1 - stats.chi2.cdf(2 * (nll[i0, 0] - nll[i, 0]), 1):.3f}")
B_old = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
B_new = ["G3", "C2", "Q1a", "Q2a", "C1", "C3", "W8", "S16"]
B_new2 = ["G3", "Q1a", "Q2a", "C1", "C3", "W8", "S16"]
eiv(B_old, "11-12 era (m02 re-cools with val-picked checkpoints, val-chosen pools): 7 files")
eiv([n for n in B_old if n != "S2"], "  same without S2", "holdout")
eiv(B_new, "11-13/14 era (G3 + families whose checkpoints were picked on the holdout): 8 files")
eiv(B_new, "  same", "holdout")
eiv(B_new2, "  same without C2 (FT seed bag of G3)")
eiv(["G3", "Q1a", "Q2a", "C1", "C2", "C3"], "  pop-family only (G3, C2, Q1a, Q2a, C1, C3)")
eiv(["G3", "Q1a", "Q2a", "C1", "C2", "C3"], "  pop-family only", "holdout")
eiv(["C3", "W8", "S16"], "  KD only (C3, W8, S16)")
eiv(["C3", "W8", "S16"], "  KD only", "holdout")

# ---------------------------------------------------------------- H pool-size curve
say("\n== H. POOL-SIZE CURVE of the 8 wide KD students (train-only branch): mean over subsets of size k")
h = hold(); yv = val_y()
W8m = KD["kd05w192"] + KD["kd05t25w192"]; S16m = W8m + KD["kd05"] + KD["kd05t25"]
ZV = {m: zf("val", np.load((P if (P / f"nn_{m}_trainonly_val.npy").exists() else W5) / f"nn_{m}_trainonly_val.npy")) for m in S16m}
ZH = {m: feed_rank_z(h["off"], h["M"], np.load(hold_path(m)).astype(np.float32))[:, 1].astype(np.float64) for m in S16m}
ZF = {m: zf("val", np.load(P / f"nn_{m}_n2cpv3s3_val_oof.npy")) for m in S16m if (P / f"nn_{m}_n2cpv3s3_val_oof.npy").exists()}
say(f" FT OOF files found for {len(ZF)}/{len(S16m)} students")
evh = lambda s: ndcg_per_feed(s, h["y"], h["off"], h["M"])
rng = np.random.default_rng(3)
def curve(members, ks, nsub=12):
    out = {}
    for k in ks:
        combs = list(itertools.combinations(members, k))
        if len(combs) > nsub:
            combs = [combs[i] for i in rng.choice(len(combs), nsub, replace=False)]
        v = [nd_val(np.mean([ZV[m] for m in c], 0)).mean() for c in combs]; hh = [evh(np.mean([ZH[m] for m in c], 0)).mean() for c in combs]
        fin = [nd_val(0.5 * zf("val", np.mean([ZF[m] for m in c], 0)) + 0.5 * zf("val", np.mean([ZV[m] for m in c], 0))).mean() for c in combs] if all(m in ZF for m in members) else [np.nan]
        out[k] = (np.mean(v), np.std(v), np.mean(hh), np.std(hh), np.mean(fin), len(combs))
        say(f"   k {k:2d} ({len(combs):2d} subsets): TO val {np.mean(v):.5f} (sd {np.std(v):.5f})  holdout {np.mean(hh):.5f} (sd {np.std(hh):.5f})  final(0.5 FT + 0.5 TO) val {np.mean(fin):.5f}")
    return out
say(" wide students (8):")
cw = curve(W8m, (1, 2, 3, 4, 6, 8))
ks = np.array(sorted(cw));
for col, lab in ((0, "TO val"), (2, "holdout"), (4, "final val")):
    yv_ = np.array([cw[k][col] for k in ks])
    if np.isnan(yv_).any(): continue
    X = np.c_[np.ones(len(ks)), 1 / ks]; bb, *_ = np.linalg.lstsq(X, yv_, rcond=None); r = yv_ - X @ bb
    say(f"   fit {lab} = a - b/k: a {bb[0]:.5f} b {-bb[1]:.5f} (max resid {np.abs(r).max():.5f}) -> k=8 {bb[0] + bb[1] / 8:.5f}, k=16 {bb[0] + bb[1] / 16:.5f} ({(bb[1] / 16 - bb[1] / 8):+.5f} vs 8), k=32 {bb[0] + bb[1] / 32:.5f} ({(bb[1] / 32 - bb[1] / 8):+.5f}), k=inf {bb[0]:.5f} ({-bb[1] / 8:+.5f})")
    # fit on k<=4 only, predict 8 (honest check of the law)
    m = ks <= 4; b2, *_ = np.linalg.lstsq(X[m], yv_[m], rcond=None); say(f"      law check: fitted on k<=4, predicted k=8 {b2[0] + b2[1] / 8:.5f} vs actual {yv_[ks == 8][0]:.5f} (miss {yv_[ks == 8][0] - (b2[0] + b2[1] / 8):+.5f})")
say(" all 16 students:")
cs = curve(S16m, (4, 8, 12, 16), nsub=10)
C = np.corrcoef(np.stack([ZV[m] for m in W8m])); iu = np.triu_indices(8, 1)
C2 = np.corrcoef(np.stack([ZV[m] for m in KD['kd05w192']])); C3_ = np.corrcoef(np.stack([ZV[m] for m in KD['kd05t25w192']]))
say(f" wide-student pairwise score corr on val: all 28 pairs {C[iu].mean():.4f}; within kd05w192 {C2[np.triu_indices(4, 1)].mean():.4f}; within kd05t25w192 {C3_[np.triu_indices(4, 1)].mean():.4f}; across arms {C[:4, 4:].mean():.4f}")
(OUT / "s4_crossteam_pool.log").write_text(buf.getvalue())
