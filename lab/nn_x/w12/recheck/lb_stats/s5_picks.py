"""Step 5: final-two-picks value and probe values.
Posterior of the true (population) per-file levels from val (optionally shrunk toward the pool mean by k) and public, then a
private realisation with the private noise covariance -> E[max of two picks], P(best), value of each second pick.
Also: private rank lottery vs the recorded leaderboard top.  Writes out/s5_picks.log."""
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
rng = np.random.default_rng(11)
FILES = ["G3", "C3", "W8", "S16", "W8C5", "K5", "KD4"]
A = np.stack([ND[n] for n in FILES]); n = len(FILES)
def popcov(w):
    w = w / w.sum(); m = A @ w; R = (A - m[:, None]) * np.sqrt(w)[None, :]; return R @ R.T
SV = np.cov(A) / NVAL; SP = popcov(wpub) / NPUB; SQ = popcov(wpriv) / NPRIV
ref = FILES.index("W8"); D = np.delete(np.eye(n) - np.eye(n)[ref][None, :].repeat(n, 0) * 0, ref, 0)
D = np.delete(np.eye(n), ref, 0) - np.eye(n)[ref]           # contrasts vs W8, (n-1) x n
others = [f for f in FILES if f != "W8"]
scored = np.array([f in PUBLIC for f in others])
def posterior(k, use_hold=False):
    """Gaussian posterior of the true deltas vs W8 (flat prior): val delta shrunk by k with covariance SV, public delta (scored files) with SP."""
    v = k * (D @ A.mean(1)); Cv = D @ SV @ D.T                 # shrinking the mean, keeping the sampling covariance (conservative)
    Pm = np.linalg.inv(Cv); b = Pm @ v
    Dp = D[scored]; pd_ = np.array([PUBLIC[f] - PUBLIC["W8"] for f in np.array(others)[scored]]); Cp = Dp @ SP @ Dp.T
    E = np.eye(n - 1)[scored]; Pp = E.T @ np.linalg.inv(Cp) @ E; Pm = Pm + Pp; b = b + E.T @ np.linalg.inv(Cp) @ pd_
    if use_hold:
        hf = np.array([f in HN for f in others]); Bh = np.stack([HN[f] - HN["W8"] for f in np.array(others)[hf]]); Ch = np.atleast_2d(np.cov(Bh)) / Bh.shape[1] * 4.0
        Eh = np.eye(n - 1)[hf]; Pm = Pm + Eh.T @ np.linalg.inv(Ch) @ Eh; b = b + Eh.T @ np.linalg.inv(Ch) @ Bh.mean(1)
    C = np.linalg.inv(Pm); return C @ b, C
say("== PICKS. posterior of the true delta vs W8 and the private realisation (20,033 feeds); 'best of two picks counts' assumed")
for lab, k, uh in [("val 1:1 + public", 1.0, False), ("val x0.4 + public", 0.4, False), ("val x0.4 + public + holdout(TO, variance x4)", 0.4, True)]:
    m, C = posterior(k, uh)
    say(f"\n -- evidence model: {lab}")
    say("    true delta vs W8: " + "  ".join(f"{f} {m[i]:+.5f}+-{np.sqrt(C[i, i]):.5f}" for i, f in enumerate(others)))
    N = 400000
    mu = rng.multivariate_normal(m, C, N)                         # true deltas vs W8
    eps = rng.multivariate_normal(np.zeros(n), SQ, N)             # private sampling noise, all files
    priv = np.zeros((N, n)); priv[:, ref] = eps[:, ref]
    oi = [FILES.index(f) for f in others]; priv[:, oi] = mu + eps[:, oi]
    rel = priv - priv[:, [ref]]                                  # private score minus W8's private score
    say("    P(file has the best private score of the 7): " + "  ".join(f"{f} {(rel.argmax(1) == i).mean():.2f}" for i, f in enumerate(FILES)))
    say("    realised private delta vs W8: " + "  ".join(f"{f} {rel[:, i].mean():+.5f}+-{rel[:, i].std():.5f} (P>0 {(rel[:, i] > 0).mean():.2f})" for i, f in enumerate(FILES) if f != "W8"))
    say("    pair of picks -> E[max private] relative to W8 alone (i.e. the value of the second pick), and P(pair contains the best of the 7)")
    res = []
    for a, b in itertools.combinations(range(n), 2):
        mx = np.maximum(rel[:, a], rel[:, b]); res.append((mx.mean(), FILES[a], FILES[b], ((rel.argmax(1) == a) | (rel.argmax(1) == b)).mean()))
    for v, a, b, pb in sorted(res, reverse=True)[:10]:
        say(f"      {a:5s}+{b:5s} {v:+.5f}   P(best in pair) {pb:.2f}")
    say("    single picks, E[private] vs W8: " + "  ".join(f"{f} {rel[:, i].mean():+.5f}" for i, f in enumerate(FILES)))

# ---------------------------------------------------------------- probe values
say("\n== PROBES. what one public reading can resolve: SE of the public delta vs W8 (or vs the stated file) and the size needed for 2 SE")
def pair(a_nd, b_nd):
    d = a_nd - b_nd; mw = np.average(d, weights=wpub); return d.mean(), se(d), np.sqrt(np.average((d - mw) ** 2, weights=wpub)) / np.sqrt(NPUB)
for a, b in [("W8C5", "W8"), ("K5", "C3"), ("KD4", "C3"), ("KD4", "W8"), ("S16", "W8"), ("W8", "C3")]:
    dv, sv, sp = pair(ND[a], ND[b]); say(f"   {a:5s} vs {b:4s}: val {dv:+.5f}+-{sv:.5f}; SE_pub {sp:.5f}; 2-SE public threshold {2 * sp:.5f}; expected |z| on public if val transfers 1:1 = {abs(dv) / sp:.2f}, at 0.4 = {0.4 * abs(dv) / sp:.2f}")
# W16 proxy: two disjoint 4+4... use W8 vs random 4-wide subsets to get the SE of a pool-size extension contrast
W8m = KD["kd05w192"] + KD["kd05t25w192"]
ZV = {m: zf("val", np.load(P / f"nn_{m}_trainonly_val.npy")) for m in W8m}; ZF = {m: zf("val", np.load(P / f"nn_{m}_n2cpv3s3_val_oof.npy")) for m in W8m}
fin = lambda ms: nd_val(0.5 * zf("val", np.mean([ZF[m] for m in ms], 0)) + 0.5 * zf("val", np.mean([ZV[m] for m in ms], 0)))
full = fin(W8m); sds = []; dms = []
for c in [W8m[:4], W8m[4:], W8m[::2], W8m[1::2], W8m[:2] + W8m[4:6], W8m[2:4] + W8m[6:]]:
    d = full - fin(c); sds.append(d.std()); dms.append(d.mean())
say(f"   pool-size doubling contrast (8 wide vs 4 of them; proxy for W16 vs W8, which is smaller): val delta mean {np.mean(dms):+.5f}; per-feed sd {np.mean(sds):.4f} -> SE_pub {np.mean(sds) / np.sqrt(NPUB):.5f}, SE_val {np.mean(sds) / np.sqrt(NVAL):.5f}")
say(f"     for W16 vs W8 expect roughly half the delta (+0.00015) and sd/sqrt(2): SE_pub ~{np.mean(sds) / np.sqrt(2) / np.sqrt(NPUB):.5f} -> expected z on public ~{0.00015 / (np.mean(sds) / np.sqrt(2) / np.sqrt(NPUB)):.2f}; P(W16 public > W8 public) ~{stats.norm.cdf(0.00015 / (np.mean(sds) / np.sqrt(2) / np.sqrt(NPUB))):.2f}")

# ---------------------------------------------------------------- rank lottery
say("\n== RANK LOTTERY on private vs the recorded leaderboard top (snapshot 2026-11-13 19:43 + our W8; other teams may have moved since)")
LB = {"#1": 0.398256, "#2": 0.395115, "us(W8)": 0.394486, "old#3": 0.393778, "old#5": 0.393081, "old#6": 0.392833}
b = 0.432
for c in (0.85, 0.90, 0.95):
    sdd = b * np.sqrt(1 - c); s_pub = sdd / np.sqrt(NPUB) / np.sqrt(2); s_priv = sdd / np.sqrt(NPRIV) / np.sqrt(2)
    for gain in (0.0, 0.0005, 0.001, 0.002):
        N = 300000; pubv = np.array(list(LB.values()))
        true = pubv[None, :] + rng.normal(0, s_pub, (N, len(LB)))          # flat-prior posterior of each team's true level (idiosyncratic part)
        pv = true + rng.normal(0, s_priv, true.shape); iu = list(LB).index("us(W8)"); pv[:, iu] += gain
        rank = (pv > pv[:, [iu]]).sum(1) + 1
        say(f"   score corr {c:.2f} (team noise public {s_pub:.5f} / private {s_priv:.5f}), our true gain {gain:+.4f}: P(rank 1) {(rank == 1).mean():.3f}  P(rank<=2) {(rank <= 2).mean():.2f}  P(rank<=3) {(rank <= 3).mean():.2f}  P(rank>=5) {(rank >= 5).mean():.2f}  E[rank among these 6] {rank.mean():.2f}")
(OUT / "s5_picks.log").write_text(buf.getvalue())
