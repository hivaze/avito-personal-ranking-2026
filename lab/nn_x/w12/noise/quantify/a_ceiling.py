"""(A) label-noise ceiling and (B) the role of the number of clicks K, on validation, CPU only.

Score = best file FD16P20@m0.4_0.2 (0.5*ft + 0.5*to). Temperature T fitted on user half A (listwise CE), applied to all.
Labels re-drawn R times from softmax(s / (T * t)) with each feed's actual K (Plackett-Luce, Gumbel top-K), ranking fixed
to the model's order. t = 1 is the calibrated model; t < 1 is a world in which the model is sharper AND right.
Out: out/a_ceiling.txt (printed), out/a_draws_t1.npz (per-feed NDCG of each draw at t = 1).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

t0 = time.time()
F, M, off = layout()
y = labels()
fid = np.repeat(np.arange(len(M)), M)
npos = np.bincount(fid, y, len(M)).astype(np.int64)
s = pool_score(BEST)
nd = ndcg_per_feed(s, y, off, M)
half = halves(F)
print(f"feeds {len(M)} rows {len(s)} positives {int(y.sum())}  K=0 feeds {(npos == 0).sum()}  "
      f"actual NDCG {nd.mean():.5f} ± {se(nd):.5f}  (record: 0.40293)")
# ties inside feeds (redraw ranks ignore tie order; pessimistic ties only matter if they exist)
srt = np.lexsort((-s, fid))
ties = ((np.diff(s[srt]) == 0) & (np.diff(fid[srt]) == 0)).sum()
print(f"tied adjacent score pairs inside feeds: {ties}")
T_A = fit_T(s, y, off, M, half == 0)
T_B = fit_T(s, y, off, M, half == 1)
print(f"T fitted on half A {T_A:.4f}, on half B {T_B:.4f}; applied: T_A")
rk = model_rank(s, off, M)
k1 = npos == 1

# ---------------------------------------------------------------- (A) ceiling at several sharpness levels
print("\n(A) expected NDCG if labels were drawn from the model's (sharpened) probabilities, same K per feed, model order")
print(f"{'t':>5} {'T_eff':>6} {'R':>4} {'E[NDCG]':>8} {'MC se':>7} {'draw sd':>8} {'half A':>8} {'half B':>8} "
      f"{'K=1 exact':>9} {'NLL/pos act':>11} {'entropy':>8} {'E[v_f]':>8} {'var mu_f':>8} {'draw share':>10}")
res = {}
grid = [1.25, 1.0, 0.98, 0.96, 0.95, 0.94, 0.92, 0.90, 0.8, 0.7, 0.6, 0.5]
for t in grid:
    R = 400 if t in (1.0, 0.8, 0.6) else 150
    x = s / (T_A * t)
    p = softmax_p(x, off, M)
    D = redraw_ndcg(x, rk, off, M, npos, R, 1234 + int(t * 1000), DISC)
    mu = D.mean(0).astype(np.float64)
    v = D.astype(np.float64).var(0, ddof=1)
    mu_ex, v_ex = exact_k1(p, rk, off, M)
    mu[k1] = mu_ex[k1]; v[k1] = v_ex[k1]          # exact where K = 1
    per_draw = D.mean(1)
    ent = entropy_per_feed(p, off, M).mean()
    nllpos = nll(x, y, off, M) / y.sum()
    share = v.mean() / (v.mean() + mu.var())
    mc_se = np.sqrt((v[~k1] / R).sum()) / len(M)     # Monte Carlo error of E[NDCG] (K>1 feeds only are simulated)
    res[t] = dict(mu=mu, v=v, per_draw=per_draw)
    print(f"{t:5.2f} {T_A * t:6.3f} {R:4d} {mu.mean():8.5f} {mc_se:7.5f} {per_draw.std(ddof=1):8.5f} "
          f"{mu[half == 0].mean():8.5f} {mu[half == 1].mean():8.5f} {mu_ex[k1].mean():9.5f} {nllpos:11.4f} {ent:8.3f} "
          f"{v.mean():8.5f} {mu.var():8.5f} {share:10.3f}", flush=True)
    if t == 1.0:
        np.savez(OUT / "a_draws_t1.npz", D=D, mu=mu, v=v, nd=nd, npos=npos, M=M, half=half, T_A=T_A)
print("columns: E[NDCG] = mean over feeds of the per-feed expected NDCG (exact for K=1, R draws for K>1); "
      "draw sd = sd over draws of the val-size mean; E[v_f] = mean per-feed draw variance; var mu_f = variance across "
      "feeds of the expected NDCG; draw share = E[v_f] / (E[v_f] + var mu_f) = share of the model-implied per-feed "
      "variance that is the label draw alone")

mu1, v1 = res[1.0]["mu"], res[1.0]["v"]
print(f"\nactual minus calibrated expectation: {nd.mean() - mu1.mean():+.5f} ± {se(nd - mu1):.5f} "
      f"(half A {(nd - mu1)[half == 0].mean():+.5f} ± {se((nd - mu1)[half == 0]):.5f}, "
      f"half B {(nd - mu1)[half == 1].mean():+.5f} ± {se((nd - mu1)[half == 1]):.5f})")
pd1 = res[1.0]["per_draw"]
print(f"actual NDCG percentile among the {len(pd1)} draws: {(pd1 < nd.mean()).mean():.3f}")
tot = nd.var()
print(f"observed per-feed variance of NDCG {tot:.5f}; model-implied total {v1.mean() + mu1.var():.5f}; "
      f"label-draw part E[v_f] {v1.mean():.5f} = {v1.mean() / tot:.3f} of the observed; "
      f"R^2 of actual on expected {np.corrcoef(nd, mu1)[0, 1] ** 2:.3f} (var mu / observed {mu1.var() / tot:.3f})")
# temperature at which the ceiling rises by the leader's public gap (0.0030464 public / 0.963 val per public unit)
gap_val = (0.3982556 - 0.3952092) / 0.963
ts = sorted(res)
em = np.array([res[t]["mu"].mean() for t in ts])
base = res[1.0]["mu"].mean()
tt = np.interp(base + gap_val, em[::-1], np.array(ts)[::-1])
print(f"sharpness needed for +{gap_val:.4f} val (= leader's +0.00305 public via the level-line slope 0.963): t = {tt:.3f}, "
      f"i.e. logits {1 / tt:.3f}x as large and still calibrated")

# ---------------------------------------------------------------- per M bin, at t = 1
print("\nper M bin (t = 1): n | actual | expected | actual-exp ± se | per-feed sd: total observed, draw-only | draw share of observed var")
Mb = [(5, 30), (31, 100), (101, 200), (201, 300)]
for a, b in Mb:
    m = (M >= a) & (M <= b)
    print(f"  M {a:3d}-{b:3d}: {m.sum():5d} | {nd[m].mean():.4f} | {mu1[m].mean():.4f} | {(nd - mu1)[m].mean():+.4f} ± {se((nd - mu1)[m]):.4f} | "
          f"{nd[m].std():.4f} {np.sqrt(v1[m].mean()):.4f} | {v1[m].mean() / nd[m].var():.3f}")


# ---------------------------------------------------------------- per-M-bin temperature (fit on half A, read on half B)
print("\n(A2) one temperature per M bin, fitted on half A (the global T is over-confident on short feeds, under-confident on long)")
Tm = np.empty(len(M))
for a, b in Mb:
    m = (M >= a) & (M <= b)
    Tb = fit_T(s, y, off, M, (half == 0) & m)
    Tm[m] = Tb
    print(f"  M {a:3d}-{b:3d}: T_A {Tb:.4f}  (T_B {fit_T(s, y, off, M, (half == 1) & m):.4f})")
xm = s / Tm[fid]
pm = softmax_p(xm, off, M)
Dm = redraw_ndcg(xm, rk, off, M, npos, 400, 999, DISC).astype(np.float64)
mum = Dm.mean(0); vm = Dm.var(0, ddof=1)
mex, vex = exact_k1(pm, rk, off, M); mum[k1] = mex[k1]; vm[k1] = vex[k1]
print(f"  E[NDCG] {mum.mean():.5f} (half A {mum[half == 0].mean():.5f}, half B {mum[half == 1].mean():.5f}); draw sd of the val mean "
      f"{Dm.mean(1).std(ddof=1):.5f}; actual - expected {nd.mean() - mum.mean():+.5f} ± {se(nd - mum):.5f} "
      f"(half B, out of sample: {(nd - mum)[half == 1].mean():+.5f} ± {se((nd - mum)[half == 1]):.5f})")
print(f"  label-draw part E[v_f] {vm.mean():.5f} = {vm.mean() / nd.var():.3f} of the observed per-feed variance; R^2 {np.corrcoef(nd, mum)[0, 1] ** 2:.3f}")
for a, b in Mb:
    m = (M >= a) & (M <= b)
    print(f"  M {a:3d}-{b:3d}: actual - expected {(nd - mum)[m].mean():+.4f} ± {se((nd - mum)[m]):.4f}  (half B {(nd - mum)[m & (half == 1)].mean():+.4f} ± {se((nd - mum)[m & (half == 1)]):.4f})")
for t in (0.8, 0.6):
    xs = xm / t
    Ds = redraw_ndcg(xs, rk, off, M, npos, 200, 31 + int(t * 10), DISC).astype(np.float64)
    mus = Ds.mean(0); mx, _ = exact_k1(softmax_p(xs, off, M), rk, off, M); mus[k1] = mx[k1]
    print(f"  sharpened t={t}: E[NDCG] {mus.mean():.5f}  draw sd of the val mean {Ds.mean(1).std(ddof=1):.5f}")
# calibration identity on K=1 feeds: for a calibrated model E[-log p_click] = E[entropy]
fk = np.where(k1)[0]
ent = entropy_per_feed(softmax_p(s / T_A, off, M), off, M)
lp = -(s / T_A - feed_lse(s / T_A, off, M)[fid]) * y
nllf = np.bincount(fid, lp, len(M))
print(f"\nK=1 feeds: mean -log p(click) {nllf[k1].mean():.4f} ± {se(nllf[k1]):.4f} vs mean entropy {ent[k1].mean():.4f}; mean log M {np.log(M[k1]).mean():.4f}"
      f"  -> information the model extracts: {np.log(M[k1]).mean() - nllf[k1].mean():.4f} nats per click")

# ---------------------------------------------------------------- (B) by K
print("\n(B) by number of positives K (t = 1)")
D1 = np.load(OUT / "a_draws_t1.npz")["D"].astype(np.float64)
loss_act = 1 - nd
loss_rd = 1 - D1           # [R, F]
Kb = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 9), (10, 400)]
print(f"{'K':>6} {'feeds':>6} {'share':>6} {'mean M':>6} {'actual':>14} {'expected':>8} {'act-exp':>16} "
      f"{'loss sh act':>11} {'loss sh rd':>10} {'feed sd act':>11} {'feed sd draw':>12} {'grp-mean draw sd':>16}")
for a, b in Kb:
    m = (npos >= a) & (npos <= b)
    ls_a = loss_act[m].sum() / loss_act.sum()
    ls_r = (loss_rd[:, m].sum(1) / loss_rd.sum(1)).mean()
    print(f"{a:>3}-{b:<3}{m.sum():6d} {m.mean():6.3f} {M[m].mean():6.1f} {nd[m].mean():.4f} ± {se(nd[m]):.4f} {mu1[m].mean():8.4f} "
          f"{(nd - mu1)[m].mean():+.4f} ± {se((nd - mu1)[m]):.4f} {ls_a:11.3f} {ls_r:10.3f} {nd[m].std():11.4f} "
          f"{np.sqrt(v1[m].mean()):12.4f} {D1[:, m].mean(1).std(ddof=1):16.5f}")

# same feeds, forced K: how the per-feed draw noise and level move with the number of clicks, M held fixed
print("\n(B2) same 14,423 feeds and model order, labels drawn with a forced K (t = 1): level and per-feed draw sd")
x = s / T_A
p = softmax_p(x, off, M)
for K in (1, 2, 3, 5):
    kk = np.minimum(np.full(len(M), K, np.int64), M)
    D = redraw_ndcg(x, rk, off, M, kk, 100, 77 + K, DISC).astype(np.float64)
    muK = D.mean(0); vK = D.var(0, ddof=1)
    if K == 1:
        muK, vK = exact_k1(p, rk, off, M)
    print(f"  K={K}: E[NDCG] {muK.mean():.5f}  per-feed draw sd {np.sqrt(vK.mean()):.4f}  draw share of model-implied var "
          f"{vK.mean() / (vK.mean() + muK.var()):.3f}  sd of a val-size mean from the draw alone {np.sqrt(vK.sum()) / len(M):.5f}")
print(f"done {time.time() - t0:.0f}s")
