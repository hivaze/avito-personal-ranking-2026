"""(A, follow-up) Where does the model's probability shape miss the realised clicks, and is part of the clicks pure noise?

1. Clicks by model rank bucket: actual count vs the count expected from softmax(s / T) (all positives; for K > 1 the
   expected count is K * p, the first-order inclusion; the K = 1 table is exact).
2. Noise-mixture model, fitted on user half A, read on half B:  p = (1 - eps) * softmax(s / T) + eps / M
   ("a share eps of clicks is uniform over the feed, unrelated to the score"). Also per M bin.
3. Expected NDCG under labels re-drawn from the fitted mixture (same K, model order) vs the actual NDCG.
"""
import sys
import time
from pathlib import Path

from scipy.optimize import minimize

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
rk = model_rank(s, off, M)
T_A = fit_T(s, y, off, M, half == 0)
p = softmax_p(s / T_A, off, M)
k1 = npos == 1

print("1. clicks by model rank (K = 1 feeds: exact expectation; all feeds: K * p)")
edges = [0, 1, 2, 3, 5, 10, 20, 50, 100, 400]
for name, rows in [("K=1", k1[fid]), ("all", np.ones(len(y), bool))]:
    Kf = npos[fid].astype(np.float64)
    exp_w = p * (1.0 if name == "K=1" else Kf)
    print(f"  {name}: {'rank':>9} {'actual':>7} {'expected':>8} {'act/exp':>7} {'z':>6}")
    for a, b in zip(edges[:-1], edges[1:]):
        m = rows & (rk >= a) & (rk < b)
        A_, E_ = y[m].sum(), exp_w[m].sum()
        print(f"       {a:4d}-{b - 1:<4d} {A_:7.0f} {E_:8.1f} {A_ / E_:7.3f} {(A_ - E_) / np.sqrt(E_):6.2f}")


def mix_nll(params, mask_feed, Mlocal=None):
    lt, le = params
    T = np.exp(lt); eps = 1 / (1 + np.exp(-le))
    q = softmax_p(s / T, off, M)
    pm = (1 - eps) * q + eps / M[fid]
    rows = mask_feed[fid] & (y > 0)
    return -np.log(pm[rows]).sum()


def fit_mix(mask_feed):
    r = minimize(mix_nll, x0=[0.0, -3.0], args=(mask_feed,), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-4, maxiter=400))
    return float(np.exp(r.x[0])), float(1 / (1 + np.exp(-r.x[1]))), r


print("\n2. noise mixture p = (1-eps) softmax(s/T) + eps/M")
A, B = half == 0, half == 1
T1A, T1B = T_A, fit_T(s, y, off, M, B)
nposB = y[B[fid]].sum()
nll1_B = nll(s / T1A, y, off, M, B)
TmA, eA, _ = fit_mix(A)
TmB, eB, _ = fit_mix(B)
nllm_B = mix_nll([np.log(TmA), np.log(eA / (1 - eA))], B)
print(f"  half A fit: T {TmA:.4f} eps {eA:.4f}   (half B fit: T {TmB:.4f} eps {eB:.4f})")
print(f"  half B NLL per positive: plain softmax (T_A {T1A:.4f}) {nll1_B / nposB:.5f} | mixture from A {nllm_B / nposB:.5f} | "
      f"gain {(nll1_B - nllm_B) / nposB:+.5f} nats/positive, total {nll1_B - nllm_B:+.1f} nats on {nposB:.0f} positives")
# per-feed SE of the NLL gain on B
q1 = softmax_p(s / T1A, off, M)
qm = (1 - eA) * softmax_p(s / TmA, off, M) + eA / M[fid]
g = np.bincount(fid, np.where(y > 0, np.log(qm) - np.log(q1), 0.0), len(M))
print(f"  per-feed NLL gain on B {g[B].mean():+.5f} ± {se(g[B]):.5f} nats/feed (z {g[B].mean() / se(g[B]):.1f})")
Mb = [(5, 30), (31, 100), (101, 200), (201, 300)]
print("  per M bin (fit on A / read on B):")
for a, b in Mb:
    m = (M >= a) & (M <= b)
    Tb, eb, _ = fit_mix(A & m)
    T1 = fit_T(s, y, off, M, A & m)
    qa = softmax_p(s / T1, off, M); qb = (1 - eb) * softmax_p(s / Tb, off, M) + eb / M[fid]
    gg = np.bincount(fid, np.where(y > 0, np.log(qb) - np.log(qa), 0.0), len(M))
    print(f"    M {a:3d}-{b:3d}: plain T {T1:.4f} | mixture T {Tb:.4f} eps {eb:.4f} | NLL gain on B {gg[B & m].mean():+.5f} ± {se(gg[B & m]):.5f} nats/feed")

print("\n3. expected NDCG when labels are drawn from the fitted mixture (fit on A), model order, same K")
for name, T, eps in [("plain softmax", T_A, 0.0), ("mixture", TmA, eA)]:
    pm = (1 - eps) * softmax_p(s / T, off, M) + eps / M[fid]
    x = np.log(pm)
    D = redraw_ndcg(x, rk, off, M, npos, 400, 4242, DISC).astype(np.float64)
    mu = D.mean(0); mx, vx = exact_k1(pm, rk, off, M); mu[k1] = mx[k1]
    v = D.var(0, ddof=1); v[k1] = vx[k1]
    print(f"  {name:14s}: E[NDCG] {mu.mean():.5f} (draw sd {D.mean(1).std(ddof=1):.5f}) | actual - expected {(nd - mu).mean():+.5f} ± {se(nd - mu):.5f} "
          f"| half B {(nd - mu)[B].mean():+.5f} ± {se((nd - mu)[B]):.5f} | draw part of observed per-feed var {v.mean() / nd.var():.3f}")
    for a, b in Mb:
        m = (M >= a) & (M <= b)
        print(f"      M {a:3d}-{b:3d}: actual - expected {(nd - mu)[m].mean():+.4f} ± {se((nd - mu)[m]):.4f}")
    if name == "mixture":
        # ceiling if the non-noise part were sharper and right
        for t in (0.8, 0.6):
            pt = (1 - eps) * softmax_p(s / (T * t), off, M) + eps / M[fid]
            Dt = redraw_ndcg(np.log(pt), rk, off, M, npos, 200, 99 + int(10 * t), DISC).astype(np.float64)
            mt = Dt.mean(0); mxt, _ = exact_k1(pt, rk, off, M); mt[k1] = mxt[k1]
            print(f"      score part sharpened t={t}: E[NDCG] {mt.mean():.5f}")
        # ceiling if the noise part were removed from the labels (eps -> 0, same softmax part)
        pt = softmax_p(s / T, off, M)
        Dt = redraw_ndcg(np.log(pt), rk, off, M, npos, 200, 7, DISC).astype(np.float64)
        mt = Dt.mean(0); mxt, _ = exact_k1(pt, rk, off, M); mt[k1] = mxt[k1]
        print(f"      noise part removed (eps = 0, T {T:.4f}): E[NDCG] {mt.mean():.5f}  (gain over the mixture {mt.mean() - mu.mean():+.5f})")
print(f"done {time.time() - t0:.0f}s")
