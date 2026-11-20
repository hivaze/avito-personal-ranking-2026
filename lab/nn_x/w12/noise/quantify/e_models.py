"""(A/E support) Across real models of different strength: NDCG, out-of-sample NLL, and the self-consistent ceiling.

For each stored val score file: T fitted on user half A, NLL per click on half B (K = 1 feeds), actual NDCG, and the
expected NDCG if labels were drawn from that model's own calibrated softmax (exact for K = 1, 100 Plackett-Luce draws
for K > 1). Question: is a better model simply a sharper calibrated one (is progress information, not denoising),
and what NLL change goes with an NDCG change, compared with the uniform-sharpening line of a_ceiling.py.
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
half = halves(F)
A, B = half == 0, half == 1
k1 = npos == 1


def single(name):
    return np.load(PRED / f"{name}_val.npy").astype(np.float64)


models = {
    "lgb_tr300k (trees, 300k feeds)": single("lgb_tr300k_none"),
    "cb_tr300k (trees, 300k feeds)": single("cb_tr300k_none"),
    "member click-label s951 (train-only)": single("nn_w5_e10_base_s951_trainonly"),
    "member click-label s952 (train-only)": single("nn_w5_e10_base_s952_trainonly"),
    "member KD0.5 s951 (train-only)": single("nn_w5_e10_kd05_s951_trainonly"),
    "member KD0.5 s952 (train-only)": single("nn_w5_e10_kd05_s952_trainonly"),
    "member all-data KD w192 s951 (train-only)": single("nn_w5_fd10_kd05w192_s951_trainonly"),
    "pool G3 (click labels)": pool_score("G3"),
    "pool W8P20": pool_score("W8P20kd"),
    "pool FD16P20": pool_score("FD16P20kd"),
    "best FD16P20@m0.4_0.2": pool_score(BEST),
}
rows = []
print(f"{'model':42s} {'NDCG':>8} {'T_A':>6} {'NLL/click B':>11} {'entropy B':>9} {'E[NDCG]':>8} {'act-exp':>16}")
for name, s in models.items():
    nd = ndcg_per_feed(s, y, off, M)
    T = fit_T(s, y, off, M, A)
    x = s / T
    pr = softmax_p(x, off, M)
    lse = feed_lse(x, off, M)
    nllf = np.bincount(fid, -(x - lse[fid]) * y, len(M))
    ent = entropy_per_feed(pr, off, M)
    rk = model_rank(s, off, M)
    D = redraw_ndcg(x, rk, off, M, npos, 100, 5, DISC).astype(np.float64)
    mu = D.mean(0); mx, _ = exact_k1(pr, rk, off, M); mu[k1] = mx[k1]
    mB = B & k1
    rows.append((name, nd.mean(), nllf[mB].mean(), ent[mB].mean(), mu.mean()))
    print(f"{name:42s} {nd.mean():8.5f} {T:6.3f} {nllf[mB].mean():11.4f} {ent[mB].mean():9.4f} {mu.mean():8.5f} "
          f"{(nd - mu).mean():+.5f} ± {se(nd - mu):.5f}", flush=True)
r = np.array([[a, b, c, d] for _, a, b, c, d in rows])
nn = slice(2, None)
sl = np.polyfit(r[nn, 1], r[nn, 0], 1)[0]
print(f"\nacross the neural models: d NDCG / d NLL-per-click (half B, K=1) = {sl:.3f} per nat "
      f"(i.e. {-1 / sl * 0.001:.4f} nats per +0.001 NDCG)")
sl2 = np.polyfit(r[:, 1], r[:, 0], 1)[0]
print(f"all models incl. trees: {sl2:.3f} per nat")
# uniform sharpening line from a_ceiling (calibrated sharper model: its expected NLL = its entropy)
print("reference, uniform sharpening of the best score (a_ceiling.py): t 1.00 -> 0.98 moved E[NDCG] 0.40137 -> 0.40390 "
      "and the mean entropy 3.722 -> 3.710 nats (all feeds) -> about 0.21 NDCG per nat")
print(f"done {time.time() - t0:.0f}s")
