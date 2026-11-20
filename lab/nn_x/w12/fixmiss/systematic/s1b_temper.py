"""Step 1b: per-feed temperature from label-free feed descriptors (changes no order inside a feed, so NDCG is
unchanged), fitted on half A. Used to tell temperature-shape deviations from rank-changing ones: a slice whose
actual/expected deviation disappears under the per-feed temperature cannot be fixed by re-ranking.

x_i = s_i * exp(w . g_f),  g_f = [1, log M, h_brk10, h_ent, n_hist_cats32, h_nov32r, H/256, guest, h_topsh, uvec_missing]
Out: data/setup_final_tf.npz (same fields as s1) + data/temper_w.npy
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

R = int(os.environ.get("R", 200))
t0 = time.time()
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["score", "label", "fidx", "h_brk10", "h_ent", "n_hist_cats32", "h_nov32r", "H", "guest", "h_topsh",
                "uvec_missing"])
y = D["label"].to_numpy().astype(np.float64)
s = D["score"].to_numpy().astype(np.float64)
npos = np.bincount(fid, y, len(M)).astype(np.int64)
first = off  # first row of each feed
G = np.column_stack([np.ones(len(M)), np.log(M)] + [np.nan_to_num(D[c].to_numpy()[first].astype(np.float64), nan=0.0)
                     for c in ["h_brk10", "h_ent", "n_hist_cats32", "h_nov32r"]] +
                    [D["H"].to_numpy()[first] / 256.0, D["guest"].to_numpy()[first].astype(float),
                     np.nan_to_num(D["h_topsh"].to_numpy()[first].astype(np.float64), nan=0.0),
                     D["uvec_missing"].to_numpy()[first].astype(float)])
mu, sd = G[:, 1:].mean(0), G[:, 1:].std(0)
G[:, 1:] = (G[:, 1:] - mu) / sd


def fit(fm):
    def fg(w):
        t = np.exp(G @ w)
        x = s * t[fid]
        lse = core_lse(x)
        p = np.exp(x - lse[fid])
        f = -(y * (x - lse[fid]))[fm[fid]].sum()
        gi = ((npos[fid] * p - y) * s)            # d f / d x_i * d x_i / d t  (x = s t)
        gf = np.bincount(fid, gi, len(M)) * t
        g = (gf[fm][:, None] * G[fm]).sum(0)
        return f, g
    return minimize(fg, np.zeros(G.shape[1]), jac=True, method="L-BFGS-B").x


def core_lse(x):
    from core import _feed_lse
    return _feed_lse(x, off, M)


wA = fit(half == 0); wB = fit(half == 1)
names = ["const", "logM", "h_brk10", "h_ent", "n_hist_cats32", "h_nov32r", "H", "guest", "h_topsh", "uvec_missing"]
print("per-feed log-temperature weights (fit A | fit B), descriptors standardised:")
for n_, a, b in zip(names, wA, wB):
    print(f"  {n_:14s} {a:+.4f} | {b:+.4f}")
np.save(HERE / "data/temper_w.npy", np.stack([wA, wB]))
t = np.exp(G @ wA)
x = s * t[fid]
for h in (0, 1):
    fm = half == h
    print(f"NLL/pos half {'AB'[h]}: global T {nll(s / 1.0036, y, off, M, fm) / npos[fm].sum():.5f}  per-feed T {nll(x, y, off, M, fm) / npos[fm].sum():.5f}")
p = softmax_p(x, off, M)
rk = model_rank(s, off, M)
L = actual_pos_loss(y, rk, off, M, DISC)
rows, loss = redraw(x, rk, off, M, npos, R, 23, DISC)
cnt = np.bincount(rows.ravel(), minlength=len(s)) / R
E = np.where(npos[fid] == 1, p, cnt)
nd = ndcg_per_feed(s, y, off, M)
nd_rd = 1 - loss.sum(1) / len(M)
print(f"NDCG {nd.mean():.5f}  redraw (per-feed T) {nd_rd.mean():.5f} ± {nd_rd.std():.5f}  ({time.time() - t0:.0f}s)")
np.savez(HERE / "data/setup_final_tf.npz", T=1.0, TB=1.0, p=p, E=E, L=L, rows=rows.astype(np.int32),
         loss=loss.astype(np.float32), rk=rk.astype(np.int16), nd=nd)
