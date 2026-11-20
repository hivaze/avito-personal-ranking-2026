"""Critic check 3 (CPU): is the stream's 'per-feed temperature' a property of the model, or an artefact of the per-feed
z-normalisation of the blend (every feed forced to sd ~1)? The KD teacher is a mean of RAW member logits
(lab/nn_x/w12/kd/dump_teacher.py sums raw logits), so a per-feed-T teacher target must be fitted on raw logits.

Same descriptors and fit as systematic/s1b_temper.py, applied to:
  (a) z-pool of the 16 FD16 train-only students (what the blend uses)
  (b) mean of their RAW logits (what a teacher softmax uses)
  (c) one raw member
Fit on user half A, NLL per positive read on half B.
"""
import sys
import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/systematic")
from core import layout, load_table, _feed_lse, zfeed, HERE  # noqa: E402

P = "<repo>/preds/"
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["label", "h_brk10", "h_ent", "n_hist_cats32", "h_nov32r", "H", "guest", "h_topsh", "uvec_missing"])
y = D["label"].to_numpy().astype(np.float64)
npos = np.bincount(fid, y, len(M))
first = off
G = np.column_stack([np.ones(len(M)), np.log(M)] + [np.nan_to_num(D[c].to_numpy()[first].astype(np.float64), nan=0.0)
                     for c in ["h_brk10", "h_ent", "n_hist_cats32", "h_nov32r"]] +
                    [D["H"].to_numpy()[first] / 256.0, D["guest"].to_numpy()[first].astype(float),
                     np.nan_to_num(D["h_topsh"].to_numpy()[first].astype(np.float64), nan=0.0),
                     D["uvec_missing"].to_numpy()[first].astype(float)])
G[:, 1:] = (G[:, 1:] - G[:, 1:].mean(0)) / G[:, 1:].std(0)
names = ["const", "logM", "h_brk10", "h_ent", "n_hist_cats32", "h_nov32r", "H", "guest", "h_topsh", "uvec_missing"]


def nll(x, fm):
    lse = _feed_lse(x, off, M)
    return float((-(x - lse[fid]) * y)[fm[fid]].sum())


def fit(s, fm):
    def fg(w):
        t = np.exp(G @ w); x = s * t[fid]
        lse = _feed_lse(x, off, M); p = np.exp(x - lse[fid])
        f = -(y * (x - lse[fid]))[fm[fid]].sum()
        gf = np.bincount(fid, (npos[fid] * p - y) * s, len(M)) * t
        return f, (gf[fm][:, None] * G[fm]).sum(0)
    return minimize(fg, np.zeros(G.shape[1]), jac=True, method="L-BFGS-B").x


FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
raw = [np.load(P + f"nn_{m}_trainonly_val.npy").astype(np.float64) for m in FD16]
cands = {"z-pool (blend form)": zfeed(np.mean([zfeed(r, off, M) for r in raw], 0), off, M),
         "raw-mean (teacher form)": np.mean(raw, 0), "raw member s951": raw[0]}
for name, s in cands.items():
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0))
    TA = float(np.exp(minimize_scalar(lambda lt: nll(s / np.exp(lt), half == 0), bounds=(-3, 3), method="bounded").x))
    wA = fit(s, half == 0)
    fB = half == 1
    g = nll(s / TA, fB) / npos[fB].sum(); pf = nll(s * np.exp(G @ wA)[fid], fB) / npos[fB].sum()
    print(f"\n[{name}] per-feed sd mean {sd.mean():.3f} (cv {sd.std() / sd.mean():.3f}, corr with logM {np.corrcoef(sd, np.log(M))[0, 1]:+.3f}); "
          f"global T_A {TA:.4f}; NLL/pos on B: global {g:.5f}  per-feed {pf:.5f}  gain {g - pf:+.5f}")
    print("   per-feed log-T weights (fit A): " + "  ".join(f"{n} {v:+.3f}" for n, v in zip(names, wA)))
    lt = G @ wA
    print(f"   implied per-feed log-T spread: sd {lt.std():.3f}, 5-95% [{np.quantile(lt, .05):+.3f}, {np.quantile(lt, .95):+.3f}]")
