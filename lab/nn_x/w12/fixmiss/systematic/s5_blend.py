"""Step 5: per-feed blend weights from label-free feed descriptors (the rank-changing counterpart of the per-feed
temperature found in s1b). Components (FD16P20 members, all-data): z(student pool) and z(pop-light pool) for the
fine-tuned (out-of-fold) and train-only branches, as build_r4.py builds them.

logit_i = sum_k exp(theta_k + beta_k . g_f) z_k,i     (k = base_ft, pop_ft, base_to, pop_to)
fitted by listwise CE on user half A, applied to B, and reverse; NDCG read against the current best file (m04_02).
Also a direct NDCG grid: pop-light weight by tercile of one descriptor (best on A applied to B and reverse).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403
from core import _feed_lse

P = ROOT / "preds"
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["label", "xp_imp_c50", "cat_newest_idx", "h_brk10", "h_ent", "n_hist_cats32", "H", "guest", "uvec_missing"])
y = D["label"].to_numpy().astype(np.float64)
npos = np.bincount(fid, y, len(M))
S8, S4 = range(951, 959), range(951, 955)
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
FDP = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]


def pool(names, suf):
    return zfeed(np.mean([zfeed(np.load(P / f"nn_{m}_{suf}.npy").astype(np.float64), off, M) for m in names], 0), off, M)


Z = {"base_ft": pool(FD16, "n2cpv3s3_val_oof"), "pop_ft": pool(FDP, "n2cpv3s3_val_oof"),
     "base_to": pool(FD16, "trainonly_val"), "pop_to": pool(FDP, "trainonly_val")}
Mr = M[fid]
wl = np.where(Mr < 50, 0.4, 0.2)
ft = zfeed((1 - wl) * Z["base_ft"] + wl * Z["pop_ft"], off, M); to = zfeed((1 - wl) * Z["base_to"] + wl * Z["pop_to"], off, M)
ref = 0.5 * ft + 0.5 * to
nd_ref = ndcg_per_feed(ref, y, off, M)
stored = 0.5 * np.load(P / "ens_FD16P20_m04_02kd_ft_val.npy") + 0.5 * np.load(P / "ens_FD16P20_m04_02kd_to_val.npy")
print(f"rebuilt m04_02 val {nd_ref.mean():.5f} (stored file {ndcg_per_feed(stored, y, off, M).mean():.5f}; corr {np.corrcoef(ref, stored)[0, 1]:.6f})")

first = off


def c(name):
    return np.nan_to_num(D[name].to_numpy().astype(np.float64)[first], nan=0.0)


cold_share = np.bincount(fid, D["xp_imp_c50"].to_numpy() == 0, len(M)) / M
cni = D["cat_newest_idx"].to_numpy()
new_share = np.bincount(fid, (cni < 0) | (cni >= 8), len(M)) / M
desc = {"logM": np.log(M), "M<50": (M < 50).astype(float), "cold_share": cold_share, "new_share": new_share,
        "h_brk10": c("h_brk10"), "h_ent": c("h_ent"), "n_hist_cats32": c("n_hist_cats32"), "H": c("H") / 256,
        "guest": c("guest"), "uvec_missing": c("uvec_missing")}
K = list(Z)
Zm = np.column_stack([Z[k] for k in K])


def fit(names, fm):
    G = np.column_stack([np.ones(len(M))] + [(desc[n] - desc[n].mean()) / (desc[n].std() + 1e-12) for n in names])
    q = G.shape[1]
    rmask = fm[fid]

    def fg(th):
        Th = th.reshape(len(K), q)
        C = np.exp(G @ Th.T)                       # feeds x K
        x = (Zm * C[fid]).sum(1)
        lse = _feed_lse(x, off, M)
        p = np.exp(x - lse[fid])
        r = (npos[fid] * p - y) * rmask
        f = -(y * (x - lse[fid]) * rmask).sum() + 1e-3 * fm.sum() * (Th[:, 1:] ** 2).sum()
        Sf = np.column_stack([np.bincount(fid, r * Zm[:, k], len(M)) for k in range(len(K))])  # feeds x K
        g = ((Sf * C).T @ G)                       # K x q
        g[:, 1:] += 2e-3 * fm.sum() * Th[:, 1:]
        return f, g.ravel()

    th0 = np.zeros((len(K), q)); th0[:, 0] = np.log([0.4, 0.1, 0.4, 0.1])
    r = minimize(fg, th0.ravel(), jac=True, method="L-BFGS-B")
    return r.x.reshape(len(K), q), G


def apply(Th, G):
    return (Zm * np.exp(G @ Th.T)[fid]).sum(1)


rows_B = half[fid] == 1
for names in [[], ["M<50"], ["logM"], ["cold_share"], ["new_share"], ["h_brk10"], ["n_hist_cats32"], ["h_ent"], ["H"],
              ["guest", "uvec_missing"], list(desc)]:
    ThA, G = fit(names, half == 0); ThB, _ = fit(names, half == 1)
    new = np.where(rows_B, apply(ThA, G), apply(ThB, G))
    report_delta(ndcg_per_feed(new, y, off, M), nd_ref, M, half, f"clogit weights x [{','.join(names) or 'const'}]"[:40])
    wA = np.exp(ThA[:, 0]); print(f"      const weights A {np.round(wA / wA.sum(), 3).tolist()} ({K})  slopes A {np.round(ThA[:, 1:], 3).tolist() if names else []}")

# direct NDCG grid: pop-light weight per tercile of one descriptor (both branches share it)
print("\n--- direct NDCG grid: pop-light weight per tercile of a descriptor (M<50 keeps 0.4 inside the grid as one option) ---")
grid = [0.1, 0.2, 0.3, 0.4, 0.5]


def build(wrow):
    a = zfeed((1 - wrow) * Z["base_ft"] + wrow * Z["pop_ft"], off, M); b = zfeed((1 - wrow) * Z["base_to"] + wrow * Z["pop_to"], off, M)
    return 0.5 * a + 0.5 * b


for dn in ["cold_share", "new_share", "h_brk10", "n_hist_cats32", "H"]:
    v = desc[dn]
    qs = np.quantile(v, [1 / 3, 2 / 3])
    t = np.searchsorted(qs, v, side="right")            # tercile per feed
    # per tercile, choose weight separately (feeds of other terciles keep the m04_02 rule)
    best = {}
    dfin = np.zeros(len(M))
    for tt in range(3):
        cur = {}
        for w in grid:
            wf = np.where(t == tt, w, np.where(M < 50, 0.4, 0.2))
            cur[w] = ndcg_per_feed(build(wf[fid]), y, off, M) - nd_ref
        for h in (0, 1):
            m_fit = (half == h) & (t == tt)
            wbest = max(grid, key=lambda w: cur[w][m_fit].mean())
            m_app = (half == 1 - h) & (t == tt)
            dfin[m_app] = cur[wbest][m_app]
            best[(tt, h)] = wbest
    report_delta(dfin + nd_ref, nd_ref, M, half, f"grid pop-light w by tercile of {dn}")
    print(f"      chosen (tercile, fit-half) -> w: {best}")
