"""Is the affinity flatness (contrast ratio CR > 1) a property of single members or created by pooling?
CR = label(top/bottom decile lift) / calibrated-model(top/bottom) for 6 inputs, for: G3 final, G3 TO pool, G3 FT pool,
single TO members (one per family) and the 7-member F7x TO pool."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fit_calib, ndcg, D  # noqa
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z

d = load("val"); P = "<repo>/preds"
fid = d["fid"]; y = d["y"].astype(np.float64); M = d["M"]
z = lambda x: feed_rank_z(d["off"], M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
S = {"G3 final": d["g"], "G3 TO pool": d["gto"], "G3 FT pool": d["gft"]}
for m in ["r6_cd_m02_r101", "r6_cd_m02_p70k_r101", "v23_gru_pre2s_s501"]:
    S[f"{m} TO"] = z(np.load(f"{P}/nn_{m}_trainonly_val.npy"))
    S[f"{m} FT"] = z(np.load(f"{P}/nn_{m}_n2cpv3_val_oof.npy"))
F7x = [f"r6_cd_m02_r{s}" for s in range(101, 108)]
S["F7x TO pool(7)"] = z(np.mean([z(np.load(f"{P}/nn_{m}_trainonly_val.npy")) for m in F7x], 0))
X = {"f3": d["f3"], "fcs": d["d_fcs"], "ctx_hs_k128": d["xt_ctx_hs_k128"], "smax": d["d_smax"], "cat_last32": d["d_cat_last32"],
     "seq_cv_pop": d["xt_seq_cv_pop"], "it_hist_all_luo": d["xt_it_hist_all_luo"]}
B = {}
for n, x in X.items():
    q = np.unique(np.quantile(np.asarray(x, np.float64), np.linspace(0, 1, 11)[1:-1]))
    B[n] = np.unique(np.searchsorted(q, np.asarray(x, np.float64), side="right"), return_inverse=True)[1]
print(f"{'score':28s} {'NDCG':>7s} " + " ".join(f"{n:>14s}" for n in X))
for k, g in S.items():
    beta, basis, p, npos = fit_calib(d, g)
    E = npos[fid] * p; E0 = npos[fid] / M[fid]
    out = []
    for n in X:
        b = B[n]; nb = b.max() + 1
        A = np.bincount(b, y, nb); Eb = np.bincount(b, E, nb); E0b = np.bincount(b, E0, nb)
        ll, ml = A / E0b, Eb / E0b
        out.append((ll[-1] / ll[0]) / (ml[-1] / ml[0]))
    print(f"{k:28s} {ndcg(d, g).mean():7.5f} " + " ".join(f"{c:14.3f}" for c in out), flush=True)
