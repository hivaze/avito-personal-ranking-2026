"""Post-hoc sanity checks (not pre-declared, no gate role): brute-force recomputation of the val signals on 40 random val
feeds vs the numba transfer, and single-feature val NDCG with a random tie-break (the gated script uses pessimistic ties)."""
import os
for _k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[_k] = "12"
import sys, numpy as np
from pathlib import Path
ROOT = Path("<repo>"); OUT = ROOT / "nn_x/pretrain2/P4"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eda/errors/synthesis"))
from metrics import ndcg_per_feed
from qt import leak_mask
src = open(OUT / "uknn_gate.py").read().replace('LOG = open(OUT / "uknn_gate.log", "w")', 'LOG = open(os.devnull, "w")')  # never touch the gated log
# reuse definitions up to (not including) the val-query section
exec(compile(src.split("# ---------------------------------------------------------------- val queries")[0], "uknn_defs", "exec"))
S = np.load(OUT / "signals_val.npz")
b = build_bank(window(16))
v_item = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64); v_cl = np.load(CLU / "cand_cl_4096_val.npy").astype(np.int64)
Qv = l2n(np.load(FLAT / "val/user_emb.npy"))
rng = np.random.default_rng(7); qf = np.sort(rng.choice(va["n"], 40, replace=False))
idx, sim = knn(Qv[qf], b["K"])
ww = np.exp(TEMP * (sim - sim[:, :1])); ww /= ww.sum(1, keepdims=True)
maxerr = 0.0; nchk = 0
for lev, keys, (ptr, bk, bp) in [("item", v_item, b["item"]), ("cl", v_cl, b["cl"])]:
    for i, q in enumerate(qf):
        d = {}
        for j in range(K_NN):
            u = idx[i, j]
            for t in range(ptr[u], ptr[u + 1]):
                a = d.setdefault(int(bk[t]), [0.0, 0.0]); a[0 if bp[t] else 1] += ww[i, j]
        for r in range(va["off"][q], va["off"][q] + va["M"][q]):
            e = d.get(int(keys[r]), [0.0, 0.0]) if keys[r] >= 0 else [0.0, 0.0]
            maxerr = max(maxerr, abs(e[0] - float(S[f"s_pos_{lev}"][r])), abs(e[1] - float(S[f"s_neg_{lev}"][r])))
            nchk += 1
print(f"brute-force check: {nchk} candidate-levels on 40 val feeds, max abs error {maxerr:.2e} (float16 storage)")
vy = np.load(FLAT / "val/cand_target.npy").astype(np.float64); keep = ~leak_mask()
tb = np.random.default_rng(0).random(len(vy)) * 1e-6
r0 = ndcg_per_feed(tb, vy, va["off"], va["M"])[keep]
print(f"random score val leak-free NDCG {r0.mean():.5f} ± {r0.std(ddof=1)/np.sqrt(len(r0)):.5f}")
for k in ["s_pos_item", "s_neg_item", "contrast_item", "s_pos_cl", "s_neg_cl", "contrast_cl"]:
    x = ndcg_per_feed(S[k].astype(np.float64) + tb, vy, va["off"], va["M"])[keep]; dd = x - r0
    print(f"single-feature (random tie-break) {k:14s} val {x.mean():.5f} ± {x.std(ddof=1)/np.sqrt(len(x)):.5f}; vs random {dd.mean():+.5f} ± {dd.std(ddof=1)/np.sqrt(len(dd)):.5f}")
