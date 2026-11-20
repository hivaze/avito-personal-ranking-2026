"""Near-duplicate 'twin' confusion: positives with a same-feed negative at item-embedding cos >= 0.95.
Oracle: lift each positive just above its highest-ranked twin. Realistic: AUC of v11 and single signals inside (pos, twin-neg) pairs."""
import json
from common_e import meta, ndcg_per_feed, ROOT, SCR
import numpy as np
import numba as nb

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
base = ndcg_per_feed(s, y, off, M)
d = f"{ROOT}/data/competition/val/item_embeddings"
meta_ = json.loads(open(f"{d}/data.meta").read())
E = np.memmap(f"{d}/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta_["shape"]))
eidx = np.load(f"{ROOT}/data/flat/val/cand_emb_idx.npy").astype(np.int64)
X = np.zeros((len(eidx), 128), np.float32)
ok = eidx >= 0
X[ok] = E[eidx[ok]].astype(np.float32)
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)


@nb.njit
def twins(X, y, s, off, M, thr):
    pa = []
    pb = []
    for f_ in range(len(M)):
        o = off[f_]
        for i in range(o, o + M[f_]):
            if y[i] <= 0:
                continue
            for j in range(o, o + M[f_]):
                if y[j] > 0 or j == i:
                    continue
                c = 0.0
                for k in range(X.shape[1]):
                    c += X[i, k] * X[j, k]
                if c >= thr:
                    pa.append(i)
                    pb.append(j)
    return np.array(pa), np.array(pb)


pa, pb = twins(X, y, s, off, M, 0.95)
npos_tw = len(np.unique(pa))
print(f"(pos, twin-neg) pairs at cos>=0.95: {len(pa)}; positives with a twin: {npos_tw} ({npos_tw / (y > 0).sum():.3f} of positives)")
above = s[pb] > s[pa]
print(f"share of twin pairs where the twin is ranked above the positive: {above.mean():.3f}")
# oracle: positive moved just above its best-scored twin (only if twin above)
s2 = s.copy()
best = np.full(len(s), -np.inf)
np.maximum.at(best, pa, s[pb])
m = best > s
s2[m] = best[m] + 1e-6
nd2 = ndcg_per_feed(s2, y, off, M)
print(f"oracle twin disambiguation: {nd2.mean():.5f} ({nd2.mean() - base.mean():+.5f}); positives moved {m.sum()}")
Z = np.load(f"{SCR}/out/attrs.npz")
S = {"v11": s, "FT5": np.load(f"{ROOT}/preds/nn_final5_val_oof.npy").astype(np.float64), "TO5": np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy").astype(np.float64)}
for k in ["f1", "f2", "f3", "f4", "f5", "ag_age", "it_hist_all_luo", "seq_cv_pop", "it_tr_ctr", "cold_item", "maxsim32", "cos_u", "pre_fe"]:
    S[k] = Z[k].astype(np.float64)
ids = np.load(f"{ROOT}/data/flat/val/cand_item_id.npy").astype(np.float64)
S["item_id"] = ids
out = []
for k, x in S.items():
    a, b = x[pa], x[pb]
    okp = ~(np.isnan(a) | np.isnan(b))
    auc = ((a > b) + 0.5 * (a == b))[okp].mean()
    out.append((auc, k, (a == b)[okp].mean()))
print("AUC inside (pos, twin-neg) pairs: " + "  ".join(f"{k} {a:.3f}" + (f"(eq {e:.2f})" if e > 0.05 else "") for a, k, e in sorted(out, reverse=True)))
