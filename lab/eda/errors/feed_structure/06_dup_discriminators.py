"""Which features separate a positive from its NEGATIVE near-duplicates (cos > 0.95, same feed)? For every val feature:
pairwise AUC on (positive, negative-duplicate) pairs, and the residual coefficient of the standardized feature difference in a
pairwise logistic model next to the base-score difference (feed-clustered SE). Leaky families (x_lh_*, als_*, non-LUO it_hist_*) skipped."""
from fs_common import *
import numba as nb

BASE = sys.argv[1] if len(sys.argv) > 1 else "nn_final2_val_oof"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
d = load_val()
X, miss = load_emb_norm(d["emb_idx"])
M, off, t, fid = d["M"], d["off"], d["target"].astype(np.int64), d["fid"]
use = ~d["leak"]
z = zfeed(np.load(PREDS / f"{BASE}.npy"), off, M)


@nb.njit(cache=True)
def collect(X, miss, off, M, t, thr, use):
    I = []; J = []
    for f in range(len(M)):
        if not use[f]: continue
        o = off[f]; m = M[f]
        Xf = X[o:o + m]; C = Xf @ Xf.T
        for i in range(m):
            if t[o + i] != 1 or miss[o + i]: continue
            for j in range(m):
                if j != i and t[o + j] == 0 and not miss[o + j] and C[i, j] > thr:
                    I.append(o + i); J.append(o + j)
    return np.array(I), np.array(J)

I, J = collect(X, miss, off, M, t, THR, use)
G = fid[I]
print(f"pairs {len(I)} in {len(np.unique(G))} feeds (cos>{THR})")


def pair_fit(dz, dx, g):
    """Symmetrised pairwise logistic without intercept: y=1 for (pos-neg) diffs. Returns beta, clustered SE."""
    Xm = np.column_stack([dz, dx])
    Xs = np.vstack([Xm, -Xm]); y = np.r_[np.ones(len(dz)), np.zeros(len(dz))]; gg = np.r_[g, g]
    b = np.zeros(2)
    for _ in range(25):
        p = 1 / (1 + np.exp(-Xs @ b))
        W = p * (1 - p)
        Hm = (Xs * W[:, None]).T @ Xs + 1e-6 * np.eye(2)
        b = b + np.linalg.solve(Hm, Xs.T @ (y - p))
    p = 1 / (1 + np.exp(-Xs @ b))
    sc = Xs * (y - p)[:, None]
    # cluster-robust sandwich
    ug, inv = np.unique(gg, return_inverse=True)
    S = np.zeros((len(ug), 2)); np.add.at(S, inv, sc)
    Hinv = np.linalg.inv((Xs * (p * (1 - p))[:, None]).T @ Xs)
    V = Hinv @ (S.T @ S) @ Hinv
    return b, np.sqrt(np.diag(V))

names = [n for n in all_names() if not (n.startswith("x_lh") or n.startswith("als") or (n.startswith("it_hist") and "luo" not in n))] if False else None
from common import all_feature_names
names = [n for n in all_feature_names("val") if not (n.startswith("x_lh") or n.startswith("als") or (n.startswith("it_hist") and "luo" not in n))]
dz = z[I] - z[J]
auc_z = (dz > 0).mean() + 0.5 * (dz == 0).mean()
print(f"base z AUC on these pairs {auc_z:.4f}")
rows = []
for n in names:
    x = np.load(ROOT / f"data/feats/val/{n}.npy", mmap_mode="r")
    xi = np.asarray(x[I], np.float64); xj = np.asarray(x[J], np.float64)
    ok = np.isfinite(xi) & np.isfinite(xj)
    if ok.mean() < 0.5: continue
    dx = np.where(ok, xi - xj, 0.0)
    sdx = dx[ok].std()
    auc = (dx[ok] > 0).mean() + 0.5 * (dx[ok] == 0).mean()
    tie = (dx[ok] == 0).mean()
    if sdx == 0:
        rows.append((n, auc, tie, 0.0, 0.0)); continue
    b, se = pair_fit(dz, dx / sdx, G)
    rows.append((n, auc, tie, b[1], b[1] / se[1]))
rows.sort(key=lambda r: -abs(r[4]))
print(f"\n{'feature':28s} {'AUC':>6s} {'tie':>6s} {'beta|z':>7s} {'t':>6s}")
for r in rows[:30]:
    print(f"{r[0]:28s} {r[1]:.4f} {r[2]:.4f} {r[3]:+.3f} {r[4]:+.2f}")
print("\nlargest |AUC-0.5| (marginal):")
for r in sorted(rows, key=lambda r: -abs(r[1] - 0.5))[:15]:
    print(f"{r[0]:28s} {r[1]:.4f} tie {r[2]:.4f} t_resid {r[4]:+.2f}")
print(f"\n#features {len(rows)}; |t_resid|>3: {sum(abs(r[4])>3 for r in rows)}; expected by chance ~{0.0027*len(rows):.1f}")
json.dump(dict(base=BASE, thr=THR, npairs=int(len(I)), auc_z=auc_z, rows=rows), open(OUT / f"out_06_{BASE}_{THR}.json", "w"), indent=1, default=float)
