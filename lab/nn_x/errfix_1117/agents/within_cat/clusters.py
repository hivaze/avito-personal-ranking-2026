"""Approximate near-duplicate clusters: items in the same feed and category with n_cand_dup95>=1 linked when their
similarity profiles (ucos, h1, h5, h10, h32, hmean5, hmax) are within a tolerance. Writes cluster id to a parquet in my folder."""
from wc import *
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
df = load()
d = df.with_row_index("rid").filter(pl.col("n_cand_dup95") >= 1)
prof = ["ucos", "h1", "h5", "h10", "h32", "hmean5", "hmax"]
P = np.nan_to_num(d.select(prof).to_numpy().astype(np.float64), nan=-1)
g = (d["fidx"] * 100 + d["cat"]).to_numpy()
rid = d["rid"].to_numpy()
ndup = d["n_cand_dup95"].to_numpy()
tol = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
order = np.argsort(g, kind="stable")
g, P, rid, ndup = g[order], P[order], rid[order], ndup[order]
st = np.r_[0, np.nonzero(g[1:] != g[:-1])[0] + 1]; en = np.r_[st[1:], len(g)]
I, J = [], []
for a, b in zip(st, en):
    if b - a < 2:
        continue
    Q = P[a:b]
    D = np.abs(Q[:, None, :] - Q[None, :, :]).max(-1)
    ii, jj = np.nonzero(np.triu(D < tol, 1))
    I.append(ii + a); J.append(jj + a)
I = np.concatenate(I); J = np.concatenate(J)
n = len(g)
A = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
nc, lab = connected_components(A, directed=False)
cs = np.bincount(lab)[lab]
print("tol", tol, "items", n, "clusters", nc, "size-1 share", (cs == 1).mean())
print("agreement: cluster size-1 == n_cand_dup95 :", (cs - 1 == ndup).mean(), " within +-1:", (np.abs(cs - 1 - ndup) <= 1).mean())
print("mean cs-1", (cs - 1).mean(), "mean ndup", ndup.mean())
out = pl.DataFrame({"rid": rid, "clu": lab.astype(np.int64), "clu_size": cs.astype(np.int64)})
out.write_parquet(f"<repo>/lab/nn_x/errfix_1117/agents/within_cat/clu_{tol}.parquet")
