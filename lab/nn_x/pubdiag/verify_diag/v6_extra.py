"""m02-part-only transfer factor (EIV profile LR), TO-half cold push, 7-mix F7x-F7b holdout by weekend,
M-only reweighted val delta (calib-style) for M2-Cd."""
from v_lib import *
from scipy import stats

fv, Mv, ov = meta("val")
g = groups_val(); _, inv = np.unique(g, return_inverse=True)
POOLS2 = dict(POOLS); POOLS2.update({"G3c": ["F10", "F7", "F7"], "M2c": ["F7", "F7"]})
V = {k: pf("val", pool_final(f, "val")) for k, f in POOLS2.items()}
cf = {"PF7x": "Pall", "G3": "G3c", "M2": "M2c"}
A = ["PF7x", "G3", "M2"]
cols = [V[cf[a]] - V["Cd"] for a in A] + [V[a] - V[cf[a]] for a in A]  # non-m02 parts, m02 parts
Dm = np.stack(cols, 1)
Sc = np.stack([np.bincount(inv, weights=Dm[:, j] - Dm[:, j].mean()) for j in range(6)], 1)
Vv = Sc.T @ Sc / len(Mv) ** 2
nP = 8562
# public covariance of the 3 total deltas
T = np.stack([V[a] - V["Cd"] for a in A], 1); Vp = np.cov(T.T) / nP
pubLB = {"Cd": 0.39208, "PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434}
p = np.array([pubLB[a] - pubLB["Cd"] for a in A]); v = Dm.mean(0)
Vvi, Vpi = np.linalg.inv(Vv), np.linalg.inv(Vp)
def prof(bn, bm):
    B = np.hstack([bn * np.eye(3), bm * np.eye(3)])  # public true = bn*nonm02 + bm*m02
    Hm = Vvi + B.T @ Vpi @ B; mu = np.linalg.solve(Hm, Vvi @ v + B.T @ Vpi @ p)
    return (v - mu) @ Vvi @ (v - mu) + (p - B @ mu) @ Vpi @ (p - B @ mu)
bs = np.linspace(-3, 4, 3501)
L = np.array([prof(1.0, b) for b in bs]); bh = bs[L.argmin()]; ci = bs[L - L.min() < 3.84]
print(f"m02-only b (non-m02 fixed 1): b_hat {bh:.2f} CI [{ci.min():.2f},{ci.max():.2f}] p(b=1) {1-stats.chi2.cdf(prof(1,1)-L.min(),1):.3f} p(b=0) {1-stats.chi2.cdf(prof(1,0)-L.min(),1):.3f}")
print("val parts non-m02 / m02:", np.round(v, 5))

# TO-half cold push
def cold(split):
    return np.load(f"<repo>/data/feats/{split}/it_hist_all_luo.npy") == 0
ft_, Mt, ot = meta("test"); pub = ft_["timestamp"].to_numpy() < 1790497920
def push(split, a, b, cflag, mask=None):
    f, M, o = meta(split)
    dz = z(split, pool_half(POOLS[a], split, "to")) - z(split, pool_half(POOLS[b], split, "to"))
    fid = np.repeat(np.arange(len(M)), M)
    nc = np.bincount(fid, weights=cflag, minlength=len(M)); nw = M - nc
    sc = np.bincount(fid, weights=dz * cflag, minlength=len(M)); sw = np.bincount(fid, weights=dz * (~cflag), minlength=len(M))
    ok = (nc > 0) & (nw > 0)
    if mask is not None: ok &= mask
    return float((sc[ok] / nc[ok] - sw[ok] / nw[ok]).mean())
cv, ct = cold("val"), cold("test")
for a in ["M2", "G3", "PF7x"]:
    print(f"TO cold push {a}-Cd: val {push('val',a,'Cd',cv):+.4f} pub {push('test',a,'Cd',ct,pub):+.4f} priv {push('test',a,'Cd',ct,~pub):+.4f}")
