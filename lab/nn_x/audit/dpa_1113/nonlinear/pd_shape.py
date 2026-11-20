"""Shape summary of label-lift vs model-lift curves (10 global quantile bins) for every input:
contrast ratio CR = (label lift top/bottom bin) / (model lift top/bottom bin): CR > 1 -> calibrated G3 is FLATTER than the
label in x (under-reacts / saturated); CR < 1 -> over-reacts. Non-monotone flags: adjacent bins where label and model move in
opposite directions by > 2 SE of the label lift. Also the same restricted to each feed's G3 top-20 rows (NDCG-relevant).
Usage: python pd_shape.py val|valto|hold"""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import *  # noqa
from scipy.stats import spearmanr

split = sys.argv[1]
d = load("val" if split.startswith("val") else "hold")
g = d["gto"] if split == "valto" else d["g"]
beta, basis, p, npos = fit_calib(d, g)
fid = d["fid"]; y = d["y"].astype(np.float64); M = d["M"]
E = npos[fid] * p; E0 = npos[fid] / M[fid]
o = np.lexsort((-g, fid)); st = np.r_[True, fid[o][1:] != fid[o][:-1]]
rk = np.empty(len(g)); rk[o] = np.arange(len(o)) - np.flatnonzero(st)[np.cumsum(st) - 1]
X = {}
for k in range(1, 6):
    X[f"f{k}"] = d[f"f{k}"]
for k in [k for k in d if k.startswith("d_") and k not in ("d_logM", "d_logH", "d_cnorm", "d_cmiss")]:
    X[k[2:]] = d[k]
for k in [k for k in d if k.startswith("xt_") and k != "xt_has_pre2_fe"]:
    X[k[3:]] = d[k]
rows = []
for n, x in X.items():
    x = np.asarray(x, np.float64)
    q = np.unique(np.quantile(x, np.linspace(0, 1, 11)[1:-1]))
    b = np.unique(np.searchsorted(q, x, side="right"), return_inverse=True)[1]; nb = b.max() + 1
    if nb < 3:
        continue
    out = []
    for sel in [np.ones(len(x), bool), rk < 20]:
        A = np.bincount(b[sel], y[sel], nb); Eb = np.bincount(b[sel], E[sel], nb); E0b = np.bincount(b[sel], E0[sel], nb)
        ll = A / np.maximum(E0b, 1e-12); ml = Eb / np.maximum(E0b, 1e-12); sl = np.sqrt(np.maximum(A, 1)) / np.maximum(E0b, 1e-12)
        cr = (ll[-1] / max(ll[0], 1e-9)) / (ml[-1] / max(ml[0], 1e-9))
        dl = np.diff(ll); dm = np.diff(ml)
        rev = int(np.sum((np.sign(dl) != np.sign(dm)) & (np.abs(dl) > 2 * sl[1:])))
        out.append((cr, spearmanr(np.arange(nb), ll)[0], spearmanr(np.arange(nb), ml)[0], rev, ll, ml))
    rows.append((n, nb, out))
print(f"[{split}] contrast ratio CR = label(top/bottom) / model(top/bottom); rev = adjacent-bin opposite moves > 2 SE")
print(f"  {'input':20s} {'CR_all':>7s} {'rhoL':>5s} {'rhoM':>5s} {'rev':>3s} | {'CR_top20':>8s} {'rev':>3s} | label lift (all rows) ... model lift")
for n, nb, out in sorted(rows, key=lambda r: -abs(np.log(r[2][0][0]))):
    (c, rl, rm, rv, ll, ml), (c2, _, _, rv2, _, _) = out
    print(f"  {n:20s} {c:7.3f} {rl:5.2f} {rm:5.2f} {rv:3d} | {c2:8.3f} {rv2:3d} | L {np.round(ll, 2).tolist()} M {np.round(ml, 2).tolist()}")
