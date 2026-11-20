"""(a) Partial dependence of the calibrated G3 pool vs the label, per input, on val (and holdout).
Calibration: within-feed conditional logit of y on a hinge basis of z(G3) (nlc.fit_calib) -> p_i; E_i = npos_f * p_i.
Per input x (global quantile bins, <= 10):
  label lift = A/E0, model lift = E/E0 (E0 = npos_f / M_f, uniform within feed), act/exp = A/E.
  LR main  = 2*(LL[offset log p + theta_bin] - LL[offset])                 (df = nb-1): what G3 leaves on x
  LR inter = 2*(LL[offset + theta_cell(stratum x bin)] - LL[offset + theta_bin + theta_stratum])  (df ~ (nb-1)(ns-1))
Strata: M3 (<50, 50-99, >=100), u0, coldrow (val), feedcold4 (val), vert (12).
Usage: python pd.py val|hold|valto
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as CHI2
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import *  # noqa

t0 = time.time()
split = sys.argv[1]
d = load("val" if split.startswith("val") else "hold")
g = d["gto"] if split == "valto" else d["g"]
beta, basis, p, npos = fit_calib(d, g)
print(f"[{split}] calib beta {np.round(beta, 3).tolist()}  ({time.time()-t0:.0f}s)")
fid = d["fid"]; y = d["y"].astype(np.float64); M = d["M"]
E = npos[fid] * p
E0 = npos[fid] / M[fid]
offl = np.log(np.maximum(p, 1e-300))


def cl_fit(idx_list, sizes, lam=1e-3):
    """conditional logit with offset offl + sum_j theta_j[idx_j]. Returns max LL."""
    tot = sum(sizes)
    starts = np.cumsum([0] + sizes[:-1])

    def f(th):
        l = offl.copy()
        for j, ix in enumerate(idx_list):
            l += th[starts[j]:starts[j] + sizes[j]][ix]
        mx = np.maximum.reduceat(l, d["off"]); e = np.exp(l - mx[fid]); Z = np.bincount(fid, e, d["nf"])
        q = e / Z[fid]
        ll = (y * l).sum() - (npos * (np.log(Z) + mx)).sum()
        r = npos[fid] * q - y
        gr = np.concatenate([np.bincount(ix, r, sizes[j]) for j, ix in enumerate(idx_list)])
        return -ll + lam * (th ** 2).sum(), gr + 2 * lam * th
    if tot == 0:
        l = offl; mx = np.maximum.reduceat(l, d["off"]); Z = np.bincount(fid, np.exp(l - mx[fid]), d["nf"])
        return (y * l).sum() - (npos * (np.log(Z) + mx)).sum()
    r = minimize(f, np.zeros(tot), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return -r.fun


LL0 = cl_fit([], [])


def bins_of(x, nb=10):
    x = np.asarray(x, np.float64)
    q = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1]))
    b = np.searchsorted(q, x, side="right")
    # merge empty
    u, b = np.unique(b, return_inverse=True)
    return b, q


# inputs
X = {}
for k in range(1, 6):
    X[f"f{k}"] = d[f"f{k}"]
    X[f"f{k}_pct"] = fpct(d, d[f"f{k}"])
for k in [k for k in d if k.startswith("d_") and k not in ("d_logM", "d_logH", "d_cnorm", "d_cmiss")]:
    X[k[2:]] = d[k]
for k in [k for k in d if k.startswith("xt_") and k != "xt_has_pre2_fe"]:
    X[k[3:]] = d[k]
if split != "hold":
    X["cold"] = d["cold"].astype(np.float64)

S = {}
S["M3"] = np.digitize(M, [50, 100])[fid]
S["u0"] = d["u0"].astype(np.int64)[fid]
S["vert"] = d["vert"]
if split != "hold":
    S["coldrow"] = d["cold"].astype(np.int64)
    cs = np.bincount(fid, d["cold"].astype(np.float64), d["nf"]) / M
    S["feedcold4"] = np.digitize(cs, [0.15, 0.30, 0.45])[fid]
else:
    # in-period proxy of coldness: no LUO history appearances (it_hist_all_luo == 0)
    S["coldrow"] = (d["xt_it_hist_all_luo"] <= 0).astype(np.int64)

res = []
detail = {}
for name, x in X.items():
    b, q = bins_of(x)
    nb_ = b.max() + 1
    if nb_ < 2:
        continue
    A = np.bincount(b, y, nb_); Eb = np.bincount(b, E, nb_); E0b = np.bincount(b, E0, nb_); n = np.bincount(b, minlength=nb_)
    LLm = cl_fit([b], [nb_])
    lr = 2 * (LLm - LL0)
    row = dict(name=name, nb=nb_, lr=lr, p=CHI2.sf(lr, nb_ - 1), maxdev=np.max(np.abs(A / Eb - 1)))
    for sn, s in S.items():
        ns = s.max() + 1
        LLr = cl_fit([b, s], [nb_, ns])
        cell = s * nb_ + b
        LLf = cl_fit([cell], [ns * nb_])
        lri = 2 * (LLf - LLr)
        dfi = (nb_ - 1) * (ns - 1)
        row[f"I_{sn}"] = lri; row[f"I_{sn}_df"] = dfi; row[f"I_{sn}_p"] = CHI2.sf(lri, dfi)
    res.append(row)
    detail[name] = (q, n / len(x), A / E0b, Eb / E0b, A / Eb, (A - Eb) / np.sqrt(Eb), np.bincount(b, g, nb_) / n)
    print(f"  {name:22s} nb {nb_:2d} LRmain {lr:8.1f} (p {row['p']:.1e}) maxdev {row['maxdev']:.3f} | " +
          " ".join(f"{sn} {row[f'I_{sn}']:.0f}/{row[f'I_{sn}_df']}" for sn in S) + f"  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== ranked by LR main / df (what G3 leaves on each input) ===")
for r in sorted(res, key=lambda r: -r["lr"] / (r["nb"] - 1)):
    print(f"  {r['name']:22s} LR {r['lr']:8.1f} df {r['nb']-1:2d} ratio {r['lr']/(r['nb']-1):6.2f} p {r['p']:.1e} maxdev {r['maxdev']:.3f}")
print("\n=== interactions ranked by LR/df (excess over df) ===")
inter = []
for r in res:
    for sn in S:
        inter.append((r["name"], sn, r[f"I_{sn}"], r[f"I_{sn}_df"], r[f"I_{sn}_p"]))
for nm, sn, lr, df, pv in sorted(inter, key=lambda t: -(t[2] - t[3]) / np.sqrt(2 * t[3]))[:40]:
    print(f"  {nm:22s} x {sn:9s} LR {lr:8.1f} df {df:3d} (LR-df)/sqrt(2df) {(lr-df)/np.sqrt(2*df):6.2f} p {pv:.1e}")
print("\n=== bin detail (edges | row share | label lift | model lift | act/exp | z | mean zG3) ===")
for r in sorted(res, key=lambda r: -r["lr"] / (r["nb"] - 1))[:25]:
    q, sh, ll, ml, ae, z, mg = detail[r["name"]]
    print(f"-- {r['name']} (edges {np.round(q, 3).tolist()})")
    for i in range(len(sh)):
        print(f"   bin{i}: share {sh[i]:.3f} label {ll[i]:.3f} model {ml[i]:.3f} act/exp {ae[i]:.3f} z {z[i]:+5.1f} meanG {mg[i]:+.3f}")
np.save(f"{D}/pd_{split}.npy", np.array(res, dtype=object), allow_pickle=True)
print(f"done {time.time()-t0:.0f}s")
