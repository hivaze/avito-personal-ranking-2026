"""A5: "FT on top of the pool", the cheapest form: a linear listwise (softmax-CE) residual on the 42 n2c_pre_v3 FT
extras (within-feed z), fitted on val labels on top of a fixed pool score, nested over the exact FT user folds
(outer fold k held out; L2 chosen by inner 4-fold CV on the training folds). CPU, ~45 parameters, not an NN.
Variants: base = G3 final / G3 TO-only; feature sets = all extras / pre_* only / LUO popularity only / ctx+seq only.
Writes a5_linear_residual.csv; stdout -> a5_linear_residual.log"""
from scipy.optimize import minimize
from lib import *  # noqa

vm = meta("val"); fold = user_folds(); fid = vm["fid"]; y = vm["y"]; n = vm["n"]
FEATS = None
for ln in open(ROOT / "nn/logs/v17_pre2s_s101_n2cpv3_ft.log"):
    if "extra features (" in ln:
        FEATS = eval(ln[ln.index("["):]); break
FC = ROOT / "nn/feat_cache"


def load_X(split, names):
    cols = []
    for nm in names:
        x = np.nan_to_num(np.load(FC / split / f"{nm}.npy").astype(np.float64), nan=0.0)
        if x.min() >= 0 and x.max() > 20:
            x = np.log1p(x)
        z = zfeed(x, split); z[~np.isfinite(z)] = 0
        cols.append(z)
    return np.stack(cols, 1)


Xall = load_X("val", FEATS)
cold = cand_cold("val").astype(np.float64)
npos = np.bincount(fid, y, n)
has = npos > 0


def make_obj(rows_mask, base, X, lam):
    """softmax-CE over feeds with >= 1 positive, restricted to candidates of feeds in rows_mask"""
    cm = rows_mask[fid] & has[fid]
    b, Xs, ys, fs = base[cm], X[cm], y[cm], fid[cm]
    nf = n
    wpos = ys / npos[fs]
    nfeeds = int((rows_mask & has).sum())

    def f(theta):
        a, w = theta[0], theta[1:]
        s = np.clip(a * b + Xs @ w, -60, 60)  # per-feed z inputs: no max-shift needed in float64
        e = np.exp(s); Z = np.bincount(fs, e, nf)
        p = e / Z[fs]
        loss = -(wpos * (s - np.log(Z[fs]))).sum() / nfeeds + lam * (w @ w)
        g_s = (p - wpos) / nfeeds
        grad = np.r_[g_s @ b, Xs.T @ g_s + 2 * lam * w]
        return loss, grad
    return f


def fit(rows_mask, base, X, lam):
    th0 = np.r_[1.0, np.zeros(X.shape[1])]
    r = minimize(make_obj(rows_mask, base, X, lam), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    return r.x


LAMS = [1e-4, 1e-3, 1e-2, 1e-1]
groups = {"all42": list(range(len(FEATS))),
          "pre_only": [j for j, nm in enumerate(FEATS) if nm.startswith("pre_")],
          "luo_only": [j for j, nm in enumerate(FEATS) if nm.startswith("it_hist")],
          "ctx_seq": [j for j, nm in enumerate(FEATS) if nm.startswith(("ctx_", "seq_", "centrality"))]}
bases = {"G3fin": final(POOLS["G3"], "val"), "G3to": zfeed(branch(POOLS["G3"], "val", "to"), "val")}
ref = pf(bases["G3fin"])
import sys
only = sys.argv[1:]  # optional "base:feats" filters to run variants in parallel processes
rows = []
for bn, base in bases.items():
    nb = pf(base)
    for gn, idx in groups.items():
        if only and f"{bn}:{gn}" not in only:
            continue
        X = np.c_[Xall[:, idx], cold] if gn == "all42" else Xall[:, idx]
        oof = np.zeros(len(y)); lams = []
        for k in range(5):
            trm = fold != k
            # inner CV for lambda
            j = (k + 1) % 5  # single inner validation fold for lambda
            sc = []
            for lam in LAMS:
                th = fit(trm & (fold != j), base, X, lam)
                s = th[0] * base + X @ th[1:]
                sc.append(pf(s)[fold == j].mean())
            lam = LAMS[int(np.argmax(sc))]; lams.append(lam)
            th = fit(trm, base, X, lam)
            cmask = (fold == k)[fid]
            oof[cmask] = (th[0] * base + X @ th[1:])[cmask]
        nd = pf(oof)
        r = dict(base=bn, feats=gn, nested_vs_base=pse(nd - nb)[0], se_b=pse(nd - nb)[1], nested_vs_G3=pse(nd - ref)[0],
                 se_g=pse(nd - ref)[1], lams=",".join(map(str, lams)))
        for var in ["M50", "day", "pos_cold(label)"]:
            S = segments("val")[var]
            for l in np.unique(S):
                r[f"{var}={l}"] = (nd - nb)[S == l].mean()
        rows.append(r)
        print(f"base {bn:5s} + linear[{gn:8s}] nested vs base {r['nested_vs_base']:+.5f}+-{r['se_b']:.5f} | vs G3 "
              f"{r['nested_vs_G3']:+.5f}+-{r['se_g']:.5f} | lams {r['lams']} | " +
              " ".join(f"{k}:{v:+.5f}" for k, v in r.items() if "=" in k), flush=True)
tag = "_".join(only).replace(":", "-") if only else "all"
pl.DataFrame(rows).write_csv(OUT / f"a5_linear_residual_{tag}.csv")
if only and "G3fin:all42" not in only:
    sys.exit(0)
# full-val fit coefficients for interpretation (all42 on G3fin)
X = np.c_[Xall, cold]
th = fit(np.ones(n, bool), bases["G3fin"], X, 1e-3)
order = np.argsort(-np.abs(th[1:]))[:12]
names = FEATS + ["cold_flag"]
print("full-val coefficients (G3fin base, lam 1e-3): a", round(th[0], 3), "|", ", ".join(f"{names[j]} {th[1 + j]:+.3f}" for j in order))
