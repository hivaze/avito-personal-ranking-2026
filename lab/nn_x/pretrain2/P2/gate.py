"""P2 step B: nested residual of ||r|| x cos_u and ||r|| x cos(c, r) on G3 (val + holdout replica), ||r|| decile
diagnostic, parity, within-feed Spearman, 'right category, buried' cell AUC. Rule: nn_x/pretrain2/P2/PREDECLARED.txt.
CPU only, 12 threads. Inputs: nn_x/pretrain2/P2/out/{val,hold}_{nr,cosu,coscr}.npy from uresid.py.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, itertools
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"; OUT = ROOT / "nn_x/pretrain2/P2/out"
T0 = time.time()
LOG = open(OUT / "gate.log", "w")
def log(*a):
    s = f"[{time.time()-T0:6.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
RES = {}


@nb.njit(parallel=True)
def nd_nb(s, y, off, M):
    out = np.ones(len(M))
    for g in nb.prange(len(M)):
        o = off[g]; m = M[g]
        ss = s[o:o + m]; yy = y[o:o + m]
        npos = 0
        for k in range(m):
            if yy[k] > 0:
                npos += 1
        if npos == 0:
            continue
        order = np.argsort(-ss)
        dcg = 0.0; i = 0
        while i < m:
            j = i
            while j + 1 < m and ss[order[j + 1]] == ss[order[i]]:
                j += 1
            npg = 0
            for k in range(i, j + 1):
                if yy[order[k]] > 0:
                    npg += 1
            for r in range(j - npg + 1, j + 1):
                dcg += 1.0 / np.log2(r + 2.0)
            i = j + 1
        idcg = 0.0
        for r in range(npos):
            idcg += 1.0 / np.log2(r + 2.0)
        out[g] = dcg / idcg
    return out


def zfeed_nan(x, M):
    """per-feed z over finite entries; NaN -> 0."""
    x = np.asarray(x, np.float64); n = len(M); fid = np.repeat(np.arange(n), M)
    f = np.isfinite(x); xv = np.where(f, x, 0.0)
    c = np.bincount(fid, f, n); mu = np.bincount(fid, xv, n) / np.maximum(c, 1)
    v = np.bincount(fid, np.where(f, (xv - mu[fid]) ** 2, 0.0), n) / np.maximum(c, 1)
    sd = np.sqrt(v); good = (c >= 2) & (sd > 0)
    z = np.where(f & good[fid], (xv - mu[fid]) / np.where(sd > 0, sd, 1)[fid], 0.0)
    return z


def se(d):
    d = np.asarray(d, np.float64); return float(d.std(ddof=1) / np.sqrt(len(d)))


def demean(x, fid, n, M):
    return x - (np.bincount(fid, x, n) / M)[fid]


class Split:
    def __init__(self, name, M, off, y, users, base, nr, cosu, coscr, okf, gatef):
        self.name, self.M, self.off, self.y = name, M, off, y
        self.n = len(M); self.fid = np.repeat(np.arange(self.n), M)
        self.users = users.copy(); u0 = self.users == 0
        self.users[u0] = -np.arange(1, u0.sum() + 1)                       # user-0 singletons
        self.base = base; self.okf = okf; self.gatef = gatef               # okf: fittable (user != 0, finite nr)
        nr_s = np.where(okf, (nr - np.nanmean(nr[okf])) / np.nanstd(nr[okf]), 0.0)
        self.nr = nr; self.nr_s = nr_s
        okr = okf[self.fid]
        x1 = zfeed_nan(cosu, M) * okr
        x3 = nr_s[self.fid] * zfeed_nan(coscr, M) * okr
        self.X = np.stack([x1, nr_s[self.fid] * x1, x3], 1)
        self.zc = {"cos_u": zfeed_nan(cosu, M), "cos_cr": zfeed_nan(coscr, M)}
        self.base_per = nd_nb(base, y, off, M)
        # within-feed demeaned design for the ridge
        self.yd = demean(y.astype(np.float64), self.fid, self.n, M)
        self.Zd = np.stack([demean(base, self.fid, self.n, M)] + [demean(self.X[:, j], self.fid, self.n, M) for j in range(3)], 1)

    def fit_ridge(self, feeds_mask):
        r = feeds_mask[self.fid]
        Z = self.Zd[r]; yd = self.yd[r]; N = len(yd)
        A = Z.T @ Z; b = Z.T @ yd
        A[1:, 1:] += 0.01 * N * np.eye(3)
        beta = np.linalg.solve(A, b)
        return beta[1:] / beta[0], beta

    def score(self, w):
        return self.base + self.X @ w

    def nested(self, seeds=(0, 1)):
        deltas, ws = [], []
        for seed in seeds:
            d = np.zeros(self.n)
            for trn, va in GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(self.n), groups=self.users):
                m = np.zeros(self.n, bool); m[trn] = True; m &= self.okf & self.gatef
                w, _ = self.fit_ridge(m); ws.append(w)
                per = nd_nb(self.score(w), self.y, self.off, self.M)
                d[va] = per[va] - self.base_per[va]
            d[~self.okf] = 0.0
            deltas.append(d)
        return np.mean(deltas, 0), np.array(ws)


# ---------------- val ----------------
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
keep = ~leak_mask()
zG3 = zfeed_nan(np.load(P / "ens_G3_final_val.npy"), Mv)
nrv = np.load(OUT / "val_nr.npy")
okv = (fv["user_id"].to_numpy() != 0) & np.isfinite(nrv)
V = Split("val", Mv, offv, yv, fv["user_id"].to_numpy(), zG3, nrv, np.load(OUT / "val_cosu.npy"), np.load(OUT / "val_coscr.npy"), okv, keep)
chk = ndcg_per_feed(zG3, yv, offv, Mv)
assert np.allclose(chk, V.base_per), "numba NDCG mismatch"
log(f"val: {V.n} feeds, leak-free {keep.sum()}, fittable {okv.sum()} (leak-free & fittable {(okv & keep).sum()}); "
    f"G3 NDCG all {V.base_per.mean():.5f} leak-free {V.base_per[keep].mean():.5f} (numba == metrics)")

# ---------------- holdout ----------------
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
hold = np.load(OUT / "hold_fidx.npy")
Mh, offh = np.load(P / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
assert (ftr["M"].to_numpy()[hold] == Mh).all()
MEM = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)] + [f"r6_cd_m02_r{s}" for s in range(101, 108)] + \
      [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]
acc = 0
for m in MEM:
    acc = acc + zfeed_nan(np.load(P / f"holdout/nn_{m}_holdout.npy"), Mh)
zH = acc / len(MEM)
nrh = np.load(OUT / "hold_nr.npy")
okh = (ftr["user_id"].to_numpy()[hold] != 0) & np.isfinite(nrh)
Hd = Split("holdout", Mh, offh, yh, ftr["user_id"].to_numpy()[hold], zH, nrh, np.load(OUT / "hold_cosu.npy"),
           np.load(OUT / "hold_coscr.npy"), okh, np.ones(len(Mh), bool))
log(f"holdout: {Hd.n} feeds, fittable {okh.sum()}; G3 (20 TO members) NDCG {Hd.base_per.mean():.5f} (recorded 0.32379)")
RES["base"] = dict(val_all=float(V.base_per.mean()), val_leakfree=float(V.base_per[keep].mean()), holdout=float(Hd.base_per.mean()))

# ---------------- ||r|| distribution, parity ----------------
uresid = json.load(open(OUT / "uresid.json"))
log(f"R2 by fold (lambda {uresid['lambda']}): {uresid['r2_by_fold'][str(uresid['lambda'])]}; R2 val {uresid['r2_val']:.4f}, "
    f"test {uresid['r2_test']:.4f}, holdout OOF {uresid['r2_holdout']:.4f}")
log(f"nr quantiles: val {uresid['nr_val']}\n   holdout {uresid['nr_holdout']}\n   train sample {uresid['nr_trainsample']}\n"
    f"   test public {uresid['nr_test_public']}\n   test private {uresid['nr_test_private']}")
par = uresid["parity_ratio_val_over_holdout_median"]
log(f"parity: median nr val / holdout {par:.3f} -> {'OK (within 10%)' if 0.9 <= par <= 1.1 else 'OUTSIDE 10% (format artefact flag)'}")
RES["parity_ratio"] = par

# ---------------- decile diagnostic ----------------
def deciles(S, Hh, pop):
    per = S.base_per; idx = np.where(pop)[0]
    q = np.quantile(S.nr[idx], np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(q, S.nr[idx], side="right") - 1, 0, 9)
    Xc = np.stack([np.ones(len(idx)), np.log(S.M[idx]), np.log1p(Hh[idx])], 1)
    beta = np.linalg.lstsq(Xc, per[idx], rcond=None)[0]; res = per[idx] - Xc @ beta
    rows = []
    for k in range(10):
        m = dec == k
        rows.append(dict(dec=k + 1, n=int(m.sum()), nr_lo=float(q[k]), nr_hi=float(q[k + 1]), ndcg=float(per[idx][m].mean()),
                         ndcg_se=se(per[idx][m]), resid=float(res[m].mean()), resid_se=se(res[m]),
                         M=float(S.M[idx][m].mean()), H=float(Hh[idx][m].mean())))
    hi = res[dec >= 7]; lo = res[dec <= 2]
    con = float(hi.mean() - lo.mean()); con_se = float(np.sqrt(se(hi) ** 2 + se(lo) ** 2))
    rawhi = per[idx][dec >= 7]; rawlo = per[idx][dec <= 2]
    return rows, con, con_se, float(rawhi.mean() - rawlo.mean()), float(np.sqrt(se(rawhi) ** 2 + se(rawlo) ** 2))

Hv = fv["H"].to_numpy().astype(np.float64); Hh = ftr["H"].to_numpy()[hold].astype(np.float64)
for nm, S, Hx, pop in [("val", V, Hv, okv & keep), ("holdout", Hd, Hh, okh)]:
    rows, con, con_se, rcon, rcon_se = deciles(S, Hx, pop)
    RES[f"deciles_{nm}"] = dict(rows=rows, resid_contrast_D8_10_minus_D1_3=(con, con_se), raw_contrast=(rcon, rcon_se))
    log(f"--- G3 NDCG by ||r|| decile, {nm} ({pop.sum()} feeds) ---")
    for r in rows:
        log(f"   D{r['dec']:2d} nr [{r['nr_lo']:.3f},{r['nr_hi']:.3f}] n {r['n']:5d}  NDCG {r['ndcg']:.4f}±{r['ndcg_se']:.4f}  "
            f"resid {r['resid']:+.4f}±{r['resid_se']:.4f}  M {r['M']:.0f} H {r['H']:.0f}")
    log(f"   contrast D8-10 minus D1-3: residual {con:+.5f} ± {con_se:.5f} (z {con/con_se:+.2f}); raw {rcon:+.5f} ± {rcon_se:.5f}")

# ---------------- within-feed Spearman ----------------
@nb.njit(parallel=True)
def feed_spear(a, b, off, M, ok):
    out = np.full(len(M), np.nan)
    for g in nb.prange(len(M)):
        if not ok[g]:
            continue
        o = off[g]; m = M[g]
        aa = a[o:o + m]; bb = b[o:o + m]
        msk = np.isfinite(aa) & np.isfinite(bb)
        k = msk.sum()
        if k < 3:
            continue
        ra = np.argsort(np.argsort(aa[msk])).astype(np.float64); rb = np.argsort(np.argsort(bb[msk])).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        if den > 0:
            out[g] = (ra * rb).sum() / den
    return out

cu_v = np.load(OUT / "val_cosu.npy").astype(np.float64); cr_v = np.load(OUT / "val_coscr.npy").astype(np.float64)
s1 = feed_spear(cr_v, cu_v, offv, Mv, okv)
abl = np.load(ROOT / "nn_x/abl/preds/nn_abl_base_s23_trainonly_val.npy").astype(np.float64)
s2 = feed_spear(cr_v, abl, offv, Mv, okv); s3 = feed_spear(cu_v, abl, offv, Mv, okv)
s4 = feed_spear(cr_v, zG3, offv, Mv, okv)
RES["spearman"] = dict(coscr_cosu=float(np.nanmean(s1)), coscr_abl_base_s23=float(np.nanmean(s2)), cosu_abl_base_s23=float(np.nanmean(s3)),
                       coscr_G3=float(np.nanmean(s4)))
log(f"within-feed Spearman (mean over val feeds): cos_cr~cos_u {np.nanmean(s1):.4f} (q10/q50/q90 {np.nanpercentile(s1,[10,50,90]).round(3)}), "
    f"cos_cr~abl_base_s23 {np.nanmean(s2):.4f}, cos_u~abl_base_s23 {np.nanmean(s3):.4f}, cos_cr~G3 {np.nanmean(s4):.4f}")
log(f"zero-shot NDCG (val leak-free fittable): cos_u {nd_nb(V.zc['cos_u'], yv, offv, Mv)[okv & keep].mean():.4f}, "
    f"cos_cr {nd_nb(V.zc['cos_cr'], yv, offv, Mv)[okv & keep].mean():.4f}, G3 {V.base_per[okv & keep].mean():.4f}")

# ---------------- PRIMARY: nested ridge ----------------
wpriv = pmix_weights()["private"]
dv, wv = V.nested()
dh, wh = Hd.nested()
dvm, dvse = float(dv[keep].mean()), se(dv[keep])
dhm, dhse = float(dh.mean()), se(dh)
pm, pse = wmean_se(dv[keep], wpriv[keep])
log("=== PRIMARY nested 3-parameter ridge (x1 z(cos_u), x2 nr_s*z(cos_u), x3 nr_s*z(cos_cr)) ===")
log(f"   val fold weights (10 fits): mean {wv.mean(0).round(4)} sd {wv.std(0).round(4)}")
log(f"   holdout fold weights: mean {wh.mean(0).round(4)} sd {wh.std(0).round(4)}")
log(f"   VAL OOF delta (14,306 leak-free, excluded at 0): {dvm:+.5f} ± {dvse:.5f} (z {dvm/dvse:+.2f})")
fo = okv & keep
log(f"   VAL OOF delta (fittable leak-free only, n {fo.sum()}): {dv[fo].mean():+.5f} ± {se(dv[fo]):.5f}")
for nm, m in [("M<50", keep & (Mv < 50)), ("M>=50", keep & (Mv >= 50))]:
    log(f"   VAL slice {nm} (n {m.sum()}): {dv[m].mean():+.5f} ± {se(dv[m]):.5f}")
log(f"   VAL pmix (private weights): {pm:+.5f} ± {pse:.5f}")
log(f"   HOLDOUT OOF delta (30,000; excluded at 0): {dhm:+.5f} ± {dhse:.5f} (z {dhm/dhse:+.2f})")
# info: full-val fit transferred to holdout
wfull, bfull = V.fit_ridge(okv & keep)
dtr = nd_nb(Hd.score(wfull), yh, offh, Mh) - Hd.base_per; dtr[~okh] = 0
log(f"   info: full-val weights {wfull.round(4)} -> holdout delta {dtr.mean():+.5f} ± {se(dtr):.5f}")
RES["primary"] = dict(val=(dvm, dvse), val_fittable=(float(dv[fo].mean()), se(dv[fo])), pmix=(float(pm), float(pse)),
                      holdout=(dhm, dhse), w_val_mean=wv.mean(0).tolist(), w_hold_mean=wh.mean(0).tolist(),
                      w_full_val=wfull.tolist(), transfer_val_to_holdout=(float(dtr.mean()), se(dtr)),
                      slices={nm: (float(dv[m].mean()), se(dv[m])) for nm, m in [("M<50", keep & (Mv < 50)), ("M>=50", keep & (Mv >= 50))]})
np.save(OUT / "delta_val.npy", dv); np.save(OUT / "delta_holdout.npy", dh)

# ---------------- secondary: NDCG-grid nested (info only) ----------------
G = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
combos = list(itertools.product(G, G, G))
def grid(S, popmask):
    pers = np.stack([nd_nb(S.score(np.array(c)), S.y, S.off, S.M) for c in combos])
    ds = []
    for seed in (0, 1):
        d = np.zeros(S.n)
        for trn, va in GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(S.n), groups=S.users):
            t = trn[(S.okf & popmask)[trn]]
            best = int(np.argmax(pers[:, t].mean(1)))
            d[va] = pers[best, va] - S.base_per[va]
        d[~S.okf] = 0; ds.append(d)
    return np.mean(ds, 0), combos[int(np.argmax(pers[:, S.okf & popmask].mean(1)))]
t = time.time()
gv, cbv = grid(V, keep); gh, cbh = grid(Hd, np.ones(Hd.n, bool))
log(f"=== secondary NDCG-grid nested (info only): val {gv[keep].mean():+.5f} ± {se(gv[keep]):.5f} (full-fit best {cbv}); "
    f"holdout {gh.mean():+.5f} ± {se(gh):.5f} (full-fit best {cbh})  [{time.time()-t:.0f}s]")
RES["grid"] = dict(val=(float(gv[keep].mean()), se(gv[keep])), holdout=(float(gh.mean()), se(gh)), best_val=cbv, best_hold=cbh)

# ---------------- 'right category, buried' cell (stage0_gate definition) ----------------
fid = V.fid; cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3)))
kf = uk // 64; o = np.lexsort((-lse, kf)); crk = np.empty(len(uk), np.int64); crk[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = crk[inv]
o2 = np.lexsort((yv, -zG3, ck)); wr = np.empty(len(yv), np.int64); wr[o2] = np.arange(len(yv)) - np.searchsorted(ck[o2], ck[o2])
cellpos = (yv > 0) & (pcr == 0) & (wr >= 3)
pi, ni = [], []
for g in np.nonzero(keep)[0]:
    s_, m_ = offv[g], Mv[g]
    pp = np.nonzero(cellpos[s_:s_ + m_])[0] + s_
    if len(pp) == 0:
        continue
    neg = np.nonzero(yv[s_:s_ + m_] == 0)[0] + s_
    for p_ in pp:
        q = neg[(cat[neg] == cat[p_]) & (zG3[neg] > zG3[p_])]
        pi.append(np.full(len(q), p_)); ni.append(q)
pi = np.concatenate(pi); ni = np.concatenate(ni)
okp = okv[fid[pi]]
def auc(d): return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")
FE = ROOT / "data/feats/val"
sig = {"z(cos_u)": V.zc["cos_u"], "z(cos_cr)": V.zc["cos_cr"], "blend(full-val ridge)": V.score(wfull),
       "ctrl -log1p(it_hist_all_luo)": zfeed_nan(-np.log1p(np.load(FE / "it_hist_all_luo.npy").astype(np.float64)), Mv),
       "ctrl -maxsim32": zfeed_nan(-np.load(FE / "maxsim32.npy").astype(np.float64), Mv)}
RES["cell"] = dict(n_pos=int(cellpos[keep[fid]].sum()), n_pairs=int(len(pi)), n_pairs_fittable=int(okp.sum()))
log(f"cell: {cellpos[keep[fid]].sum()} positives, {len(pi)} beater pairs on leak-free feeds ({okp.sum()} on fittable feeds)")
for k_, x in sig.items():
    a = auc((x[pi] - x[ni])[okp]); RES["cell"][k_] = a
    log(f"   pair AUC {k_:32s} {a:.4f}")
json.dump(RES, open(OUT / "gate.json", "w"), indent=1, default=float)

# ---------------- verdict ----------------
task_pass = (dvm >= 0.001) and (dvm / dvse >= 3) and (dhm > 0)
con_v = RES["deciles_val"]["resid_contrast_D8_10_minus_D1_3"][0]
plan_ok = (pm >= 0) and (con_v < 0)
log(f"VERDICT: task gate {'PASS' if task_pass else 'FAIL'} (val {dvm:+.5f} ± {dvse:.5f}, need >= +0.001 and z >= 3; holdout {dhm:+.5f}, need > 0); "
    f"plan conditions pmix>=0 {pm >= 0}, decile contrast<0 {con_v < 0} -> {'PASS' if task_pass and plan_ok else 'CLOSED'}")
