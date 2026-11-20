"""P1 stage B: does trajectory type (label-free clusters) change how a feed should be ranked, beyond G3?
Pre-declared gate: nn_x/pretrain2/P1/PREDECLARED.txt. Needs out/clusters.npz from cluster_build.py. CPU only, 12 threads.
Test 1: nested per-cluster vs global weighting of the 3 G3 families (val leak-free + holdout).
Test 2: nested per-cluster vs global FT/TO mix (val only).
Info: per-cluster G3 NDCG / covariate residual / ANOVA, buried-cell concentration, val->holdout transfer.
Test 3: label-free cluster-share shift val vs test public vs test private.
Outputs: out/cluster_diag.log (stdout), out/cluster_diag.json."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json, itertools
from pathlib import Path
import numpy as np, polars as pl
from scipy import stats
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"; OUT = ROOT / "nn_x/pretrain2/P1/out"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
RES = {}
BOUND = 1790497920
ZTHR = 3.58; DMIN = 0.001; TAU = 1000.0
FAM = {"F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
       "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
       "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]}
FN = list(FAM)
NF = np.array([len(FAM[f]) for f in FN], float)
WG3 = NF / NF.sum()
GRID = [tuple(WG3)] + [(a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)]
GRID = np.array(GRID)
MIX = np.round(np.arange(11) / 10, 2)
C = np.load(OUT / "clusters.npz")


def se(d): d = np.asarray(d, np.float64); return float(d.std(ddof=1) / np.sqrt(len(d)))
def fmt(m, s): return f"{m:+.5f} ± {s:.5f} (z {m / s if s > 0 else 0:+.2f})"


def zf(off, M, x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)


def groups_of(users):
    g = users.astype(np.int64).copy(); z0 = g == 0
    g[z0] = -np.arange(1, z0.sum() + 1)
    return g


def folds_of(groups, seed):
    u, inv = np.unique(groups, return_inverse=True)
    return (np.random.default_rng(seed).permutation(len(u)) % 5)[inv]


# ---------------- val ----------------
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64); nv = len(Mv)
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
gv = groups_of(fv["user_id"].to_numpy()); keep = ~leak_mask()
FT = {f: np.mean([zf(ov, Mv, np.load(P / f"nn_{m}_n2cpv3_val_oof.npy")) for m in FAM[f]], 0) for f in FN}
TO = {f: np.mean([zf(ov, Mv, np.load(P / f"nn_{m}_trainonly_val.npy")) for m in FAM[f]], 0) for f in FN}
def sv(w): return 0.5 * zf(ov, Mv, sum(wi * FT[f] for wi, f in zip(w, FN))) + 0.5 * zf(ov, Mv, sum(wi * TO[f] for wi, f in zip(w, FN)))
NDv = np.stack([ndcg_per_feed(sv(w), yv, ov, Mv) for w in GRID])
g3v = ndcg_per_feed(np.load(P / "ens_G3_final_val.npy"), yv, ov, Mv)
log(f"val: {nv} feeds, leak-free {keep.sum()}; G3 file {g3v.mean():.5f} (leak-free {g3v[keep].mean():.5f}); "
    f"rebuilt G3 point {NDv[0].mean():.5f}; max |per-feed diff| {np.abs(NDv[0] - g3v).max():.2e}; grid {len(GRID)} points")
zft = zf(ov, Mv, np.load(P / "ens_G3_ft_val.npy")); zto = zf(ov, Mv, np.load(P / "ens_G3_to_val.npy"))
NDmix = np.stack([ndcg_per_feed(a * zft + (1 - a) * zto, yv, ov, Mv) for a in MIX])
zG3v = 0.5 * zft + 0.5 * zto
log(f"FT/TO mix grid (val leak-free): " + " ".join(f"{a:.1f}:{NDmix[i][keep].mean():.5f}" for i, a in enumerate(MIX)))
fam_single = {f: float(ndcg_per_feed(0.5 * zf(ov, Mv, FT[f]) + 0.5 * zf(ov, Mv, TO[f]), yv, ov, Mv)[keep].mean()) for f in FN}
log("single-family val (leak-free):", {k: round(v, 5) for k, v in fam_single.items()})
RES["val_base"] = dict(g3=float(g3v[keep].mean()), g3_rebuilt=float(NDv[0][keep].mean()), fam_single=fam_single)

# ---------------- holdout ----------------
hm = np.load(P / "holdout/holdout_meta.npy"); Mh = hm[0].astype(np.int64); oh = hm[1].astype(np.int64); nh = len(Mh)
yh = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
ftr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id", "H"])
hold = C["feeds_hold"]
gh = groups_of(ftr["user_id"].to_numpy()[hold])
Hh = ftr["H"].to_numpy()[hold].astype(np.int64)
HF = {f: np.mean([zf(oh, Mh, np.load(P / f"holdout/nn_{m}_holdout.npy")) for m in FAM[f]], 0) for f in FN}
NDh = np.stack([ndcg_per_feed(sum(wi * HF[f] for wi, f in zip(w, FN)), yh, oh, Mh) for w in GRID])
log(f"holdout: {nh} feeds, {len(np.unique(gh))} user groups, user0 {(ftr['user_id'].to_numpy()[hold] == 0).sum()}; "
    f"G3 point {NDh[0].mean():.5f}; single family " + " ".join(f"{f} {ndcg_per_feed(HF[f], yh, oh, Mh).mean():.5f}" for f in FN))
RES["hold_base"] = dict(g3=float(NDh[0].mean()))
W = pmix_weights(); wpriv = W["private"]; wpub = W["public"]


# ---------------- nested machinery ----------------
def nested(ND, lab, K, sel, ev, groups, tau=None):
    """Per-feed NDCG of the per-cluster arm and of the global arm (both nested), averaged over 2 fold seeds."""
    n = ND.shape[1]; oh_ = np.zeros((n, K)); oh_[np.arange(n), lab] = 1.0
    pc = np.zeros(n); gl = np.zeros(n); picks = []
    for seed in (0, 1):
        fold = folds_of(groups, seed)
        for k in range(5):
            tr = (fold != k) & sel; te = (fold == k) & ev
            S_all = ND[:, tr].sum(1); n_all = tr.sum(); gb = int(np.argmax(S_all))
            S_c = ND[:, tr] @ oh_[tr]; n_c = oh_[tr].sum(0)
            if tau is None:
                obj = np.where(n_c > 0, S_c / np.maximum(n_c, 1), -np.inf)
            else:
                obj = (S_c + tau * (S_all / n_all)[:, None]) / (n_c + tau)
            best = np.where(n_c > 0, np.argmax(obj, 0), gb)
            gl[te] += ND[gb, te] / 2
            pc[te] += ND[best[lab[te]], te] / 2
            picks.append((gb, best.tolist()))
    return pc, gl, picks


def arms(lab_by_key, keys):
    return [(key, K, sel) for key in keys for K in (8, 32) for sel in ("raw", "shrunk")]


rng = np.random.default_rng(0)
LAB = {}
for K in (8, 32):
    for key in ("ue", "hp"):
        LAB[(key, K)] = {s: C[f"{key}{K}_{s}"].astype(np.int64) for s in ("fit", "hold", "val", "test")}
    LAB[("rnd", K)] = {"val": rng.integers(K, size=nv), "hold": rng.integers(K, size=nh)}
assert (C["feeds_val"] == np.arange(nv)).all()

# ---------------- test 1 ----------------
log("=" * 30, "TEST 1: per-cluster vs global family weighting (nested, user-grouped 5-fold x 2 seeds) ", "=" * 30)
RES["test1"] = {}
gate_rows = []
for key in ("ue", "hp", "rnd"):
    for K in (8, 32):
        for sel in ("raw", "shrunk"):
            tau = None if sel == "raw" else TAU
            lv, lh = LAB[(key, K)]["val"], LAB[(key, K)]["hold"]
            pv, gvv, pk = nested(NDv, lv, K, keep, keep, gv, tau)
            ph, ghh, _ = nested(NDh, lh, K, np.ones(nh, bool), np.ones(nh, bool), gh, tau)
            dv = (pv - gvv)[keep]; dh = ph - ghh
            pm, pse = wmean_se(dv, wpriv[keep])
            r = dict(val=(float(dv.mean()), se(dv)), hold=(float(dh.mean()), se(dh)), pmix=(float(pm), float(pse)),
                     val_pc_vs_g3=(float((pv - NDv[0])[keep].mean()), se((pv - NDv[0])[keep])),
                     val_gl_vs_g3=(float((gvv - NDv[0])[keep].mean()), se((gvv - NDv[0])[keep])),
                     hold_pc_vs_g3=(float((ph - NDh[0]).mean()), se(ph - NDh[0])),
                     hold_gl_vs_g3=(float((ghh - NDh[0]).mean()), se(ghh - NDh[0])),
                     global_picks=[GRID[g].round(3).tolist() for g, _ in pk[:3]])
            RES["test1"][f"{key}{K}_{sel}"] = r
            log(f"{key}{K:<3d}{sel:7s} pc-gl val {fmt(*r['val'])} | hold {fmt(*r['hold'])} | priv-mix {fmt(*r['pmix'])} || "
                f"vs G3: val pc {r['val_pc_vs_g3'][0]:+.5f} gl {r['val_gl_vs_g3'][0]:+.5f}; hold pc {r['hold_pc_vs_g3'][0]:+.5f} gl {r['hold_gl_vs_g3'][0]:+.5f}")
            if key != "rnd":
                m, s_ = r["val"]; hm_ = r["hold"][0]
                task = m >= DMIN and m / s_ >= ZTHR and hm_ > 0 and r["pmix"][0] >= -r["pmix"][1]
                plan = task and hm_ >= 0.0005
                gate_rows.append((f"{key}{K}_{sel}", task, plan))
n_task = sum(t for _, t, _ in gate_rows); n_plan = sum(p for _, _, p in gate_rows)
verdict = "PASS" if n_plan else ("PASS (task gate) / FAIL (plan gate)" if n_task else "CLOSED")
RES["gate"] = dict(rows=gate_rows, verdict=verdict)
log(f"GATE (pre-declared): arms passing task gate {n_task}/8, plan gate {n_plan}/8 -> VERDICT {verdict}")

# ---------------- test 2 ----------------
log("=" * 30, "TEST 2: per-cluster vs global FT/TO mix (val only, nested) ", "=" * 30)
RES["test2"] = {}
for key in ("ue", "hp", "rnd"):
    for K in (8, 32):
        for sel in ("raw", "shrunk"):
            lv = LAB[(key, K)]["val"]
            pv, gvv, pk = nested(NDmix, lv, K, keep, keep, gv, None if sel == "raw" else TAU)
            dv = (pv - gvv)[keep]; pm, pse = wmean_se(dv, wpriv[keep])
            g3d = (gvv - NDmix[5])[keep]
            RES["test2"][f"{key}{K}_{sel}"] = dict(val=(float(dv.mean()), se(dv)), pmix=(float(pm), float(pse)),
                                                    gl_vs_g3=(float(g3d.mean()), se(g3d)), global_a=[float(MIX[g]) for g, _ in pk])
            log(f"{key}{K:<3d}{sel:7s} pc-gl val {fmt(dv.mean(), se(dv))} | priv-mix {fmt(pm, pse)} | global a picks {sorted(set(float(MIX[g]) for g, _ in pk))} "
                f"| gl vs G3(a=.5) {g3d.mean():+.5f}")

# ---------------- information: per-cluster G3 level, residual, ANOVA ----------------
log("=" * 30, "INFO: per-cluster G3 NDCG, covariate residual, ANOVA ", "=" * 30)
segv = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet")
cs_v = segv["cold_share"].to_numpy().astype(np.float64)
Hv = fv["H"].to_numpy().astype(np.float64); u0v = (fv["user_id"].to_numpy() == 0).astype(float)
Xv = np.column_stack([np.ones(nv), np.log(Mv), np.log1p(Hv), u0v, cs_v])[keep]
u0h = (ftr["user_id"].to_numpy()[hold] == 0).astype(float)
Xh = np.column_stack([np.ones(nh), np.log(Mh), np.log1p(Hh), u0h])
def resid(X, yy): b = np.linalg.lstsq(X, yy, rcond=None)[0]; return yy - X @ b
rv = resid(Xv, NDv[0][keep]); rh = resid(Xh, NDh[0])
def anova(r, lab, K):
    n = len(r); mu = r.mean(); cnt = np.bincount(lab, minlength=K); m_ = np.bincount(lab, r, K) / np.maximum(cnt, 1)
    ssb = (cnt * (m_ - mu) ** 2).sum(); ssw = ((r - m_[lab]) ** 2).sum(); kk = (cnt > 0).sum()
    F = (ssb / (kk - 1)) / (ssw / (n - kk)); return float(F), float(stats.f.sf(F, kk - 1, n - kk)), float(ssb / (ssb + ssw)), m_, cnt
RES["info_level"] = {}
for key in ("ue", "hp", "rnd"):
    for K in (8, 32):
        lv = LAB[(key, K)]["val"][keep]; lh = LAB[(key, K)]["hold"]
        Fv, pv_, etav, mv_, cv_ = anova(rv, lv, K); Fh, ph_, etah, mh_, ch_ = anova(rh, lh, K)
        ok = (cv_ >= 30) & (ch_ >= 30)
        corr = float(np.corrcoef(mv_[ok], mh_[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        lvl_v = np.bincount(lv, NDv[0][keep], K) / np.maximum(cv_, 1); lvl_h = np.bincount(lh, NDh[0], K) / np.maximum(ch_, 1)
        RES["info_level"][f"{key}{K}"] = dict(F_val=Fv, p_val=pv_, eta2_val=etav, F_hold=Fh, p_hold=ph_, eta2_hold=etah,
                                              corr_resid_val_hold=corr, n_clusters_corr=int(ok.sum()),
                                              lvl_val=lvl_v.round(5).tolist(), lvl_hold=lvl_h.round(5).tolist(),
                                              res_val=mv_.round(5).tolist(), res_hold=mh_.round(5).tolist(),
                                              n_val=cv_.tolist(), n_hold=ch_.tolist())
        log(f"{key}{K:<3d} residual ANOVA val F {Fv:.2f} p {pv_:.2g} eta2 {etav:.4f} | hold F {Fh:.2f} p {ph_:.2g} eta2 {etah:.4f} | "
            f"corr(cluster resid val, hold) {corr:+.3f} over {ok.sum()} clusters; G3 level range val {lvl_v[cv_ >= 30].min():.4f}-{lvl_v[cv_ >= 30].max():.4f} "
            f"hold {lvl_h[ch_ >= 30].min():.4f}-{lvl_h[ch_ >= 30].max():.4f}")
        if K == 8 and key != "rnd":
            for c in range(K):
                sv_ = se(NDv[0][keep][lv == c]) if cv_[c] > 1 else float("nan"); sh_ = se(NDh[0][lh == c]) if ch_[c] > 1 else float("nan")
                log(f"    c{c}: val n {cv_[c]:5d} G3 {lvl_v[c]:.4f}±{sv_:.4f} resid {mv_[c]:+.4f} | hold n {ch_[c]:5d} G3 {lvl_h[c]:.4f}±{sh_:.4f} resid {mh_[c]:+.4f}")

# ---------------- information: full-fit per-cluster weights (descriptive) + val->holdout transfer ----------------
log("=" * 30, "INFO: full-fit per-cluster family weights on val leak-free, transferred to holdout ", "=" * 30)
RES["transfer"] = {}
for key in ("ue", "hp"):
    for K in (8, 32):
        lv = LAB[(key, K)]["val"]; lh = LAB[(key, K)]["hold"]
        S_all = NDv[:, keep].sum(1); gb = int(np.argmax(S_all)); n_all = keep.sum()
        oh_ = np.zeros((nv, K)); oh_[np.arange(nv), lv] = 1
        S_c = NDv[:, keep] @ oh_[keep]; n_c = oh_[keep].sum(0)
        out = {}
        for sel in ("raw", "shrunk"):
            obj = S_c / np.maximum(n_c, 1) if sel == "raw" else (S_c + TAU * (S_all / n_all)[:, None]) / (n_c + TAU)
            best = np.where(n_c > 0, np.argmax(obj, 0), gb)
            d = NDh[best[lh], np.arange(nh)] - NDh[gb]
            out[sel] = dict(hold=(float(d.mean()), se(d)), weights=[GRID[b].round(2).tolist() for b in best])
            log(f"{key}{K:<3d}{sel:7s} val-global w {GRID[gb].round(2).tolist()} (val {S_all[gb]/n_all:.5f} vs G3 {S_all[0]/n_all:.5f}); "
                f"transfer per-cluster - global on holdout {fmt(*out[sel]['hold'])}")
            if K == 8:
                log("      per-cluster w (F10,F7x,F7x70): " + " ".join(f"c{c}:{GRID[b].round(1).tolist()}" for c, b in enumerate(best)))
        RES["transfer"][f"{key}{K}"] = out

# ---------------- information: buried-cell concentration (val) ----------------
log("=" * 30, "INFO: 'right category, buried' cell concentration by cluster (val leak-free) ", "=" * 30)
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64); fid = np.repeat(np.arange(nv), Mv)
ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3v)))
kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv]
o2 = np.lexsort((yv, -zG3v, ck)); wr = np.empty(len(yv), np.int64); wr[o2] = np.arange(len(yv)) - np.searchsorted(ck[o2], ck[o2])
cellpos = (yv > 0) & (pcr == 0) & (wr >= 3) & keep[fid]
pos = (yv > 0) & keep[fid]
log(f"cell positives {cellpos.sum():,} of {pos.sum():,} leak-free positives ({cellpos.sum()/pos.sum():.3f})")
RES["cell"] = {}
for key in ("ue", "hp", "rnd"):
    for K in (8, 32):
        lab = LAB[(key, K)]["val"][fid]
        cp = np.bincount(lab[cellpos], minlength=K); pp = np.bincount(lab[pos], minlength=K)
        frac = cp / np.maximum(pp, 1); ok = pp >= 50
        chi2, pchi, _, _ = stats.chi2_contingency(np.vstack([cp[ok], (pp - cp)[ok]]))
        RES["cell"][f"{key}{K}"] = dict(frac=frac.round(4).tolist(), n_pos=pp.tolist(), chi2=float(chi2), p=float(pchi))
        log(f"{key}{K:<3d} cell share of positives by cluster: min {frac[ok].min():.3f} max {frac[ok].max():.3f} "
            f"(overall {cellpos.sum()/pos.sum():.3f}); chi2 {chi2:.1f} p {pchi:.2g}")

# ---------------- test 3: shift diagnostic ----------------
log("=" * 30, "TEST 3: cluster composition val vs test public vs test private (label-free) ", "=" * 30)
tsT = pl.read_parquet(FLAT / "test/feeds.parquet", columns=["timestamp"])["timestamp"].to_numpy()
pubm = tsT < BOUND
def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b): a_ = a[a > 0]; return float((a_ * np.log(a_ / b[a > 0])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
RES["shift"] = {}
log(f"test public {pubm.sum()} feeds, private {(~pubm).sum()}")
for key in ("ue", "hp"):
    for K in (8, 32):
        L = LAB[(key, K)]
        sets = {"fit": L["fit"], "hold": L["hold"], "val": L["val"], "val_lf": L["val"][keep], "pub": L["test"][pubm], "priv": L["test"][~pubm]}
        sh = {s: np.bincount(v, minlength=K) / len(v) for s, v in sets.items()}
        nn_ = {s: len(v) for s, v in sets.items()}
        J = {f"{a}-{b}": js(sh[a], sh[b]) for a, b in [("val", "pub"), ("val", "priv"), ("pub", "priv"), ("fit", "val"), ("hold", "val"), ("fit", "priv")]}
        chi_pp = stats.chi2_contingency(np.vstack([np.bincount(sets["pub"], minlength=K), np.bincount(sets["priv"], minlength=K)]) + 0)
        chi_vp = stats.chi2_contingency(np.vstack([np.bincount(sets["val"], minlength=K), np.bincount(sets["priv"], minlength=K)]) + 0)
        dpv = sh["priv"] - sh["val"]; sdp = np.sqrt(sh["priv"] * (1 - sh["priv"]) / nn_["priv"] + sh["val"] * (1 - sh["val"]) / nn_["val"])
        dpp = sh["priv"] - sh["pub"]; sdq = np.sqrt(sh["priv"] * (1 - sh["priv"]) / nn_["priv"] + sh["pub"] * (1 - sh["pub"]) / nn_["pub"])
        top = np.argsort(-np.abs(dpv))[:3]; topq = np.argsort(-np.abs(dpp))[:3]
        # val G3 NDCG reweighted to the public / private cluster mix (label-free weights on val leak-free feeds)
        lvk = L["val"][keep]; base = NDv[0][keep]
        rw = {}
        for nm, s in (("pub", "pub"), ("priv", "priv")):
            w = (sh[s] / np.maximum(sh["val_lf"], 1e-12))[lvk]
            m_, s_ = wmean_se(base, w)
            dsh = (base - base.mean()) * (w / w.mean() - 1)  # weighted - unweighted mean as a per-feed contribution
            rw[nm] = dict(level=(float(m_), float(s_)), shift=(float(m_ - base.mean()), se(dsh)))
        pm_pub = wmean_se(base, wpub[keep]); pm_priv = wmean_se(base, wpriv[keep])
        RES["shift"][f"{key}{K}"] = dict(shares={s: v.round(4).tolist() for s, v in sh.items()}, n=nn_, js=J,
                                         chi2_pub_priv=(float(chi_pp[0]), float(chi_pp[1])), chi2_val_priv=(float(chi_vp[0]), float(chi_vp[1])),
                                         reweighted=rw, pmix_level=dict(pub=pm_pub, priv=pm_priv), val_lf_level=float(base.mean()))
        log(f"{key}{K:<3d} JS " + " ".join(f"{k} {v:.4f}" for k, v in J.items()) +
            f" | chi2 pub-priv {chi_pp[0]:.1f} p {chi_pp[1]:.2g}; val-priv {chi_vp[0]:.1f} p {chi_vp[1]:.2g}")
        log(f"      largest priv-val share shifts: " + "; ".join(f"c{c} {sh['val'][c]:.3f}->{sh['priv'][c]:.3f} ({dpv[c]:+.3f}±{sdp[c]:.3f})" for c in top))
        log(f"      largest priv-pub share shifts: " + "; ".join(f"c{c} {sh['pub'][c]:.3f}->{sh['priv'][c]:.3f} ({dpp[c]:+.3f}±{sdq[c]:.3f})" for c in topq))
        log(f"      val G3 {base.mean():.5f}; cluster-mix reweighted: pub {rw['pub']['level'][0]:.5f} (shift {fmt(*rw['pub']['shift'])}), "
            f"priv {rw['priv']['level'][0]:.5f} (shift {fmt(*rw['priv']['shift'])}); pmix levels pub {pm_pub[0]:.5f} priv {pm_priv[0]:.5f}")

json.dump(RES, open(OUT / "cluster_diag.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("done; json", OUT / "cluster_diag.json")
