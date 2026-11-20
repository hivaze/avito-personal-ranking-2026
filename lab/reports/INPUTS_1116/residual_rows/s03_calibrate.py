"""INVENTORY B, step 3: calibration of the uniform lens (lens.py).
(a) POSITIVE CONTROLS: an input block the ranker really reads, offered to the pool of members trained WITHOUT it.
    True value = NDCG(pool of the with-input twins) - NDCG(pool of the members without the block), measured here from the stored predictions.
(b) PLACEBOS on the current pool (16 held-out wide students, train-only): N(0,1) columns and within-feed permutations of real columns.
Usage: s03_calibrate.py val|hold      Output: out/calib_<split>.json + out/s03_calibrate_<split>.log (stdout)
CPU only, 12 threads, read-only."""
import sys, json, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
from lens import *

sp = sys.argv[1]; t0 = time.time()
FE = [ROOT / "data/feats", ROOT / "data/feats_new"]
POP8 = "seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo".split(",")
PRE2 = "pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32".split(",")
CTXND = "ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,ctx_nd_top5u,ctx_nd_ucent".split(",")
CTXHN = "ctx_hn_bestrk,ctx_hn_e1_rrw,ctx_hn_margin,ctx_hn_rrw,ctx_hn_soft05,ctx_hn_top1w,ctx_hs_e1,ctx_hs_k128,ctx_hs_mutual,ctx_hs_t10".split(",")


def pf(v, suf=""):
    """prediction file of a member on this split"""
    d = W5 if v.startswith("w5_") else AB
    return d / (f"nn_{v}{suf}_trainonly_val.npy" if sp == "val" else f"holdout/nn_{v}{suf}_holdout.npy")


CONTROLS = [
    # tag, members WITHOUT the block, with-input twins, columns, note
    ("pop8_hard10ep", [f"abl_e10_xdrop_pop_s{s}" for s in (951, 952, 953, 954)], [f"w5_e10_base_s{s}" for s in (951, 952, 953, 954)], POP8,
     "8 popularity / co-visitation inputs, hard-label 10-epoch members, 4 seeds (the control of COVERAGE_MAP l.323)"),
    ("pop8_student_w192", [f"abl_e10_pop1kd05w192_s{s}" for s in (951, 952, 953, 954)], [f"w5_e10_kd05w192_s{s}" for s in (951, 952, 953, 954)], POP8,
     "the same 8 inputs under the CURRENT recipe: popularity-light wide students (which read xp_imp_c50 instead) vs full-view wide students"),
    ("pre2_hard10ep", [f"abl_e10_xdrop_pre2_s{s}" for s in (951, 952)], [f"w5_e10_base_s{s}" for s in (951, 952)], PRE2,
     "5 encoder scalars (+ flag), hard-label 10-epoch members, 2 seeds"),
    ("ctxnd_hard2ep", [f"abl_xdrop_ctxnd_s{s}" for s in (23, 24)], [f"abl_base_s{s}" for s in (23, 24)], CTXND, "10 ctx_nd inputs, 2-epoch members, 2 seeds"),
    ("ctxhn_hs_hard2ep", [f"abl_xdrop_ctxhn_hs_s{s}" for s in (23, 24)], [f"abl_base_s{s}" for s in (23, 24)], CTXHN, "ctx_hn + ctx_hs (10 inputs), 2-epoch members, 2 seeds"),
]


def col(name, D):
    for d in FE:
        p = d / ("val" if sp == "val" else "train") / f"{name}.npy"
        if p.exists():
            x = np.load(p, mmap_mode="r")
            return np.asarray(x if D["rows"] is None else x[D["rows"]], np.float64)
    raise FileNotFoundError(name)


res = dict(split=sp, controls={}, placebo=[])
D = load_split(sp, "TO16")
print(f"[{time.time() - t0:.0f}s] split {sp}: {D['n']} feeds, {len(D['y'])} rows; pool of the 16 held-out wide students (train-only) NDCG {D['ndcg0'].mean():.5f}", flush=True)
if sp == "val":
    import pmix
    D["wpriv"] = pmix.weights()["private"]
res["base_ndcg"] = float(D["ndcg0"].mean())

# ---------------- (a) positive controls
for tag, abl, twin, cols, note in CONTROLS:
    if not all(pf(v).exists() for v in abl + twin):
        print(f"control {tag}: prediction files missing on {sp}, skipped"); continue
    Dc = dict(D); Dc.pop("fit0", None); Dc.pop("fit0_fold", None)
    Dc["base"] = zpool([pf(v) for v in abl], D["off"], D["M"]); Dc["ndcg0"] = ndcg(Dc["base"], Dc)
    tw = ndcg(zpool([pf(v) for v in twin], D["off"], D["M"]), Dc); true = tw - Dc["ndcg0"]
    out = dict(note=note, n_members=len(abl), ndcg_without=float(Dc["ndcg0"].mean()), ndcg_with=float(tw.mean()), true=float(true.mean()), true_se=float(true.std() / np.sqrt(len(true))), cols={})
    print(f"\n== CONTROL {tag} ({note})\n   pool without the block {Dc['ndcg0'].mean():.5f} | with-input twins {tw.mean():.5f} | TRUE trained-input value {fmt(true)} (z {true.mean() / (true.std() / np.sqrt(len(true))):+.1f})", flush=True)
    Fz = []
    for c in cols:
        F = forms(col(c, Dc), Dc); r = read_column(Dc, F=F); out["cols"][c] = {k: (float(v) if not isinstance(v, (str, bool)) else v) for k, v in r.items()}; Fz.append(F["z"])
        print(f"   {c:20s} LR_z {r['LR_z']:8.1f} coef {r['coef_z']:+.4f} zrob {r['zrob_z']:+6.2f} | LR_rk {r['LR_rk']:8.1f} | nats/feed {r['nats_z']:+.5f} +- {r['nats_se']:.5f} | nested NDCG {r['nest']:+.5f} +- {r['nest_se']:.5f}"
              + (f" | LR_z+nan {r['LR_znan']:8.1f}" if "LR_znan" in r else ""), flush=True)
    f0 = base_fit(Dc); rb = fit(Dc, Fz, theta0=[f0["theta"][0]] + [0.0] * len(Fz)); dl = rb["ll"] - f0["ll"]
    d = nested(Dc, Fz, rb["theta"])
    out["block"] = dict(k=len(Fz), LR=float(2 * (rb["L"] - f0["L"])), nats=float(dl.mean()), nats_se=float(dl.std() / np.sqrt(len(dl))), nest=float(d.mean()), nest_se=float(d.std() / np.sqrt(len(d))))
    if "wpriv" in D:
        out["block"]["nest_pmix"] = float(np.average(d, weights=D["wpriv"]))
    b = out["block"]
    print(f"   BLOCK ({b['k']} columns, z form, joint): LR {b['LR']:.1f} ({b['k']} df) | nats/feed {b['nats']:+.5f} +- {b['nats_se']:.5f} | nested NDCG {b['nest']:+.5f} +- {b['nest_se']:.5f}"
          f" = {b['nest'] / out['true']:.2f} of the true value; best single-column nested = {max(v['nest'] for v in out['cols'].values()) / out['true']:.2f} of true", flush=True)
    res["controls"][tag] = out
    json.dump(res, open(OUT / f"calib_{sp}.json", "w"), indent=1)

# ---------------- (b) placebos on the current pool
@nb.njit(cache=True)
def perm_within(x, off, M, seed):
    np.random.seed(seed); out = x.copy()
    for f in range(len(M)):
        o = off[f]; m = M[f]
        for i in range(m - 1, 0, -1):
            j = np.random.randint(0, i + 1); t = out[o + i]; out[o + i] = out[o + j]; out[o + j] = t
    return out

NPL = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rng = np.random.default_rng(20261116)
print(f"\n== PLACEBOS on the current pool ({sp}): {NPL} N(0,1) columns + {NPL} within-feed permutations of real columns", flush=True)
for i in range(NPL):
    x = rng.standard_normal(len(D["y"])); r = read_column(D, x=x, do_rk=False); r["kind"] = "normal"; res["placebo"].append({k: (float(v) if not isinstance(v, (str, bool)) else v) for k, v in r.items()})
REAL = ["f4", "f5", "cos_u", "seq_max_all", "it_hist_all_luo", "tte_it_ctr", "trd_kde_sday", "clu_pmi1024_mean", "ctx_nd_knn_h", "cat_cnt_all", "ixh_good", "seq_cv_invd20", "pre2_fe_z", "tte_it_imp", "attn32"]
Zr = {}
for i in range(NPL):
    c = REAL[i % len(REAL)]
    if c not in Zr: Zr[c] = forms(col(c, D), D)["z"]
    x = perm_within(Zr[c].astype(np.float64), D["off"], D["M"], 1000 + i); r = read_column(D, x=x, do_rk=False); r["kind"] = "perm:" + c
    res["placebo"].append({k: (float(v) if not isinstance(v, (str, bool)) else v) for k, v in r.items()})
    if i % 25 == 0: print(f"   [{time.time() - t0:.0f}s] placebo {i}", flush=True)
for kind in ("normal", "perm"):
    P_ = [p for p in res["placebo"] if p["kind"].startswith(kind) and "LR_z" in p]
    lr = np.array([p["LR_z"] for p in P_]); zr = np.array([p["zrob_z"] for p in P_]); ne = np.array([p["nest"] for p in P_]); nse = np.array([p["nest_se"] for p in P_])
    print(f"   {kind:6s} n {len(P_)}: LR mean {lr.mean():.2f} (chi2_1: 1.00) p95 {np.quantile(lr, 0.95):.2f} (3.84) max {lr.max():.2f} | zrob sd {zr.std():.2f} max|z| {np.abs(zr).max():.2f}"
          f" | nested NDCG mean {ne.mean():+.5f} sd {ne.std():.5f} max {ne.max():+.5f} min {ne.min():+.5f} mean SE {nse.mean():.5f} | share nested z >= 2: {(ne / np.maximum(nse, 1e-12) >= 2).mean():.3f}")
json.dump(res, open(OUT / f"calib_{sp}.json", "w"), indent=1)
print(f"[{time.time() - t0:.0f}s] done", flush=True)
