"""(b) Nested additive residual tests on G3: s = g + w * x, x = per-feed z of a label-free construction.
w chosen per outer fold on the other 4 folds (user-grouped 5-fold, fixed seed 0) from grid W; OOF per-feed NDCG delta vs G3.
Holdout: w chosen on ALL val (w_full) applied to the 30K in-period holdout (g = train-only pool z). Rule (task): a
construction is information G3 lacks only if OOF > 2 SE AND > +0.0003 (project bar) AND holdout z >= 0 (if computable).
Usage: python resid.py <group>   (groups: A B C D)
"""
import sys, time
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, fpct, group_pct, fsum, folds5, ndcg, wmean_se, pmix_w, D  # noqa
from featmat import entropy_by_feed, hist_aggs, decayed_catshare, XT_REL_DONE

t0 = time.time()
grp = sys.argv[1]
dv = load("val"); dh = load("hold")
fold = folds5(dv, seed=0)
W = [-0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
PW = pmix_w()["private"]
Mv = dv["M"]; tue = (dv["ts"].astype(np.int64) + 10800) // 86400 < 20720
SL = [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue)]
base_v = ndcg(dv, dv["g"]); base_h = ndcg(dh, dh["g"])


def cat_rank_desc(d, x):
    """0-based rank of x (descending) within (feed, category)."""
    key = d["fid"].astype(np.int64) * 64 + d["cat"].astype(np.int64)
    o = np.lexsort((-np.asarray(x, np.float64), key)); ks = key[o]
    st = np.r_[True, ks[1:] != ks[:-1]]; start = np.flatnonzero(st)[np.cumsum(st) - 1]
    r = np.empty(len(x)); r[o] = np.arange(len(o)) - start
    return r


def lastk_samecat_type(d, k, types):
    nf = d["nf"]; hoff = d["hoff"]; Hc = d["Hc"]; hcat = d["hcat"].astype(np.int64); het = d["het"].astype(np.int64)
    hfid = np.repeat(np.arange(nf), Hc); pos = np.arange(len(hfid)) - np.repeat(hoff, Hc)
    m = (pos < k) & np.isin(het, types)
    c = np.bincount(hfid[m] * 64 + hcat[m], minlength=nf * 64)
    return np.log1p(c[d["fid"].astype(np.int64) * 64 + d["cat"].astype(np.int64)])


def coldp(d):
    return (d["xt_it_hist_all_luo"] <= 0).astype(np.float64)


def constructions(d, g):
    """dict name -> raw row-level construction (label-free; own feed/history only)."""
    C = {}
    fid = d["fid"]; M = d["M"]
    if g == "A":  # affinity-contrast family from the PD lens
        aff = ["f3", "d_fcs", "d_cshare", "d_cat_last32", "xt_ctx_hs_k128", "d_smax", "xt_ctx_hn_margin", "xt_ctx_nd_knn_u"]
        P = {a: fpct(d, d[a]) for a in aff}
        for a in aff:
            C[f"z_{a}"] = d[a]
            C[f"bot20_{a}"] = np.minimum(P[a] - 0.2, 0)        # penalise the bottom 20% only
            C[f"top20_{a}"] = np.maximum(P[a] - 0.8, 0)        # boost the top 20% only
        comp = np.mean([P[a] for a in aff], 0)
        C["aff_comp"] = comp
        C["aff_comp_bot20"] = np.minimum(fpct(d, comp) - 0.2, 0)
        C["aff_comp_sq"] = (fpct(d, comp) - 0.5) ** 2
    elif g == "B":  # aggregates
        C["g_catrank_neglog"] = -np.log1p(cat_rank_desc(d, d["g"]))
        C["g_catleader"] = (cat_rank_desc(d, d["g"]) == 0).astype(np.float64)
        key = fid.astype(np.int64) * 64 + d["cat"].astype(np.int64)
        for n, x in [("f5", d["f5"]), ("f4", d["f4"]), ("f1", d["f1"]), ("pop", d["xt_seq_cv_pop"]), ("luo", d["xt_it_hist_all_luo"]),
                     ("cos_u", d["d_cos_u"]), ("smax", d["d_smax"]), ("pre2fe", d["xt_pre2_fe_s"])]:
            C[f"incat_pct_{n}"] = group_pct(d, key, x)[0]
        C["inject_ratio"] = np.log((d["d_fcs"] + 1.0 / M[fid]) / (d["d_cshare"] + 0.01))
        ent = entropy_by_feed(d, d["cat"].astype(np.int64), 64)[fid]
        C["fcs_x_lowent"] = d["d_fcs"] * (ent.max() - ent)
        hent, hncat, sh = hist_aggs(d)
        C["cshare_x_hent"] = d["d_cshare"] * hent[fid]
        C["cat_last8"] = d["d_cat_last8"]; C["cat_last32"] = d["d_cat_last32"]
        for hl in (4, 16, 64):
            C[f"cat_decay{hl}"] = decayed_catshare(d, hl)
        C["samecat_last32_contact_e2e3"] = lastk_samecat_type(d, 32, [2, 3])
        C["samecat_last32_e4"] = lastk_samecat_type(d, 32, [4])
        C["samecat_last8_e1"] = lastk_samecat_type(d, 8, [1])
        for n in ["f5", "f4", "seq_cv_pop", "it_hist_all_luo", "d_cos_u", "d_smax"]:
            x = d[n] if n in d else d["xt_" + n]
            x = np.asarray(x, np.float64)
            mu = fsum(d, x) / M; s = np.sqrt(np.maximum(fsum(d, x * x) / M - mu ** 2, 0))
            C[f"{n}_x_feedsd"] = fz(d, x) * s[fid]          # relative position scaled by the feed's spread
    elif g == "C":  # within-feed relative versions of the xt inputs that lack them
        for n in [k for k in d if k.startswith("xt_") and k[3:] not in XT_REL_DONE and k != "xt_has_pre2_fe"]:
            C[f"rel_{n[3:]}"] = d[n]                       # fz() is applied below
    elif g == "D":  # interactions suggested by the PD lens (modifier centred on val mean)
        cp = coldp(d); fcp = (fsum(d, cp) / M)[fid]
        lM = np.log(M)[fid]; u0 = d["u0"][fid].astype(np.float64)
        C["ctx_hn_e1_rrw_x_feedcold"] = fz(d, d["xt_ctx_hn_e1_rrw"]) * (fcp - 0.25)
        C["ctx_hs_e1_x_feedcold"] = fz(d, d["xt_ctx_hs_e1"]) * (fcp - 0.25)
        C["ctx_hs_t10_x_feedcold"] = fz(d, d["xt_ctx_hs_t10"]) * (fcp - 0.25)
        C["pre2_hmax32_x_logM"] = fz(d, d["xt_pre2_hmax32"]) * (lM - 4.4)
        C["smax8_x_logM"] = fz(d, d["d_smax8"]) * (lM - 4.4)
        C["f3pct_x_u0"] = fz(d, d["f3"]) * (u0 - 0.1)
        C["knnf1_x_u0"] = fz(d, d["xt_ctx_nd_knn_f1"]) * (u0 - 0.1)
        C["pre2fez_x_coldrow"] = d["xt_pre2_fe_z"] * cp
        C["pre2fes_x_coldrow"] = fz(d, d["xt_pre2_fe_s"]) * cp
        C["top5h_x_coldrow"] = fz(d, d["xt_ctx_nd_top5h"]) * cp
        C["cos_u_x_u0"] = fz(d, d["d_cos_u"]) * (u0 - 0.1)
        C["f3_x_logM"] = fz(d, d["f3"]) * (lM - 4.4)
    return C


Cv = constructions(dv, grp); Ch = constructions(dh, grp)
print(f"[{grp}] {len(Cv)} constructions ({time.time()-t0:.0f}s)", flush=True)
rows = []
for name in Cv:
    xv = fz(dv, Cv[name]); xh = fz(dh, Ch[name])
    per = np.stack([ndcg(dv, dv["g"] + w * xv) for w in W])  # [nW, nf]
    out = np.empty(dv["nf"]); ws = []
    for k in range(5):
        sel = fold == k
        wb = int(np.argmax(per[:, ~sel].mean(1)))
        ws.append(W[wb]); out[sel] = per[wb, sel]
    dl = out - base_v
    wf = W[int(np.argmax(per.mean(1)))]
    ins = per.mean(1).max() - base_v.mean()
    pm, ps = wmean_se(dl, PW)
    hd = ndcg(dh, dh["g"] + wf * xh) - base_h if wf != 0 else np.zeros(dh["nf"])
    hz = hd.mean() / max(hd.std() / np.sqrt(len(hd)), 1e-12)
    sl = " ".join(f"{n} {dl[m].mean()/max(dl[m].std()/np.sqrt(m.sum()),1e-12):+.1f}" for n, m in SL)
    z = dl.mean() / max(dl.std() / np.sqrt(len(dl)), 1e-12)
    rows.append((name, dl.mean(), dl.std() / np.sqrt(len(dl)), z, ws, wf, ins, pm, ps, hd.mean(), hd.std() / np.sqrt(len(hd)), hz, sl))
    flag = "PASS" if (z > 2 and dl.mean() > 0.0003 and hz >= 0) else ""
    print(f"  {name:28s} OOF {dl.mean():+.5f}+-{dl.std()/np.sqrt(len(dl)):.5f} (z {z:+.2f}) w {ws} | in-sample w {wf} {ins:+.5f} | "
          f"pmix {pm:+.5f}+-{ps:.5f} | HOLD {hd.mean():+.5f} (z {hz:+.2f}) | {sl} {flag} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{D}/resid_{grp}.npy", np.array(rows, dtype=object), allow_pickle=True)
print("done")
