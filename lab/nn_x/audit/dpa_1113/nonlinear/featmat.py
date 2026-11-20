"""Feature matrices for the residual GBDT / MLP (C0 = g only; C1 = g + the model's inputs; C2 = C1 + aggregates)."""
import numpy as np
from nlc import fz, fpct, group_pct, fsum

XT_REL_DONE = {"seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"}
DMODEL = ["cos_u", "smax", "smax8", "smax32", "smean", "lse", "top3", "ccnt", "cshare", "firstrec", "fcs", "logM", "logH",
          "mx_e1", "mx_e2", "mx_e3", "mx_e4", "cc_e1", "cc_e2", "cc_e3", "cc_e4"]


def entropy_by_feed(d, key, nkey):
    """Shannon entropy of the distribution of `key` (row-level int) within each feed."""
    c = np.bincount(d["fid"] * nkey + key, minlength=d["nf"] * nkey).reshape(d["nf"], nkey).astype(np.float64)
    p = c / np.maximum(c.sum(1, keepdims=True), 1)
    return -(np.where(p > 0, p * np.log(p), 0)).sum(1)


def hist_aggs(d):
    """user-level history aggregates per feed: history category entropy, distinct cats, type shares (last 32)."""
    nf = d["nf"]; hoff = d["hoff"]; Hc = d["Hc"]; hcat = d["hcat"].astype(np.int64); het = d["het"].astype(np.int64)
    hfid = np.repeat(np.arange(nf), Hc)
    c = np.bincount(hfid * 64 + hcat, minlength=nf * 64).reshape(nf, 64).astype(np.float64)
    p = c / np.maximum(c.sum(1, keepdims=True), 1)
    hent = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
    hncat = (c > 0).sum(1)
    k = np.arange(len(hfid)) - np.repeat(hoff, Hc)
    last32 = k < 32
    sh = {}
    for t in (1, 2, 3, 4):
        sh[t] = np.bincount(hfid, (het == t) & last32, nf) / np.maximum(np.minimum(Hc, 32), 1)
    return hent, hncat, sh


def decayed_catshare(d, hl):
    """sum_k [cat_k == c] 0.5^(k/hl) / sum_k 0.5^(k/hl) for each candidate row (history newest-first)."""
    nf = d["nf"]; hoff = d["hoff"]; Hc = d["Hc"]; hcat = d["hcat"].astype(np.int64)
    hfid = np.repeat(np.arange(nf), Hc)
    k = np.arange(len(hfid)) - np.repeat(hoff, Hc)
    w = 0.5 ** (k / hl)
    num = np.bincount(hfid * 64 + hcat, w, nf * 64)
    den = np.bincount(hfid, w, nf)
    key = d["fid"] * 64 + d["cat"].astype(np.int64)
    return num[key] / np.maximum(den[d["fid"]], 1e-9)


def build(d, level):
    cols, names, cats = [], [], []

    def add(n, x):
        cols.append(np.asarray(x, np.float32)); names.append(n)
    g = d["g"]
    add("g", g)
    if level == 0:
        return np.stack(cols, 1), names, cats
    for k in range(1, 6):
        x = d[f"f{k}"]
        add(f"f{k}", x); add(f"f{k}_z", fz(d, x)); add(f"f{k}_pct", fpct(d, x))
    for n in DMODEL:
        add(n, d["d_" + n])
    for n in [k for k in d if k.startswith("xt_")]:
        add(n[3:], d[n])
    for n in XT_REL_DONE:
        add(n + "_relz", fz(d, d["xt_" + n])); add(n + "_relp", fpct(d, d["xt_" + n]))
    for n in ["cos_u", "smax", "lse", "top3", "ccnt"]:
        add(n + "_relz", fz(d, d["d_" + n])); add(n + "_relp", fpct(d, d["d_" + n]))
    add("u0", d["u0"][d["fid"]].astype(np.float32))
    cats.append(len(names)); add("cat", d["cat"].astype(np.float32))
    cats.append(len(names)); add("vert", d["vert"].astype(np.float32))
    if level == 1:
        return np.stack(cols, 1), names, cats
    # ---- level 2: aggregates
    fid = d["fid"]; M = d["M"]
    key = fid.astype(np.int64) * 64 + d["cat"].astype(np.int64)
    for n, x in [("f1", d["f1"]), ("f2", d["f2"]), ("f4", d["f4"]), ("f5", d["f5"]), ("pop", d["xt_seq_cv_pop"]),
                 ("luo", d["xt_it_hist_all_luo"]), ("cos_u", d["d_cos_u"]), ("smax", d["d_smax"]),
                 ("pre2fe", d["xt_pre2_fe_s"]), ("g", g)]:
        p, ncat = group_pct(d, key, x)
        add(f"{n}_incat_pct", p)
    add("ncat_in_feed_rows", ncat)
    for n in [k for k in d if k.startswith("xt_") and k[3:] not in XT_REL_DONE and k != "xt_has_pre2_fe"]:
        add(n[3:] + "_relp", fpct(d, d[n]))
    for n, x in [("f5", d["f5"]), ("f4", d["f4"]), ("pop", d["xt_seq_cv_pop"]), ("luo", d["xt_it_hist_all_luo"]),
                 ("cos_u", d["d_cos_u"]), ("smax", d["d_smax"]), ("pre2fe", d["xt_pre2_fe_s"])]:
        x = np.asarray(x, np.float64)
        m = fsum(d, x) / M; s = np.sqrt(np.maximum(fsum(d, x * x) / M - m ** 2, 0))
        add(f"feed_mean_{n}", m[fid]); add(f"feed_sd_{n}", s[fid])
    cent = entropy_by_feed(d, d["cat"].astype(np.int64), 64)
    add("feed_cat_entropy", cent[fid])
    add("feed_ncat", (np.bincount(key, minlength=d["nf"] * 64).reshape(d["nf"], 64) > 0).sum(1)[fid])
    coldp = (d["xt_it_hist_all_luo"] <= 0).astype(np.float64)
    add("feed_coldproxy", (fsum(d, coldp) / M)[fid])
    hent, hncat, sh = hist_aggs(d)
    add("hist_cat_entropy", hent[fid]); add("hist_ncat", hncat[fid])
    for t in (1, 2, 3, 4):
        add(f"hist_last32_share_e{t}", sh[t][fid])
    add("cat_last8", d["d_cat_last8"]); add("cat_last32", d["d_cat_last32"])
    for hl in (4, 16, 64):
        add(f"cat_decay{hl}", decayed_catshare(d, hl))
    add("inject_ratio", np.log((d["d_fcs"] + 1.0 / M[fid]) / (d["d_cshare"] + 0.01)))
    return np.stack(cols, 1), names, cats
