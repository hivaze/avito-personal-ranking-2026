"""ENRICH-1113 evaluation: items per PREDECLARED.txt through the CCB harness (nested val LIN, full-val theta -> holdout, pmix).
usage: python run.py ITEM [ITEM ...]   (ITEM in S1 C1 C2 C3 R1 R2 U1 U2 MEAS)"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, numba as nb
import harness as Hs
from harness import evaluate, log_to, sanity, fit_lin, b, bh, yv, yh, ov, oh, Mv, Mh, fid, npos_f, nF
import ext_tables as ET

T0 = time.time()
LOG = open(HERE / "logs/run.log", "a"); log = log_to(LOG)
V = np.load(HERE / "feats/val.npz"); Hd = np.load(HERE / "feats/hold.npz")
V = {k: V[k] for k in V.files}; Hd = {k: Hd[k] for k in Hd.files}
assert np.array_equal(V["y"], yv) and np.array_equal(Hd["y"], yh)
DAY0 = 20702; LAP = 50.0


@nb.njit(cache=False)
def feed_auc(x, y, off, M):
    tot = 0.0; n = 0
    for i in range(len(M)):
        o = off[i]; m = M[i]; npos = 0
        for k in range(m):
            if y[o + k] > 0:
                npos += 1
        if npos == 0 or npos == m:
            continue
        v = np.empty(m)
        for k in range(m):
            t = x[o + k]
            v[k] = -1e30 if t != t else t
        order = np.argsort(v)
        ranks = np.empty(m); i0 = 0
        while i0 < m:
            j = i0
            while j + 1 < m and v[order[j + 1]] == v[order[i0]]:
                j += 1
            avg = 0.5 * (i0 + j) + 1.0
            for k in range(i0, j + 1):
                ranks[order[k]] = avg
            i0 = j + 1
        s = 0.0
        for k in range(m):
            if y[o + k] > 0:
                s += ranks[k]
        tot += (s - npos * (npos + 1) / 2) / (npos * (m - npos)); n += 1
    return tot / max(n, 1), n


def aucs(names, Xv):
    out = {}
    for n, x in zip(names, Xv):
        a, _ = feed_auc(np.asarray(x, np.float64), yv, ov, Mv); out[n] = float(a)
    return out


def hbin(minh):
    bb = np.full(len(minh), 8, np.int64)
    for lo, hi, k in [(0, 1, 0), (2, 3, 1), (4, 7, 2), (8, 15, 3), (16, 31, 4), (32, 63, 5), (64, 127, 6), (128, 255, 7)]:
        bb[(minh >= lo) & (minh <= hi)] = k
    return bb


def lhbin(lh):
    return np.where(lh < 6, 0, np.where(lh < 11, 1, np.where(lh < 18, 2, 3)))


def loglift_from(tab, cat, key, dayidx=None):
    """tab = (n[day, cat, key], p[day, cat, key]). dayidx None -> all days; else per-row day index: use days < dayidx."""
    n, p = tab
    if dayidx is None:
        N = n.sum(0)[None]; P = p.sum(0)[None]; di = np.zeros(len(cat), np.int64)
    else:
        cn = np.concatenate([np.zeros((1,) + n.shape[1:]), np.cumsum(n, 0)], 0)  # cn[d] = sum over days < d
        cp = np.concatenate([np.zeros((1,) + p.shape[1:]), np.cumsum(p, 0)], 0)
        N, P, di = cn, cp, dayidx
    nc = N.sum(2); pc = P.sum(2)
    rc = np.where(nc > 0, pc / np.maximum(nc, 1), 0.0)
    r_c = rc[di, cat]
    ncell = N[di, cat, key]; pcell = P[di, cat, key]
    rate = (pcell + LAP * r_c) / (ncell + LAP)
    lift = np.where(r_c > 0, rate / np.maximum(r_c, 1e-12), 1.0)
    return np.log(np.maximum(lift, 1e-6))


def tab(nm):
    z = np.load(HERE / f"out/tab_{nm}.npz"); return z["n"], z["p"]


def hold_dayidx():
    return (Hd["day"] - DAY0)[Hd["fid"]]


def item_S1():
    t = ET.table(ET.S1_SEASON)
    return ["ext_season"], [t[V["cat"]]], [t[Hd["cat"]]]


def item_C1():
    cyc = ET.CYCLE; long_t = ET.table({k: 1.0 for k, v in cyc.items() if v == "long"}); short_t = ET.table({k: 1.0 for k, v in cyc.items() if v == "short"})
    def cols(D):
        mh = D["minh"]; has = mh < 999; lg = long_t[D["cat"]]; sh = short_t[D["cat"]]
        return [lg * np.where(has, np.log1p(mh), 0.0), lg * (mh <= 8), sh * (mh <= 8)]
    return ["long_x_loghpos", "long_x_recent8", "short_x_recent8"], cols(V), cols(Hd)


def item_C2():
    T = tab("C2")
    return ["cat_recency_lift"], [loglift_from(T, V["cat"], hbin(V["minh"]))], [loglift_from(T, Hd["cat"], hbin(Hd["minh"]), hold_dayidx())]


def item_C3():
    life = ET.table(ET.LIFE_D, default=30.0)
    def cols(D):
        a = D["age"]; L = life[D["cat"]]
        return [np.log(np.maximum(a, 0.1) / (24.0 * L)) * np.where(np.isnan(a), np.nan, 1.0), (a < 24) * (L >= 30) * 1.0]
    return ["log_age_over_life", "fresh_x_slowcat"], cols(V), cols(Hd)


def item_R1():
    def cols(D):
        lh = D["local_hour"][D["fid"]]; home = D["home"]
        return [home * (lh < 6), home * (lh >= 18)]
    return ["home_x_localnight", "home_x_localevening"], cols(V), cols(Hd)


def item_R2():
    T = tab("R2")
    kv = lhbin(V["local_hour"])[V["fid"]]; kh = lhbin(Hd["local_hour"])[Hd["fid"]]
    return ["cat_x_localhour_lift"], [loglift_from(T, V["cat"], kv)], [loglift_from(T, Hd["cat"], kh, hold_dayidx())]


def item_U1():
    def cols(D):
        return [D["s2"].astype(np.float64), D["s2r"].astype(np.float64), D["s4"].astype(np.float64)]
    return ["cat_s2_cnt", "cat_s2_recent32", "cat_s4_cnt"], cols(V), cols(Hd)


def item_U2():
    T = tab("U2")
    def cols(D, dayidx):
        u0 = (D["user"] == 0).astype(np.int64)[D["fid"]]
        ll = loglift_from(T, D["cat"], np.ones(len(D["cat"]), np.int64), dayidx)
        return [u0 * ll]
    return ["guest_x_catlift"], cols(V, None), cols(Hd, hold_dayidx())


def measurements():
    # (a) per-category CTR val vs train random-like (trs sample, p_random > 0.9 & M >= 50) and vs all trs
    Tr = np.load(HERE / "feats/trs.npz"); pr = np.load(HERE / "feats/trs_prand.npy")
    catT = Tr["cat"]; yT = Tr["y"]; fidT = Tr["fid"]; MT = Tr["M"]
    rl = ((pr > 0.9) & (MT >= 50))[fidT]
    def ctr(c, y, m=None):
        if m is not None:
            c, y = c[m], y[m]
        n = np.bincount(c, minlength=64); p = np.bincount(c, weights=y, minlength=64); return n, p
    nv, pv = ctr(V["cat"], yv); nr, prr = ctr(catT, yT, rl); na, pa = ctr(catT, yT)
    # (b) G3 act/exp per category on val: p_i = npos_f * softmax(beta b) within feed, beta = full-val control fit
    th = fit_lin(b[:, None], np.ones(len(yv), bool)); s = th[0] * b
    e = np.exp(s - np.repeat(np.maximum.reduceat(s, ov), Mv)); Z = np.bincount(fid, weights=e, minlength=nF)
    p = npos_f[fid] * e / Z[fid]
    ep = np.bincount(V["cat"], weights=p, minlength=64)
    s1 = ET.table(ET.S1_SEASON)
    log("MEAS per-category: code name | val rows | CTR val | CTR train random-like | CTR train all | log(val/trR) | G3 act/exp (z) | S1 idx")
    rows = []
    for c in range(2, 54):
        if nv[c] < 2000:
            continue
        cv = pv[c] / nv[c]; cr = prr[c] / max(nr[c], 1); ca = pa[c] / max(na[c], 1)
        ae = pv[c] / max(ep[c], 1e-9); zz = (pv[c] - ep[c]) / np.sqrt(max(ep[c], 1e-9))
        rows.append(dict(code=c, name=ET.NAMES.get(c, "?"), n_val=int(nv[c]), ctr_val=cv, ctr_trR=cr, ctr_trA=ca, logratio=float(np.log(cv / max(cr, 1e-9))), act_exp=ae, z=zz, s1=int(s1[c])))
        log(f"  c{c:<3d} {ET.NAMES.get(c, '?'):24s} {int(nv[c]):8d} {cv:.4f} {cr:.4f} {ca:.4f} {np.log(cv / max(cr, 1e-9)):+.3f} {ae:.3f} ({zz:+.1f}) {int(s1[c]):+d}")
    lr = np.array([r["logratio"] for r in rows]); ae = np.array([r["act_exp"] for r in rows]); s1v = np.array([r["s1"] for r in rows])
    w = np.array([r["n_val"] for r in rows], np.float64)
    log(f"MEAS drift summary over {len(rows)} categories (>= 2000 val rows): sd of log(CTR val / CTR trainR) {lr.std():.3f} (row-weighted {np.sqrt(np.average((lr - np.average(lr, weights=w)) ** 2, weights=w)):.3f}); "
        f"sd of G3 act/exp {ae.std():.3f}, max |z| {np.abs([r['z'] for r in rows]).max():.1f}; "
        f"Spearman(S1 idx, log drift) {__import__('scipy.stats').stats.spearmanr(s1v, lr)[0]:+.3f}; Spearman(S1 idx, act/exp) {__import__('scipy.stats').stats.spearmanr(s1v, ae)[0]:+.3f}; "
        f"Pearson(log drift, log act/exp) {np.corrcoef(lr, np.log(ae))[0, 1]:+.3f}")
    # local-hour coverage
    for nm, D in [("val", V), ("hold", Hd)]:
        log(f"MEAS {nm}: feeds with home component {np.mean(D['hcomp'] >= 0):.3f}; non-zero tz shift {np.mean(D['shift'] != 0):.3f}; "
            f"local night (0-6) share {np.mean(D['local_hour'] < 6):.3f} vs MSK night {np.mean(D['msk_hour'] < 6):.3f}")
    json.dump(rows, open(HERE / "out/meas_category.json", "w"), indent=1)


ITEMS = dict(S1=item_S1, C1=item_C1, C2=item_C2, C3=item_C3, R1=item_R1, R2=item_R2, U1=item_U1, U2=item_U2)
if __name__ == "__main__":
    sanity(log)
    for it in sys.argv[1:]:
        if it == "MEAS":
            measurements(); continue
        names, Xv, Xh = ITEMS[it]()
        a = aucs(names, Xv)
        log(f"{it} within-feed AUC (val, raw columns): " + ", ".join(f"{k} {v:.4f}" for k, v in a.items()))
        res = evaluate(it, names, Xv, Xh, log, extra=dict(auc=a))
        ok = (res["val_delta"] >= 0.0005) and (res["val_z"] >= 2.5) and (res["hold_delta"] >= 0) and (res["pmix_delta"] >= -res["pmix_se"])
        res["verdict"] = "PASS" if ok else "CLOSED"
        log(f"{it} VERDICT {res['verdict']} (bar: val >= +0.0005 & z >= 2.5 & holdout >= 0 & pmix >= -1 SE)")
        with open(HERE / "out/results.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
    log(f"DONE {time.time()-T0:.0f}s")
