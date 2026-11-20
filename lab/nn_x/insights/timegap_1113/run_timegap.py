"""TIMEGAP-1113 evaluation: PSI train-sample vs val (raw / within-feed z / within-feed pct rank), within-feed AUC on val,
nested LIN residual on G3 (harness = copy of enrich_1113/harness.py) for items A B C D E ALL4 per PREDECLARED_timegap.txt."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, numba as nb
import harness as Hs
from harness import evaluate, log_to, sanity, feed_z_nanmin, pctrank, yv, yh, ov, oh, Mv, Mh

T0 = time.time()
LOG = open(HERE / "run.log", "a"); log = log_to(LOG)
V = np.load(HERE / "feats/val.npz"); Hd = np.load(HERE / "feats/hold.npz"); Tr = np.load(HERE / "feats/trs.npz")
V = {k: V[k] for k in V.files}; Hd = {k: Hd[k] for k in Hd.files}; Tr = {k: Tr[k] for k in Tr.files}
assert np.array_equal(V["y"], yv) and np.array_equal(Hd["y"], yh)
assert np.array_equal(V["M"], Mv) and np.array_equal(Hd["M"], Mh)


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


def psi(p, q, eps=1e-6):
    p = np.maximum(p / p.sum(), eps); q = np.maximum(q / q.sum(), eps)
    return float(((p - q) * np.log(p / q)).sum())


def hist_raw(x):
    """integer bins 0..20 (clipped) + NaN bin"""
    h = np.zeros(22)
    isn = np.isnan(x); v = np.clip(np.nan_to_num(x, nan=0.0), 0, 20).astype(int)
    h[:21] = np.bincount(v[~isn], minlength=21)[:21]; h[21] = isn.sum()
    return h


def hist_z(x):
    edges = np.concatenate([[-np.inf], np.arange(-2.5, 2.51, 0.5), [np.inf]])
    return np.histogram(x, bins=edges)[0].astype(np.float64)


def hist_p(x):
    return np.histogram(x, bins=np.linspace(-0.5, 0.5, 11))[0].astype(np.float64)


COLS = ["gap_last", "gap_last_any", "gap_first", "never_seen", "n_days_active", "n_days_active_any", "gap_first_any"]


def norm_versions(D, col):
    x = np.asarray(D[col], np.float64); off = D["loc_off"].astype(np.int64); M = D["M"].astype(np.int64)
    return x, feed_z_nanmin(x, off, M), pctrank(x, off, M)


def psi_block():
    rowday_tr = Tr["day"][Tr["fid"]]
    out = {}
    log("PSI train-sample vs val (raw: integer bins + NaN bin; z: 12 bins of width 0.5 on [-2.5, 2.5] + tails; pct: 10 bins on [-0.5, 0.5])")
    log(f"{'column':18s} {'PSI raw':>8s} {'PSI z':>8s} {'PSI pct':>8s} | {'raw d>=7':>8s} {'z d>=7':>8s} {'pct d>=7':>8s} | hold-vs-val raw / z / pct")
    for c in COLS:
        xv, zv_, pv_ = norm_versions(V, c); xt, zt, pt = norm_versions(Tr, c); xh, zh_, ph_ = norm_versions(Hd, c)
        late = rowday_tr >= 7
        r = dict(raw=psi(hist_raw(xt), hist_raw(xv)), z=psi(hist_z(zt), hist_z(zv_)), pct=psi(hist_p(pt), hist_p(pv_)),
                 raw_d7=psi(hist_raw(xt[late]), hist_raw(xv)), z_d7=psi(hist_z(zt[late]), hist_z(zv_)), pct_d7=psi(hist_p(pt[late]), hist_p(pv_)),
                 hraw=psi(hist_raw(xh), hist_raw(xv)), hz=psi(hist_z(zh_), hist_z(zv_)), hpct=psi(hist_p(ph_), hist_p(pv_)))
        out[c] = r
        log(f"{c:18s} {r['raw']:8.3f} {r['z']:8.3f} {r['pct']:8.3f} | {r['raw_d7']:8.3f} {r['z_d7']:8.3f} {r['pct_d7']:8.3f} | {r['hraw']:.3f} / {r['hz']:.3f} / {r['hpct']:.3f}")
        # raw distribution summary per split
        hv = hist_raw(xv) / len(xv); ht = hist_raw(xt) / len(xt)
        log(f"    raw share by value (train | val): NaN {ht[21]:.3f} | {hv[21]:.3f}; " +
            " ".join(f"{k}:{ht[k]:.3f}|{hv[k]:.3f}" for k in range(0, 21) if ht[k] + hv[k] > 0.004))
        log(f"    z sd within-feed-normalised: train {zt.std():.3f} val {zv_.std():.3f} hold {zh_.std():.3f}; share of feeds with z all 0 (constant): "
            f"train {np.mean(np.bincount(Tr['fid'], weights=np.abs(zt)) == 0):.3f} val {np.mean(np.bincount(V['fid'], weights=np.abs(zv_)) == 0):.3f}")
    json.dump(out, open(HERE / "psi.json", "w"), indent=1)
    return out


def aucs(names, Xv):
    out = {}
    for n, x in zip(names, Xv):
        a, nf = feed_auc(np.asarray(x, np.float64), yv, ov, Mv); out[n] = float(a)
    return out


ITEMS = {
    "A": ["gap_last"], "B": ["gap_last_any"], "C": ["gap_first"], "D": ["never_seen"], "E": ["n_days_active"],
    "ALL4": ["gap_last", "gap_last_any", "gap_first", "n_days_active"],
}
if __name__ == "__main__":
    sanity(log)
    P = psi_block()
    a_all = aucs(COLS, [V[c] for c in COLS])
    log("within-feed AUC on val (raw; NaN below min): " + ", ".join(f"{k} {v:.4f}" for k, v in a_all.items()))
    xz = feed_z_nanmin(np.asarray(V["gap_last"], np.float64), V["loc_off"].astype(np.int64), Mv)
    log(f"AUC check of the within-feed z of gap_last (NaN -> feed min): {feed_auc(xz, yv, ov, Mv)[0]:.4f}")
    for it in sys.argv[1:] if len(sys.argv) > 1 else list(ITEMS):
        names = ITEMS[it]
        Xv = [np.asarray(V[c], np.float64) for c in names]; Xh = [np.asarray(Hd[c], np.float64) for c in names]
        res = evaluate(f"TG_{it}", names, Xv, Xh, log, extra=dict(auc={c: a_all[c] for c in names}, psi={c: P[c] for c in names}))
        ok = (res["val_delta"] >= 0.0005) and (res["val_z"] >= 2.5) and (res["hold_delta"] >= 0) and (res["pmix_delta"] >= -res["pmix_se"])
        res["verdict"] = "PASS" if ok else "CLOSED"
        log(f"TG_{it} VERDICT {res['verdict']} (bar: val >= +0.0005 & z >= 2.5 & holdout >= 0 & pmix >= -1 SE)")
        with open(HERE / "results.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
    log(f"DONE {time.time()-T0:.0f}s")
