"""ENRICH-1113 batch 2: C4 complement categories (pre-declared addendum) and X1 category act/exp offset (information)."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, numba as nb, polars as pl
from run import *  # noqa: F401,F403  (harness objects, V, Hd, log, aucs, evaluate)
from run import V, Hd, log, aucs, feed_auc
import harness as Hs
import ext_tables as ET

FLAT = Path("<repo>/data/flat")


@nb.njit(cache=False)
def cat_presence(hist_cat, hist_hpos, hist_off, H, nF):
    rec = np.zeros((nF, 64)); anyc = np.zeros((nF, 64))
    for f in range(nF):
        ho = hist_off[f]
        for j in range(H[f]):
            c = hist_cat[ho + j]
            if c < 0 or c >= 64:
                continue
            anyc[f, c] += 1.0
            if hist_hpos[ho + j] < 32:
                rec[f, c] += 1.0
    return rec, anyc


def presence(split, feeds_idx):
    f = pl.read_parquet(FLAT / split / "feeds.parquet", columns=["hist_off", "H"])
    ho = f["hist_off"].to_numpy().astype(np.int64)[feeds_idx]; H = f["H"].to_numpy().astype(np.int64)[feeds_idx]
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, H)])
    hcat = np.asarray(np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r")[hrows]).astype(np.int64)
    hhp = np.asarray(np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")[hrows]).astype(np.int64)
    loc_hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    return cat_presence(hcat, hhp, loc_hoff, H, len(H))


def item_C4():
    CM = ET.complement_matrix()
    def cols(split, idx, D):
        rec, _ = presence(split, idx)
        cnt = rec @ CM                      # [feed, cand_cat] = # recent events in complement-source categories
        x = cnt[D["fid"], D["cat"]]
        return [(x > 0) * 1.0, np.log1p(x)]
    Xv = cols("val", np.arange(len(V["M"])), V)
    Xh = cols("train", Hs.HOLD, Hd)
    return ["comp_recent", "log1p_comp_cnt"], Xv, Xh


def x1_information():
    """fold-wise: offsets from the training folds' act/exp of the control model; LIN on [b, offset]; nested."""
    from harness import b, bh, yv, yh, ov, oh, Mv, Mh, fid, npos_f, nF, rowfold, fit_lin, nd, ndh, SL, stat
    catv = V["cat"]; cath = Hd["cat"]
    def offsets(tr):
        th = fit_lin(b[:, None], tr); s = th[0] * b
        e = np.exp(s - np.repeat(np.maximum.reduceat(s, ov), Mv)); Z = np.bincount(fid, weights=e, minlength=nF)
        p = npos_f[fid] * e / Z[fid]
        ep = np.bincount(catv[tr], weights=p[tr], minlength=64); ap = np.bincount(catv[tr], weights=yv[tr], minlength=64)
        off = np.log((ap + 20.0) / (ep + 20.0))
        return off
    pred = np.empty(len(yv)); predc = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k
        off = offsets(tr); x = off[catv]
        G = np.stack([b, Hs.zv(x), Hs.pctrank(x, ov, Mv)], 1)
        th = fit_lin(G, tr); pred[~tr] = G[~tr] @ th
        thc = fit_lin(b[:, None], tr); predc[~tr] = b[~tr] * thc[0]
    d = nd(pred) - nd(predc); dv, se, z = stat(d)
    import pmix
    pm, pse = pmix.wmean_se(d, pmix.weights()["private"])
    line = f"X1 INFORMATION nested val delta {dv:+.5f} (SE {se:.5f}) z {z:+.2f} | pmix {pm:+.5f} ({pse:.5f})"
    for sn, m in SL.items():
        line += f" | {sn} {d[m].mean():+.5f}"
    log(line)
    off = offsets(np.ones(len(yv), bool)); x = off[catv]
    G = np.stack([b, Hs.zv(x), Hs.pctrank(x, ov, Mv)], 1); th = fit_lin(G, np.ones(len(yv), bool))
    xh = off[cath]; Gh = np.stack([bh, Hs.zh(xh), Hs.pctrank(xh, oh, Mh)], 1)
    dh = ndh(Gh @ th) - ndh(bh); hd, hse, hz = stat(dh)
    log(f"X1 INFORMATION theta_full {np.round(th, 4).tolist()}; holdout transfer {hd:+.5f} (SE {hse:.5f}) z {hz:+.2f} | LATE {dh[Hs.H_LATE].mean():+.5f} | EARLY {dh[Hs.H_EARLY].mean():+.5f}")
    log("X1 offsets (full val) by code: " + ", ".join(f"c{c}:{off[c]:+.3f}" for c in range(2, 54) if np.bincount(catv, minlength=64)[c] >= 2000))
    a, _ = feed_auc(x, yv, ov, Mv); log(f"X1 within-feed AUC of the full-val offset column (in-sample, information) {a:.4f}")


if __name__ == "__main__":
    sanity(log)
    for it in sys.argv[1:]:
        if it == "X1":
            x1_information(); continue
        names, Xv, Xh = item_C4()
        a = aucs(names, Xv)
        log(f"{it} within-feed AUC (val, raw columns): " + ", ".join(f"{k} {v:.4f}" for k, v in a.items()))
        res = evaluate(it, names, Xv, Xh, log, extra=dict(auc=a))
        ok = (res["val_delta"] >= 0.0005) and (res["val_z"] >= 2.5) and (res["hold_delta"] >= 0) and (res["pmix_delta"] >= -res["pmix_se"])
        res["verdict"] = "PASS" if ok else "CLOSED"
        log(f"{it} VERDICT {res['verdict']} (bar: val >= +0.0005 & z >= 2.5 & holdout >= 0 & pmix >= -1 SE)")
        with open(HERE / "out/results.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
    log(f"DONE batch2 {time.time()-T0:.0f}s")
