"""W25 evaluation (CPU): single-model read-outs, paired arm deltas, and the pre-declared ensemble test with the G3 pool
(nn_x/cb26/cbrule.py rule; nested user-grouped selection of the mix weight over the VF folds).
Usage: eval_w25.py single <tag> [<tag> ...]            per-arm val / halves / trH / slices
       eval_w25.py pair <tagA> <tagB>                    paired per-feed deltas A - B (val, trH, slices)
       eval_w25.py ens <tag> [pool]                     member test vs the pool (default G3)
"""
import json, sys
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, ens_feats, zf, ROOT  # noqa: E402
from cbrule import judge, pse, val_slices  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
import pmix  # noqa: E402

OUT = ROOT / "nn_x/w25/out"
fv, Mv, ov, yv = meta("val"); fh, Mh, oh, yh = meta("trH"); ft_, Mt, ot, _ = meta("test")
nv = lambda s: ndcg_per_feed(s, yv, ov, Mv); nh = lambda s: ndcg_per_feed(s, yh, oh, Mh)
zv = lambda x: zf(ov, Mv, x); zh = lambda x: zf(oh, Mh, x); zt = lambda x: zf(ot, Mt, x)


def vf_folds():
    u = fv["user_id"].to_numpy().astype(np.int64); g = np.where(u == 0, -1 - np.arange(len(u)), u)
    return [np.sort(te) for _, te in GroupKFold(n_splits=5).split(np.arange(len(Mv)), groups=g)]


def load(tag):
    return {s: np.load(OUT / f"{tag}_{s}.npy") for s in ("val", "test", "trH")}


def single(tag):
    j = json.loads((OUT / f"{tag}.json").read_text()); s = load(tag)
    pv, ph = nv(s["val"]), nh(s["trH"])
    sl = val_slices()
    txt = (f"[{tag}] feeds {j['n_train_feeds']} rows {j['n_train_rows']} best_iter {j['best_iter']} time {j['time'] / 60:.0f}m | "
           f"val {pv.mean():.5f} +- {pse(pv)[1]:.5f} (A {j['val_halfA_ES']:.5f} B {j['val_halfB']:.5f}) trH {ph.mean():.5f} +- {pse(ph)[1]:.5f} | "
           + " ".join(f"{k} {pv[m].mean():.4f}" for k, m in sl.items()))
    print(txt)
    return pv, ph


def pair(a, b):
    sa, sb = load(a), load(b)
    dv = nv(sa["val"]) - nv(sb["val"]); dh = nh(sa["trH"]) - nh(sb["trH"])
    m, se = pse(dv); hm, hs = pse(dh)
    print(f"[{a} - {b}] val {m:+.5f} +- {se:.5f} (z {m / se:+.2f}) | trH {hm:+.5f} +- {hs:.5f} (z {hm / hs:+.2f})")
    for k, msk in val_slices().items():
        sm, ss = pse(dv[msk]); print(f"    slice {k:9s} n={msk.sum():5d} {sm:+.5f} +- {ss:.5f}")
    pm = pmix.wmean_se(dv, pmix.weights()["private"]); print(f"    pmix private {pm[0]:+.5f} +- {pm[1]:.5f}")
    return dv, dh


def corr_feed(a, b, M):
    fid = np.repeat(np.arange(len(M)), M)
    return float((np.bincount(fid, a * b, len(M)) / M).mean())


def ens(tag, pool="G3"):
    ev, eh, et = ens_feats(pool, "val"), ens_feats(pool, "trH"), ens_feats(pool, "test")
    pool_v = zv(0.5 * zv(ev["ft"]) + 0.5 * zv(ev["to"])); to_h = zh(eh["to"])
    bv, bh = nv(pool_v), nh(to_h)
    s = load(tag); zc_v, zc_h = zv(s["val"]), zh(s["trH"])
    single_v = nv(s["val"]).mean()
    rho_v, rho_h = corr_feed(zc_v, pool_v, Mv), corr_feed(zc_h, to_h, Mh)
    gap = bv.mean() - single_v
    bar = bv.mean() - 0.049 * (1 - rho_v)
    print(f"[{tag} vs {pool}] pool val {bv.mean():.5f} trH {bh.mean():.5f}; member val {single_v:.5f}; rho val {rho_v:.3f} trH {rho_h:.3f}; "
          f"gap {gap:.4f}; law gain {0.0186 * (1 - rho_v) - 0.378 * gap:+.4f}; member bar (val >= {bar:.4f}) {'MET' if single_v >= bar else 'NOT met'}")
    ws = (0.05, 0.1, 0.2, 0.3)
    dv = {w: nv(pool_v + w * zc_v) - bv for w in ws}
    dh = {w: nh(to_h + w * zc_h) - bh for w in ws}
    for w in ws:
        m, se = pse(dv[w]); hm, hs = pse(dh[w])
        print(f"   w={w:.2f}: val {m:+.5f} +- {se:.5f} (z {m / se:+.2f}) | trH {hm:+.5f} +- {hs:.5f}")
    # nested selection over the VF folds
    folds = vf_folds(); oof = np.zeros(len(Mv)); chosen = []
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(Mv)), te)
        wbest = max(ws, key=lambda w: dv[w][tr].mean())
        oof[te] = dv[wbest][te]; chosen.append(wbest)
    wall = max(ws, key=lambda w: dv[w].mean())
    print(f"   nested w per fold {chosen}; full-val best w {wall}")
    ok, txt = judge(f"{tag} nested member (holdout at w={wall})", oof, dh[wall], pmix.wmean_se(oof, pmix.weights()["private"]),
                    extra_cond=(dv[0.05].mean() > 0) and (single_v >= bar))
    print(txt)
    return ok


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "single":
        for t in sys.argv[2:]:
            single(t)
    elif mode == "pair":
        pair(sys.argv[2], sys.argv[3])
    elif mode == "ens":
        ens(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "G3")
