"""E8: confirm the cause. Train candidates whose item never appears in any history had NO content embedding in the
pretraining item table (build_corpus.py added val/test candidate embeddings but not train candidate embeddings), so the
encoders scored them with the learned 'missing' vector. Patch on CPU: pre_* = NaN for those rows and the within-feed z
features recomputed over the remaining candidates. Train-fit LightGBM (all windows, random-like feeds) base vs
base + pre9 (as built) vs base + pre9 (patched); evaluate on val (val features unchanged).
Usage: python pretrain/diag_patch.py"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/pretrain")
from common import swap_luo, FLAT, FEATS
from metrics import ndcg_per_feed
from diag_trainstage import PARAMS, ROUNDS, PRE9, rows_of, load

ROOT = Path("<repo>"); PT = ROOT / "data/pretrain"
MODE = sys.argv[1] if len(sys.argv) > 1 else "missing"   # "missing": NaN where content was missing (train only); "cold": NaN where it_hist_all_luo == 0 (train and val)


@nb.njit(parallel=True, cache=True)
def zscore_nan(off, M, x):
    out = np.full(len(x), np.nan, np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if not np.isnan(v):
                s += v; s2 += v * v; c += 1
        if c == 0:
            continue
        mu = s / c; sd = np.sqrt(max(s2 / c - mu * mu, 0.0)) + 1e-9
        for k in range(m):
            v = x[o + k]
            if not np.isnan(v):
                out[o + k] = (v - mu) / sd
    return out


def main():
    t0 = time.time()
    base = swap_luo(json.loads((ROOT / "data/eval_cache/base_features.json").read_text()))
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    has = np.load(PT / "item_has.npy")
    item = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    y_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    tr, ho = [], []
    for k in [1, 2, 3]:
        wf = rng.permutation((lambda w: w[pr[w] > 0.9])(np.load(PT / f"win{k}/window_feeds.npy")))
        tr.append(wf[:25000]); ho.append(wf[25000:33000])
    tr, ho = np.sort(np.concatenate(tr)), np.sort(np.concatenate(ho))

    def block(feeds):
        r, Mf = rows_of(off, M, feeds)
        o = np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
        Xb = load("train", base, r, Mf)
        Xp = load("train", PRE9, r, Mf)
        miss = ~has[np.asarray(item[r])]
        if MODE == "cold":
            miss = np.asarray(np.load(FEATS / "train/it_hist_all_luo.npy", mmap_mode="r")[r]) == 0
        Xq = Xp.copy()
        Xq[miss] = np.nan
        for j, (zn, src) in enumerate([("pre_fe_z", "pre_fe"), ("pre_new_z", "pre_new")]):
            Xq[:, PRE9.index(zn)] = zscore_nan(o, Mf, Xq[:, PRE9.index(src)].copy())
        return Xb, Xp, Xq, y_all[r].astype(np.float32), Mf, o, miss

    Xb, Xp, Xq, y, Mtr, _, miss_tr = block(tr)
    Hb, Hp, Hq, yh, Mh, oh, miss_ho = block(ho)
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    Vb = load("val", base, None, Mv); Vp = load("val", PRE9, None, Mv)
    Vq = Vp.copy()
    if MODE == "cold":
        vm = np.load(FEATS / "val/it_hist_all_luo.npy") == 0
        Vq[vm] = np.nan
        for zn, src in [("pre_fe_z", "pre_fe"), ("pre_new_z", "pre_new")]:
            Vq[:, PRE9.index(zn)] = zscore_nan(offv, Mv, Vq[:, PRE9.index(src)].copy())
    print(f"[data] {time.time()-t0:.0f}s; train rows {len(y)}, missing-content share {miss_tr.mean():.3f}", flush=True)
    res = {}
    for arm, (A, H, V, names) in {"base": (Xb, Hb, Vb, base), "pre9_asbuilt": (np.hstack([Xb, Xp]), np.hstack([Hb, Hp]), np.hstack([Vb, Vp]), base + PRE9),
                                  "pre9_patched": (np.hstack([Xb, Xq]), np.hstack([Hb, Hq]), np.hstack([Vb, Vq]), base + PRE9)}.items():
        pv, ph = [], []
        for s in range(2):
            b = lgb.train(dict(PARAMS, num_threads=24, seed=s), lgb.Dataset(A, y, group=Mtr, feature_name=names, categorical_feature=["category_id"]), ROUNDS)
            pv.append(ndcg_per_feed(b.predict(V, num_threads=24), yv, offv, Mv)); ph.append(ndcg_per_feed(b.predict(H, num_threads=24), yh, oh, Mh))
        res[arm] = (np.mean(pv, 0), np.mean(ph, 0))
        print(f"  {arm}: val {res[arm][0].mean():.4f} holdout {res[arm][1].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    for arm in ["pre9_asbuilt", "pre9_patched"]:
        dv = res[arm][0] - res["base"][0]; dh = res[arm][1] - res["base"][1]
        out[arm] = dict(val_delta=round(float(dv.mean()), 5), val_se=round(float(dv.std() / np.sqrt(len(dv))), 5),
                        holdout_delta=round(float(dh.mean()), 5), holdout_se=round(float(dh.std() / np.sqrt(len(dh))), 5))
        print(f"E8[{MODE}] {arm} vs base: VAL {dv.mean():+.5f} ({dv.std()/np.sqrt(len(dv)):.5f}) | HOLDOUT {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f})", flush=True)
    with open(ROOT / "logs/pretrain_diag_trainstage.json", "a") as fh:
        fh.write(json.dumps({"E8_patch_" + MODE: out}) + "\n")


if __name__ == "__main__":
    main()
