"""Robustness copy of the pretrain/gate_pre2.py g1 setup for it_hist_hz_* (copied, not imported from pretrain/).
Training: 25K random-like (p_random > 0.9) feeds per pretraining window (data/pretrain/win{1,2,3}/window_feeds.npy, rng 0),
window holdout = the next 8K per window; frozen base = swap_luo(data/eval_cache/base_features.json) (122 features);
LightGBM lambdarank (diag_trainstage PARAMS, 350 rounds), 2 seeds, 16 threads.
Arms: G0 = base (as built). G2 = base with it_hist_{all,et1..4}_luo -> it_hist_hz_{all,et1..4} and
it_hist_all_luo_{rk,z,gap} recomputed from it_hist_hz_all (src/stack_val.feed_rank_z, gap = x - feed max);
it_hist_et5..10_luo stay as built in both arms. Val rows are identical between arms.
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/it_hist_hz_gate_g1.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "16")
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "<repo>/src")
from common import swap_luo
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; NEW = ROOT / "data/feats_new"
PT = ROOT / "data/pretrain"
SC = Path("/tmp/scratchpad/featrev/it_hist_hz")
TH = 16
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1, num_threads=TH)
ROUNDS = 350
SWAP = {f"it_hist_{n}_luo": f"it_hist_hz_{n}" for n in ["all", "et1", "et2", "et3", "et4"]}


def rows_of(off, M, feeds):
    Mf = M[feeds]
    return np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum()), Mf


def load_block(split, names, rows, Mf):
    """returns dict name -> float32 column; G2 columns prefixed 'G2:' for the swapped / recomputed ones."""
    o = np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
    X = {}
    for n in names:
        x = np.load(FEATS / split / f"{n}.npy", mmap_mode="r")
        X[n] = np.asarray(x if rows is None else x[rows], np.float32)
    for a, b in SWAP.items():
        x = np.load(NEW / split / f"{b}.npy", mmap_mode="r")
        X["G2:" + a] = np.asarray(x if rows is None else x[rows], np.float32)
    h = X["G2:it_hist_all_luo"]
    rz = feed_rank_z(o, Mf, h)
    X["G2:it_hist_all_luo_rk"] = rz[:, 0].astype(np.float32); X["G2:it_hist_all_luo_z"] = rz[:, 1].astype(np.float32)
    X["G2:it_hist_all_luo_gap"] = (h - np.repeat(np.maximum.reduceat(h, o), Mf)).astype(np.float32)
    return X


def main():
    T0 = time.time()
    base = swap_luo(json.loads((ROOT / "data/eval_cache/base_features.json").read_text()))
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    y_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    tr, ho = [], []
    for k in (1, 2, 3):
        wf = np.load(PT / f"win{k}/window_feeds.npy"); wf = rng.permutation(wf[pr[wf] > 0.9])
        tr.append(wf[:25000]); ho.append(wf[25000:33000])
    tr, ho = np.sort(np.concatenate(tr)), np.sort(np.concatenate(ho))
    rt, Mt = rows_of(off, M, tr); rh, Mh = rows_of(off, M, ho)
    Xt = load_block("train", base, rt, Mt); Xh = load_block("train", base, rh, Mh)
    yt = y_all[rt].astype(np.float32); yh = y_all[rh].astype(np.float32); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    Xv = load_block("val", base, None, Mv)
    swapped = [n for n in base if ("G2:" + n) in Xt]
    print("loaded", len(yt), len(yh), len(yv), "swapped in G2:", swapped, f"{time.time()-T0:.0f}s", flush=True)
    for n in swapped:
        assert np.array_equal(Xv[n], Xv["G2:" + n]) or n.endswith(("_rk", "_z", "_gap")), n
        if n.endswith(("_rk", "_z", "_gap")):
            print(f"  val {n}: max |as-built - recomputed| {np.abs(Xv[n] - Xv['G2:' + n]).max():.2e}")
    arms = {"G0": base, "G2": [("G2:" + n) if ("G2:" + n) in Xt else n for n in base]}
    res = {}
    for arm, cols in arms.items():
        A = np.stack([Xt[c] for c in cols], 1)
        pv, ph = [], []
        for s in range(2):
            b = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(A, yt, group=Mt, feature_name=base, categorical_feature=["category_id"]), ROUNDS)
            pv.append(ndcg_per_feed(b.predict(np.stack([Xv[c] for c in cols], 1), num_threads=TH), yv, offv, Mv))
            ph.append(ndcg_per_feed(b.predict(np.stack([Xh[c] for c in cols], 1), num_threads=TH), yh, oh, Mh))
        res[arm] = (np.mean(pv, 0), np.mean(ph, 0), [float(x.mean()) for x in pv], [float(x.mean()) for x in ph])
        print(f"{arm}: val {res[arm][0].mean():.5f} window-holdout {res[arm][1].mean():.5f} seeds val {np.round(res[arm][2], 5).tolist()} {time.time()-T0:.0f}s", flush=True)
    dv = res["G2"][0] - res["G0"][0]; dh = res["G2"][1] - res["G0"][1]
    out = dict(val=(float(dv.mean()), float(dv.std() / np.sqrt(len(dv)))), holdout=(float(dh.mean()), float(dh.std() / np.sqrt(len(dh)))),
               per_seed_val=[a - b for a, b in zip(res["G2"][2], res["G0"][2])])
    print(f"G2 - G0: VAL {dv.mean():+.5f} ({dv.std()/np.sqrt(len(dv)):.5f}) | WINDOW-HOLDOUT {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f}) | per-seed val {np.round(out['per_seed_val'], 5).tolist()}")
    (SC / "gate_g1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
