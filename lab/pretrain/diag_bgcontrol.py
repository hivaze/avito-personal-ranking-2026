"""E5: is the in-window corpus (future cross-user behaviour at full production density) the cause of the val drop?
Train-fit LightGBM on W1 (or W3) random-like train feeds: base vs base + z(fe), z(new) of
  (a) the windowed single-config model w<k>_v3 (in-window snapshots at 100% density), or
  (b) the background-only control w<k>bg (same config, NO in-window snapshot).
Val side for both: z(fe), z(new) of v3A (val snapshots at ~3% density). Paired per-feed val deltas + train holdout.
Usage: python pretrain/diag_bgcontrol.py --window 1 --ctrl w1bg_v3"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/pretrain")
from common import swap_luo, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from diag_trainstage import PARAMS, ROUNDS, rows_of, load

ROOT = Path("<repo>"); PT = ROOT / "data/pretrain"; WS = PT / "winscores"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--ctrl", default="w1bg_v3")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    k = a.window
    base = swap_luo(json.loads((ROOT / "data/eval_cache/base_features.json").read_text()))
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    wf_all = np.load(PT / f"win{k}/window_feeds.npy")
    win_rows = np.load(PT / f"win{k}/cand_rows.npy")
    rng = np.random.default_rng(0)
    wf = rng.permutation(wf_all[pr[wf_all] > 0.9])
    tr, ho = np.sort(wf[:a.n]), np.sort(wf[a.n:a.n + 10000])
    y_all = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")

    def block(feeds):
        r, Mf = rows_of(off, M, feeds)
        X = load("train", base, r, Mf)
        o = np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
        pos = np.searchsorted(win_rows, r)
        extra = {}
        for tag in [f"w{k}_v3", a.ctrl]:
            z = [feed_rank_z(o, Mf, np.load(WS / f"pre_{tag}_{key}_win{k}.npy", mmap_mode="r")[pos].astype(np.float32))[:, 1] for key in ["fe", "new"]]
            extra[tag] = np.stack(z, 1)
        return X, y_all[r].astype(np.float32), Mf, o, extra

    Xtr, ytr, Mtr, _, Etr = block(tr)
    Xho, yho, Mho, oho, Eho = block(ho)
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    Xv = load("val", base, None, Mv)
    Ev = np.stack([feed_rank_z(offv, Mv, np.load(ROOT / f"preds/pre_v3A_{key}_val.npy").astype(np.float32))[:, 1] for key in ["fe", "new"]], 1)
    res = {}
    for arm in ["base", f"w{k}_v3", a.ctrl]:
        pv, ph = [], []
        for s in range(a.seeds):
            names = base + ([] if arm == "base" else ["pz_fe", "pz_new"])
            Xa = Xtr if arm == "base" else np.hstack([Xtr, Etr[arm]])
            b = lgb.train(dict(PARAMS, num_threads=a.threads, seed=s),
                          lgb.Dataset(Xa, ytr, group=Mtr, feature_name=names, categorical_feature=["category_id"]), ROUNDS)
            pv.append(ndcg_per_feed(b.predict(Xv if arm == "base" else np.hstack([Xv, Ev]), num_threads=a.threads), yv, offv, Mv))
            ph.append(ndcg_per_feed(b.predict(Xho if arm == "base" else np.hstack([Xho, Eho[arm]]), num_threads=a.threads), yho, oho, Mho))
        res[arm] = (np.mean(pv, 0), np.mean(ph, 0))
        print(f"window {k} {arm}: val {res[arm][0].mean():.4f} holdout {res[arm][1].mean():.4f}", flush=True)
    out = {}
    for arm in [f"w{k}_v3", a.ctrl]:
        dv = res[arm][0] - res["base"][0]; dh = res[arm][1] - res["base"][1]
        out[arm] = dict(val_delta=round(float(dv.mean()), 5), val_se=round(float(dv.std() / np.sqrt(len(dv))), 5),
                        holdout_delta=round(float(dh.mean()), 5), holdout_se=round(float(dh.std() / np.sqrt(len(dh))), 5))
        print(f"window {k} {arm} vs base: VAL {dv.mean():+.5f} ({dv.std()/np.sqrt(len(dv)):.5f}) | HOLDOUT {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f})", flush=True)
    with open(ROOT / "logs/pretrain_diag_trainstage.json", "a") as fh:
        fh.write(json.dumps({f"E5_bgcontrol_w{k}": out}) + "\n")


if __name__ == "__main__":
    main()
