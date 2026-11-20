"""ap02: AGE-PROD arms in the TRUNC fresh-sample proxy harness (nn_x/insights/review/r2_trunc_replicate.py: U2 150K-feed
sample, LightGBM lambdarank, 400 rounds, readouts at 100/200/300/400). Read-only on every cache; writes out/gb/.

usage: OMP_NUM_THREADS=12 python ap02_arms.py ARM SEED [SEED ...]
Arm tokens joined by '+':
  base    no offset
  tprod<a> r2's truncation offset: a * L_trunc[group, rkend 0-2] on production rows (lifts estimated on U2 itself)
  age<a>   a * L_age[M bucket (M<50 / M>=50), age bin] on ALL production (cls 0) rows; L_age from out/age_lifts.json
           (full train, production vs exploration-like cls 1, holdout excluded; age bins / proxy as a08)
  ageh<a>  the same on M>=50 production rows only
Offsets are init_score at training only. Readouts: val (live features), val-stale ('both' coverage replay: the LUO /
co-vis columns replaced by nn_x/r6/stale_cf/both_*.npy, it_hist_all_luo_z recomputed), HO. Per-feed NDCG saved."""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

DU = ROOT / "nn_x/insights/data_usage/out"
G = DU / "gb"
RV = ROOT / "nn_x/insights/review/out/gb"
AP = ROOT / "nn_x/audit/ageprod/out"
OUT = AP / "gb"
OUT.mkdir(parents=True, exist_ok=True)
CF = ROOT / "nn_x/r6/stale_cf"
FEATS = json.load(open(G / "feats.json"))
THREADS = 12
ROUNDS = 400

KNOTS = np.load(ROOT / "data/feats/life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6
EDGES = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]


def age_bin(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = np.where(x < CAP, KNOTS[k] * (1 - w) + KNOTS[k + 1] * w, np.nan)
    return np.digitize(np.nan_to_num(a, nan=1e8), EDGES) - 1


def feed_z(x, fid, n):
    c = np.bincount(fid, minlength=n).astype(np.float64)
    mu = np.bincount(fid, weights=x, minlength=n) / np.maximum(c, 1)
    v = np.bincount(fid, weights=(x - mu[fid]) ** 2, minlength=n) / np.maximum(c, 1)
    sd = np.sqrt(v)
    sd[sd < 1e-9] = 1.0
    return (x - mu[fid]) / sd[fid]


def main():
    arm = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]
    t0 = time.time()
    X = np.load(RV / "U2_X.npy")
    m = dict(np.load(RV / "U2_meta.npz"))
    nf = len(m["M"]); fid = np.repeat(np.arange(nf), m["M"])
    Mr = m["M"][fid]; c = m["cls"][fid]; y = m["y"]; re = m["rkend"].astype(np.int64)
    prow = c == 0
    init = np.zeros(len(fid))
    LA = json.load(open(AP / "age_lifts.json"))["lifts"]
    L_age = np.array([LA["prod_M<50"]["L"], LA["prod_M>=50"]["L"]])  # [M bucket, age bin]
    info = {}
    for tok in arm.split("+"):
        if tok == "base":
            continue
        elif tok.startswith("tprod"):
            a = float(tok[5:])
            grp = np.where(c >= 1, 2, np.where(Mr < 50, 0, 1))
            L = np.zeros((3, 3))
            for gg in range(3):
                base = y[(grp == gg) & (re >= 3)].mean()
                for k in range(3):
                    L[gg, k] = np.log(max(y[(grp == gg) & (re == k)].mean(), 1e-4) / base)
            info["trunc_lifts"] = np.round(L, 3).tolist()
            init += np.where((re <= 2) & (grp <= 1), a * L[grp, np.minimum(re, 2)], 0)
        elif tok.startswith("ageh") or tok.startswith("age"):
            a = float(tok[4:] if tok.startswith("ageh") else tok[3:])
            ab = age_bin(X[:, FEATS.index("f4")])
            mb = (Mr >= 50).astype(int)
            sel = prow & (Mr >= 50) if tok.startswith("ageh") else prow
            off = a * L_age[mb, ab]
            init += np.where(sel, off, 0.0)
            info["age_lifts"] = np.round(L_age, 3).tolist()
            info["age_rows_offset"] = int(sel.sum())
            info["age_offset_mean_abs"] = float(np.abs(off[sel]).mean())
        else:
            raise ValueError(tok)
    print(arm, "offset rows", int((init != 0).sum()), "of", len(init), json.dumps(info), flush=True)
    # readout sets
    Xv = np.load(G / "val_X.npy"); mv = dict(np.load(G / "val_meta.npz"))
    Xh = np.load(G / "HO_X.npy"); mh = dict(np.load(G / "HO_meta.npz"))
    Mv = mv["M"].astype(np.int64); offv = np.concatenate([[0], np.cumsum(Mv)[:-1]])
    Mh = mh["M"].astype(np.int64); offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    # stale val ('both' coverage replay)
    Xs = Xv.copy()
    fidv = np.repeat(np.arange(len(Mv)), Mv)
    for nm in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_pop", "seq_cv_norm"]:
        Xs[:, FEATS.index(nm)] = np.load(CF / f"both_{nm}.npy").astype(np.float32)
    Xs[:, FEATS.index("it_hist_all_luo_z")] = feed_z(Xs[:, FEATS.index("it_hist_all_luo")].astype(np.float64), fidv, len(Mv)).astype(np.float32)
    print(f"stale val: changed cells {(Xs != Xv).mean():.4f}; loaded in {time.time() - t0:.0f}s", flush=True)
    params = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, num_threads=THREADS, verbose=-1, max_bin=255)
    for seed in seeds:
        fn = OUT / f"res_{arm.replace('+', '_')}_s{seed}.npz"
        if fn.exists():
            print(arm, seed, "exists", flush=True)
            continue
        t1 = time.time()
        ds = lgb.Dataset(X, y, group=m["M"], init_score=init if np.any(init != 0) else None, feature_name=FEATS)
        b = lgb.train(dict(params, seed=seed), ds, ROUNDS)
        ttrain = time.time() - t1
        out = {}
        res = dict(arm=arm, seed=seed, train_secs=round(ttrain), **info)
        for k in (100, 200, 300, 400):
            pv = b.predict(Xv, num_iteration=k, num_threads=THREADS)
            ps = b.predict(Xs, num_iteration=k, num_threads=THREADS)
            ph = b.predict(Xh, num_iteration=k, num_threads=THREADS)
            nv = ndcg_per_feed(pv, mv["y"], offv, Mv); ns = ndcg_per_feed(ps, mv["y"], offv, Mv); nh = ndcg_per_feed(ph, mh["y"], offh, Mh)
            out[f"val{k}"] = nv.astype(np.float32); out[f"stale{k}"] = ns.astype(np.float32); out[f"ho{k}"] = nh.astype(np.float32)
            res[f"val{k}"] = round(float(nv.mean()), 5); res[f"stale{k}"] = round(float(ns.mean()), 5); res[f"ho{k}"] = round(float(nh.mean()), 5)
            if k == 200:
                out["pred_val200"] = pv.astype(np.float32)
        np.savez(fn, **out)
        res["secs"] = round(time.time() - t1)
        json.dump(res, open(fn.with_suffix(".json"), "w"))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
