"""CPU GBDT (LightGBM lambdarank, full-list truncation) proxy for TRAIN-DATA-USAGE arms. Read-only on the cache.

usage: OMP_NUM_THREADS=12 python 07_gbdt_arms.py ARM [ARM ...] --seed S [--rounds 400]
Every arm trains on the same fixed 150K-feed natural-mix train sample U (or a documented subset/superset) and is read
out on val (exact metric, pessimistic ties) and on 10K train-period random-like holdout feeds; per-feed NDCG saved for
paired deltas. Offsets (init_score) act only at training time (prediction = f(x) without offset), like a fixed PAL
tower; weights are per-feed constants (LightGBM multiplies lambdas/hessians per row).

Arm grammar (combine with '+'): base | exam<a> (a * log theta_rand(rank) on production rows; theta from random-like M>=50
feeds) | prodctr<a> (a * log production CTR(rank), the confounded curve) | trunc<a> (a * log end-of-feed lift by
class, rkend 0-2) | ips<c> (per-feed weight 1/theta_rand(rank of positives), clipped to [1/c, c], production only) |
lgbpos (LightGBM native position GAM) | rho<k> (keep top k% of production feeds by RHO excess r101) | rnd<k> (keep a random
k% of production feeds) | pool<w> (feed weight w on random-like feeds) | addX (append 30K extra pool feeds) |
droptop (drop production feeds whose first positive is at rank <= 2) | droplast (drop feeds with a last-item positive)
| short<w> (feed weight w on M<50 feeds).
"""
import argparse
import json
import time
import sys
import numpy as np
import lightgbm as lgb
from pathlib import Path

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed

D = ROOT / "nn_x/insights/data_usage/out"
G = D / "gb"


def load(name):
    X = np.load(G / f"{name}_X.npy", mmap_mode="r")
    m = dict(np.load(G / f"{name}_meta.npz"))
    return X, m


THETA = np.clip(np.load(D / "theta_rand_rank.npy"), 0.3, 5.0)
PCTR = np.clip(np.load(D / "ctr_prod_rank.npy"), 0.1, 10.0)


def end_lifts(m):
    """log lift of rkend 0/1/2 vs rkend>=3, per group (prod M<50, prod M>=50, random-like) from the sample itself."""
    fid = np.repeat(np.arange(len(m["M"])), m["M"])
    Mr = m["M"][fid]; c = m["cls"][fid]; y = m["y"]; re = m["rkend"]
    grp = np.where(c >= 1, 2, np.where(Mr < 50, 0, 1))
    L = np.zeros((3, 3))
    for g in range(3):
        base = y[(grp == g) & (re >= 3)].mean()
        for k in range(3):
            L[g, k] = np.log(max(y[(grp == g) & (re == k)].mean(), 1e-4) / base)
    return L, grp


def build(arm, seed):
    X, m = load("U")
    nf = len(m["M"])
    keep = np.ones(nf, bool)
    fw = np.ones(nf)
    off_parts = []
    prod = m["cls"] == 0
    rng = np.random.default_rng(1000 + seed)
    fid = np.repeat(np.arange(nf), m["M"])
    rank = m["rank"].astype(np.int64); prow = prod[fid]
    init = np.zeros(len(fid))
    position = None
    extra = None
    addcols, addinf = [], []  # training-only extra feature columns and their constant inference values
    for tok in arm.split("+"):
        if tok == "base":
            continue
        if tok.startswith("exam"):
            a = float(tok[4:]); init += np.where(prow, a * np.log(THETA[np.minimum(rank, 299)]), 0)
        elif tok.startswith("prodctr"):
            a = float(tok[7:]); init += np.where(prow, a * np.log(PCTR[np.minimum(rank, 299)]), 0)
        elif tok.startswith("tprod") or tok.startswith("trand"):  # truncation offset on production / random-like rows only
            a = float(tok[5:]); L, grp = end_lifts(m); re = m["rkend"].astype(np.int64)
            sel = (grp <= 1) if tok.startswith("tprod") else (grp == 2)
            init += np.where((re <= 2) & sel, a * L[grp, np.minimum(re, 2)], 0)
        elif tok.startswith("trunc"):
            a = float(tok[5:]); L, grp = end_lifts(m); re = m["rkend"].astype(np.int64)
            init += np.where(re <= 2, a * L[grp, np.minimum(re, 2)], 0)
        elif tok.startswith("ips"):
            c = float(tok[3:]); y = m["y"]
            inv = np.bincount(fid, weights=y / THETA[np.minimum(rank, 299)], minlength=nf) / np.maximum(np.bincount(fid, weights=y, minlength=nf), 1)
            w = np.clip(inv, 1 / c, c)
            w = w / w[prod].mean()
            fw = np.where(prod, fw * w, fw)
        elif tok == "lgbpos":
            rb = np.minimum(rank, 19)
            rb = np.where(rank >= 20, 20, rb); rb = np.where(rank >= 30, 21, rb); rb = np.where(rank >= 50, 22, rb); rb = np.where(rank >= 100, 23, rb)
            eb = np.minimum(m["rkend"].astype(np.int64), 3)
            position = np.where(prow, rb * 4 + eb, 96 + eb).astype(np.int32)
        elif tok.startswith("rhoinv"):  # keep the LOWEST k% excess (production-agreeing / easy feeds)
            k = float(tok[6:]) / 100; ex = m["excess101"]; pm = prod & np.isfinite(ex)
            cut = np.quantile(ex[pm], k)
            keep &= ~(pm & (ex > cut))
        elif tok.startswith("rho"):
            k = float(tok[3:]) / 100; ex = m["excess101"]; pm = prod & np.isfinite(ex)
            cut = np.quantile(ex[pm], 1 - k)
            keep &= ~(pm & (ex < cut))
        elif tok.startswith("rnd"):
            k = float(tok[3:]) / 100
            keep &= ~(prod & (rng.random(nf) > k))
        elif tok.startswith("pool"):
            w = float(tok[4:]); fw = np.where(m["cls"] >= 1, fw * w, fw)
        elif tok.startswith("coldw"):  # label-free: weight w on feeds whose CANDIDATE cold share is in the top quartile
            import polars as pl
            w = float(tok[5:])
            cs = pl.read_parquet(D / "train_feeds.parquet", columns=["coldshare"])["coldshare"].to_numpy()[m["fidx"]]
            fw = np.where(cs >= np.quantile(cs, 0.75), fw * w, fw)
        elif tok.startswith("short"):
            w = float(tok[5:]); fw = np.where(m["M"] < 50, fw * w, fw)
        elif tok == "droptop":
            keep &= ~(prod & (m["r0"] <= 2))
        elif tok == "droplast":
            keep &= ~m["pos_last"]
        elif tok == "addX":
            extra = "X"
        elif tok == "dom":  # NN-style traffic-domain input: production 0, random-like p_random; inference 1 (random-like)
            addcols.append(m["prand"][fid].astype(np.float32)); addinf.append(1.0)
        elif tok == "posfeat":  # EDA b4: shown rank and rank-from-end as training features, fixed (20, 50) at inference
            addcols.append(rank.astype(np.float32)); addinf.append(20.0)
            addcols.append(m["rkend"].astype(np.float32)); addinf.append(50.0)
        else:
            raise ValueError(tok)
    rows = keep[fid]
    Xt = np.asarray(X)[rows]
    if addcols:
        Xt = np.hstack([Xt, np.stack([c[rows] for c in addcols], 1)])
    yt = m["y"][rows]; gt = m["M"][keep]; it = init[rows]; wt = fw[fid][rows]
    pt = position[rows] if position is not None else None
    if extra:
        X2, m2 = load(extra)
        X2a = np.asarray(X2)
        if addcols:  # extra pool feeds: dom = their p_random, posfeat = their own shown rank / rank-from-end
            ex = []
            for tok in arm.split("+"):
                if tok == "dom":
                    ex.append(np.repeat(m2["prand"], m2["M"]).astype(np.float32))
                elif tok == "posfeat":
                    ex += [m2["rank"].astype(np.float32), m2["rkend"].astype(np.float32)]
            X2a = np.hstack([X2a, np.stack(ex, 1)])
        Xt = np.concatenate([Xt, X2a]); yt = np.concatenate([yt, m2["y"]]); gt = np.concatenate([gt, m2["M"]])
        it = np.concatenate([it, np.zeros(len(m2["y"]))])
        w2 = np.ones(len(m2["y"]))
        if "pool" in arm:  # same random-like weight for the extra pool feeds
            w2 *= float([t for t in arm.split("+") if t.startswith("pool")][0][4:])
        wt = np.concatenate([wt, w2])
        if pt is not None:
            pt = np.concatenate([pt, 96 + np.minimum(m2["rkend"].astype(np.int32), 3)])
    return Xt, yt, gt, it, wt, pt, addinf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    Xv, mv = load("val")
    Xh, mh = load("HO")
    Mv = mv["M"].astype(np.int64); offv = np.concatenate([[0], np.cumsum(Mv)[:-1]])
    Mh = mh["M"].astype(np.int64); offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
    Xv = np.asarray(Xv); Xh = np.asarray(Xh)
    feats = json.load(open(G / "feats.json"))
    for arm in a.arms:
        t0 = time.time()
        Xt, yt, gt, it, wt, pt, addinf = build(arm, a.seed)
        Xva, Xha = Xv, Xh
        if addinf:
            Xva = np.hstack([Xv, np.tile(np.array(addinf, np.float32), (len(Xv), 1))])
            Xha = np.hstack([Xh, np.tile(np.array(addinf, np.float32), (len(Xh), 1))])
        fn_ = feats + [f"extra{i}" for i in range(len(addinf))]
        params = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=10.0, num_threads=a.threads, verbose=-1, seed=a.seed, max_bin=255,
                      lambdarank_position_bias_regularization=0.0)
        ds = lgb.Dataset(Xt, yt, group=gt, init_score=it if np.any(it != 0) else None,
                         weight=wt if np.any(wt != 1) else None, position=pt, feature_name=fn_, free_raw_data=True)
        b = lgb.train(params, ds, a.rounds)
        res = dict(arm=arm, seed=a.seed, n_feeds=int(len(gt)), n_rows=int(len(yt)), secs=0.0)
        out = {}
        for k in range(100, a.rounds + 1, 100):
            pv = b.predict(Xva, num_iteration=k, num_threads=a.threads)
            ph = b.predict(Xha, num_iteration=k, num_threads=a.threads)
            nv = ndcg_per_feed(pv, mv["y"], offv, Mv); nh = ndcg_per_feed(ph, mh["y"], offh, Mh)
            out[f"val{k}"] = nv.astype(np.float32); out[f"ho{k}"] = nh.astype(np.float32)
            res[f"val{k}"] = float(nv.mean()); res[f"ho{k}"] = float(nh.mean())
        out["pred_val"] = pv.astype(np.float32)
        res["secs"] = time.time() - t0
        fn = f"res_{arm.replace('+', '_')}_s{a.seed}{a.tag}"
        np.savez(G / f"{fn}.npz", **out)
        json.dump(res, open(G / f"{fn}.json", "w"))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
