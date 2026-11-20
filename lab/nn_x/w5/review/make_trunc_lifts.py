"""Per-M-bucket TRUNC lifts for train.py --trunc_lifts (CPU, read-only on data; writes nn_x/w5/trunc_lifts_mbucket.json).

Why: the review (lift_by_m.py) found that the production group 'prod_M<50' of trunc_lift.json mixes regimes (M1-16
rkend0 < 0, M33-49 rkend1/2 < 0 while the pooled M<50 lifts are all > 0). This script splits the production feeds into
the M buckets [1,16], [17,32], [33,49], [50,inf) and recomputes the lifts per bucket.

Definition (= nn_x/insights/data_usage/15_nn_artifacts.py, which wrote trunc_lift.json, restricted to one bucket, and
weighted exactly as nn_x/w5/review/lift_by_m.py):
    lift[b, k] = log( rate(b, rkend == k) / rate(b, rkend >= 3) ),  k = 0, 1, 2
    rate(b, cell) = sum_i w_f(i) y_i / sum_i w_f(i)  over the train candidates i of bucket-b production feeds in the cell,
    w_f = 1 - p_soft_f, p_soft = EDA posterior p_random (eda/labels_position/out/train_prandom.npz, NaN -> 0, no M<50
    rule; the same p_soft that scales the training offset), rkend = TRUE rank from the end
    (nn_artifacts/train_rkend_true.npy). The base rate is the bucket's own rkend >= 3 rate (15_nn_artifacts.py: the
    group's own rkend >= 3 rate).
Production feeds = the TRUNC-PROD rule of the training harness: NOT (M >= 50 and NN p_random > 0.5), where the NN
p_random is train.py's float32 p_random with M < 50 -> 0 (the pool / random-like feeds get no offset and are excluded
here); the 30000 selection-holdout feeds (train.py --holdout 30000: rng 12345 over NN p > 0.9 & M >= 50) are excluded
explicitly (they are a subset of the pool feeds anyway). Only train rows are used (no val / test).

Uncertainty / shrink: feed-clustered delta-method SE of each log lift (ratio estimators with per-feed sums, covariance of
the two cells of the same feeds included). Empirical-Bayes normal shrink toward 0: lift * tau2 / (tau2 + se^2), tau2 =
method-of-moments prior variance over the 12 cells. Applied ONLY to cells with few positives (< MIN_POS unweighted clicks
in the rkend cell or the base cell); otherwise the raw lift is used (factor reported for every cell).
Diagnostic only (NOT applied): 'mh_lift' = log Mantel-Haenszel rate ratio of rkend k vs rkend >= 3 stratified by exact M
(M capped at MH_CAP) inside the bucket, i.e. the lift with the between-M baseline differences removed (the pooled ratio
above mixes feeds of different M: the base cell over-weights long feeds, which have more rkend >= 3 rows).
  usage: .venv/bin/python nn_x/w5/review/make_trunc_lifts.py [out.json] [--buckets 1-16,17-32,33-49,50-]
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import polars as pl

ROOT = "<repo>"
ART = f"{ROOT}/nn_x/insights/data_usage/out/nn_artifacts"
_args = [x for x in sys.argv[1:]]
_bk = "1-16,17-32,33-49,50-"
if "--buckets" in _args:
    _i = _args.index("--buckets")
    _bk = _args[_i + 1]
    del _args[_i:_i + 2]
OUT = _args[0] if _args else f"{ROOT}/nn_x/w5/trunc_lifts_mbucket.json"
BUCKETS = tuple((int(a), int(b) if b else None) for a, b in (x.split("-") for x in _bk.split(",")))  # inclusive; None = inf
assert BUCKETS[0][0] == 1 and BUCKETS[-1][1] is None and all(BUCKETS[i + 1][0] == BUCKETS[i][1] + 1
                                                             for i in range(len(BUCKETS) - 1)), BUCKETS
MH_CAP = 1000
MIN_POS = 1000  # "few positives": fewer unweighted clicks than this in a cell -> EB shrink applied
POOL_P, HOLDOUT, HOLD_SEED = 0.5, 30000, 12345


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    t0 = time.time()
    M = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    n = len(M)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(n)).all(), "train_prandom.npz not in train feed order"
    p_soft = np.nan_to_num(pr["p_random"].astype(np.float64), nan=0.0)  # cd_arms5.p_soft_all
    p_nn = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)  # train.py prand_np
    p_nn[M < 50] = 0.0
    cand = np.where((p_nn > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(HOLD_SEED).choice(cand, min(HOLDOUT, len(cand)), replace=False))
    pool = (M >= 50) & (p_nn.astype(np.float64) > POOL_P)  # trunc_groups: group 2 (no offset)
    prod = ~pool
    prod[hold] = False
    assert pool[hold].all(), "holdout feeds should all be pool-eligible"
    # lift_by_m.py's float64 pool rule, for the record
    pn64 = p_soft.copy()
    pn64[M < 50] = 0
    n_rule_diff = int(((M >= 50) & (pn64 > 0.5) != pool).sum())

    rk = np.load(f"{ART}/train_rkend_true.npy")
    y = np.load(f"{ROOT}/data/flat/train/cand_target.npy")
    assert rk.shape == y.shape == (int(M.sum()),)
    fid = np.repeat(np.arange(n, dtype=np.int64), M)
    key = fid * 4 + np.minimum(rk, 3).astype(np.int64)
    del rk
    npos = np.bincount(key, weights=y.astype(np.float64), minlength=4 * n).reshape(n, 4)  # clicks per (feed, cell)
    nrow = np.bincount(key, minlength=4 * n).reshape(n, 4).astype(np.float64)  # rows per (feed, cell)
    del key, fid, y
    w = 1.0 - p_soft

    rows = []
    for lo, hi in BUCKETS:
        fsel = prod & (M >= lo) & ((M <= hi) if hi is not None else True)
        assert fsel.any(), f"bucket M {lo}-{hi}: no production train feeds"
        wf = w[fsel]
        A = npos[fsel] * wf[:, None]  # weighted clicks per feed and cell
        B = nrow[fsel] * wf[:, None]  # weighted rows per feed and cell
        SA, SB = A.sum(0), B.sum(0)
        rate = SA / SB
        lift = np.log(rate[:3] / rate[3])
        # feed-clustered delta method: z_fc = (A_fc - r_c B_fc) / SA_c  ->  var(log r_c) = sum_f z_fc^2
        Z = (A - rate[None, :] * B) / SA[None, :]
        se = np.array([np.sqrt(((Z[:, k] - Z[:, 3]) ** 2).sum()) for k in range(3)])
        # Mantel-Haenszel rate ratio stratified by exact M (person-time form: a = weighted clicks, T = weighted rows)
        st = np.minimum(M[fsel], MH_CAP)
        aS = np.stack([np.bincount(st, A[:, c], MH_CAP + 1) for c in range(4)], 1)
        TS = np.stack([np.bincount(st, B[:, c], MH_CAP + 1) for c in range(4)], 1)
        mh = []
        for k in range(3):
            tt = TS[:, k] + TS[:, 3]
            ok = (TS[:, k] > 0) & (TS[:, 3] > 0)
            mh.append(float(np.log((aS[ok, k] * TS[ok, 3] / tt[ok]).sum() / (aS[ok, 3] * TS[ok, k] / tt[ok]).sum())))
        rows.append(dict(
            name=f"M{lo}-{hi}" if hi is not None else f"M>={lo}", lo=lo, hi=hi, n_feeds=int(fsel.sum()),
            n_feeds_excluded_pool=int((pool & (M >= lo) & ((M <= hi) if hi is not None else True)).sum()),
            mean_w=float(wf.mean()), sum_w=float(wf.sum()),
            rows=[int(x) for x in nrow[fsel].sum(0)], pos=[int(round(x)) for x in npos[fsel].sum(0)],
            wrows=SB.tolist(), wpos=SA.tolist(), rate=rate.tolist(), raw=lift.tolist(), se=se.tolist(), mh_lift=mh))
    raw = np.array([r["raw"] for r in rows])
    assert np.isfinite(raw).all(), raw
    se = np.array([r["se"] for r in rows])
    tau2 = max(float((raw ** 2).mean() - (se ** 2).mean()), 1e-6)
    out_b = []
    for r, lr, s in zip(rows, raw, se):
        fac = tau2 / (tau2 + s ** 2)
        few = [min(r["pos"][k], r["pos"][3]) < MIN_POS for k in range(3)]
        app = np.where(few, lr * fac, lr)
        r.update(shrink_factor=fac.tolist(), few_pos=few, shrink_applied=bool(any(few)),
                 rkend0=float(app[0]), rkend1=float(app[1]), rkend2=float(app[2]))
        out_b.append(r)
    res = dict(
        description="per-M-bucket TRUNC log lifts for nn_x/w5 train.py --trunc_lifts (production feeds only; pool / "
                    "random-like (M>=50 and NN p_random > 0.5) and holdout feeds get no offset). Applied values: "
                    "buckets[i].rkend0/1/2. See nn_x/w5/review/make_trunc_lifts.py for the definition.",
        definition="lift[b,k] = log(rate(b, rkend==k) / rate(b, rkend>=3)), rate = sum (1-p_soft) y / sum (1-p_soft) "
                   "over train candidates of bucket-b production feeds (holdout + pool feeds excluded)",
        buckets=out_b, tau2=tau2, min_pos=MIN_POS, mh_cap=MH_CAP,
        n_train_feeds=n, n_holdout=int(len(hold)), n_pool_excluded=int(pool.sum()),
        n_pool_rule_diff_vs_float64=n_rule_diff,
        inputs={k: md5(v) for k, v in dict(
            rkend=f"{ART}/train_rkend_true.npy", prandom=f"{ROOT}/eda/labels_position/out/train_prandom.npz",
            target=f"{ROOT}/data/flat/train/cand_target.npy", feeds=f"{ROOT}/data/flat/train/feeds.parquet").items()},
        script="nn_x/w5/review/make_trunc_lifts.py", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)
    print(f"tau2 {tau2:.4f}; pool feeds excluded {int(pool.sum())} (holdout {len(hold)}); float32 vs float64 pool-rule "
          f"diffs {n_rule_diff}; {time.time() - t0:.0f}s")
    for r in out_b:
        print(f"{r['name']:>7}: feeds {r['n_feeds']:>8} mean w {r['mean_w']:.3f} | pos rk0/1/2/>=3 {r['pos']} | wpos "
              f"{np.round(r['wpos'], 0).tolist()} | rate {np.round(r['rate'], 4).tolist()} | raw lift "
              f"{np.round(r['raw'], 3).tolist()} se {np.round(r['se'], 3).tolist()} | EB factor "
              f"{np.round(r['shrink_factor'], 4).tolist()} applied {r['shrink_applied']} -> "
              f"{[round(r[f'rkend{k}'], 4) for k in range(3)]} | MH(M-strat, diag) {np.round(r['mh_lift'], 3).tolist()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
