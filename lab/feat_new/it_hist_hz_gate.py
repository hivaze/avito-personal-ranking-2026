"""CPU screen for it_hist_hz_* (train-fit LightGBM gate; harness = review LEAKAGE-7 lgb_causal.py, copied, not imported).

Training rows: 50K random-like non-holdout train feeds (p_random > 0.5, M >= 50, rng 7; = review/leakage
train_sample_feeds.npy), LightGBM lambdarank (eval_family params, 350 rounds), 2 seeds, 16 threads.
Arms (pre-declared):
  R0 = BASE + it_hist_{all,et1..4}_luo + within-feed rank/z of it_hist_all_luo        (as built)
  R2 = BASE + it_hist_hz_{all,et1..4}  + within-feed rank/z of it_hist_hz_all         (train rows thinned; holdout
       rows use L = 0; val rows are byte-identical to R0's)
  Rn = BASE only (no popularity), reference.
Evaluated on: all val feeds (paired per-feed delta R2 - R0, SE), the 30K random-like train holdout (per-feed SE ~2.3x too
small there), val splits (Tue/Thu, M<50/M>=50, feeds whose positives are all young/all old), and the review's trunc17
stale-val copy (val it_hist/seq_cv with train snapshots of days >= Sep 17 removed; diagnostic only, never acceptance).
PASS rule: val paired delta R2 - R0 > 2 SE.
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/it_hist_hz_gate.py [--seeds 2] [--deterministic --arms R0,R2 --tag _det4]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "16")
import sys, time, json, argparse
from pathlib import Path
import numpy as np, polars as pl, numba as nb, lightgbm as lgb
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
sys.path.insert(0, "<repo>/feat_new")
import it_hist_hz as H

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; NEW = ROOT / "data/feats_new"
SC = Path("/tmp/scratchpad/featrev/it_hist_hz")
CF = Path("/tmp/scratchpad/review/shift/out/cf")
TH = 16
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1, num_threads=TH)
BASE = ["f1", "f2", "f3", "f4", "f5", "f1_rk", "f2_rk", "f3_rk", "f4_rk", "f5_rk", "f1_z", "f2_z", "f3_z", "f4_z", "f5_z",
        "cos_u", "cos_u_rk", "cos_u_z", "maxsim32", "maxsim32_rk", "maxsim32_z", "attn32", "cos_h10", "cos_hall",
        "meansim32", "cos_het1", "cat_cnt_all", "cat_cnt_all_rk", "cat_cnt_et1", "cat_share_all", "cat_min_hpos",
        "feed_cat_share", "centrality", "H", "M", "item_norm", "dot_u", "category_id", "ctx_hs_k128", "ctx_hn_soft05",
        "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_top5h", "seq_cv_invd20",
        "seq_cv_pop", "seq_cv_norm", "u_het1_share"]
LUO = [f"it_hist_{n}_luo" for n in H.NAMES]
HZ = [f"it_hist_hz_{n}" for n in H.NAMES]
SEQ = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm"]


@nb.njit(parallel=True)
def rank_z(x, off, M, out):
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std()
        for i in range(m):
            lo = 0; eq = 0
            for j in range(m):
                if v[j] < v[i]:
                    lo += 1
                elif v[j] == v[i]:
                    eq += 1
            out[o + i, 0] = (lo + 0.5 * (eq - 1)) / max(m - 1, 1)
            out[o + i, 1] = (v[i] - mu) / (sd + 1e-6)


def feeds(split):
    return pl.read_parquet(FLAT / split / "feeds.parquet")


def rows_of(split, fids):
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    Ms = M[fids]
    rows = np.repeat(off[fids] - np.concatenate([[0], np.cumsum(Ms)[:-1]]), Ms) + np.arange(Ms.sum())
    return rows, np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64), Ms


def load(split, name, rows):
    for p in (NEW / split / f"{name}.npy", FEATS / split / f"{name}.npy", ROOT / "nn/feat_cache" / split / f"{name}.npy"):
        if p.exists():
            x = np.load(p, mmap_mode="r")
            return np.asarray(x if rows is None else x[rows], np.float32)
    raise FileNotFoundError(name)


def block(split, fids, stale=False):
    rows, off, M = rows_of(split, fids)
    X = {n: load(split, n, rows) for n in BASE + LUO + HZ}
    if stale:  # trunc17 stale-val copy (val only): popularity / co-vis inputs with train Sep 17-19 removed
        for n in H.NAMES:
            t = np.load(CF / f"trunc17_it_hist_{n}_luo.npy").astype(np.float32)
            X[f"it_hist_{n}_luo"] = t; X[f"it_hist_hz_{n}"] = t
        for n in SEQ:
            X[n] = np.load(CF / f"trunc17_{n}.npy").astype(np.float32)
    for n in ["it_hist_all_luo", "it_hist_hz_all"]:
        o = np.zeros((len(rows), 2), np.float64)
        rank_z(X[n].astype(np.float64), off, M, o)
        X[n + "_rk"] = o[:, 0].astype(np.float32); X[n + "_zz"] = o[:, 1].astype(np.float32)
    y = np.asarray(np.load(FLAT / split / "cand_target.npy", mmap_mode="r")[rows], np.float32)
    return X, y, off, M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true", help="LightGBM deterministic=True, force_row_wise (run-to-run exact)")
    ap.add_argument("--arms", default="Rn,R0,R2"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.deterministic:
        PARAMS.update(deterministic=True, force_row_wise=True)
    T0 = time.time()
    hold = H.holdout_feeds()
    f = feeds("train"); Mtr = f["M"].to_numpy().astype(np.int64)
    pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[Mtr < 50] = 0.0
    pool = np.setdiff1d(np.where((pr > 0.5) & (Mtr >= 50))[0], hold)
    trs = np.sort(np.random.default_rng(7).choice(pool, 50000, replace=False))
    ref = Path("/tmp/scratchpad/review/leakage/train_sample_feeds.npy")
    if ref.exists():
        print("train sample identical to review/leakage:", bool(np.array_equal(np.load(ref), trs)))
    fv = feeds("val")
    D = {"tr": block("train", trs), "ho": block("train", hold), "va": block("val", np.arange(len(fv))),
         "va17": block("val", np.arange(len(fv)), stale=True)}
    print("loaded", {k: len(v[1]) for k, v in D.items()}, f"{time.time()-T0:.0f}s", flush=True)
    for k in ("tr", "ho", "va"):
        X = D[k][0]
        print(f"  {k}: zero share luo {np.mean(X['it_hist_all_luo'] == 0):.4f} hz {np.mean(X['it_hist_hz_all'] == 0):.4f}; "
              f"rows changed {np.mean(X['it_hist_all_luo'] != X['it_hist_hz_all']):.4f}")
    ARMS = {"Rn": BASE, "R0": BASE + LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"],
            "R2": BASE + HZ + ["it_hist_hz_all_rk", "it_hist_hz_all_zz"]}
    ARMS = {k: v for k, v in ARMS.items() if k in a.arms.split(",")}
    res = {}
    for arm, names in ARMS.items():
        Xtr = np.stack([D["tr"][0][n] for n in names], 1)
        acc = {k: [] for k in ("va", "ho", "va17")}
        for s in range(a.seeds):
            ds = lgb.Dataset(Xtr, D["tr"][1], group=D["tr"][3], feature_name=names, categorical_feature=["category_id"])
            b = lgb.train(dict(PARAMS, seed=s), ds, 350)
            for key in acc:
                X, y, off, M = D[key]
                acc[key].append(ndcg_per_feed(b.predict(np.stack([X[n] for n in names], 1), num_threads=TH), y, off, M))
            if arm == "R2" and s == 0:
                imp = dict(zip(names, b.feature_importance("gain")))
                print("  R2 gain share of hz block", round(sum(v for k, v in imp.items() if "hz" in k) / sum(imp.values()), 4))
            if arm == "R0" and s == 0:
                imp = dict(zip(names, b.feature_importance("gain")))
                print("  R0 gain share of luo block", round(sum(v for k, v in imp.items() if "luo" in k) / sum(imp.values()), 4))
        res[arm] = {k: np.mean(v, 0) for k, v in acc.items()}
        res[arm]["seeds"] = {k: [float(x.mean()) for x in v] for k, v in acc.items()}
        print(f"{arm}: val {res[arm]['va'].mean():.5f} holdout {res[arm]['ho'].mean():.5f} stale-val {res[arm]['va17'].mean():.5f} "
              f"seeds {res[arm]['seeds']}  {time.time()-T0:.0f}s", flush=True)
    # splits of val
    tsv = fv["timestamp"].to_numpy().astype(np.int64); dayv = (tsv + H.MSK) // 86400
    Mv = fv["M"].to_numpy()
    knots = np.load(FEATS / "life_f4_age_knots.npy")
    _, cutv = H.cutoff_per_feed("val")
    fiv = np.load(FLAT / "val/cand_fidx.npy").astype(np.int64)
    Ev = (cutv[fiv] - tsv[fiv]) / 3600.0 + H.age_hours(np.load(FLAT / "val/cand_f4.npy"), knots)
    yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    npos = np.bincount(fiv, yv, len(fv)); nyp = np.bincount(fiv, yv * (Ev <= 0), len(fv))
    splits = {"Tue": dayv == dayv.min(), "Thu": dayv != dayv.min(), "M<50": Mv < 50, "M>=50": Mv >= 50,
              "pos all young": (npos > 0) & (nyp == npos), "pos some young": (nyp > 0) & (nyp < npos),
              "pos none young": (npos > 0) & (nyp == 0), "no pos": npos == 0}
    out = {}
    for a1, a0 in [p for p in [("R2", "R0"), ("R0", "Rn"), ("R2", "Rn")] if p[0] in res and p[1] in res]:
        o = {}
        for key, lab in [("va", "VAL"), ("ho", "HOLDOUT"), ("va17", "STALE-VAL trunc17")]:
            d = res[a1][key] - res[a0][key]
            o[key] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
            per_seed = [s1 - s0 for s1, s0 in zip(res[a1]["seeds"][key], res[a0]["seeds"][key])]
            print(f"{a1} - {a0} {lab:18s}: {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f})  z {d.mean()/(d.std()/np.sqrt(len(d))):+.2f}  per-seed {np.round(per_seed, 5).tolist()}")
        d = res[a1]["va"] - res[a0]["va"]
        for sn, m in splits.items():
            o[f"va_{sn}"] = (float(d[m].mean()), float(d[m].std() / np.sqrt(m.sum())), int(m.sum()))
            print(f"    {a1} - {a0} val {sn:15s} n={m.sum():5d}: {d[m].mean():+.5f} ({d[m].std()/np.sqrt(m.sum()):.5f})")
        out[f"{a1}-{a0}"] = o
    dv = res["R2"]["va"] - res["R0"]["va"]
    verdict = "PASS" if dv.mean() > 2 * dv.std() / np.sqrt(len(dv)) else "FAIL"
    print("GATE (val R2 - R0 > 2 SE):", verdict)
    out["verdict"] = verdict
    out["arms"] = {k: {kk: float(v[kk].mean()) for kk in ("va", "ho", "va17")} for k, v in res.items()}
    (SC / f"gate{a.tag}.json").write_text(json.dumps(out, indent=1))
    np.savez(SC / f"gate_perfeed{a.tag}.npz", **{f"{k}_{kk}": v[kk] for k, v in res.items() for kk in ("va", "ho", "va17")})


if __name__ == "__main__":
    main()
