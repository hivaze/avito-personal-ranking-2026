"""CPU measurement (kd_literature): how much of the click label sits inside the IN-SAMPLE teacher target, and is the
teacher's softmax calibrated at T = 1?

Preconditions being tested
  * cross-fitted KD (Dao et al. 2021, arXiv 2104.09732) pays when the in-sample teacher over-fits its own labels;
  * temperature arms (kwT07 / kwT15) pay when softmax(teacher / 1) is mis-calibrated on unseen feeds;
  * top-k focused rank distillation (RankDistil / RRD) pays when the target mass is concentrated in the head.
Matched comparison for the first point: the 15 cross-fit fold models (teacher_xf<K> = SUM of the 3 models that did NOT
train on user fold K). For a train feed of fold j: OOF = xf_j / 3 (never saw it), IN = mean of xf_k, k != j (12 models
that all trained on it). Same recipe, same data share -> the IN - OOF difference is pure memorisation.
Feed sets (30K each): holdout (unseen by every model), exploration train feeds (M >= 50, p_random > 0.9), production
train feeds M >= 50 (p_random < 0.1), short train feeds M < 50.
No model input is built here; train labels are used only to score stored train-only predictions.
"""
import sys

sys.path.insert(0, "<repo>/src")
import numpy as np
import polars as pl
from metrics import ndcg_per_feed

ROOT = "<repo>"
KD = f"{ROOT}/nn_x/w12/kd/out"
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["user_id", "M", "cand_off"])
uid = f["user_id"].to_numpy().astype(np.int64)
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
n = len(uid)
xf = np.where(uid == 0, np.arange(n) % 5, (uid * 2654435761 % 4294967296) % 5)
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[M < 50] = 0.0
sel = np.where((pr > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))  # == train.py --holdout 30000
rng = np.random.default_rng(7)
sets = {
    "holdout (unseen)": hold,
    "exploration train": np.sort(rng.choice(np.setdiff1d(sel, hold), 30000, replace=False)),
    "production M>=50": np.sort(rng.choice(np.where((pr < 0.1) & (M >= 50))[0], 30000, replace=False)),
    "short M<50": np.sort(rng.choice(np.where(M < 50)[0], 30000, replace=False)),
}
y_mm = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
T_mm = {k: np.load(f"{KD}/teacher_{k}.npy", mmap_mode="r") for k in
        ["mean11", "mean25", "w8g1", "oof", "xf0", "xf1", "xf2", "xf3", "xf4"]}
TG = np.round(np.arange(0.5, 2.55, 0.05), 2)
se = lambda d: d.std() / np.sqrt(len(d))


def lsm(x, o, fk):
    x = x.astype(np.float64)
    mx = np.maximum.reduceat(x, o)[fk]
    return x - (np.log(np.add.reduceat(np.exp(x - mx), o))[fk] + mx)


for name, fe in sets.items():
    Ms = M[fe]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[fe], Ms)])
    o2 = np.concatenate([[0], np.cumsum(Ms)[:-1]])
    fk = np.repeat(np.arange(len(fe)), Ms)
    y = np.asarray(y_mm[rows]).astype(np.float32)
    ys = np.add.reduceat(y, o2)
    okf = ys > 0
    fold = xf[fe]
    S = {k: np.asarray(T_mm[k][rows]).astype(np.float64) for k in T_mm}
    xs = np.stack([S[f"xf{k}"] for k in range(5)])  # sums of 3
    fold_row = fold[fk]
    oof = xs[fold_row, np.arange(len(rows))] / 3.0
    ins = (xs.sum(0) - xs[fold_row, np.arange(len(rows))]) / 12.0
    V = {"mean11": S["mean11"], "mean25": S["mean25"], "w8g1 (8 wide students)": S["w8g1"],
         "xf OOF (3 models)": oof, "xf IN-sample (12 models)": ins,
         "xf IN-sample, 3 of the 12": None}
    # a 3-model in-sample teacher of matched size: the fold sum (j+1) % 5 (all 3 models trained on fold j)
    V["xf IN-sample, 3 of the 12"] = xs[(fold_row + 1) % 5, np.arange(len(rows))] / 3.0
    print(f"\n== {name}: {len(fe)} feeds, {len(rows)} rows, feeds with a positive {int(okf.sum())}, mean M {Ms.mean():.1f}, "
          f"positives per feed {ys[okf].mean():.2f}")
    res = {}
    for k, s in V.items():
        nd = ndcg_per_feed(s.astype(np.float32), y, o2, Ms)
        nll = {}
        for T in TG:
            lp = lsm(s / T, o2, fk)
            nll[T] = (-(np.add.reduceat(y * lp, o2) / np.maximum(ys, 1)))[okf]
        best = min(TG, key=lambda T: nll[T].mean())
        lp1 = lsm(s, o2, fk)
        p1 = np.exp(lp1)
        ent = -np.add.reduceat(p1 * lp1, o2)
        pmax = np.maximum.reduceat(p1, o2)
        # mass of the 10 largest target probabilities of each feed
        order = np.lexsort((-p1, fk))
        rk = np.empty(len(rows), np.int64)
        rk[order] = np.arange(len(rows)) - o2[fk[order]]
        top10 = np.add.reduceat(p1 * (rk < 10), o2)
        ppos = np.add.reduceat(p1 * y, o2)[okf]  # target mass on the clicked candidates
        res[k] = dict(nd=nd, nll1=nll[1.0], lp1=lp1, ppos=ppos)
        print(f" {k:<28s} NDCG {nd.mean():.5f}  NLL(T=1) {nll[1.0].mean():.4f}  T* {best:.2f} (NLL {nll[best].mean():.4f}, "
              f"gain {nll[1.0].mean() - nll[best].mean():+.4f}+-{se(nll[1.0] - nll[best]):.4f})  entropy {ent.mean():.3f} nats "
              f"(log M {np.log(Ms).mean():.3f})  max p median {np.median(pmax):.3f}  top-10 mass {top10.mean():.3f}  "
              f"mass on clicks {ppos.mean():.4f}")
    if name != "holdout (unseen)":
        for a, b in (("xf IN-sample (12 models)", "xf OOF (3 models)"), ("xf IN-sample, 3 of the 12", "xf OOF (3 models)")):
            d_nd = res[a]["nd"] - res[b]["nd"]
            d_nll = (res[b]["nll1"] - res[a]["nll1"])
            # mean within-feed-centred log-prob shift of clicked candidates = memorisation offset delta (nats)
            dl = res[a]["lp1"] - res[b]["lp1"]
            dpos = (np.add.reduceat(dl * y, o2) / np.maximum(ys, 1))[okf]
            pa, pb = res[a]["ppos"], res[b]["ppos"]
            lam = (pa - pb) / np.maximum(1 - pb, 1e-9)  # q_in = (1 - lam) q_oof + lam onehot  (per feed, implied)
            print(f"   {a} minus {b}: NDCG {d_nd.mean():+.5f}+-{se(d_nd):.5f}; log p(click) {dpos.mean():+.4f}+-{se(dpos):.4f} nats; "
                  f"mass on clicks {pb.mean():.4f} -> {pa.mean():.4f}; implied hard-label share lambda {lam.mean():+.4f}+-{se(lam):.4f} "
                  f"(median {np.median(lam):+.4f})")
