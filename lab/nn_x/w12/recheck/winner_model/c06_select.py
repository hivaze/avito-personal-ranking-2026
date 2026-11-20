"""winner_model recheck, check 6: SELECT. Does the production ranker's candidate SELECTION carry ranking signal that the
pool does not have?  Production train feeds show the top-M of the production order; exploration feeds show a random
subset of a wider pool (check 4: even a random subset of a top-200+ production list is separable from a val/test set at
AUC 0.85-0.95). A classifier "candidate comes from a production feed (1) vs an exploration feed (0)" is therefore a
label-free estimate of how likely the production ranker is to put a candidate into the shown top.
  - trained on TRAIN rows only, no click target, no position, per-candidate RAW columns only (no within-feed relative
    column, no feed-level or user-level column); the two classes are M-matched feed by feed; the 30K holdout feeds are
    never in the fit; user-disjoint 10% of the sampled feeds are the test fold.
  - read-outs: held-out candidate AUC (also by production position band); on val and on the in-period holdout:
    NDCG of the score alone, within-feed Spearman to the pool, and the nested residual on W8 (weight chosen on one half
    of the users, read on the other half, both directions; nn_x/w12/forensics/fx.py).
Usage: NT=8 CUDA_VISIBLE_DEVICES="" python c06_select.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
NT = int(os.environ.get("NT", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("NUMBA_NUM_THREADS", str(NT))
os.environ.setdefault("POLARS_MAX_THREADS", str(NT))
sys.path.insert(0, "<repo>/src")
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from metrics import ndcg_per_feed

R = "<repo>/"
OUT = R + "nn_x/w12/recheck/winner_model/out/"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261114)
FA = [("flat", "cand_f1"), ("flat", "cand_f2"), ("flat", "cand_f3"), ("flat", "cand_f4"), ("flat", "cand_f5"),
      ("flat", "cand_category_id"), ("feats", "cos_u"), ("feats", "cos_hall"), ("feats", "cos_h10"), ("feats", "cos_h50"),
      ("feats", "cos_het1"), ("feats", "cos_het2"), ("feats", "cos_het3"), ("feats", "cos_het4"), ("feats", "cat_cnt_all"),
      ("feats", "cat_cnt_5"), ("feats", "cat_cnt_20"), ("feats", "cat_share_all"), ("feats", "cat_min_hpos"),
      ("feats", "attn32"), ("feats", "dot_hall"), ("feats", "maxsim32"), ("feats", "emb_missing")]
FB = FA + [("feats", "it_hist_all_luo"), ("feats", "it_hist_et1_luo"), ("feats", "seq_cv_invd20"), ("feats", "seq_cv_pop"),
           ("feats", "seq_cv_norm")]


def meta(sp):
    f = pl.read_parquet(f"{R}data/flat/{sp}/feeds.parquet")
    return f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64), f["user_id"].to_numpy().astype(np.int64)


def load(sp, rows, cols):
    """rows sorted ascending. -> float32 [n, F]"""
    X = np.empty((len(rows), len(cols)), np.float32)
    for j, (d, c) in enumerate(cols):
        X[:, j] = np.asarray(np.load(f"{R}data/{d}/{sp}/{c}.npy", mmap_mode="r")[rows], np.float32)
    return X


def zfeed(x, off, M):
    x = np.asarray(x, np.float64); fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, x) / M; sd = np.sqrt(np.maximum(np.bincount(fid, x * x) / M - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def nested(base, s, y, off, M, uid, ws=(-0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), seed=0):
    p0 = ndcg_per_feed(base, y, off, M)
    P = np.stack([ndcg_per_feed(base + w * s, y, off, M) for w in ws])
    r = np.random.default_rng(seed); uu = np.unique(uid); h = r.random(len(uu)) < 0.5
    half = h[np.searchsorted(uu, uid)]
    z0 = uid == 0; half[z0] = r.random(z0.sum()) < 0.5
    d = np.zeros(len(M)); wc = []
    for a in (True, False):
        tr = half == a; te = ~tr
        k = int(np.argmax(P[:, tr].mean(1))); wc.append(ws[k]); d[te] = P[k, te] - p0[te]
    best = int(np.argmax(P.mean(1)))
    curve = " ".join(f"{w:+.2f}:{(P[i] - p0).mean():+.5f}" for i, w in enumerate(ws))
    return d, wc, ws[best], (P[best] - p0), curve


se = lambda d: d.std() / np.sqrt(len(d))
f_ = lambda d: f"{d.mean():+.5f} +- {se(d):.5f} (z {d.mean() / se(d):+.1f})"

# ------------------------------------------------------------------ train sample
Mt, ot, ut = meta("train")
p2 = np.nan_to_num(np.load(R + "eda/labels_position/out/train_prandom_v2.npz")["p_random"].astype(np.float32), nan=0.5)
pr = np.nan_to_num(np.load(R + "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mt < 50] = 0.0
cand = np.where((pr > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))  # nn_x/w5/train.py:289-290
hold_rows = np.concatenate([np.arange(o, o + m) for o, m in zip(ot[hold], Mt[hold])])
yh = np.load(R + "nn_x/w5/preds/holdout/holdout_y.npy")
assert len(yh) == len(hold_rows) and (np.asarray(np.load(R + "data/flat/train/cand_target.npy", mmap_mode="r")[hold_rows]) == yh).all()
print(f"holdout reproduced: {len(hold)} feeds, {len(hold_rows)} rows", flush=True)

is_hold = np.zeros(len(Mt), bool); is_hold[hold] = True
E = np.where((p2 > 0.9) & ~is_hold)[0]
NE = int(os.environ.get("NE", "80000"))
E = rng.choice(E, min(NE, len(E)), replace=False)
Pall = np.where(p2 < 0.1)[0]
byM = {}
order = np.argsort(Mt[Pall], kind="stable"); Ps = Pall[order]; Ms = Mt[Ps]
lo = np.searchsorted(Ms, np.arange(0, 302));
Psel = []
for m, c in zip(*np.unique(Mt[E], return_counts=True)):
    grp = Ps[lo[m]:lo[m + 1]]
    Psel.append(rng.choice(grp, c, replace=len(grp) < c))
Psel = np.concatenate(Psel)
print(f"classifier feeds: exploration {len(E)} (mean M {Mt[E].mean():.1f}), production M-matched {len(Psel)} (mean M {Mt[Psel].mean():.1f})", flush=True)
KC = 40  # candidates per feed


def rows_of(feeds):
    out, fid = [], []
    for f in feeds:
        m = Mt[f]
        sel = np.arange(m) if m <= KC else rng.choice(m, KC, replace=False)
        out.append(ot[f] + sel); fid.append(np.full(len(sel), f))
    return np.concatenate(out), np.concatenate(fid)


rE, fE = rows_of(E); rP, fP = rows_of(Psel)
rows = np.r_[rE, rP]; fid = np.r_[fE, fP]; lab = np.r_[np.zeros(len(rE)), np.ones(len(rP))]
o = np.argsort(rows, kind="stable"); rows, fid, lab = rows[o], fid[o], lab[o]
u = ut[fid]
test = np.where(u == 0, fid % 10 == 0, (u * 2654435761 % 4294967296) % 10 == 0)
pos = np.asarray(np.load(R + "data/flat/train/cand_pos.npy", mmap_mode="r")[rows]).astype(np.int64)
print(f"rows {len(rows)} (test fold {int(test.sum())}); loading features", flush=True)
XB = load("train", rows, FB)

# ------------------------------------------------------------------ eval splits
Mv, ov, uv = meta("val")
yv = np.load(R + "data/flat/val/cand_target.npy")
XvB = load("val", np.arange(Mv.sum()), FB)
XhB = load("train", hold_rows, FB)
Mh, oh = np.load(R + "nn_x/w5/preds/holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
assert (Mh == Mt[hold]).all()
uh = ut[hold]
ft, to = np.load(R + "preds/ens_W8kd_ft_val.npy"), np.load(R + "preds/ens_W8kd_to_val.npy")
w8v = zfeed(0.5 * zfeed(ft, ov, Mv) + 0.5 * zfeed(to, ov, Mv), ov, Mv)
tov = zfeed(to, ov, Mv)
W8 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in (951, 952, 953, 954)]
w8h = zfeed(np.mean([zfeed(np.load(R + f"nn_x/w5/preds/holdout/nn_{v}_holdout.npy"), oh, Mh) for v in W8], 0), oh, Mh)
print(f"bases: W8 val {ndcg_per_feed(w8v, yv, ov, Mv).mean():.5f} | W8 TO val {ndcg_per_feed(tov, yv, ov, Mv).mean():.5f} "
      f"| W8 TO holdout {ndcg_per_feed(w8h, yh, oh, Mh).mean():.5f}", flush=True)


def spear(a, b, off, M, n=3000):
    r = np.random.default_rng(0).choice(len(M), min(n, len(M)), replace=False); out = []
    for f in r:
        x, y_ = a[off[f]:off[f] + M[f]], b[off[f]:off[f] + M[f]]
        out.append(np.corrcoef(x.argsort().argsort(), y_.argsort().argsort())[0, 1])
    return float(np.nanmean(out))


for name, cols in (("A content+affinity", FA), ("B A+popularity/co-vis", FB)):
    k = len(cols)
    cat = [5]
    ds = lgb.Dataset(XB[~test][:, :k], lab[~test], categorical_feature=cat, free_raw_data=False)
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, num_threads=NT, verbose=-1), ds, 300)
    pt = m.predict(XB[test][:, :k], num_threads=NT)
    lt, post = lab[test], pos[test]
    print(f"\n=== variant {name} ({k} columns): held-out candidate AUC production-vs-exploration {roc_auc_score(lt, pt):.4f} "
          f"(n {len(lt)})", flush=True)
    for a, b in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 400)):
        sel = (lt == 0) | ((post >= a) & (post < b))
        print(f"    production position [{a},{b}) vs exploration candidates: AUC {roc_auc_score(lt[sel], pt[sel]):.4f}; "
              f"mean score {pt[(lt == 1) & (post >= a) & (post < b)].mean():.3f} (exploration {pt[lt == 0].mean():.3f})")
    imp = sorted(zip(m.feature_importance("gain"), [c for _, c in cols]), reverse=True)[:8]
    print("    top gain: " + ", ".join(f"{c} {g / sum(x for x, _ in zip(m.feature_importance('gain'), cols)):.2f}" for g, c in imp))
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    ev_, eh_ = logit(m.predict(XvB[:, :k], num_threads=NT)), logit(m.predict(XhB[:, :k], num_threads=NT))
    np.save(OUT + f"select_{name[0]}_val.npy", ev_.astype(np.float32)); np.save(OUT + f"select_{name[0]}_holdout.npy", eh_.astype(np.float32))
    sv, sh = zfeed(ev_, ov, Mv), zfeed(eh_, oh, Mh)
    print(f"    score alone: val NDCG {ndcg_per_feed(ev_, yv, ov, Mv).mean():.5f} (random 0.2890; f1 alone "
          f"{ndcg_per_feed(XvB[:, 0], yv, ov, Mv).mean():.4f}) | holdout NDCG {ndcg_per_feed(eh_, yh, oh, Mh).mean():.5f}")
    print(f"    within-feed Spearman to the pool: val {spear(ev_, w8v, ov, Mv):.3f} | holdout {spear(eh_, w8h, oh, Mh):.3f}")
    for bn, base, y_, off_, M_, uid_, s_ in (("W8 (0.5 FT + 0.5 TO), val", w8v, yv, ov, Mv, uv, sv), ("W8 TO pool, val", tov, yv, ov, Mv, uv, sv),
                                             ("W8 TO pool, holdout", w8h, yh, oh, Mh, uh, sh)):
        d, wc, wb, db, curve = nested(base, s_, y_, off_, M_, uid_)
        print(f"    nested on {bn}: {f_(d)}; w chosen {wc}; in-sample best w {wb} {db.mean():+.5f} +- {se(db):.5f} | "
              f"M<50 {d[M_ < 50].mean() if (M_ < 50).any() else float('nan'):+.5f} M>=50 {d[M_ >= 50].mean():+.5f}")
        print(f"       curve {curve}", flush=True)
print("C06DONE")
