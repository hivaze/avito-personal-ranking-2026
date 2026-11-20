"""CPU pre-gate for privileged-feature distillation (PFD; Xu et al. KDD 2020 arXiv 1907.05171, Yang et al. NeurIPS 2022
arXiv 2209.08754): would a TEACHER-ONLY item-level label aggregate make the teacher better on train-period feeds?

Privileged feature (train rows only, never a student input, never computed for val/test): leave-one-out click statistics
of the same item over the whole train window, from OTHER feeds (both time directions).
  ctr   = (C - y + a p0) / (N - 1 + a)                       a = 20 impressions
  coec  = (C - y + b) / (E - e + b)                          clicks over expected clicks; e = click rate of the row's
                                                             (domain, position) cell for production rows and of its
                                                             M-bucket for exploration rows; b = 0.25 expected clicks
  ctr_x = same as ctr from exploration (random-order) impressions only
Gate (TOP1_GAP_EXTERNAL_1114 lever 3): nested residual on the stored teacher logits over the 30K holdout feeds
(the teacher never trained on them) must be >= +0.003 NDCG.  Nested = z(teacher) + w z(feature), w picked on one
half of the users and read on the other half, both directions; paired per-feed SE.
Same-user exclusion is NOT applied here (own row only): re-shown (user, item) pairs are 0.18% of rows
(TOP1_GAP_FORENSICS_1114 section 3); a real PFD teacher must exclude the user's other feeds.
"""
import sys

sys.path.insert(0, "<repo>/src")
import numpy as np
import polars as pl
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = "<repo>"
TR = f"{ROOT}/data/flat/train"
KD = f"{ROOT}/nn_x/w12/kd/out"
f = pl.read_parquet(f"{TR}/feeds.parquet", columns=["user_id", "M", "cand_off"])
uid = f["user_id"].to_numpy().astype(np.int64)
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
n = len(M)
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[M < 50] = 0.0
sel = np.where((pr > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))

item = np.load(f"{TR}/cand_item_id.npy")
y = np.load(f"{TR}/cand_target.npy")
pos = np.load(f"{TR}/cand_pos.npy")
N_ROWS = len(y)
expl_feed = pr > 0.5
print(f"rows {N_ROWS}, feeds {n}, exploration-like feeds (p_random > 0.5, M >= 50) {int(expl_feed.sum())}", flush=True)

# expected clicks per row
e = np.empty(N_ROWS, np.float32)
CH = 20_000_000
mb = np.minimum(M // 25, 12)
# pass 1: cell rates
cp, np_ = np.zeros(300), np.zeros(300)
cm, nm = np.zeros(13), np.zeros(13)
for s in range(0, n, 200_000):
    fe = np.arange(s, min(n, s + 200_000))
    r0, r1 = off[fe[0]], off[fe[-1]] + M[fe[-1]]
    fk = np.repeat(fe, M[fe])
    yy = y[r0:r1].astype(np.float64)
    ex = expl_feed[fk]
    pp = np.clip(pos[r0:r1].astype(np.int64), 0, 299)
    cp += np.bincount(pp[~ex], yy[~ex], 300)
    np_ += np.bincount(pp[~ex], minlength=300)
    cm += np.bincount(mb[fk][ex], yy[ex], 13)
    nm += np.bincount(mb[fk][ex], minlength=13)
rp = cp / np.maximum(np_, 1)
rm = cm / np.maximum(nm, 1)
p0 = (cp.sum() + cm.sum()) / N_ROWS
print(f"global click rate {p0:.5f}; production rate at pos 0 / 10 / 50 / 150: {rp[0]:.4f} {rp[10]:.4f} {rp[50]:.4f} {rp[150]:.4f}; "
      f"exploration rate by M-bucket (first 6): {np.round(rm[:6], 4)}", flush=True)
for s in range(0, n, 200_000):
    fe = np.arange(s, min(n, s + 200_000))
    r0, r1 = off[fe[0]], off[fe[-1]] + M[fe[-1]]
    fk = np.repeat(fe, M[fe])
    ex = expl_feed[fk]
    pp = np.clip(pos[r0:r1].astype(np.int64), 0, 299)
    e[r0:r1] = np.where(ex, rm[mb[fk]], rp[pp])
del pos

K = int(item.max()) + 1
C = np.bincount(item, y, K)
Nn = np.bincount(item, minlength=K)
E = np.bincount(item, e, K)
xrow = np.repeat(expl_feed, M)
Cx = np.bincount(item[xrow], y[xrow], K)
Nx = np.bincount(item[xrow], minlength=K)
print(f"items {int((Nn > 0).sum())}; with >= 5 impressions {int((Nn >= 5).sum())}", flush=True)


def feats(fe):
    Ms = M[fe]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[fe], Ms)])
    it, yy, ee = item[rows], y[rows].astype(np.float64), e[rows].astype(np.float64)
    isx = np.repeat(expl_feed[fe], Ms)
    n1 = Nn[it] - 1
    F = {"ctr (a 20)": (C[it] - yy + 20 * p0) / (n1 + 20),
         "coec (b 0.25)": (C[it] - yy + 0.25) / (E[it] - ee + 0.25),
         "ctr_x expl-only (a 20)": (Cx[it] - yy * isx + 20 * p0) / (Nx[it] - isx + 20)}
    return Ms, rows, yy.astype(np.float32), n1, F


def auc_feed(x, yy, o2, Ms):
    fk = np.repeat(np.arange(len(Ms)), Ms)
    order = np.lexsort((x, fk))
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) - o2[fk[order]] + 1.0
    npos = np.add.reduceat(yy, o2)
    sr = np.add.reduceat(r * yy, o2)
    ok = (npos > 0) & (npos < Ms)
    return ((sr - npos * (npos + 1) / 2) / np.maximum(npos * (Ms - npos), 1))[ok].mean()


GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
se = lambda d: d.std() / np.sqrt(len(d))
for name, fe, teachers in [("HOLDOUT (teacher never trained on these feeds)", hold, ["mean11", "w8g1", "oof"]),
                           ("exploration train feeds, cross-fitted teacher only", np.sort(np.random.default_rng(7).choice(np.setdiff1d(sel, hold), 30000, replace=False)), ["oof"])]:
    Ms, rows, yy, n1, F = feats(fe)
    o2 = np.concatenate([[0], np.cumsum(Ms)[:-1]])
    half = (uid[fe] * 2654435761 % 4294967296) % 2
    half = np.where(uid[fe] == 0, np.arange(len(fe)) % 2, half)
    print(f"\n== {name}: {len(fe)} feeds, {len(rows)} rows; other-feed impressions of the item: >=1 {np.mean(n1 >= 1):.3f}, >=5 {np.mean(n1 >= 5):.3f}, >=20 {np.mean(n1 >= 20):.3f}")
    for k, x in F.items():
        print(f" within-feed AUC of {k}: {auc_feed(x, yy, o2, Ms):.4f}")
    for t in teachers:
        s = np.asarray(np.load(f"{KD}/teacher_{t}.npy", mmap_mode="r")[rows]).astype(np.float32)
        zt = feed_rank_z(o2, Ms, s)[:, 1]
        base = ndcg_per_feed(zt.astype(np.float32), yy, o2, Ms)
        print(f" teacher {t}: NDCG {base.mean():.5f}; within-feed AUC {auc_feed(s, yy, o2, Ms):.4f}")
        for k, x in F.items():
            zf = feed_rank_z(o2, Ms, x.astype(np.float32))[:, 1]
            nd = np.stack([ndcg_per_feed((zt + w * zf).astype(np.float32), yy, o2, Ms) for w in GRID])
            wa = int(np.argmax(nd[:, half == 0].mean(1)))
            wb = int(np.argmax(nd[:, half == 1].mean(1)))
            d = np.where(half == 1, nd[wa], nd[wb]) - base
            wi = int(np.argmax(nd.mean(1)))
            print(f"   + {k:24s} nested {d.mean():+.5f}+-{se(d):.5f} (w picked {GRID[wa]} / {GRID[wb]}); in-sample best w {GRID[wi]} "
                  f"{(nd[wi] - base).mean():+.5f}; curve " + " ".join(f"{w}:{(nd[i] - base).mean():+.4f}" for i, w in enumerate(GRID) if w))
