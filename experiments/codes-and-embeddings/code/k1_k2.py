# Audit of the one decoding cue with a positive sign on validation (the share of an item's card views
# that came from the recommendation feed, counted over other users). K1 rebuilds the counts from strictly
# earlier days and without the test split; K1b measures how well the forward-only part of the counts
# separates positives from negatives; K2 recomputes the holdout transfer with a user-clustered SE.
# The checks were declared in PREDECLARED.txt at 06:12:25 UTC on 2026-11-13, before the run.
# Run without arguments on CPU with 12 threads, 3 minutes; the command line is not on record.
# Produced k1_k2.log and k1_k2.json: the validation gain +0.00033 falls to -0.00001 once leak-free.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/codes/exploit/06_skeptic/k1_k2.py
#
"""06_skeptic K1/K2 (PREDECLARED.txt in this directory, written first). CPU only, 12 threads.
K1: A1 X1 on val with leak-free (V-lf: strictly earlier days) and no-test (train+val, any day) counts vs V-exact (parity with lensA).
K1b: pair AUC of the forward-only increment of n_s1. K2: A1 holdout transfer with a user-clustered SE.
Outputs: k1_k2.log, k1_k2.json (this directory only)."""
import os, sys, time, json
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *

HERE = ROOT / "nn_x/codes/exploit/06_skeptic"
T0 = time.time(); LOGF = open(HERE / "k1_k2.log", "w")


def log(s=""):
    s = f"[{time.time()-T0:6.0f}s] {s}"; print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


D0 = 20702
day_of = lambda ts: ((np.asarray(ts).astype(np.int64) + 10800) // 86400 - D0).astype(np.int64)
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); uv = fv["user_id"].to_numpy().astype(np.int64); dv = day_of(fv["timestamp"].to_numpy())
item_v = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64); fid_v = np.repeat(np.arange(len(Mv)), Mv)
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
want_items = np.unique(item_v)
want = np.zeros(int(max(want_items.max(), 2 ** 27)) + 1, bool); want[want_items] = True
parts = []
for si, s in enumerate(["train", "val", "test"]):
    fs = pl.read_parquet(FLAT / s / "feeds.parquet")
    u = fs["user_id"].to_numpy().astype(np.int64); fd = day_of(fs["timestamp"].to_numpy())
    hi = np.load(FLAT / s / "hist_item_id.npy"); t = np.load(FLAT / s / "hist_event_type.npy")
    keep = want[hi] & (t == 1); del t
    so = np.load(FLAT / s / "hist_event_source.npy")[keep]
    hi = hi[keep].astype(np.int64); hf = np.load(FLAT / s / "hist_fidx.npy")[keep].astype(np.int64)
    uk = u[hf]; uk = np.where(uk > 0, uk, -(si * 10_000_000 + hf + 1))
    base = dict(uk=uk, item=hi, day=fd[hf].astype(np.int16), sp=np.full(len(hi), si, np.int8))
    parts.append(pl.DataFrame({**base, "c": np.zeros(len(hi), np.int8)}))                       # class 0 = e1 any source
    m = so == 1
    parts.append(pl.DataFrame({k: v[m] for k, v in base.items()} | {"c": np.ones(int(m.sum()), np.int8)}))  # class 1 = e1 & s1
    log(f"  {s}: e1 rows on val-candidate items {keep.sum():,}")
    del hi, so, hf, uk, keep
D = pl.concat(parts).group_by(["uk", "item", "c"]).agg(pl.col("day").min(), pl.col("sp").min().alias("spmin"))
del parts
log(f"distinct (user key, item, class) {len(D):,}")
# no-test variant: a (key, item, class) counts iff it appears in train or val (spmin <= 1)
cu = pl.DataFrame({"row": np.arange(len(item_v)), "uk": uv[fid_v], "item": item_v, "fday": dv[fid_v].astype(np.int16)})
iidx = np.full(len(want), -1, np.int64); iidx[want_items] = np.arange(len(want_items)); nI = len(want_items)
g = D.group_by(["item", "c", "day"]).len(); g2 = D.filter(pl.col("spmin") <= 1).group_by(["item", "c"]).len()
ND = 25
CNT = {}
for c in (0, 1):
    m = (g["c"] == c).to_numpy()
    H_ = np.zeros((ND, nI), np.int64)
    H_[g["day"].to_numpy()[m], iidx[g["item"].to_numpy()[m]]] = g["len"].to_numpy()[m]
    cum = np.vstack([np.zeros((1, nI), np.int64), np.cumsum(H_, 0)])
    ii = iidx[item_v]
    tot = H_.sum(0)[ii].astype(np.float64)
    lf = cum[np.minimum(dv[fid_v], ND), ii].astype(np.float64)
    m2 = (g2["c"] == c).to_numpy(); nt = np.zeros(nI); nt[iidx[g2["item"].to_numpy()[m2]]] = g2["len"].to_numpy()[m2]
    CNT[("exact", c)], CNT[("lf", c)], CNT[("notest", c)] = tot, lf, nt[ii].astype(np.float64)
own = cu.filter(pl.col("uk") > 0).join(D, on=["uk", "item"], how="inner")
for var, filt in [("exact", None), ("lf", pl.col("day") < pl.col("fday")), ("notest", pl.col("spmin") <= 1)]:
    o = own if filt is None else own.filter(filt)
    o = o.group_by(["row", "c"]).len()
    for c in (0, 1):
        oc = o.filter(pl.col("c") == c)
        CNT[(var, c)][oc["row"].to_numpy()] -= oc["len"].to_numpy(); assert CNT[(var, c)].min() >= 0
ref = np.load(ROOT / "nn_x/codes/exploit/lensA_funnel/itsrc_val.npz")
par = (float(np.abs(CNT[("exact", 0)] - ref["e1"]).max()), float(np.abs(CNT[("exact", 1)] - ref["s1"]).max()))
log(f"parity V-exact vs lensA itsrc_val e1/s1: max |diff| {par}")
X = {}
for var in ("exact", "lf", "notest"):
    n1, ns1 = CNT[(var, 0)], CNT[(var, 1)]
    pc = np.bincount(cat, ns1, 64) / np.maximum(np.bincount(cat, n1, 64), 1)
    X[var] = (ns1 + 2 * pc[cat]) / (n1 + 2)
    log(f"  {var:6s}: mean n_e1 {n1.mean():.3f} n_s1 {ns1.mean():.3f}; share n_e1>0 {np.mean(n1 > 0):.4f}")
chk = float(np.abs(X["exact"] - np.load(ROOT / "nn_x/codes/exploit/lensA_funnel/cues_val.npz")["X1"]).max())
log(f"X1 exact vs lensA cues_val X1: max |diff| {chk:.2e}")

RES = dict(parity=par, x1_parity=chk)
V = Split("val"); g3 = g3_score(V); gz = V.fz(g3); base = V.nd(gz)
log(f"G3 val {base.mean():.5f}")
TL = Tail(V, g3)


def se_cl(S, v):
    """user-clustered SE of the mean of a per-feed vector (user-0 feeds singletons)."""
    r, se = ratio_cl(v, np.ones_like(v), S.cl); return se


for var in ("exact", "notest", "lf"):
    x = X[var]; tw = TL.win(x); dl, ws = nested(V, gz, x)
    se = dl.std() / np.sqrt(V.n); sec = se_cl(V, dl)
    log(f"K1 X1[{var:6s}] tail win {tw['win']:.4f} +- {tw['se']:.4f} | nested {dl.mean():+.5f} (SE {se:.5f}, z {dl.mean()/max(se,1e-12):+.2f}; "
        f"clustered SE {sec:.5f}) w {ws} | corr with exact {np.corrcoef(x, X['exact'])[0,1]:.3f}")
    RES[f"X1_{var}"] = dict(tail=tw["win"], tail_se=tw["se"], nested=float(dl.mean()), se=float(se), se_cl=float(sec), w=ws)
# fixed-weight curve (info): val delta at w = +0.05 for each variant
for var in ("exact", "notest", "lf"):
    d = V.nd(gz + 0.05 * V.fz(X[var])) - base
    log(f"K1 X1[{var:6s}] fixed w=+0.05 val delta {d.mean():+.5f} (SE {d.std()/np.sqrt(V.n):.5f})")
    RES[f"X1_{var}_w005"] = float(d.mean())
# K1b: forward-only increment of n_s1 (exact - lf) as a cue
inc = CNT[("exact", 1)] - CNT[("lf", 1)]
log(f"K1b forward increment n_s1: share > 0 {np.mean(inc > 0):.4f}, mean {inc.mean():.3f}")
tw = TL.win(inc)
# all within-feed (positive, negative) pairs: AUC via per-feed ranks
y = V.y; fid = V.fid
o = np.lexsort((inc, fid)); xs = inc[o]; fs = fid[o]
# mid-ranks within feed
r = np.empty(len(inc)); i = 0; N = len(inc)
starts = np.r_[0, np.flatnonzero((np.diff(fs) != 0) | (np.diff(xs) != 0)) + 1, N]
for a_, b_ in zip(starts[:-1], starts[1:]):
    r[o[a_:b_]] = (a_ + b_ - 1) / 2.0
fstart = np.r_[0, np.cumsum(V.M)[:-1]]; r = r - fstart[fid]
npos = np.bincount(fid, y, V.n); nneg = V.M - npos
sumr = np.bincount(fid, r * y, V.n)
ok = (npos > 0) & (nneg > 0)
U = sumr - npos * (npos - 1) / 2.0
auc_num = U[ok]; auc_den = (npos * nneg)[ok]
auc = auc_num.sum() / auc_den.sum()
log(f"K1b forward increment: tail win {tw['win']:.4f} +- {tw['se']:.4f}; pooled within-feed pos-vs-neg AUC {auc:.4f}")
RES["K1b"] = dict(share_pos=float(np.mean(inc > 0)), tail=tw["win"], auc_all=float(auc))
# same for the leak-free n_s1 level (reference)
# K2: holdout transfer with clustered SE
H = Split("hold"); gh = g3_score(H); ghz = H.fz(gh); bh = H.nd(ghz)
log(f"G3 holdout {bh.mean():.5f}; holdout feeds {H.n}, distinct user keys {H.cl.max()+1}")
A1 = ROOT / "nn_x/codes/exploit/01_feedpull"
for nm in ("X1", "Y2"):
    xh = np.load(A1 / f"hold_{nm}.npy").astype(np.float64)
    t = H.nd(ghz + 0.05 * H.fz(xh)) - bh
    se = t.std() / np.sqrt(H.n); sec = se_cl(H, t)
    log(f"K2 {nm} holdout transfer w=+0.05: {t.mean():+.5f} (feed SE {se:.5f}, z {t.mean()/se:+.2f}; user-clustered SE {sec:.5f}, z {t.mean()/sec:+.2f})")
    RES[f"K2_{nm}"] = dict(delta=float(t.mean()), se=float(se), se_cl=float(sec))
json.dump(RES, open(HERE / "k1_k2.json", "w"), indent=1, default=float)
log("done")
