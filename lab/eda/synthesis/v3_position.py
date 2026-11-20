"""V3: position bias / random-like train feeds / f4 sign flip, plus expected LB offset from feed-size composition.
Samples: all train feeds with M>=50 and P(random)>0.9 (eda_labels_position detector), an M-matched-size random sample of
production-like feeds (P<0.05, M>=50), and all of val."""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

FLAT = Path("<repo>/data/flat")
rng = np.random.default_rng(0)
ft = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ft["M"].to_numpy().astype(np.int64); offtr = ft["cand_off"].to_numpy().astype(np.int64)
z = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
pr = np.full(len(Mtr), np.nan); pr[z["fidx"]] = z["p_random"]
cpos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
ctg = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
cf4 = np.load(FLAT / "train/cand_f4.npy", mmap_mode="r")
cf1 = np.load(FLAT / "train/cand_f1.npy", mmap_mode="r")

rnd = np.where((pr > 0.9) & (Mtr >= 50))[0]
prod_all = np.where((pr < 0.05) & (Mtr >= 50))[0]
prod = np.sort(rng.choice(prod_all, size=min(len(rnd), len(prod_all)), replace=False))
print(f"train feeds M>=50: {(Mtr>=50).sum()}, P>0.9: {len(rnd)}, P<0.05: {len(prod_all)} (sampled {len(prod)}); "
      f"mean P over M>=50 (non-NaN): {np.nanmean(pr[Mtr>=50]):.3f}", flush=True)


def gather(feeds):
    M = Mtr[feeds]; off0 = offtr[feeds]
    idx = np.concatenate([np.arange(o, o + m) for o, m in zip(off0, M)])
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    return M, off, idx


def nd(s, y, off, M):
    return ndcg_per_feed(s + 1e-9 * rng.random(len(s)), y, off, M)  # random tie-break


def describe(name, M, off, y, pos, f4, f1):
    fid = np.repeat(np.arange(len(M)), M)
    o = np.lexsort((pos, fid))
    rk = np.empty(len(o), np.int64); rk[o] = np.arange(len(o)) - np.repeat(off, M)
    rkend = np.repeat(M, M) - 1 - rk
    rates = {r: y[rk == r].mean() for r in [0, 1, 2, 5, 10, 20, 30, 100]}
    out = f"{name}: feeds {len(M)}, base {y.mean()*100:.2f}%, rank0 {rates[0]*100:.2f}%, r1 {rates[1]*100:.2f}, r2 {rates[2]*100:.2f}, " \
          f"r5 {rates[5]*100:.2f}, r10 {rates[10]*100:.2f}, r30 {rates[30]*100:.2f}, r100 {rates[100]*100:.2f}, last {y[rkend==0].mean()*100:.1f}%"
    print(out, flush=True)
    rand = nd(np.zeros(len(y)), y, off, M).mean()
    posasc = nd(-pos.astype(np.float64), y, off, M).mean()
    print(f"   NDCG random {rand:.4f}  pos-asc {posasc:.4f} (+{posasc-rand:.4f})  +f4 {nd(f4,y,off,M).mean():.4f}  -f4 {nd(-f4,y,off,M).mean():.4f}"
          f"  +f1 {nd(f1,y,off,M).mean():.4f}", flush=True)
    # f4 decile ratio of target rate (deciles of global val-like thresholds computed within the group)
    q = np.quantile(f4, np.linspace(0, 1, 11)); d = np.clip(np.searchsorted(q, f4, side="right") - 1, 0, 9)
    print("   f4 decile rate / mean:", " ".join(f"{y[d==k].mean()/y.mean():.2f}" for k in range(10)), flush=True)
    return np.bincount(np.digitize(M, [50, 75, 100, 150, 200, 250, 301]), minlength=8)[1:7] / len(M)


shares = {}
for name, feeds in [("train random-like (P>0.9,M>=50)", rnd), ("train production-like (P<0.05,M>=50)", prod)]:
    M, off, idx = gather(feeds)
    shares[name] = describe(name, M, off, np.asarray(ctg[idx], np.float64), np.asarray(cpos[idx]), np.asarray(cf4[idx], np.float64),
                            np.asarray(cf1[idx], np.float64))

fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy()
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
f4v = np.load(FLAT / "val/cand_f4.npy").astype(np.float64); f1v = np.load(FLAT / "val/cand_f1.npy").astype(np.float64)
mv = Mv >= 50
sel = np.repeat(mv, Mv)
Mv50 = Mv[mv]; offv50 = np.concatenate([[0], np.cumsum(Mv50)[:-1]])
print(f"val M>=50: feeds {mv.sum()}, base {yv[sel].mean()*100:.2f}%  +f4 {nd(f4v[sel],yv[sel],offv50,Mv50).mean():.4f}  "
      f"-f4 {nd(-f4v[sel],yv[sel],offv50,Mv50).mean():.4f}  +f1 {nd(f1v[sel],yv[sel],offv50,Mv50).mean():.4f}  random {nd(np.zeros(sel.sum()),yv[sel],offv50,Mv50).mean():.4f}")
q = np.quantile(f4v[sel], np.linspace(0, 1, 11)); d = np.clip(np.searchsorted(q, f4v[sel], side="right") - 1, 0, 9)
print("   val f4 decile rate / mean:", " ".join(f"{yv[sel][d==k].mean()/yv[sel].mean():.2f}" for k in range(10)))
vsh = np.bincount(np.digitize(Mv50, [50, 75, 100, 150, 200, 250, 301]), minlength=8)[1:7] / len(Mv50)
print("M-bin shares within M>=50 [50,75,100,150,200,250,300]: val", np.round(vsh, 3))
for k, v in shares.items(): print("   ", k, np.round(v, 3))

# ---- expected random NDCG (closed form) per val feed; reweight per M-bin to public / private composition
npos = np.add.reduceat(yv, offv).astype(np.int64)
disc = 1 / np.log2(np.arange(2, 302))
cd = np.concatenate([[0], np.cumsum(disc)])
exp_rand = (npos / Mv) * cd[Mv] / cd[npos]
bins = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
bv = np.digitize(Mv, bins) - 1
per_bin = np.array([exp_rand[bv == b].mean() for b in range(len(bins) - 1)])
fte = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = fte["M"].to_numpy(); tst = fte["timestamp"].to_numpy()
for nm, m in [("val", None), ("test public", tst < 1790497920), ("test private", tst >= 1790497920), ("test all", np.ones(len(Mt), bool))]:
    if m is None:
        print(f"expected random NDCG {nm}: {exp_rand.mean():.4f}")
    else:
        bt = np.digitize(Mt[m], bins) - 1
        w = np.bincount(bt, minlength=len(bins) - 1) / m.sum()
        print(f"expected random NDCG {nm} (val per-M-bin, reweighted): {(w*per_bin).sum():.4f}  (n={m.sum()}, mean M {Mt[m].mean():.1f})")
# same for the best NN OOF
nn = np.load("<repo>/preds/nn_final_val_oof.npy").astype(np.float64)
pn = ndcg_per_feed(nn, yv, offv, Mv)
pb = np.array([pn[bv == b].mean() for b in range(len(bins) - 1)])
for nm, m in [("test public", tst < 1790497920), ("test private", tst >= 1790497920)]:
    bt = np.digitize(Mt[m], bins) - 1
    w = np.bincount(bt, minlength=len(bins) - 1) / m.sum()
    print(f"nn_final val {pn.mean():.4f} -> composition-adjusted {nm}: {(w*pb).sum():.4f}")
