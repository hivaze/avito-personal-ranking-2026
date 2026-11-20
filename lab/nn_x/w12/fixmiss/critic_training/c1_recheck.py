"""Critic re-check of the training stream (CPU only, stored predictions only).
Independent code: own per-feed z, own NDCG (binary gain, log2, pessimistic ties, no-positive feeds -> 1), pmix weights
from src/pmix.py (the only shared helper), own SEs.
 A. rebuild FD16P20@m0.4_0.2 from members and compare with the stored ens files (val NDCG, test corr)
 B. item 1 (pop-light part of FT branch = train-only pop-light preds on M>=thr) + per-M-bucket contributions
 C. selection-aware read: nested threshold/direction choice by user folds, and a sign-flip null of the max over the menu
 D. replication independence: per-feed corr of the item-1 delta on FD16P20 vs on W16+PD4w
 E. item 2 analog already on disk: click-label pop-light abl_e10_pop1_s951-954 fine-tuned with the nopop preset (25 extras)
    vs the standard preset (42 extras), single-seed n2cpv3, as the pop-light part on W16 and FD16 bases
 F. popularity-tercile O/E of the FINAL file (stored vs item 1) on M>=50
"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from pathlib import Path
import pmix

P = Path("<repo>/preds"); FL = Path("<repo>/data/flat")
fv = pl.read_parquet(FL / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy()
yv = np.load(FL / "val/cand_target.npy").astype(np.float64)
N = len(Mv); fid = np.repeat(np.arange(N), Mv); MROW = Mv[fid]; big = Mv >= 50
assert ov[0] == 0 and (np.diff(ov) == Mv[:-1]).all() and len(yv) == Mv.sum()
W = pmix.weights()

def zf(x, fid=fid, M=Mv):
    x = np.asarray(x, np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    var = np.maximum(np.bincount(fid, x * x, len(M)) / M - mu ** 2, 0)
    return (x - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)

def ndcg(s, y=yv, fid=fid, M=Mv, off=ov):
    s = np.asarray(s, np.float64)
    order = np.lexsort((y, -s, fid))          # within feed: score desc, ties -> negatives first (pessimistic)
    pos = np.arange(len(s)) - off[fid]        # rank position inside feed (fid sorted already)
    d = 1 / np.log2(pos + 2)
    dcg = np.bincount(fid, y[order] * d, len(M))
    k = np.bincount(fid, y, len(M))
    idcg = np.bincount(fid, (pos < k[fid]) * d, len(M))
    out = np.ones(len(M)); m = idcg > 0; out[m] = dcg[m] / idcg[m]
    return out

def se(d): return d.std(ddof=1) / np.sqrt(len(d))
def fmt(d, w=None):
    if w is None: return f"{d.mean():+.5f}+-{se(d):.5f}"
    m, s = pmix.wmean_se(d, w); return f"{m:+.5f}+-{s:.5f}"
def line(d): return (f"val {fmt(d)} | pub {fmt(d, W['public'])} | priv {fmt(d, W['private'])} | "
                     f"M<50 {fmt(d[~big])} | M>=50 {fmt(d[big])}")

_c = {}
def mem(m, suf):
    k = (m, suf)
    if k not in _c: _c[k] = zf(np.load(P / f"nn_{m}_{suf}_val{'_oof' if suf != 'trainonly' else ''}.npy"))
    return _c[k]
def pool(ms, suf): return np.mean([mem(m, suf) for m in ms], 0)

S4, S8 = range(951, 955), range(951, 959)
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
FDP4 = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]
PD4w = [f"abl_e10_pop1kd05w192_s{s}" for s in S4]
FT, TO = "n2cpv3s3", "trainonly"
wM = np.where(MROW >= 50, 0.2, 0.4)

def final(Fb, Fp, Tb, Tp, w=wM):  # as build_r4.py: pools re-z'd, branch = (1-w) z(base) + w z(pop), final = 0.5 z(FT) + 0.5 z(TO)
    return 0.5 * zf((1 - w) * zf(Fb) + w * zf(Fp)) + 0.5 * zf((1 - w) * zf(Tb) + w * zf(Tp))

print("=== A. rebuild check")
Fb, Fp, Tb, Tp = pool(FD16, FT), pool(FDP4, FT), pool(FD16, TO), pool(FDP4, TO)
cur = final(Fb, Fp, Tb, Tp)
st = 0.5 * np.load(P / "ens_FD16P20_m04_02kd_ft_val.npy") + 0.5 * np.load(P / "ens_FD16P20_m04_02kd_to_val.npy")
e_cur, e_st = ndcg(cur), ndcg(st)
print(f"rebuilt {e_cur.mean():.5f} stored {e_st.mean():.5f} | per-feed max|diff| {np.abs(e_cur - e_st).max():.4f}, "
      f"feeds differing {(np.abs(e_cur - e_st) > 1e-9).mean():.4f}")
# member z of the stored file uses rank-z? compare correlation
print(f"corr(rebuilt, stored) val {np.corrcoef(cur, st)[0, 1]:.6f}")

print("\n=== B. item 1 on FD16P20@m0.4_0.2 (ref = rebuilt)")
TpZ = Tp
def item1(Fb, Fp, Tb, Tp, sel):  # sel = row mask where the FT pop part takes the TO pop preds
    return final(Fb, np.where(sel, Tp, Fp), Tb, Tp)
e1 = ndcg(item1(Fb, Fp, Tb, Tp, MROW >= 50)); d1 = e1 - e_cur
print("thr 50 :", line(d1))
dall = ndcg(item1(Fb, Fp, Tb, Tp, np.ones_like(MROW, bool))) - e_cur
print("all M  :", line(dall))
edges = [2, 30, 50, 75, 100, 150, 300, 10 ** 6]
print("per-M-bucket contribution of the all-feeds swap (sum over bucket / N) and bucket mean:")
for a, b in zip(edges[:-1], edges[1:]):
    m = (Mv >= a) & (Mv < b)
    print(f"  M [{a},{b}): share {m.mean():.3f}  contrib {dall[m].sum() / N:+.5f}  mean {dall[m].mean():+.5f}+-{se(dall[m]):.5f}")
print(f"  identity check: thr-50 swap equals all-swap on M>=50 rows: {np.allclose(d1[big], dall[big])}")

print("\n=== C. selection-aware reads")
uid = fv["user_id"].to_numpy()
rng = np.random.default_rng(0)
u_un, u_inv = np.unique(uid, return_inverse=True)
grp = np.where(uid == 0, -(np.arange(N) + 1), u_inv)  # u0 feeds = singletons
g_un, g_inv = np.unique(grp, return_inverse=True)
fold = rng.integers(0, 5, len(g_un))[g_inv]
TH = [30, 50, 75, 100, 150]
menu = [("none", np.zeros(N, bool)), ("all", np.ones(N, bool))]
menu += [(f">={t}", Mv >= t) for t in TH] + [(f"<{t}", Mv < t) for t in TH]
# delta of option o per feed = dall where the option swaps, 0 elsewhere (swap is per feed and per-feed NDCG depends only on the feed)
D = np.stack([np.where(m, dall, 0.0) for _, m in menu])
nested = np.zeros(N); picks = []
for k in range(5):
    tr, te = fold != k, fold == k
    j = int(np.argmax(D[:, tr].mean(1))); picks.append(menu[j][0]); nested[te] = D[j, te]
print(f"nested choice over {len(menu)} options by 5 user folds: picks {picks} | {line(nested)}")
nested2 = np.zeros(N); picks2 = []
menu2 = [0, 2, 3, 4, 5, 6]  # none + the '>=t' family only (stream's stated family)
for k in range(5):
    tr, te = fold != k, fold == k
    j = menu2[int(np.argmax(D[menu2][:, tr].mean(1)))]; picks2.append(menu[j][0]); nested2[te] = D[j, te]
print(f"nested choice, '>=t' family + none: picks {picks2} | {line(nested2)}")
# sign-flip null by user group: max over the menu of the in-sample mean
obs = D.mean(1).max()
flips = rng.choice([-1.0, 1.0], size=(2000, len(g_un)))
mx = np.empty(2000)
for i in range(2000):
    s = flips[i][g_inv]
    mx[i] = (D * s).mean(1).max()
print(f"in-sample best option {menu[int(np.argmax(D.mean(1)))][0]} {obs:+.5f}; sign-flip null (user groups) of the max over the menu: "
      f"mean {mx.mean():+.5f}, 95% {np.quantile(mx, 0.95):+.5f}; p = {(mx >= obs).mean():.3f}")
obs50 = D[3].mean()
print(f"thr-50 alone vs the max-over-menu null: p = {(mx >= obs50).mean():.3f}")

print("\n=== D. replication independence (W16+PD4w)")
Gb, Gp, Hb, Hp = pool(W16, FT), pool(PD4w, FT), pool(W16, TO), pool(PD4w, TO)
c2 = ndcg(final(Gb, Gp, Hb, Hp)); d2 = ndcg(item1(Gb, Gp, Hb, Hp, MROW >= 50)) - c2
print("W16+PD4w thr 50:", line(d2))
nz = big & ((d1 != 0) | (d2 != 0))
print(f"per-feed corr of the two deltas on M>=50: {np.corrcoef(d1[big], d2[big])[0, 1]:.3f} (feeds with any change {nz.sum()} of {big.sum()})")
avg = 0.5 * (d1 + d2)
print("mean of the two deltas:", line(avg))
# the pop-light members are 4+4 different students; is the delta driven by shared FT recipe? corr of FT-TO pop-pool shifts
sh1, sh2 = Fp - Tp, Gp - Hp
c = np.corrcoef(sh1[MROW >= 50], sh2[MROW >= 50])[0, 1]
print(f"row-level corr of (FT pop pool - TO pop pool) between FDP4 and PD4w on M>=50 rows: {c:.3f}")

print("\n=== E. item 2 analog on disk: click-label pop-light (abl_e10_pop1_s951-954), FT std vs nopop preset (n2cpv3, 1 seed)")
PC4a = [f"abl_e10_pop1_s{s}" for s in S4]
PC4np = [f"abl_e10_pop1_s{s}np" for s in S4]
for m, mn in zip(PC4a, PC4np):
    t = ndcg(mem(m, TO)).mean(); a = ndcg(mem(m, "n2cpv3")).mean(); b = ndcg(mem(mn, "n2cpv3")).mean()
    print(f"  {m}: TO {t:.5f} | FT std {a:.5f} ({a - t:+.5f}) | FT nopop {b:.5f} ({b - t:+.5f}) | nopop - std {b - a:+.5f}")
for bn, (B_ft, B_to) in (("W16", (Gb, Hb)), ("FD16", (Fb, Tb))):
    Pt = pool(PC4a, TO); Ps = pool(PC4a, "n2cpv3"); Pn = pool(PC4np, "n2cpv3")
    for rn, w in (("w0.2", 0.2), ("m0.4_0.2", wM)):
        es = ndcg(final(B_ft, Ps, B_to, Pt, w)); en = ndcg(final(B_ft, Pn, B_to, Pt, w))
        eto = ndcg(final(B_ft, Pt, B_to, Pt, w)); e50 = ndcg(final(B_ft, np.where(MROW >= 50, Pt, Ps), B_to, Pt, w))
        print(f"  {bn}+PC4a {rn}: nopop-FT minus std-FT : {line(en - es)}")
        print(f"  {bn}+PC4a {rn}: no-FT (TO) minus std-FT: {line(eto - es)}")
        print(f"  {bn}+PC4a {rn}: item1-form minus std-FT: {line(e50 - es)}")
        print(f"  {bn}+PC4a {rn}: nopop-FT minus item1-form: {line(en - e50)}")

print("\n=== F. O/E by within-feed popularity tercile, final files, M>=50 / M<50 (fitted temperature)")
from scipy.optimize import minimize_scalar
tab = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet", columns=["it_hist_all_luo", "fidx"])
assert (tab["fidx"].to_numpy() == fid).all()
popv = tab["it_hist_all_luo"].to_numpy().astype(np.float64)
popv = np.nan_to_num(popv, nan=-1.0)
# average rank within feed (ties shared), as a percentile
rk = np.empty(len(popv))
order = np.lexsort((popv, fid)); sp = popv[order]; sf = fid[order]
pos = np.arange(len(sp)) - ov[sf]
# tie-average positions
key = np.concatenate([[True], (sp[1:] != sp[:-1]) | (sf[1:] != sf[:-1])])
gid = np.cumsum(key) - 1
avgpos = (np.bincount(gid, pos) / np.bincount(gid))[gid]
rk[order] = avgpos
pct = rk / np.maximum(MROW - 1, 1); terc = np.digitize(pct, [1 / 3, 2 / 3])
K = np.bincount(fid, yv, N); yK = yv / np.maximum(K, 1)[fid]
def sm(x):
    mx = np.maximum.reduceat(x, ov); e = np.exp(x - mx[fid]); return e / np.bincount(fid, e, N)[fid]
def oe(x):
    x = zf(x)
    t = minimize_scalar(lambda a: -(yK * np.log(sm(a * x) + 1e-30)).sum(), bounds=(0.1, 5), method="bounded").x
    E = sm(t * x) * K[fid]; out = []
    for bm, bn in ((MROW >= 50, "M>=50"), (MROW < 50, "M<50")):
        out.append(bn + " " + " ".join(f"{yv[bm & (terc == k)].sum() / E[bm & (terc == k)].sum():.3f}" for k in range(3)))
    return f"t {t:.3f} | " + " | ".join(out)
print("  stored final :", oe(cur))
print("  item-1 final :", oe(item1(Fb, Fp, Tb, Tp, MROW >= 50)))
print("  FT branch    :", oe(zf((1 - wM) * Fb + wM * Fp)))
print("  FT branch it1:", oe(zf((1 - wM) * Fb + wM * np.where(MROW >= 50, Tp, Fp))))
print("  TO branch    :", oe(zf((1 - wM) * Tb + wM * Tp)))
