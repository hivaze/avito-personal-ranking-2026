"""Critic check 3 (CPU): (a) does item 1 move test scores in the same direction as the public-refuted M-rule
(FD16P20@m0.2_0.4, public 0.3947465 vs FD16P20 0.394892) on M>=50 feeds? (b) item-1 delta by val day (test is later).
(c) item-1 delta by FT fold (the fold model that produced the OOF prediction is unknown here, so by user-fold proxy: skipped).
"""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/critic_training")
from c1_lib import *  # noqa
import polars as pl

ftst = pl.read_parquet(FL / "test/feeds.parquet"); Mt = ftst["M"].to_numpy().astype(np.int64); ot = ftst["cand_off"].to_numpy()
fidt = np.repeat(np.arange(len(Mt)), Mt); MTR = Mt[fidt]
zt = lambda x: zf(x, fidt, Mt)
def tpool(ms, suf): return np.mean([zt(np.load(P / f"nn_{m}_{suf}_test.npy")) for m in ms], 0)
Fb, Fp, Tb, Tp = tpool(FD16, FT), tpool(FDP4, FT), tpool(FD16, TO), tpool(FDP4, TO)
wt = np.where(MTR >= 50, 0.2, 0.4)
def tfin(Fp_, w):
    return 0.5 * zt((1 - w) * zt(Fb) + w * zt(Fp_)) + 0.5 * zt((1 - w) * zt(Tb) + w * zt(Tp))
cur = tfin(Fp, wt); new = tfin(np.where(MTR >= 50, Tp, Fp), wt)
st = lambda tag: 0.5 * np.load(P / f"ens_{tag}_ft_test.npy") + 0.5 * np.load(P / f"ens_{tag}_to_test.npy")
fd, mrule = st("FD16P20kd"), st("FD16P20_m02_04kd")
b = MTR >= 50
d_item1 = (new - cur)[b]; d_mrule = (zt(mrule) - zt(fd))[b]
# pure 'more pop-light' direction on test: TO pop pool minus base mix, per-feed z
print(f"test M>=50 rows: corr(item-1 change, M-rule change) = {np.corrcoef(d_item1, d_mrule)[0, 1]:+.3f}")
# projection: regress item-1 change on M-rule change
beta = (d_item1 * d_mrule).sum() / (d_mrule ** 2).sum()
print(f"  regression coef of item-1 change on M-rule change {beta:+.3f}; share of item-1 change variance explained {np.corrcoef(d_item1, d_mrule)[0, 1] ** 2:.3f}")
# same on val, with the val NDCG deltas of each
fv_ = fv
cur_v = final(pool(FD16, FT), pool(FDP4, FT), pool(FD16, TO), pool(FDP4, TO))
Fbv, Fpv, Tbv, Tpv = pool(FD16, FT), pool(FDP4, FT), pool(FD16, TO), pool(FDP4, TO)
new_v = final(Fbv, np.where(MROW >= 50, Tpv, Fpv), Tbv, Tpv)
stv = lambda tag: 0.5 * np.load(P / f"ens_{tag}_ft_val.npy") + 0.5 * np.load(P / f"ens_{tag}_to_val.npy")
fdv, mrv = stv("FD16P20kd"), stv("FD16P20_m02_04kd")
bv = MROW >= 50
print(f"val M>=50 rows: corr(item-1 change, M-rule change) = {np.corrcoef((new_v - cur_v)[bv], (zf(mrv) - zf(fdv))[bv])[0, 1]:+.3f}")
e_fd, e_mr = ndcg(fdv), ndcg(mrv)
print("M-rule vs FD16P20 on val (public read -0.00015):", line(e_mr - e_fd))
# (b) by val day
ts = fv["timestamp"].to_numpy(); day = (ts - ts.min()) // 86400
d1 = ndcg(new_v) - ndcg(cur_v)
print("item-1 delta by val day:")
for k in np.unique(day):
    m = day == k
    print(f"  day {k}: feeds {m.sum()} | all {d1[m].mean():+.5f}+-{se(d1[m]):.5f} | M>=50 {d1[m & big].mean():+.5f}+-{se(d1[m & big]):.5f}")
