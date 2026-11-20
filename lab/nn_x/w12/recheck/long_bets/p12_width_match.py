"""P12: where does extra student capacity go? Single-member corr to the arm's OWN teacher on the holdout (teacher honest there),
member holdout / val NDCG, member - teacher. Arms at kd_w 0.5 (d128 / d192 / d256, 20 epochs, no dropout, FFN x4, 2 history layers)
and at kd_w 0.8 (d128 / d192, one seed each). Log: out/p12_width_match.txt"""
from lb_common import *
import polars as pl
T = ROOT / 'nn_x/w12/kd/out'
ftr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['M', 'cand_off']); M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0
sel = np.where((pr > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[hold], M[hold])])
t11 = zh(np.asarray(np.load(T / 'teacher_mean11.npy', mmap_mode='r')[rows], np.float32)); t25 = zh(np.asarray(np.load(T / 'teacher_mean25.npy', mmap_mode='r')[rows], np.float32))
print('P12: single-member function matching vs width (holdout; teacher honest). corr to the arm\'s own teacher.')
for arm, tt, seeds in (('kd08', t11, (951,)), ('kd08w192', t11, (951,)), ('kd05', t11, S4), ('kd05w192', t11, S4), ('kd05t25', t25, S4), ('kd05t25w192', t25, S4), ('kd05t25w256', t25, (951, 952)), ('kd05t25w192b3', t25, (951, 952)), ('kw_e20', t11, (951, 952)), ('kw_drop0', t11, (951, 952)), ('kw_ff4', t11, (951, 952)), ('kw_hl2', t11, (951, 952))):
    H = [zh(np.load(W / f'holdout/nn_w5_e10_{arm}_s{s}_holdout.npy')) for s in seeds]
    V = [ev(zv(np.load(W / f'nn_w5_e10_{arm}_s{s}_trainonly_val.npy'))).mean() for s in seeds]
    print(f' {arm:15s} n={len(seeds)} member hold {np.mean([eh(h).mean() for h in H]):.5f} val {np.mean(V):.5f} | corr(member, own teacher) {np.mean([wcorr(h, tt) for h in H]):.4f} | member - teacher (hold) {np.mean([eh(h).mean() - eh(tt).mean() for h in H]):+.5f}')
