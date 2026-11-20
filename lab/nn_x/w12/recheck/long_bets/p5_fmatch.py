"""P5: how well do students match the teacher FUNCTION on unseen feeds (holdout)? teacher_mean11 holdout logits vs students.
If pure-teacher students (kd10) are far from the teacher, function matching (aug / longer / wider) has headroom."""
from lb_common import *
import polars as pl
T = ROOT / 'nn_x/w12/kd/out'
ftr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['M', 'cand_off']); M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0
sel = np.where((pr > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[hold], M[hold])]); assert (M[hold] == Mh).all()
teach = {k: zh(np.asarray(np.load(T / f'teacher_{k}.npy', mmap_mode='r')[rows], np.float32)) for k in ('mean11', 'mean25', 'w8g1', 'oof')}
for k, t in teach.items(): print(f'teacher {k:7s} holdout NDCG {eh(t).mean():.5f}')
t11 = teach['mean11']; e11 = eh(t11)
print(f'{"arm":14s} {"member hold":>11s} {"corr(member,T11)":>16s} {"pool hold":>10s} {"corr(pool,T11)":>14s} {"pool - T11":>20s} {"member-member":>13s}')
for arm in ('base', 'kd03', 'kd05', 'kd08', 'kd10', 'kd05w192', 'kd05t25w192', 'kx05', 'kd05g2w192'):
    try: H = [zh(np.load(W / f'holdout/nn_w5_e10_{arm}_s{s}_holdout.npy')) for s in S4]
    except Exception as e: print(arm, 'missing', e); continue
    p = np.mean(H, 0); iu = np.triu_indices(4, 1)
    print(f'{arm:14s} {np.mean([eh(h).mean() for h in H]):11.5f} {np.mean([wcorr(h, t11) for h in H]):16.4f} {eh(p).mean():10.5f} {wcorr(p, t11):14.4f} {f(eh(p) - e11):>20s} {np.corrcoef(H)[iu].mean():13.4f}')
# teacher + students: does the student pool add to the teacher (label info beyond the teacher)?
for arm in ('kd10', 'kd05w192'):
    H = np.mean([zh(np.load(W / f'holdout/nn_w5_e10_{arm}_s{s}_holdout.npy')) for s in S4], 0)
    print(f' 0.5 z(T11) + 0.5 z({arm} 4-pool) vs T11: {f(eh(0.5 * t11 + 0.5 * zh(H)) - e11)}')
