"""u20wit: SENSITIVITY of the declared U20 read to the definition of the cross-user column (information only; u20_read.py decides).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python nn_x/w12/next/u20wit/u20_read_sens.py [--A u20a --B u20b --C u20c --seeds 951,952]
Stored GPU train-only predictions (GPU files vs GPU files), member delta paired per feed, pooled over the seeds (the statistic of the rule), selected and @last.
Columns:
  val  : declared cross-user (13,571) | strict: user hash fold != 0, known or unknown user, user-0 bucket kept | unseen users only (no train feed) |
         seen users of folds 1-4 | fold-0 hash but no train feed (cross-user by substance, fold 0 by hash)
  hold : declared cross-user (24,026) | each user fold 1..4 | cross-user without the user-0 bucket | cross-user by M (50-99, 100-300)
Light: one thread is enough (no bootstrap)."""
from wit_common import *
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('--A', default='u20a'); ap.add_argument('--B', default='u20b'); ap.add_argument('--C', default='u20c'); ap.add_argument('--seeds', default='951,952')
a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(',')]
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); yh = np.load(W5P / 'holdout/holdout_y.npy').astype(np.float32)
xfh = tm['xf'][hold]; uh = tm['uid'][hold]
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32); uv = fv['user_id'].to_numpy().astype(np.int64)
tru = np.unique(tm['uid'][tm['uid'] != 0]); seen = np.isin(uv, tru) & (uv != 0); xv = (uv * 2654435761 % 4294967296) % 5; v0 = seen & (xv == 0)
COL = {'val': (('declared cross-user', ~v0), ('strict: hash fold != 0 or user 0', (uv == 0) | (xv != 0)), ('unseen users (no train feed)', ~seen), ('seen users, folds 1-4', seen & (xv != 0)),
               ('unseen users with fold-0 hash', ~seen & (uv != 0) & (xv == 0)), ('fold-0 users (declared)', v0)),
       'hold': (('declared cross-user', xfh != 0),) + tuple((f'user fold {k}', xfh == k) for k in (1, 2, 3, 4)) + (('cross-user, no user-0 bucket', (xfh != 0) & (uh != 0)),
               ('cross-user M 50-99', (xfh != 0) & (Mh < 100)), ('cross-user M 100-300', (xfh != 0) & (Mh >= 100)), ('fold-0 users (declared)', xfh == 0))}
SP = {'val': (yv, ov, Mv), 'hold': (yh, oh, Mh)}; name = lambda arm, s: f'w5_e10_{arm}_s{s}'
def path(arm, s, suf, sp): return W5P / (f'nn_{name(arm, s)}{suf}_trainonly_val.npy' if sp == 'val' else f'holdout/nn_{name(arm, s)}{suf}_holdout.npy')
done = {arm: [s for s in seeds if (W5L / f'{name(arm, s)}.json').exists() and all(path(arm, s, suf, sp).exists() for suf in ('', '@last') for sp in ('val', 'hold'))] for arm in (a.A, a.B, a.C)}
print(f'U20 sensitivity read: A = {a.A}, B = {a.B}, C = {a.C}; finished: ' + ', '.join(f'{arm} {done[arm]}' for arm in (a.A, a.B, a.C)))
for suf, tag in (('', 'SELECTED'), ('@last', '@last')):
    print(f'\n=== {tag} checkpoints: member delta pooled over seeds, paired per feed ===')
    for sp in ('val', 'hold'):
        y, off, M = SP[sp]; e = {(arm, s): ndcg_feeds(np.load(path(arm, s, suf, sp)).astype(np.float32), y, off, M) for arm in (a.A, a.B, a.C) for s in done[arm]}
        for cn, mk in COL[sp]:
            t = f'  {"val" if sp == "val" else "holdout":7s} {cn:34s} n {int(mk.sum()):5d} |'
            for X, Y, lab in ((a.B, a.A, 'B-A'), (a.C, a.A, 'C-A'), (a.B, a.C, 'B-C')):
                ss = [s for s in done[X] if s in done[Y]]
                if ss:
                    d = np.mean([e[X, s][mk] - e[Y, s][mk] for s in ss], 0); t += f' {lab} {fmt(d)} ({d.mean() / se(d):+.1f} SE; {len(ss)} seeds) |'
            print(t)
print('DONE')
