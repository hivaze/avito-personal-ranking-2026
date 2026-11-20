"""u20wit read-out (1): the DECLARED U20 read-out with the user-fold split, from the STORED train-only predictions (GPU files vs GPU files).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/u20_read.py [--A u20a --B u20b --C u20c --seeds 951,952 --norule]
Rule (nn_x/w5/logs/E_kd_predeclared.txt 20:21Z item 1 + 21:26Z addendum; thresholds are NOT changed here):
  statistic = member delta, paired per feed, pooled over the seeds (per-feed mean over seeds of NDCG_B,s - NDCG_A,s), CROSS-USER columns only
  GO   iff  B - A >= +0.0008 on the holdout  AND  B - A >= 0 on val  AND  (B - A) / (C - A) >= 0.5
  KILL iff  B - A <  +0.0004 on the holdout;   otherwise 2 more seeds.
My reading of what the text leaves open (stated in the report): the ratio is taken on the cross-user HOLDOUT column (the witness the rule names first, where
C - A was forecast); it is defined only if C - A > 0 there; the SELECTED checkpoints decide (the convention of every other rule in the file), @last is printed
next to it and a disagreement between the two is flagged.
Columns: fold-0 users = holdout feeds with train.py fold 0 (5,974) / val feeds whose user has a train feed and hashes to fold 0 (852);
         cross-user = all other feeds (val: includes unseen users and the user-0 bucket)."""
from wit_common import *
import argparse
sys.path.insert(0, str(ROOT / 'nn_x/w12/recheck/pool_composition'))
from stack_val import feed_rank_z
ap = argparse.ArgumentParser(); ap.add_argument('--A', default='u20a'); ap.add_argument('--B', default='u20b'); ap.add_argument('--C', default='u20c')
ap.add_argument('--seeds', default='951,952'); ap.add_argument('--norule', action='store_true', help='known-answer tests on other arms: print no GO / KILL line')
a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(',')]
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
hm = np.load(W5P / 'holdout/holdout_meta.npy'); assert (hm[0] == Mh).all() and (hm[1] == oh).all(); yh = np.load(W5P / 'holdout/holdout_y.npy').astype(np.float32)
xfh = tm['xf'][hold]; assert (xfh == np.load(ROOT / 'nn_x/w12/recheck/kd_audit/out/xfold_holdout.npy')).all(), 'fold ids differ from kd_audit'
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32); uv = fv['user_id'].to_numpy().astype(np.int64)
tru = np.unique(tm['uid'][tm['uid'] != 0]); seen = np.isin(uv, tru) & (uv != 0); xv = (uv * 2654435761 % 4294967296) % 5
xa = np.load(ROOT / 'nn_x/w12/recheck/kd_audit/out/xfold_val.npy'); assert ((xa == -1) == (uv == 0)).all() and (xa[uv != 0] == xv[uv != 0]).all(), 'val fold ids differ from kd_audit'
v0 = seen & (xv == 0)
COL = {'val': (('fold-0 users', v0), ('cross-user', ~v0), ('all', np.ones(len(Mv), bool)), ('cross-user M<50', ~v0 & (Mv < 50)), ('cross-user M>=50', ~v0 & (Mv >= 50))),
       'hold': (('fold-0 users', xfh == 0), ('cross-user', xfh != 0), ('all', np.ones(len(Mh), bool)))}
SP = {'val': (yv, ov, Mv), 'hold': (yh, oh, Mh)}
print(f'U20 declared read-out: A = {a.A}, B = {a.B}, C = {a.C}, seeds {seeds} (stored GPU train-only predictions; paired per feed)')
print(f'  holdout feeds {len(Mh)}: fold-0 users {int((xfh == 0).sum())}, cross-user {int((xfh != 0).sum())} | val feeds {len(Mv)}: train user {int(seen.sum())}, fold-0 users {int(v0.sum())}, cross-user {int((~v0).sum())}')
name = lambda arm, s: f'w5_e10_{arm}_s{s}'
def path(arm, s, suf, sp): return W5P / (f'nn_{name(arm, s)}{suf}_trainonly_val.npy' if sp == 'val' else f'holdout/nn_{name(arm, s)}{suf}_holdout.npy')
done = {arm: [s for s in seeds if (W5L / f'{name(arm, s)}.json').exists() and all(path(arm, s, suf, sp).exists() for suf in ('', '@last') for sp in ('val', 'hold'))] for arm in (a.A, a.B, a.C)}
miss = [name(arm, s) for arm in (a.A, a.B, a.C) for s in seeds if s not in done[arm]]
print('  finished runs: ' + ', '.join(f'{arm} {done[arm]}' for arm in (a.A, a.B, a.C)) + (f' | MISSING: {miss}' if miss else ' | all present'))
rng = np.random.default_rng(20261115)
def boot_ratio(dB, dC, nb=1000):  # 1000 bootstrap draws (4000 until 03:03 UTC on 11-15; lowered because the machine load tripled the run time; the ratio interval is information only)
    n = len(dB); ix = rng.integers(0, n, (nb, n)); r = dB[ix].mean(1) / dC[ix].mean(1); return np.quantile(r, [0.025, 0.16, 0.5, 0.84, 0.975]), float(np.mean(r >= 0.5)), float(np.mean(dC[ix].mean(1) <= 0))
verdicts = {}
for suf, tag in (('', 'SELECTED checkpoints'), ('@last', '@last checkpoints')):
    print(f'\n=== {tag} ===')
    R = {}
    for sp in ('val', 'hold'):
        y, off, M = SP[sp]; e = {}; zz = {}
        for arm in (a.A, a.B, a.C):
            for s in done[arm]:
                x = np.load(path(arm, s, suf, sp)).astype(np.float32); assert len(x) == len(y)
                e[arm, s] = ndcg_feeds(x, y, off, M); zz[arm, s] = feed_rank_z(off, M, x)[:, 1]
        pool = {arm: ndcg_feeds(np.mean([zz[arm, s] for s in done[arm]], 0), y, off, M) for arm in (a.A, a.B, a.C) if done[arm]}
        print(f'  -- {"val" if sp == "val" else "holdout"}')
        for cn, mk in COL[sp]:
            t = f'     {cn:17s} n {int(mk.sum()):5d} |'
            for arm in (a.A, a.B, a.C):
                if done[arm]: t += f' {arm} ' + ' '.join(f'{e[arm, s][mk].mean():.5f}' for s in done[arm]) + f' pool{len(done[arm])} {pool[arm][mk].mean():.5f} |'
            print(t)
        for cn, mk in COL[sp]:
            t = f'     {cn:17s} deltas |'
            for X, Y, lab in ((a.B, a.A, 'B-A'), (a.C, a.A, 'C-A'), (a.B, a.C, 'B-C')):
                ss = [s for s in done[X] if s in done[Y]]
                if not ss: continue
                d = np.mean([e[X, s][mk] - e[Y, s][mk] for s in ss], 0); R[sp, cn, lab] = d
                t += f' {lab} member({len(ss)} seeds) {fmt(d)} [per seed ' + ' '.join(f'{np.mean(e[X, s][mk] - e[Y, s][mk]):+.5f}' for s in ss) + ']'
                if len(done[X]) == len(done[Y]) == len(seeds): t += f' pool {fmt(pool[X][mk] - pool[Y][mk])}'
                t += ' |'
            if (sp, cn, 'B-A') in R and (sp, cn, 'C-A') in R:
                dB, dC = R[sp, cn, 'B-A'], R[sp, cn, 'C-A']; q, p05, pneg = boot_ratio(dB, dC)
                t += f' ratio (B-A)/(C-A) {dB.mean() / dC.mean():+.2f} (bootstrap median {q[2]:+.2f}, 68% [{q[1]:+.2f}, {q[3]:+.2f}], 95% [{q[0]:+.2f}, {q[4]:+.2f}]; P(ratio >= 0.5) {p05:.2f}; P(C-A <= 0) {pneg:.2f})'
            print(t)
    if ('hold', 'cross-user', 'B-A') in R and ('hold', 'cross-user', 'C-A') in R and not a.norule and not miss:
        bh, bv, ch = R['hold', 'cross-user', 'B-A'], R['val', 'cross-user', 'B-A'], R['hold', 'cross-user', 'C-A']
        ratio = bh.mean() / ch.mean() if ch.mean() > 0 else float('nan')
        go = bh.mean() >= 0.0008 and bv.mean() >= 0 and ch.mean() > 0 and ratio >= 0.5; kill = bh.mean() < 0.0004
        v = 'GO' if go else ('KILL' if kill else '2 MORE SEEDS'); verdicts[tag] = v
        print(f'  RULE on the CROSS-USER columns ({tag}): B-A holdout {fmt(bh)} (bar GO >= +0.0008, KILL < +0.0004) | B-A val {fmt(bv)} (bar >= 0) | C-A holdout {fmt(ch)} | '
              f'ratio {ratio:+.2f} (bar >= 0.5{"" if ch.mean() > 0 else "; NOT DEFINED: C - A <= 0"})  ->  {v}')
        print(f'    for the record, the same rule on the POOLED columns (not the decision): B-A holdout {fmt(R["hold", "all", "B-A"])}, val {fmt(R["val", "all", "B-A"])}, C-A holdout {fmt(R["hold", "all", "C-A"])}')
        print(f'    same-user channel (fold-0 users; reported, not evidence): B-A holdout {fmt(R["hold", "fold-0 users", "B-A"])}, val {fmt(R["val", "fold-0 users", "B-A"])}; C-A holdout {fmt(R["hold", "fold-0 users", "C-A"])}, val {fmt(R["val", "fold-0 users", "C-A"])}')
if verdicts:
    vs = list(verdicts.values())
    print(f'\nDECISION (selected checkpoints decide): {vs[0]}' + ('' if len(set(vs)) == 1 else f'   ** @last reads {vs[1]}: the two checkpoint choices DISAGREE, report both **'))
elif miss and not a.norule:
    print(f'\nNO DECISION: runs missing {miss}')
