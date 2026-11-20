"""u20wit_verify: independent re-derivation of the declared U20 read (stored GPU prediction files only; CPU; own code).
Nothing is imported from nn_x/w12/next/u20wit/ nor from pc_common / src.metrics / src.stack_val: fold hash, holdout draw, NDCG and
rank-z are re-implemented here.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit_verify/v_read.py > nn_x/w12/next/u20wit_verify/out_v_read.txt
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '4'
import json, sys
import numpy as np, polars as pl, numba as nb
from pathlib import Path

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; W = ROOT / 'nn_x/w5/preds'; LOGS = ROOT / 'nn_x/w5/logs'
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else '951,952').split(',')]


@nb.njit(parallel=True, cache=False)
def my_ndcg(s, y, off, M):
    """Per-feed full-list NDCG, binary gain, pessimistic ties (a positive ranks below every negative of equal score).
    Own formulation: rank of a positive = #(score > s) + #(negatives with score == s) + (index among tied positives)."""
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0:
            continue
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk:
                        if y[o + j] <= 0: r += 1
                        elif j < k: r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def my_z(x, off, M):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        mu = 0.0
        for k in range(m): mu += x[o + k]
        mu /= m
        v = 0.0
        for k in range(m): v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m) + 1e-9
        for k in range(m): out[o + k] = (x[o + k] - mu) / sd
    return out


def fold_of(uid, idx):
    uid = uid.astype(np.uint64)
    h = ((uid * np.uint64(2654435761)) % np.uint64(4294967296)) % np.uint64(5)
    return np.where(uid == 0, idx % 5, h.astype(np.int64))


def ms(d):
    d = np.asarray(d, np.float64)
    return f'{d.mean():+.7f} +- {d.std(ddof=1) / np.sqrt(len(d)):.7f}'


# ---------------------------------------------------------------- train meta, holdout identity, folds
ft = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['user_id', 'M', 'cand_off'])
n = ft.height; Mtr = ft['M'].to_numpy().astype(np.int64); uid = ft['user_id'].to_numpy().astype(np.int64); otr = ft['cand_off'].to_numpy().astype(np.int64)
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); assert (pr['fidx'] == np.arange(n)).all()
pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
cand = np.where((pold > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
assert (Mtr[hold] == Mh).all(), 'holdout M mismatch'
yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
ytr = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')
# alignment of the stored holdout labels with my reconstruction of the holdout rows (sample of 3000 feeds + global count)
rng = np.random.default_rng(7)
for j in rng.choice(30000, 3000, replace=False):
    assert (np.asarray(ytr[otr[hold[j]]:otr[hold[j]] + Mh[j]]).astype(np.float32) == yh[oh[j]:oh[j] + Mh[j]]).all(), j
xf_all = fold_of(uid, np.arange(n))
xfh = xf_all[hold]; uh = uid[hold]
print(f'train feeds {n}; holdout {len(hold)} (M match, labels match on 3000 sampled feeds); holdout by fold {np.bincount(xfh, minlength=5)}; user-0 holdout feeds {int((uh == 0).sum())}')
print(f'train feeds by fold {np.bincount(xf_all, minlength=5)}; fold-0 train feeds outside the holdout {int(((xf_all == 0)).sum() - (xfh == 0).sum())}')

# ---------------------------------------------------------------- val meta
fv = pl.read_parquet(FLAT / 'val/feeds.parquet', columns=['user_id', 'M', 'cand_off'])
Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64); uv = fv['user_id'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
tr_users = np.unique(uid)
has_tr = np.isin(uv, tr_users)
hv = fold_of(uv, np.zeros(len(uv), np.int64) - 1)          # user-0 bucket gets -1 % 5 = 4 -> never fold 0 here
v_f0_strict = has_tr & (uv != 0) & (hv == 0)                # val feeds of a train user who hashes to fold 0
v_hash0_any = (uv != 0) & (hv == 0)                         # every val user hashing to fold 0 (seen or unseen)
print(f'val feeds {len(Mv)}; with a train user {int(has_tr.sum())} (user-0 bucket {int((uv == 0).sum())}); fold-0 train users {int(v_f0_strict.sum())}; hash-0 any {int(v_hash0_any.sum())}')

cols = {
    'hold cross-user': ('h', xfh != 0), 'hold fold-0': ('h', xfh == 0), 'hold all': ('h', np.ones(30000, bool)),
    'hold cross-user, no user-0 bucket': ('h', (xfh != 0) & (uh != 0)),
    'val cross-user': ('v', ~v_f0_strict), 'val fold-0': ('v', v_f0_strict), 'val all': ('v', np.ones(len(Mv), bool)),
    'val strict (no hash-0 user)': ('v', ~v_hash0_any), 'val cross-user M<50': ('v', ~v_f0_strict & (Mv < 50)), 'val cross-user M>=50': ('v', ~v_f0_strict & (Mv >= 50)),
}

res = {}
for suf in ('', '@last'):
    nd = {}; zz = {}
    for arm in 'abc':
        for s in SEEDS:
            v = f'w5_e10_u20{arm}_s{s}'
            pv = np.load(W / f'nn_{v}{suf}_trainonly_val.npy').astype(np.float32); ph = np.load(W / f'holdout/nn_{v}{suf}_holdout.npy').astype(np.float32)
            assert len(pv) == len(yv) and len(ph) == len(yh) and np.isfinite(pv).all() and np.isfinite(ph).all()
            nd[arm, s, 'v'] = my_ndcg(pv, yv, ov, Mv); nd[arm, s, 'h'] = my_ndcg(ph, yh, oh, Mh)
            zz[arm, s, 'v'] = my_z(pv, ov, Mv); zz[arm, s, 'h'] = my_z(ph, oh, Mh)
            if suf == '':
                J = json.load(open(LOGS / f'{v}.json'))
                keys = {k: J[k] for k in J if any(t in k for t in ('best', 'val_ndcg', 'holdout', 'step', 'select'))} if isinstance(J, dict) else {}
                print(f'  {v}: my val {nd[arm, s, "v"].mean():.5f} my holdout {nd[arm, s, "h"].mean():.5f} | json keys {list(J.keys())[:14] if isinstance(J, dict) else type(J)}')
                for k in ('best_step', 'best_val', 'best', 'val_ndcg', 'holdout_ndcg', 'best_holdout', 'select_on', 'steps', 'total_steps'):
                    if isinstance(J, dict) and k in J and not isinstance(J[k], (list, dict)): print(f'      json {k} = {J[k]}')
    pool = {}
    for arm in 'abc':
        for w, y, off, M in (('v', yv, ov, Mv), ('h', yh, oh, Mh)):
            zm = np.mean([zz[arm, s, w] for s in SEEDS], 0).astype(np.float32)
            pool[arm, w] = my_ndcg(zm, y, off, M)
    print(f'\n===== checkpoint kind: {"selected" if suf == "" else "@last"}; seeds {SEEDS} =====')
    print('column | n | B-A member pooled over seeds | per seed | B-A pool | C-A member | per seed | C-A pool | ratio member | ratio pool')
    for name, (w, mk) in cols.items():
        dBA = np.mean([nd['b', s, w][mk] - nd['a', s, w][mk] for s in SEEDS], 0); dCA = np.mean([nd['c', s, w][mk] - nd['a', s, w][mk] for s in SEEDS], 0)
        ps_b = ', '.join(f'{(nd["b", s, w][mk] - nd["a", s, w][mk]).mean():+.5f}' for s in SEEDS); ps_c = ', '.join(f'{(nd["c", s, w][mk] - nd["a", s, w][mk]).mean():+.5f}' for s in SEEDS)
        pBA = pool['b', w][mk] - pool['a', w][mk]; pCA = pool['c', w][mk] - pool['a', w][mk]
        print(f'{name} | {int(mk.sum())} | {ms(dBA)} | ({ps_b}) | {ms(pBA)} | {ms(dCA)} | ({ps_c}) | {ms(pCA)} | {dBA.mean() / dCA.mean():+.3f} | {pBA.mean() / pCA.mean():+.3f}')
        res[suf, name] = (dBA.mean(), dCA.mean(), pBA.mean(), pCA.mean(), int(mk.sum()))
    print('levels (member mean over seeds / pool):')
    for name in ('hold cross-user', 'val cross-user', 'hold fold-0', 'val fold-0'):
        w, mk = cols[name]
        print('  ', name, ' | '.join(f'u20{arm} ' + ', '.join(f'{nd[arm, s, w][mk].mean():.5f}' for s in SEEDS) + f' pool {pool[arm, w][mk].mean():.5f}' for arm in 'abc'))
    # seed-level heterogeneity of B-A on the decision column: difference of the two seeds' deltas, paired per feed
    if len(SEEDS) == 2:
        for name in ('hold cross-user', 'val cross-user'):
            w, mk = cols[name]
            d1 = nd['b', SEEDS[0], w][mk] - nd['a', SEEDS[0], w][mk]; d2 = nd['b', SEEDS[1], w][mk] - nd['a', SEEDS[1], w][mk]
            print(f'  seed heterogeneity {name}: delta(s{SEEDS[1]}) - delta(s{SEEDS[0]}) = {ms(d2 - d1)}; per-feed corr of the two seed deltas {np.corrcoef(d1, d2)[0, 1]:+.3f}')
            # cross-seed pairings (B seed i vs A seed j), all four
            xs = ', '.join(f'b{sb}-a{sa} {(nd["b", sb, w][mk] - nd["a", sa, w][mk]).mean():+.5f}' for sb in SEEDS for sa in SEEDS)
            print(f'  cross-seed pairings {name}: {xs}')

print('\n===== RULE (20:21Z + 21:26Z addendum; cross-user columns; member deltas pooled over the 2 seeds) =====')
for suf in ('', '@last'):
    bh, ch, pbh, pch, nh = res[suf, 'hold cross-user']; bv, cv, pbv, pcv, nv = res[suf, 'val cross-user']
    go = (bh >= 0.0008) and (bv >= 0) and (bh / ch >= 0.5); kill = bh < 0.0004
    print(f'{"selected" if suf == "" else "@last"}: B-A holdout {bh:+.7f} (n {nh}), val {bv:+.7f} (n {nv}), C-A holdout {ch:+.7f}, ratio {bh / ch:+.3f} -> GO {go} KILL {kill} -> {"GO" if go else "KILL" if kill else "2 MORE SEEDS"}')
    gop = (pbh >= 0.0008) and (pbv >= 0) and (pbh / pch >= 0.5); killp = pbh < 0.0004
    print(f'   same rule on the 2-POOL statistic (not the declared one): B-A holdout {pbh:+.7f}, val {pbv:+.7f}, ratio {pbh / pch:+.3f} -> {"GO" if gop else "KILL" if killp else "2 MORE SEEDS"}')
    bh, ch, pbh, pch, nh = res[suf, 'hold all']; bv = res[suf, 'val all'][0]
    print(f'   same rule on ALL feeds, member (20:21Z text before the addendum): B-A holdout {bh:+.7f}, val {bv:+.7f}, ratio {bh / ch:+.3f} -> {"GO" if (bh >= 0.0008 and bv >= 0 and bh / ch >= 0.5) else "KILL" if bh < 0.0004 else "2 MORE SEEDS"}')
