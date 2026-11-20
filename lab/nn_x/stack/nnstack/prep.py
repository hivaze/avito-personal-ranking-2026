"""Prepare per-candidate stacker inputs: family z's (ft, to), base pool score, context; val/test/holdout + FT folds."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path
from sklearn.model_selection import GroupKFold
from common import feed_meta
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
R = Path('<repo>'); P = R / 'preds'; D = R / 'nn_x/stack/nnstack/data'; D.mkdir(exist_ok=True)
FAM = {
 'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82'] + [f'v15_cd_allprew_s{i}' for i in (83, 84, 85)],
 'F6': ['v13_lr1_mix1'] + [f'v14_s{i}' for i in range(92, 97)],
 'F7x': [f'r6_cd_m02_r{i}' for i in range(101, 108)],
 'F9': [f'v22_pre2s_lr1_s{i}' for i in range(401, 407)],
 'F10': [f'v23_gru_pre2s_s{i}' for i in range(501, 507)],
 'F12': [f'v26_d192_pre2s_s{i}' for i in range(701, 708)],
 'F7x70': [f'r6_cd_m02_p70k_r{i}' for i in range(101, 108)],
}
POOLS = {'PF7x': ['F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'], 'G3': ['F10', 'F7x', 'F7x70']}

def z(o, M, x): return feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]

# ---- val/test member z's -> family means
for split, ftn, ton in [('val', 'n2cpv3_val_oof', 'trainonly_val'), ('test', 'n2cpv3_test', 'trainonly_test')]:
    f, M, o = feed_meta(split)
    fam = {}
    for F, ms in FAM.items():
        for half, suf in [('ft', ftn), ('to', ton)]:
            zs = [z(o, M, np.load(P / f'nn_{m}_{suf}.npy')) for m in ms]
            fam[(F, half)] = (np.sum(zs, 0), len(ms))
            np.save(D / f'{split}_fam_{F}_{half}.npy', z(o, M, np.mean(zs, 0)).astype(np.float32))
    for pool, Fs in POOLS.items():
        for half in ('ft', 'to'):
            s = sum(fam[(F, half)][0] for F in Fs) / sum(fam[(F, half)][1] for F in Fs)
            np.save(D / f'{split}_pool_{pool}_{half}.npy', z(o, M, s).astype(np.float32))
        spec = 0.5 * np.load(D / f'{split}_pool_{pool}_ft.npy') + 0.5 * np.load(D / f'{split}_pool_{pool}_to.npy')
        np.save(D / f'{split}_pool_{pool}_spec.npy', z(o, M, spec).astype(np.float32))
    # context
    H = f['H'].to_numpy().astype(np.float32); u0 = (f['user_id'].to_numpy() == 0).astype(np.float32)
    luo = np.nan_to_num(np.asarray(np.load(R / f'nn_x/r6/feat_cache/{split}/it_hist_all_luo.npy', mmap_mode='r'), np.float64))
    cold = (luo == 0).astype(np.float32)
    np.save(D / f'{split}_cold.npy', cold)
    np.save(D / f'{split}_feedctx.npy', np.stack([M.astype(np.float32), H, u0], 1))
    if split == 'val':
        y = np.load(R / 'data/flat/val/cand_target.npy').astype(np.float32); np.save(D / 'val_y.npy', y)
        for pool in POOLS:
            sp = np.load(D / f'val_pool_{pool}_spec.npy'); print(pool, 'val spec ndcg', ndcg_per_feed(sp, y, o, M).mean())
        g = f['user_id'].to_numpy().astype(np.int64); u = g == 0; g = np.where(u, -1 - np.arange(len(M)), g)
        fold = np.zeros(len(M), np.int8)
        for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(M)), groups=g)): fold[ho] = k
        np.save(D / 'val_fold.npy', fold); print('fold sizes', np.bincount(fold))
        ts = f['timestamp'].to_numpy().astype(np.int64); np.save(D / 'val_tue.npy', ((ts + 10800) // 86400 < 20720))
    print(split, 'done', flush=True)

# ---- holdout (train-only members): reconstruct holdout feed ids as nn_x/r6/holdout_eval.py
tr = pl.read_parquet(R / 'data/flat/train/feeds.parquet')
Mtr = tr['M'].to_numpy(); offtr = tr['cand_off'].to_numpy()
pr = np.nan_to_num(np.load(R / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
pr[Mtr < 50] = 0.0
cand = np.where((pr > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, oh = np.load(P / 'holdout/holdout_meta.npy'); assert (Mh == Mtr[hold]).all()
rows = np.concatenate([np.arange(a, a + m) for a, m in zip(offtr[hold], Mtr[hold])])
yh = np.load(P / 'holdout/holdout_y.npy').astype(np.float32); np.save(D / 'hold_y.npy', yh)
fam = {}
for F, ms in FAM.items():
    zs = [z(oh, Mh, np.load(P / f'holdout/nn_{m}_holdout.npy')) for m in ms]
    fam[F] = (np.sum(zs, 0), len(ms))
    np.save(D / f'hold_fam_{F}_to.npy', z(oh, Mh, np.mean(zs, 0)).astype(np.float32))
for pool, Fs in POOLS.items():
    s = sum(fam[F][0] for F in Fs) / sum(fam[F][1] for F in Fs)
    np.save(D / f'hold_pool_{pool}_to.npy', z(oh, Mh, s).astype(np.float32))
    print(pool, 'holdout to-pool ndcg', ndcg_per_feed(np.load(D / f'hold_pool_{pool}_to.npy'), yh, oh, Mh).mean())
luo = np.nan_to_num(np.asarray(np.load(R / 'nn_x/r6/feat_cache/train/it_hist_all_luo.npy', mmap_mode='r')[rows], np.float64))
np.save(D / 'hold_cold.npy', (luo == 0).astype(np.float32))
ht = tr[hold]
np.save(D / 'hold_feedctx.npy', np.stack([Mh.astype(np.float32), ht['H'].to_numpy().astype(np.float32),
                                         (ht['user_id'].to_numpy() == 0).astype(np.float32)], 1))
np.save(D / 'hold_user.npy', ht['user_id'].to_numpy().astype(np.int64))
np.save(D / 'hold_meta.npy', np.stack([Mh, oh]))
print('hold cold share', (luo == 0).mean(), 'done')
