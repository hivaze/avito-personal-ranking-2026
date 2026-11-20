"""ft_1115 shared loaders (read-only on preds/, subs/, data/). CPU only."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'; W5 = R / 'nn_x/w5/preds'; ABL = R / 'nn_x/abl/preds'
D = R / 'nn_x/w12/ft_1115'; OUT = D / 'out'; OUT.mkdir(exist_ok=True)
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
uv = fv['user_id'].to_numpy().astype(np.int64); tt = ft_['timestamp'].to_numpy()
PUB = tt < pmix.BOUND                      # public test feeds (8,562)
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
zt = lambda x: feed_rank_z(ot, Mt, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
se = lambda d: d.std() / np.sqrt(len(d))
WT = pmix.weights(); WPUB, WPRIV = WT['public'], WT['private']
def f(d, w=None):
    if w is None: return f'{d.mean():+.5f}+-{se(d):.5f}'
    m, s = pmix.wmean_se(d, w); return f'{m:+.5f}+-{s:.5f}'
def ms(d, w=None):
    if w is None: return float(d.mean()), float(se(d))
    m, s = pmix.wmean_se(d, w); return float(m), float(s)
S8 = range(951, 959); S4 = range(951, 955)
MEM = {
    'W8': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'W8b': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(955, 959)],
    'P1d': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
    'G3': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)] + [f'r6_cd_m02_r{s}' for s in range(101, 108)] + [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
}
MEM['W16'] = MEM['W8'] + MEM['W8b']
def stored(tag, sp):
    """stored branch files of a built candidate: (FT pool, TO pool), each a member-equal mean of per-feed rank-z (pools with the 0.2 admixture: already mixed per branch)"""
    return np.load(P / f'ens_{tag}_ft_{sp}.npy'), np.load(P / f'ens_{tag}_to_{sp}.npy')
def raw(m, kind, sp):
    return np.load(P / f'nn_{m}_{kind}_{"val_oof" if sp == "val" and kind != "trainonly" else sp}.npy') if kind != 'trainonly' else np.load(P / f'nn_{m}_trainonly_{sp}.npy')
def mpool(ms_, kind, sp):
    z = zv if sp == 'val' else zt
    return np.mean([z(raw(m, kind, sp)) for m in ms_], 0)
def test_folds(m):
    for d in (P, W5, ABL):
        p = d / f'nn_{m}_n2cpv3s3_test_folds.npy'
        if p.exists(): return np.load(p).astype(np.float32)
    raise FileNotFoundError(m)
def user_groups():
    """user-grouped groups; every user-0 feed is its own group (as in finetune.py --u0_singleton)"""
    g = uv.copy(); u0 = g == 0; g[u0] = -1 - np.arange(len(g))[u0]; return g
def feed_mean(x, M, off): return np.add.reduceat(x, off) / M
def feed_corr(a, b, M, off):
    ma = np.repeat(feed_mean(a, M, off), M); mb = np.repeat(feed_mean(b, M, off), M); da, db = a - ma, b - mb
    return np.add.reduceat(da * db, off) / (np.sqrt(np.add.reduceat(da * da, off) * np.add.reduceat(db * db, off)) + 1e-12)
