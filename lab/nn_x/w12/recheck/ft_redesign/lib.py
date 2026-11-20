"""Shared loaders for the ft_redesign recheck (read-only on preds/)."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
P = Path('<repo>/preds'); W = Path('<repo>/nn_x/w5/preds')
OUT = Path('<repo>/nn_x/w12/recheck/ft_redesign/out'); OUT.mkdir(exist_ok=True)
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
uv = fv['user_id'].to_numpy(); Hv = fv['H'].to_numpy(); tv = fv['timestamp'].to_numpy()
ut = ft_['user_id'].to_numpy(); Ht = ft_['H'].to_numpy(); tt = ft_['timestamp'].to_numpy()
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1].astype(np.float64)
rkv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 0].astype(np.float64)
rkt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 0].astype(np.float64)
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, ov, Mv)
se = lambda d: d.std() / np.sqrt(len(d)); f = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
sd4 = (951, 952, 953, 954)
GROUPS = {
    'S:kd05w192': [f'w5_e10_kd05w192_s{s}' for s in sd4], 'S:kd05t25w192': [f'w5_e10_kd05t25w192_s{s}' for s in sd4],
    'S:kd05': [f'w5_e10_kd05_s{s}' for s in sd4], 'S:kd05t25': [f'w5_e10_kd05t25_s{s}' for s in sd4], 'S:kd08': [f'w5_e10_kd08_s{s}' for s in sd4],
    'H:m02': [f'r6_cd_m02_r{s}' for s in range(101, 108)], 'H:m02p70k': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
    'H:gru': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)], 'H:pop1': [f'abl_e10_pop1_s{s}' for s in range(951, 959)]}
W8 = GROUPS['S:kd05w192'] + GROUPS['S:kd05t25w192']; N8 = GROUPS['S:kd05'] + GROUPS['S:kd05t25']
C3M = GROUPS['H:gru'] + GROUPS['H:m02'] + GROUPS['H:m02p70k'] + GROUPS['H:pop1']
def raw(m, kind, sp):
    """kind: 'ft' (n2cpv3s3) or 'to'."""
    if kind == 'ft': return np.load(P / f'nn_{m}_n2cpv3s3_{"val_oof" if sp == "val" else "test"}.npy')
    return np.load(P / f'nn_{m}_trainonly_{sp}.npy')
_c = {}
def z(m, kind, sp):
    k = (m, kind, sp)
    if k not in _c: _c[k] = (zv if sp == 'val' else zt)(raw(m, kind, sp))
    return _c[k]
def pool(ms, kind, sp): return np.mean([z(m, kind, sp) for m in ms], 0)
def feed_mean(x, M, off):
    """per-feed mean of a per-candidate array"""
    return np.add.reduceat(x, off) / M
def feed_corr(a, b, M, off):
    """per-feed Pearson correlation between two per-candidate arrays"""
    ma = np.repeat(feed_mean(a, M, off), M); mb = np.repeat(feed_mean(b, M, off), M)
    da, db = a - ma, b - mb
    num = np.add.reduceat(da * db, off); den = np.sqrt(np.add.reduceat(da * da, off) * np.add.reduceat(db * db, off)) + 1e-12
    return num / den
