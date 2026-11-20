"""skeptic leak_overfit: shared loaders (CPU only, read-only on prediction files)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
R = Path('<repo>'); W = R/'nn_x/w5/preds'; AB = R/'nn_x/abl/preds'; P = R/'preds'
fv, Mv, ov = feed_meta('val'); ft, Mt, ot = feed_meta('test')
Mv = np.asarray(Mv, np.int64); ov = np.asarray(ov, np.int64); Mt = np.asarray(Mt, np.int64); ot = np.asarray(ot, np.int64)
yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(W/'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W/'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1].astype(np.float64)
se = lambda d: d.std(ddof=1)/np.sqrt(len(d))
f = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def vpath(v, suf=''):
    for D in (W, AB, P):
        p = D/f'nn_{v}{suf}_trainonly_val.npy'
        if p.exists(): return p
    raise FileNotFoundError(v+suf)
def hpath(v, suf=''):
    for D in (W/'holdout', AB/'holdout', P/'holdout'):
        p = D/f'nn_{v}{suf}_holdout.npy'
        if p.exists(): return p
    raise FileNotFoundError(v+suf)
def tpath(v, suf=''):
    for D in (W, AB, P):
        p = D/f'nn_{v}{suf}_trainonly_test.npy'
        if p.exists(): return p
    raise FileNotFoundError(v+suf)
def ftpath(v, sp):
    n = f'nn_{v}_n2cpv3s3_val_oof.npy' if sp == 'val' else f'nn_{v}_n2cpv3s3_test.npy'
    for D in (P, W, AB):
        if (D/n).exists(): return D/n
    raise FileNotFoundError(v)
LV = lambda v, suf='': zv(np.load(vpath(v, suf)))
LH = lambda v, suf='': zh(np.load(hpath(v, suf)))
S4 = (951, 952, 953, 954)
E = lambda arm, seeds=S4: [f'w5_e10_{arm}_s{s}' for s in seeds]
W8 = E('kd05w192') + E('kd05t25w192')
N8 = E('kd05') + E('kd05t25')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]
F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
G3M = F10 + F7x + F7x70
POP1 = [f'abl_e10_pop1_s{s}' for s in range(951, 959)]
POP1KD = [f'abl_e10_pop1kd05t25_s{s}' for s in S4]
def poolv(ms, suf=''): return np.mean([LV(m, suf) for m in ms], 0)
def poolh(ms, suf=''): return np.mean([LH(m, suf) for m in ms], 0)
