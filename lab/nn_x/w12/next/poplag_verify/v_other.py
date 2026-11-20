"""poplag_verify step M (LABEL-FREE): the 26 inputs the study left as stored ("own feed / own history only"): is there a holdout-vs-test shift in them
that a lag would have to explain? Mean on the model scale (attach_xt transform), in private-test SDs, and missing share, subset holdout rows vs private test rows (M >= 50).
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_other.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


NAMES = ['ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'ctx_hn_top1w', 'ctx_hn_rrw',
         'ctx_hn_bestrk', 'ctx_hn_margin', 'ctx_hn_soft05', 'ctx_hn_e1_rrw', 'ctx_hs_mutual', 'ctx_hs_k128', 'ctx_hs_e1', 'ctx_hs_t10', 'pre2_fe_z', 'pre2_new_z', 'pre2_fe_s', 'pre2_new_s', 'pre2_hmax32', 'pre2_fe']
ftr, hold, _ = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fco = ftr['cand_off'].to_numpy().astype(np.int64)
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
hM = fM[hold]; fid = np.repeat(np.arange(len(hold)), hM)
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[hold], hM)]); late = (hd >= SEP6 + 7)[fid]
fte = pl.read_parquet(FLAT / 'test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
t_fx = np.load(FLAT / 'test/cand_fidx.npy').astype(np.int64); ref = ((tts >= PUB_TS) & (tM >= 50))[t_fx]
fva = pl.read_parquet(FLAT / 'val/feeds.parquet'); vM = fva['M'].to_numpy().astype(np.int64); v_fx = np.load(FLAT / 'val/cand_fidx.npy').astype(np.int64); vref = (vM >= 50)[v_fx]


def tr(n, x):
    x = np.asarray(x, np.float32)
    x = np.clip(x, -5, 5) if n.endswith('_z') else (x if n.endswith('_rk') else np.clip(x, -10, 10))
    return np.nan_to_num(x, nan=0.0).astype(np.float64)


P('column | missing share holdout subset / private test / val (M >= 50) | (mean holdout subset - mean private test) / SD(private test) | same for val - private test | same for holdout complement - private test')
worst = (0.0, '')
for n in NAMES:
    h = np.asarray(np.load(FC / 'train' / f'{n}.npy', mmap_mode='r')[rows]); t = np.load(FC / 'test' / f'{n}.npy')[ref]; v = np.load(FC / 'val' / f'{n}.npy')[vref]
    sd = tr(n, t).std() + 1e-12
    a = (tr(n, h[late]).mean() - tr(n, t).mean()) / sd; b = (tr(n, v).mean() - tr(n, t).mean()) / sd; c = (tr(n, h[~late]).mean() - tr(n, t).mean()) / sd
    if abs(a) > worst[0]: worst = (abs(a), n)
    P(f'{n:17s} | {np.isnan(h[late]).mean():.4f} / {np.isnan(t).mean():.4f} / {np.isnan(v).mean():.4f} | {a:+.3f} | {b:+.3f} | {c:+.3f}')
P(f'largest |standardised mean gap| holdout subset vs private test over the 26 stored inputs: {worst[0]:.3f} ({worst[1]})')
(HERE / 'logs' / 'v_other.txt').write_text('\n'.join(OUT) + '\n')
