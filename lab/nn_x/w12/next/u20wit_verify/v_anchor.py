"""u20wit_verify: anchor of the CPU fp32 path on the stored bf16 GPU holdout files (H0 = 5,974 fold-0 holdout feeds, XH = 8,000 cross-user holdout feeds), own code."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '4'
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path('<repo>')
src = open(ROOT / 'nn_x/w12/next/u20wit_verify/v_cmp.py').read().split("f = pl.read_parquet")[0]; exec(src)
W = ROOT / 'nn_x/w5/preds'; ST = ROOT / 'nn_x/w12/next/u20wit/cache'
ft = pl.read_parquet(ROOT / 'data/flat/train/feeds.parquet', columns=['user_id', 'M', 'cand_off']); Mtr = ft['M'].to_numpy().astype(np.int64); otr = ft['cand_off'].to_numpy().astype(np.int64)
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pold > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
sm = np.load(ST / 'sub_meta.npz'); subf = sm['src_fidx']; Msub = Mtr[subf]; osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]); h0 = np.where(sm['set_id'] == 2)[0]
xhf = np.load(ST / 'subxh/src_fidx.npy'); Mx = Mtr[xhf]; ox = np.concatenate([[0], np.cumsum(Mx)[:-1]])
def take(pred_h, feeds):   # holdout-array rows of train feeds `feeds` (must be holdout feeds)
    j = np.searchsorted(hold, feeds); assert (hold[j] == feeds).all()
    return np.concatenate([pred_h[oh[k]:oh[k] + Mh[k]] for k in j]), np.concatenate([yh[oh[k]:oh[k] + Mh[k]] for k in j])
res = {}
for a in 'abc':
    for s in (951, 952):
        v = f'w5_e10_u20{a}_s{s}'; g = np.load(W / f'holdout/nn_{v}_holdout.npy').astype(np.float32)
        gh, yy = take(g, subf[h0]); Mq = Msub[h0]; oq = np.concatenate([[0], np.cumsum(Mq)[:-1]])
        c = np.load(ST / f'sc/{v}__tr.npy'); ch = np.concatenate([c[osub[k]:osub[k] + Msub[k]] for k in h0])
        e_g = my_ndcg(gh, yy, oq, Mq); e_c = my_ndcg(ch.astype(np.float32), yy, oq, Mq)
        gx, yx = take(g, xhf); cx = np.load(ST / f'scxh/{v}__tr.npy'); ex_g = my_ndcg(gx, yx, ox, Mx); ex_c = my_ndcg(cx.astype(np.float32), yx, ox, Mx)
        res[a, s] = (e_g, e_c, ex_g, ex_c)
        cc = feed_corr(gh.astype(np.float64), ch.astype(np.float64), oq, Mq)
        print(f'{v}: H0 CPU - GPU file {ms(e_c - e_g)} (within-feed logit corr {np.nanmean(cc):.5f}) | XH CPU - GPU file {ms(ex_c - ex_g)}', flush=True)
for j, nm in ((0, 'H0'), (2, 'XH')):
    dg = np.mean([res['b', s][j] - res['a', s][j] for s in (951, 952)], 0); dc = np.mean([res['b', s][j + 1] - res['a', s][j + 1] for s in (951, 952)], 0)
    print(f'{nm}: B-A member from the GPU files {ms(dg)} | from the CPU replay {ms(dc)} | CPU - GPU of the delta {ms(dc - dg)}')
