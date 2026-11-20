"""why/where W1: per-feed segment covariates for the 14,423 val feeds and the 30,000 holdout feeds (label use: only to locate the
positive rows of a feed, for the positive-item segments). CPU, read-only; writes out/seg_val.npz, out/seg_hold.npz.
Covariates (feed level):
  M, H, user0, day
  user_ntr   : number of TRAINING feeds of the same non-zero user (train split minus the 30,000 holdout feeds); -1 for user 0
  pos_seen   : share of the feed's positive items that occur in a training candidate row (train minus holdout feeds), any day
  pos_seen_c : same, causal: the item occurs in a train candidate row (any train feed) of a day before the feed's day
  cand_seen  : share of the feed's candidates seen in training candidate rows (model-perspective warm share)
  pos_pop0   : share of the feed's positives whose it_hist_all_luo input is 0 (as fed to the model: fresh on holdout, stale on val)
  pos_age_d  : mean listing age (days, f4 knots; capped f4 -> 999) of the positives; cand_age_med: median candidate age
  cat_mode   : most frequent candidate category of the feed; pos_cat: category of the first positive
  npos       : number of positives"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np, polars as pl, numba as nb, time

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; FC = ROOT / 'nn/feat_cache'
HERE = ROOT / 'lab/nn_x/w12/moe/why/where'; OUT = HERE / 'out'
MINI = ROOT / 'lab/nn_x/w12/next/poplag/cache/mini'
MSK, DAY = 3 * 3600, 86400
T0 = time.time(); lg = lambda *a: print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)
day_of = lambda ts: (np.asarray(ts, np.int64) + MSK) // DAY


def age_days(f4):
    kn = np.load(ROOT / 'data/feats/life_f4_age_knots.npy'); x = f4.astype(np.float64); u = x * 64
    k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k; a = kn[k] * (1 - w) + kn[k + 1] * w
    return np.where(x < 63 / 64, a / 24.0, 999.0)


@nb.njit(parallel=False, cache=True)
def first_day(item, rday, n_items):
    out = np.full(n_items, 1 << 30, np.int64)
    for i in range(len(item)):
        if rday[i] < out[item[i]]: out[item[i]] = rday[i]
    return out


@nb.njit(cache=True)
def feed_mode(x, off, M, nc):
    out = np.zeros(len(M), np.int64); cnt = np.zeros(nc, np.int64)
    for f in range(len(M)):
        cnt[:] = 0
        for r in range(off[f], off[f] + M[f]): cnt[x[r]] += 1
        out[f] = np.argmax(cnt)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pl.read_parquet(FLAT / 'train/feeds.parquet'); hold = np.load(MINI / 'hold.npy')
    nfe = len(tr); is_h = np.zeros(nfe, bool); is_h[hold] = True
    tday = day_of(tr['timestamp'].to_numpy()); tuser = tr['user_id'].to_numpy()
    citem = np.load(FLAT / 'train/cand_item_id.npy'); cfidx = np.load(FLAT / 'train/cand_fidx.npy')
    lg('train cand loaded', len(citem))
    nI = int(max(citem.max(), np.load(FLAT / 'val/cand_item_id.npy').max())) + 1
    rtrain = ~is_h[cfidx]
    seen = np.zeros(nI, bool); seen[citem[rtrain]] = True
    fd = first_day(citem, tday[cfidx], nI)  # any train feed (holdout included), causal first day
    lg('item tables built; seen items', int(seen.sum()))
    # user training-feed counts (non-holdout train feeds)
    uc = np.bincount(tuser[~is_h & (tuser > 0)], minlength=int(tuser.max()) + 1)
    del citem, cfidx, rtrain

    def build(sp):
        if sp == 'val':
            f = pl.read_parquet(FLAT / 'val/feeds.parquet'); M = f['M'].to_numpy().astype(np.int64); H = f['H'].to_numpy()
            off = f['cand_off'].to_numpy().astype(np.int64); u = f['user_id'].to_numpy(); d = day_of(f['timestamp'].to_numpy())
            item = np.load(FLAT / 'val/cand_item_id.npy'); cat = np.load(FLAT / 'val/cand_category_id.npy').astype(np.int64)
            f4 = np.load(FLAT / 'val/cand_f4.npy'); y = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
            pop = np.load(FC / 'val/it_hist_all_luo.npy')
        else:
            M = np.load(MINI / 'M.npy').astype(np.int64); H = np.load(MINI / 'H.npy'); u = np.load(MINI / 'user_id.npy')
            d = day_of(np.load(MINI / 'ts.npy')); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
            crow = np.load(MINI / 'crow.npy'); item = np.load(MINI / 'c_item.npy'); cat = np.load(MINI / 'c_cat.npy').astype(np.int64)
            f4 = np.load(MINI / 'c_f.npy')[:, 3]; y = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
            assert (np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')[crow] == y).all(), 'holdout row order'
            assert (np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[crow]) == item).all()
            pop = np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[crow])
        fid = np.repeat(np.arange(len(M)), M); rday = d[fid]
        ok = item >= 0
        s_any = np.where(ok, seen[np.maximum(item, 0)], False)
        s_c = np.where(ok, fd[np.maximum(item, 0)] < rday, False)
        age = age_days(f4)
        pos = y > 0; npos = np.bincount(fid, weights=pos, minlength=len(M))
        pm = lambda x: np.bincount(fid, weights=x * pos, minlength=len(M)) / np.maximum(npos, 1)
        # first positive's category
        fp = np.full(len(M), -1); pr = np.flatnonzero(pos); fpos = fid[pr]; first = np.unique(fpos, return_index=True)
        fp[first[0]] = cat[pr[first[1]]]
        med_age = np.array([np.median(age[o:o + m]) for o, m in zip(off, M)])
        out = dict(M=M, H=H, user0=(u == 0), day=d, user_ntr=np.where(u > 0, uc[np.minimum(u, len(uc) - 1)] * (u < len(uc)), -1),
                   pos_seen=pm(s_any.astype(np.float64)), pos_seen_c=pm(s_c.astype(np.float64)),
                   cand_seen=np.bincount(fid, weights=s_any, minlength=len(M)) / M,
                   pos_pop0=pm((np.nan_to_num(pop, nan=0) <= 0).astype(np.float64)),
                   pos_age_d=pm(age), cand_age_med=med_age, cat_mode=feed_mode(cat, off, M, 64), pos_cat=fp, npos=npos)
        np.savez(OUT / f'seg_{sp}.npz', **out)
        lg(sp, 'feeds', len(M), {k: float(np.mean(v)) for k, v in out.items() if k not in ('cat_mode', 'pos_cat', 'day')})

    build('val'); build('hold')


if __name__ == '__main__':
    main()
