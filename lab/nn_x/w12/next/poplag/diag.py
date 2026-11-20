"""poplag: LABEL-FREE prediction-shift diagnostics of the staged lagged predictions (no target file is opened; not a reading of the rule).
How far does each pool's within-feed ranking move when the popularity inputs are made test-like?
Usage: CUDA_VISIBLE_DEVICES="" python diag.py <view> [<view> ...]   -> logs/diag_<views>.txt"""
import sys, time
from pl_common import *

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
fid = np.repeat(np.arange(len(Mh)), Mh); OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def z(x):
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, weights=x) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / Mh - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def pool(view, mem):
    return z(np.mean([z(np.load(CACHE / f'pred_{view}_{v}.npy')) for v in mem], 0))


def fcorr(a, b):
    return np.bincount(fid, weights=a * b) / Mh          # both are per-feed z (mean 0, sd 1)


def top1(a):
    o = np.lexsort((-a, fid)); return o[oh]


def topk_overlap(a, b, k=10):
    ra = np.empty(len(a), np.int64); o = np.lexsort((-a, fid)); ra[o] = np.arange(len(a)) - oh[fid[o]]
    rb = np.empty(len(b), np.int64); o = np.lexsort((-b, fid)); rb[o] = np.arange(len(b)) - oh[fid[o]]
    return np.bincount(fid, weights=((ra < k) & (rb < k))) / k


def main():
    mini = CACHE / 'mini'; crow = np.load(mini / 'crow.npy'); ts = np.load(mini / 'ts.npy'); late = day_of(ts) >= 20709
    w0, p0 = pool('lag0', W8), pool('lag0', POP1KD)
    P(f'=== poplag label-free prediction shift {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; 30,000 holdout feeds; +- = 1 SE over feeds ===')
    c0 = fcorr(w0, p0); P(f' lag 0: per-feed corr(W8 pool, pop1kd pool) {c0.mean():.4f} +- {se(c0):.4f}')
    x0 = np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[crow])
    for view in sys.argv[1:]:
        P(f'\n##### view {view} vs lag 0 (both CPU fp32) #####')
        w1, p1 = pool(view, W8), pool(view, POP1KD)
        for nm, a, b in (('W8 pool', w0, w1), ('pop1kd pool', p0, p1)):
            c = fcorr(a, b); t1 = (top1(a) != top1(b)).astype(np.float64); ov = topk_overlap(a, b)
            P(f' {nm:12s}: per-feed corr(lag 0, lagged) {c.mean():.4f} +- {se(c):.4f} | top-1 changed in {t1.mean():.4f} +- {se(t1):.4f} of feeds | top-10 overlap {ov.mean():.4f} | '
              f'days >= Sep 13: corr {c[late].mean():.4f}, top-1 changed {t1[late].mean():.4f}')
        c1 = fcorr(w1, p1); P(f' per-feed corr(W8 pool, pop1kd pool): lagged {c1.mean():.4f} +- {se(c1):.4f} (lag 0 {c0.mean():.4f}); paired change {fmt(c1 - c0)}')
        xl = np.load(CACHE / f'lag_{view}_it_hist_all_luo.npy')
        lost = (x0 > 0) & (xl == 0); kept = (x0 > 0) & (xl > 0); cold0 = x0 == 0
        for nm, a, b in (('W8 pool', w0, w1), ('pop1kd pool', p0, p1)):
            d = b - a
            P(f' {nm:12s}: mean change of the within-feed z: rows that lose their history count (lag 0 > 0 -> lagged 0; {lost.mean():.3f} of rows) {d[lost].mean():+.4f}; '
              f'rows that keep a count ({kept.mean():.3f}) {d[kept].mean():+.4f}; rows cold at lag 0 ({cold0.mean():.3f}) {d[cold0].mean():+.4f}; '
              f'mean z at lag 0 of the rows that lose it {a[lost].mean():+.4f}, lagged {b[lost].mean():+.4f}')
    (LOGS / f'diag_{"_".join(sys.argv[1:])}.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
