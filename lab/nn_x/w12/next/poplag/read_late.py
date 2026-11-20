"""poplag: STAGED, NOT RUN BY THE AUTHOR. Labelled read of a lagged view restricted to holdout days >= Sep 13 (13,424 feeds), the subset on which the
label-free criteria G1-G5 hold for both views (logs/gate_v3_late.txt). It needs a pre-declaration by the lead before it is run (the declared gate on the
30,000 feeds FAILED for the private-like view, so no lagged view was read). Prints nothing about the other 16,576 feeds.
Usage: CUDA_VISIBLE_DEVICES="" python read_late.py priv_v3"""
import sys, time
from pl_common import *
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
fid = np.repeat(np.arange(len(Mh)), Mh)


def z(x):
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, weights=x) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / Mh - mu * mu, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


pool = lambda view, mem: np.mean([z(np.load(CACHE / f'pred_{view}_{v}.npy')) for v in mem], 0)
view = sys.argv[1]
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
sel = day_of(np.load(CACHE / 'mini/ts.npy')) >= 20709
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)[sel]
stat = lambda pw, pp, w: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)
print(f'=== poplag read_late {view} {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; holdout days >= Sep 13: {int(sel.sum())} feeds; +- = 1 SE paired per feed ===')
w0, p0, w1, p1 = pool('lag0', W8), pool('lag0', POP1KD), pool(view, W8), pool(view, POP1KD)
print(f' W8 pool: lag 0 {ev(w0).mean():.5f} lagged {ev(w1).mean():.5f}: {fmt(ev(w1) - ev(w0))} | pop1kd pool: lag 0 {ev(p0).mean():.5f} lagged {ev(p1).mean():.5f}: {fmt(ev(p1) - ev(p0))}')
for w in (0.1, 0.2, 0.3):
    a, b = stat(w0, p0, w), stat(w1, p1, w)
    print(f' S({w}): lag 0 (CPU) {fmt(a)} | lagged {fmt(b)} | lagged - lag 0 {fmt(b - a)}')
print(' Note: the CPU lag-0 statistic on 30,000 feeds is -0.00047 against -0.00063 from the stored bf16 files (logs/control.txt); judge the paired change, not the level alone.')
