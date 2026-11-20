"""poplag_verify step A: CONTROL re-derived with my own metric / pool code. Labelled, but LAG 0 ONLY (stored files and the study's CPU lag-0 replay).
No lagged prediction file is opened here.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/poplag_verify/v_control.py"""
from v_common import *
import time

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy')
Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
assert oh[0] == 0 and (oh[1:] == np.cumsum(Mh)[:-1]).all() and len(yh) == Mh.sum()
print(f'=== poplag_verify CONTROL {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: holdout feeds {len(Mh)}, rows {int(Mh.sum())}, positives {int(yh.sum())}, '
      f'feeds without positive {int((np.add.reduceat(yh, oh) == 0).sum())}; +- = 1 SE paired per feed ===')
# identity of the holdout: my own re-derivation of the selection vs holdout_meta and vs the study's mini split
f, hold, p = my_holdout()
assert (f['M'].to_numpy()[hold] == Mh).all()
assert np.array_equal(hold, np.load(STUDY / 'mini/hold.npy'))
co = f['cand_off'].to_numpy().astype(np.int64)[hold]
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co, Mh)])
assert np.array_equal(rows, np.load(STUDY / 'mini/crow.npy'))
ytr = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')[rows]
print(f'holdout identity: my selection == holdout_meta M == study mini split (hold, crow): True; holdout_y == train targets of those rows: {bool((ytr == yh).all())}')

ev = lambda s: ndcg_feed(s, yh, oh, Mh)
z = lambda x: zfeed(x, oh, Mh)
# cross-check of my vectorised NDCG against a plain loop on 1,500 feeds, and against src/metrics on all feeds for one member
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402  (only as a cross-check of my own function)
s0 = np.load(stored_path(W8[0]))
sub = np.random.default_rng(7).choice(len(Mh), 1500, replace=False)
a = ev(s0)
print(f'NDCG self-check: |vectorised - loop| max on 1500 feeds {np.abs(a[sub] - ndcg_feed_slow(s0, yh, oh, Mh, sub)).max():.2e}; |mine - src/metrics| max on 30000 feeds {np.abs(a - ndcg_per_feed(s0, yh, oh, Mh)).max():.2e}')

res = {}
print('\nmember                        stored bf16 | study CPU lag-0 | paired CPU - stored | rows tied within feed (stored / CPU) | row corr')
worst = 0.0
for v in W8 + P4:
    xs = np.load(stored_path(v)); xc = np.load(STUDY / f'pred_lag0_{v}.npy')
    assert xs.shape == xc.shape == yh.shape and np.isfinite(xs).all() and np.isfinite(xc).all()
    es, ec = ev(xs), ev(xc); d = ec - es; worst = max(worst, abs(d.mean()))
    res[v] = (xs, xc)
    print(f' {v:28s} {es.mean():.5f} | {ec.mean():.5f} | {f5(d)} | max |z diff| {np.abs(z(xs) - z(xc)).max():.3f} | corr {np.corrcoef(xs, xc)[0, 1]:.6f}')
print(f' largest |member CPU - stored| = {worst:.5f} (bar 0.0010)')


def pool(mem, k):
    return np.mean([z(res[v][k]) for v in mem], 0)


def stat(pw, pp, w):
    return ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)


sw, sp, cw, cp = pool(W8, 0), pool(P4, 0), pool(W8, 1), pool(P4, 1)
print(f'\n W8 pool: stored {ev(sw).mean():.5f} | CPU {ev(cw).mean():.5f} | paired {f5(ev(cw) - ev(sw))}')
print(f' pop1kd pool: stored {ev(sp).mean():.5f} | CPU {ev(cp).mean():.5f} | paired {f5(ev(cp) - ev(sp))}')
print(f' pop1kd pool - W8 pool: stored {f5(ev(sp) - ev(sw))} | CPU {f5(ev(cp) - ev(cw))}')
for w in (0.1, 0.2, 0.3):
    a, b = stat(sw, sp, w), stat(cw, cp, w)
    print(f' statistic w {w}: stored {f5(a)} | CPU lag-0 {f5(b)} | CPU - stored {f5(b - a)}')
a, b = stat(sw, sp, 0.2), stat(cw, cp, 0.2)
print(f' CONTROL (rule: members within 0.0010, statistic within +-0.0003 of the stored one): members {worst <= 0.0010}; |CPU - stored| = {abs(b.mean() - a.mean()):.5f} -> {abs(b.mean() - a.mean()) <= 0.0003}')
# variant readings of the statistic (does the result depend on the exact pooling convention?)
alt = ev(0.8 * sw.astype(np.float64) + 0.2 * sp.astype(np.float64)) - ev(sw)
print(f' variant (pools blended without the second z): stored {f5(alt)}')
# SE that a read restricted to holdout days >= Sep 13 would have (lag-0 CPU statistic; SE only, the mean of this slice is not printed)
ts = f['timestamp'].to_numpy().astype(np.int64)[hold]; late = (ts + TZ) // DAY >= SEP6 + 7
print(f' holdout days >= Sep 13: {int(late.sum())} feeds; SE of the lag-0 CPU statistic (w 0.2) on that subset {b[late].std(ddof=1) / np.sqrt(late.sum()):.5f}; on the rest ({int((~late).sum())} feeds) {b[~late].std(ddof=1) / np.sqrt((~late).sum()):.5f}')
