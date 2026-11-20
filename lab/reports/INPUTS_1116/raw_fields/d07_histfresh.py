"""INVENTORY D: how fresh is the history of a train row vs a val / test row? (parity fact for every history-derived input.)
No event timestamps exist; a lower bound of the newest event time is the newest POSTING time among the feed's history items, where the posting time
of an item comes from f4 (age) of any candidate appearance of that item (decode of feats_v2/life_/post_time.py, knots data/feats/life_f4_age_knots.npy).
Also: event-type / source mix and repeat structure of the newest 32 events by split. CPU only, read-only, label-free."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
import numpy as np, polars as pl
from pathlib import Path
R = Path('<repo>'); F = R / 'data/flat'; OUT = R / 'reports/INPUTS_1116/raw_fields'
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
knots = np.load(R / 'data/feats/life_f4_age_knots.npy'); K = 64; CAP = 63 / 64 - 1e-6
def age_hours(x):
    x = x.astype(np.float64); u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)
NI = 111_400_000; tp = np.full(NI, np.nan, np.float32); T0 = 1788000000   # offsets keep float32 exact to 128 s
fe = {sp: pl.read_parquet(F / sp / 'feeds.parquet') for sp in ('train', 'val', 'test')}
for sp in ('train', 'val', 'test'):
    ts = fe[sp]['timestamp'].to_numpy().astype(np.float64); fi = ld(sp, 'cand_fidx'); it = ld(sp, 'cand_item_id'); f4 = ld(sp, 'cand_f4')
    for a in range(0, len(it), 50_000_000):
        t = ts[np.asarray(fi[a:a + 50_000_000])] - age_hours(np.asarray(f4[a:a + 50_000_000])) * 3600.0; ok = np.isfinite(t)
        tp[np.asarray(it[a:a + 50_000_000])[ok]] = (t[ok] - T0).astype(np.float32)
P(f'items with a decoded posting time: {int(np.isfinite(tp).sum())}')
day = lambda t: (t + 10800) / 86400.0
for sp in ('train', 'val', 'test'):
    f = fe[sp]; n = len(f); ts = f['timestamp'].to_numpy().astype(np.float64); H = f['H'].to_numpy().astype(np.int64); ho = f['hist_off'].to_numpy().astype(np.int64)
    sel = np.arange(n) if sp != 'train' else np.sort(np.random.default_rng(0).choice(n, 200_000, replace=False))
    hi = ld(sp, 'hist_item_id'); het = ld(sp, 'hist_event_type'); hes = ld(sp, 'hist_event_source')
    gap = np.full(len(sel), np.nan); gap8 = np.full(len(sel), np.nan); known = np.zeros(len(sel)); e1_32 = np.full(len(sel), np.nan); s1_32 = np.full(len(sel), np.nan); e1_all = np.full(len(sel), np.nan)
    for q, fidx in enumerate(sel):
        h = H[fidx]
        if h == 0: continue
        items = np.asarray(hi[ho[fidx]:ho[fidx] + h]); t = tp[items].astype(np.float64) + T0; ok = np.isfinite(t)
        et = np.asarray(het[ho[fidx]:ho[fidx] + h]); es = np.asarray(hes[ho[fidx]:ho[fidx] + h])
        e1_32[q] = np.mean(et[:32] == 1); s1_32[q] = np.mean(es[:32] == 1); e1_all[q] = np.mean(et == 1)
        known[q] = ok.mean()
        if ok.any(): gap[q] = (ts[fidx] - t[ok].max()) / 86400.0
        if ok[:8].any(): gap8[q] = (ts[fidx] - t[:8][ok[:8]].max()) / 86400.0
    g = gap[np.isfinite(gap)]
    P(f'{sp}: feeds {len(sel)} | history items with a known posting time (mean share) {np.nanmean(known):.3f} | days from the newest posting time among history items to the feed: '
      f'q05 {np.quantile(g, 0.05):.2f} q25 {np.quantile(g, 0.25):.2f} median {np.median(g):.2f} q75 {np.quantile(g, 0.75):.2f} | share < 1 day {np.mean(g < 1):.3f} < 2 days {np.mean(g < 2):.3f} < 0.25 day {np.mean(g < 0.25):.3f}')
    P(f'{sp}: newest 32 events: mean share of e1 {np.nanmean(e1_32):.4f}, of source s1 {np.nanmean(s1_32):.4f} | whole history e1 share {np.nanmean(e1_all):.4f}')
    if sp != 'train':
        # absolute date of the newest posting time among history items, by feed day
        d_feed = np.floor(day(ts[sel])).astype(int); newest = ts[sel] - gap * 86400.0
        for d in np.unique(d_feed):
            m = (d_feed == d) & np.isfinite(gap)
            if m.sum() < 200: continue
            dd = day(newest[m]); P(f'   {sp} feeds of MSK day {d} (Sep {d - 20696}): newest history posting time, days before the feed: q05 {np.quantile(gap[m], 0.05):.2f} median {np.median(gap[m]):.2f}; newest posting date (Sep, MSK) q95 {np.quantile(dd, 0.95) - 20696:.2f} max {dd.max() - 20696:.2f}')
(OUT / 'd07_histfresh.txt').write_text('\n'.join(L) + '\n')
