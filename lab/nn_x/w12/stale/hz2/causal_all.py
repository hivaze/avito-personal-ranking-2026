"""hz2 check C1 + C3 (DECLARATION.txt ADDENDUM 5): full-coverage structural causality check and file integrity. Label-free, single thread, chunked.
C1: for ALL train rows of a view: every dense day slot in the row's mask (cache/row_dmask[_L].npy) is < the feed's day slot and is one of the 12 dense train days;
    for ALL train feeds: every sparse day slot (cache/feed_smask.npy) is <= the feed's day slot. The kernels read DM / IM / postings only on the days of dmask and
    SM / sparse postings only on the days of smask & ~dmask (kernels.py), so 0 violations = no counted snapshot is dated after the row's own day, on every row.
C3: column files: length, dtype, NaN / inf / negative values; per-feed lag files.
Usage: python causal_all.py   -> logs/causal_all.txt
"""
from hz2_common import *

out = []


def P(*a):
    s = ' '.join(str(x) for x in a); out.append(s); print(f'[{time.time() - T0:5.0f}s] ' + s, flush=True)


ftr = feeds('train'); nF = len(ftr)
M = ftr['M'].to_numpy().astype(np.int64)
d = (day_of(ftr['timestamp'].to_numpy()) - DAY0).astype(np.int64)
N = int(M.sum())
P(f'train feeds {nF}, rows {N}; feed day slots {int(d.min())}..{int(d.max())}')
# highest set bit of a uint16 / uint32 mask (-1 if empty) through a float log2 on small integers (exact below 2^24)
hb = lambda x: np.where(x > 0, np.floor(np.log2(np.maximum(x, 1).astype(np.float64))).astype(np.int64), -1)
sm = np.load(CACHE / 'feed_smask.npy')
assert len(sm) == nF
hs = hb(sm.astype(np.int64))
P(f'C1 sparse days: feeds with a sparse day slot after the own day: {int((hs > d).sum())} [must be 0]; feeds whose own day is a sparse day: {float(((sm >> d.astype(np.uint32)) & 1).mean()):.4f}; '
  f'feeds with a sparse day at all: {float((sm != 0).mean()):.4f}')
ok = int((hs > d).sum()) == 0
nondense = np.uint16(~np.uint16(DENSE_BITS) & np.uint16(0xFFFF))
for tag in ('', '_L'):
    dm = np.load(CACHE / f'row_dmask{tag}.npy', mmap_mode='r')
    assert len(dm) == N
    bad_after = 0; bad_nondense = 0; gap_hist = np.zeros(16, np.int64); empty = 0
    off = np.concatenate([[0], np.cumsum(M)])
    CH = 200000
    for a in range(0, nF, CH):
        b = min(nF, a + CH)
        x = np.asarray(dm[off[a]:off[b]])
        dr = np.repeat(d[a:b], M[a:b])
        h = hb(x.astype(np.int64))
        bad_after += int((h >= dr).sum())
        bad_nondense += int(((x & nondense) != 0).sum())
        nz = h >= 0
        empty += int((~nz).sum())
        gap_hist += np.bincount(np.minimum(dr[nz] - h[nz], 15), minlength=16)
    name = 'feats (primary)' if tag == '' else 'feats_L (lag-first)'
    P(f'C1 dense days, {name}: rows with a dense day slot >= the own day: {bad_after} [must be 0]; rows with a non-dense slot in the mask: {bad_nondense} [must be 0]; '
      f'rows without any dense day: {empty} ({empty / N:.4f})')
    P('   rows by distance (own day - latest dense day), 1..15+: ' + ' '.join(f'{x / max(gap_hist.sum(), 1):.3f}' for x in gap_hist[1:]) + f' ; distance 0: {int(gap_hist[0])}')
    ok = ok and bad_after == 0 and bad_nondense == 0
P(f'C1 {"PASS" if ok else "FAIL"}')
# C3
dist, ppub = test_lag_dist()
ok3 = True
for vd in ('feats', 'feats_L'):
    L = np.load(HERE / vd / 'train/hz2_lag.npy'); pp = np.load(HERE / vd / 'train/hz2_part_pub.npy'); cs = np.load(HERE / vd / 'train/hz2_cut_slot.npy'); kp = np.load(HERE / vd / 'train/hz2_keep_rows.npy', mmap_mode='r')
    good = len(L) == nF and len(pp) == nF and len(cs) == nF and len(kp) == N and set(np.unique(L).tolist()) <= {6, 7, 8, 9} and np.array_equal(cs.astype(np.int64), d - L.astype(np.int64))
    line = f'C3 {vd}: lag file ok {good}; public-like share {float(pp.mean()):.4f} (test {ppub:.4f}); keep rows {float(np.asarray(kp).mean()):.4f};'
    for nm, sel in (('pub', pp), ('priv', ~pp)):
        u, c = dist[nm]
        sh_t = c / c.sum(); sh_v = np.array([float((L[sel] == x).mean()) for x in u])
        line += f' {nm} lags {u.tolist()} view shares ' + ' / '.join(f'{x:.4f}' for x in sh_v) + ' test ' + ' / '.join(f'{x:.4f}' for x in sh_t) + ';'
        good = good and float(np.abs(sh_v - sh_t).max()) <= 0.005
    good = good and abs(float(pp.mean()) - ppub) <= 0.005
    P(line)
    for n in COLS9:
        x = np.load(HERE / vd / 'train' / f'{HZ2_NAME[n]}.npy', mmap_mode='r')
        g = x.dtype == np.float32 and len(x) == N
        nbad = 0; nneg = 0; mx = 0.0; nzero = 0
        for a in range(0, N, 50_000_000):
            y = np.asarray(x[a:a + 50_000_000])
            nbad += int((~np.isfinite(y)).sum()); nneg += int((y < -1e-4).sum()); mx = max(mx, float(y.max())); nzero += int((y == 0).sum())
        g = g and nbad == 0 and nneg == 0
        good = good and g
        P(f'   {vd}/{HZ2_NAME[n]:17s} rows {len(x)} dtype {x.dtype} non-finite {nbad} below -1e-4 {nneg} max {mx:.4g} zero share {nzero / N:.4f} -> {"ok" if g else "BAD"}')
    ok3 = ok3 and good
P(f'C3 {"PASS" if ok3 else "FAIL"}')
open(LOGS / 'causal_all.txt', 'w').write('\n'.join(out) + '\n')
