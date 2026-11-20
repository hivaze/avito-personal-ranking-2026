"""hz2 ADDENDUM 7 (label-free, information): the trajectory-encoder inputs pre2_fe* are outside the hz2 view. How far are train rows from test rows there, by listing age?
Usage: python pre2_gap.py -> logs/pre2_gap.txt"""
from hz2_common import *
CF = ROOT / 'nn_x/r6/stale_cf'
kn = knots()
out = []


def P(*a):
    s = ' '.join(str(x) for x in a); out.append(s); print(s, flush=True)


def rows_of(off, m):
    return np.repeat(off - np.concatenate([[0], np.cumsum(m)[:-1]]), m) + np.arange(int(m.sum()))


def stats(name, b, fe, fez, newz, idc):
    has = ~np.isnan(fe)
    fz = np.clip(np.nan_to_num(fez, nan=0.0), -5, 5); nz = np.clip(np.nan_to_num(newz, nan=0.0), -5, 5)
    line1 = f'{name:34s} rows {len(b):8d} | no encoder score (has:pre2_fe = 0): all {1 - has.mean():.3f} | by age ' + ' '.join(f'{1 - has[b == k].mean():.3f}' for k in range(7))
    P(line1)
    if idc is not None:
        z0 = np.nan_to_num(idc, nan=0.0) == 0
        P(f'{"":34s}               | item-ID contribution exactly 0:       all {z0.mean():.3f} | by age ' + ' '.join(f'{z0[b == k].mean():.3f}' for k in range(7)))
    d = np.abs(fz - nz)
    P(f'{"":34s}               | mean |pre2_fe_z - pre2_new_z|:        all {d.mean():.3f} | by age ' + ' '.join(f'{d[b == k].mean():.3f}' for k in range(7)))


P('age buckets: ' + ' '.join(AGE_NAMES))
# train: trPool private-like feeds of days >= Sep 13
ftr = feeds('train'); hold, p = holdout_feeds(ftr)
Mf = ftr['M'].to_numpy().astype(np.int64)
pool = (p > 0.5) & (Mf >= 50); pool[hold] = False
part_pub, L, _, _ = assign_lags(ftr['feed_id'].to_numpy())
d = (day_of(ftr['timestamp'].to_numpy()) - DAY0).astype(np.int64)
for nm, selm in (('train trPool priv-like, >= Sep 13', pool & ~part_pub & (d >= 7)), ('train trPool priv-like, Sep 17-19', pool & ~part_pub & (d >= 11)),
                 ('train trPool priv-like, Sep 9-12', pool & ~part_pub & (d >= 3) & (d <= 6)), ('train trPool priv-like, Sep 6-8', pool & ~part_pub & (d <= 2))):
    sel = np.flatnonzero(selm)
    crow = rows_of(ftr['cand_off'].to_numpy().astype(np.int64)[sel], Mf[sel])
    g = lambda n: np.asarray(np.load(FC / 'train' / f'{n}.npy', mmap_mode='r')[crow])
    b = age_bucket(age_hours(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[crow]), kn))
    stats(nm, b, g('pre2_fe'), g('pre2_fe_z'), g('pre2_new_z'), g('pre2_id_contrib'))
# val
fv = feeds('val'); Mv = fv['M'].to_numpy().astype(np.int64); sel = np.flatnonzero(Mv >= 50)
vrow = rows_of(fv['cand_off'].to_numpy().astype(np.int64)[sel], Mv[sel])
vb = age_bucket(age_hours(np.load(FLAT / 'val/cand_f4.npy')[vrow], kn))
g = lambda n: np.load(FC / 'val' / f'{n}.npy')[vrow]
stats('val M >= 50, stored', vb, g('pre2_fe'), g('pre2_fe_z'), g('pre2_new_z'), g('pre2_id_contrib'))
try:
    c = lambda n: np.load(CF / f'pre2both_{n}.npy')[vrow]
    stats('val M >= 50, stale-val override', vb, c('pre2_fe'), c('pre2_fe_z'), g('pre2_new_z'), None)
    ch = (np.nan_to_num(c('pre2_fe_z'), nan=0.0) != np.nan_to_num(g('pre2_fe_z'), nan=0.0))
    P(f'{"":34s}               | rows whose pre2_fe_z the stale-val override changes: all {ch.mean():.3f} | by age ' + ' '.join(f'{ch[vb == k].mean():.3f}' for k in range(7)))
except FileNotFoundError as e:
    P(f'stale-val pre2 override not found: {e}')
# test
fte = feeds('test'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
for part in ('priv', 'pub'):
    ts_ = np.flatnonzero(((tts < PUB_TS) if part == 'pub' else (tts >= PUB_TS)) & (tM >= 50))
    trow = rows_of(fte['cand_off'].to_numpy().astype(np.int64)[ts_], tM[ts_])
    tb = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy')[trow], kn))
    g = lambda n: np.load(FC / 'test' / f'{n}.npy')[trow]
    stats(f'test {part} M >= 50', tb, g('pre2_fe'), g('pre2_fe_z'), g('pre2_new_z'), g('pre2_id_contrib'))
open(LOGS / 'pre2_gap.txt', 'w').write('\n'.join(out) + '\n')
