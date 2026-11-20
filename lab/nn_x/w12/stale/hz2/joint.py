"""hz2: JOINT parity checks beyond the declared gate (label-free, information): (1) per-feed cold share (share of rows with it_hist_all = 0): quantiles and
share of feeds >= 0.45; (2) joint zero pattern of it_hist_all x seq_cv_pop by age bucket: row-weighted total-variation distance to private rows (the
poplag_verify v_joint statistic). Populations: trPool / trHold, private-like feeds, all days and days >= Sep 13; reference: private test feeds with M >= 50."""
from hz2_common import *
L_ = []
def P(s):
    print(s, flush=True); L_.append(s)
ftr = feeds('train'); nF = len(ftr); M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
hold, p = holdout_feeds(ftr)
pool = (p > 0.5) & (M >= 50); pool[hold] = False
part_pub = np.load(OUT / 'train' / 'hz2_part_pub.npy'); d = (day_of(ftr['timestamp'].to_numpy()) - DAY0)
kn = knots()
def rows_of(off, m):
    return np.repeat(off - np.concatenate([[0], np.cumsum(m)[:-1]]), m) + np.arange(int(m.sum()))
fte = feeds('test'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
tsel = np.flatnonzero((tts >= PUB_TS) & (tM >= 50)); tcrow = rows_of(fte['cand_off'].to_numpy().astype(np.int64)[tsel], tM[tsel])
tb = age_bucket(age_hours(np.load(FLAT / 'test/cand_f4.npy')[tcrow], kn))
ta = np.load(FC / 'test/it_hist_all_luo.npy')[tcrow] == 0; tq = np.load(FC / 'test/seq_cv_pop.npy')[tcrow] == 0
tfid = np.repeat(np.arange(len(tsel)), tM[tsel])
tcs = np.bincount(tfid, weights=ta) / tM[tsel]
def pat(a, q, b):
    c = np.bincount(b * 4 + a.astype(np.int64) * 2 + q.astype(np.int64), minlength=28).reshape(7, 4).astype(np.float64)
    return c / np.maximum(c.sum(1, keepdims=True), 1), c.sum(1) / c.sum()
tp, tw = pat(ta, tq, tb)
P(f'reference private feeds M >= 50: {len(tsel)}; per-feed cold share q10 / q50 / q90 = {np.quantile(tcs, .1):.3f} / {np.quantile(tcs, .5):.3f} / {np.quantile(tcs, .9):.3f}; feeds >= 0.45: {float((tcs >= 0.45).mean()):.3f}')
VIEWS = {'as built': (FC / 'train/it_hist_all_luo.npy', FC / 'train/seq_cv_pop.npy'), 'hz': (FNEW / 'train/it_hist_hz_all.npy', FC / 'train/seq_cv_pop.npy'),
         'hz2': (HERE / 'feats/train/it_hist_hz_all.npy', HERE / 'feats/train/seq_cv_hz_pop.npy'), 'hz2L': (HERE / 'feats_L/train/it_hist_hz_all.npy', HERE / 'feats_L/train/seq_cv_hz_pop.npy')}
keep_all = np.load(OUT / 'train/hz2_keep_rows.npy', mmap_mode='r')
for pname, fmask in (('trPool', pool), ('trHold', np.isin(np.arange(nF), hold))):
    for sub, dm in (('all days', np.ones(nF, bool)), ('days >= Sep 13', d >= 7)):
        sel = np.flatnonzero(fmask & ~part_pub & dm)
        crow = rows_of(coff[sel], M[sel]); fid = np.repeat(np.arange(len(sel)), M[sel])
        b = age_bucket(age_hours(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[crow]), kn))
        keep = np.asarray(keep_all[crow])
        P(f'\n{pname}, private-like, {sub}: {len(sel)} feeds, {len(crow)} rows')
        base = None
        for vn, (pa, pq) in VIEWS.items():
            a = np.asarray(np.load(pa, mmap_mode='r')[crow]) == 0; q = np.asarray(np.load(pq, mmap_mode='r')[crow]) == 0
            if vn == 'as built':
                base = (a, q)
            for tag, (aa, qq) in ((vn, (a, q)),) + (((vn + 'k', (np.where(keep, base[0], a), np.where(keep, base[1], q))),) if vn in ('hz2', 'hz2L') and sub == 'all days' else ()):
                cs = np.bincount(fid, weights=aa) / M[sel]
                vp, vw = pat(aa, qq, b)
                tv = float((vw * 0.5 * np.abs(vp - tp).sum(1)).sum())
                P(f'  {tag:9s} per-feed cold share q10 / q50 / q90 = {np.quantile(cs, .1):.3f} / {np.quantile(cs, .5):.3f} / {np.quantile(cs, .9):.3f}; feeds >= 0.45: {float((cs >= 0.45).mean()):.3f}; '
                  f'joint zero pattern (it_hist_all x seq_cv_pop) TV distance to private rows: {tv:.4f}')
open(LOGS / 'joint.txt', 'w').write('\n'.join(L_) + '\n')
