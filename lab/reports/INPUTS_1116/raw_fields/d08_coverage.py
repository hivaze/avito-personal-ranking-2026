"""INVENTORY D: label-free coverage / parity numbers of the candidate inputs on val, public, private and holdout rows. CPU only, read-only."""
import sys
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
BOUND = 1790497920; FE = R / 'data/feats'
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
tf = pl.read_parquet(F / 'train/feeds.parquet'); tu = tf['user_id'].to_numpy(); tts = tf['timestamp'].to_numpy().astype(np.int64)
last = pl.DataFrame({'u': tu, 't': tts}).filter(pl.col('u') > 0).group_by('u').agg(pl.col('t').max().alias('tl'), pl.len().alias('nf'))
hold, _ = holdout_feeds()
def rowsets(sp):
    f = pl.read_parquet(F / sp / 'feeds.parquet'); fi = np.asarray(ld(sp, 'cand_fidx')); ts = f['timestamp'].to_numpy()
    if sp == 'val': return f, {'val': np.ones(len(fi), bool)}, {'val': np.ones(len(f), bool)}
    return f, {'public': (ts < BOUND)[fi], 'private': (ts >= BOUND)[fi]}, {'public': ts < BOUND, 'private': ts >= BOUND}
for sp in ('val', 'test'):
    f, rs, fs = rowsets(sp); fi = np.asarray(ld(sp, 'cand_fidx'))
    uci = np.nan_to_num(np.load(FE / sp / 'tte_uc_imp.npy')); uii = np.nan_to_num(np.load(FE / sp / 'tte_ui_imp.npy')); iti = np.load(FE / sp / 'tte_it_imp.npy')
    f1 = np.asarray(ld(sp, 'cand_f1')); f2 = np.asarray(ld(sp, 'cand_f2'))
    j = f.select('user_id', 'timestamp').join(last, left_on='user_id', right_on='u', how='left')
    gap = (j['timestamp'].to_numpy().astype(np.float64) - j['tl'].to_numpy().astype(np.float64)) / 86400.0; nf = j['nf'].fill_null(0).to_numpy()
    for k in rs:
        m = rs[k]; fm = fs[k]; g = gap[fm & np.isfinite(gap)]
        P(f'{k}: rows {int(m.sum())} feeds {int(fm.sum())} | feeds whose user has a train feed {np.mean(nf[fm] > 0):.4f} (median {np.median(nf[fm][nf[fm] > 0]):.0f} train feeds) | days since that user\'s LAST train feed: q10 {np.quantile(g, 0.1):.1f} median {np.median(g):.1f} q90 {np.quantile(g, 0.9):.1f} | '
          f'rows with user x category train impressions {np.mean(uci[m] > 0):.4f} | same (user, item) shown in train {np.mean(uii[m] > 0):.5f} | item has train impressions {np.mean(iti[m] > 0):.4f} | f1 at floor {np.mean(f1[m] < 0.018):.4f} | f2 == 0 {np.mean(f2[m] == 0):.4f}')
# train rows: the same quantities as they would be built causally (earlier-day feeds), all train and the holdout
fi = ld('train', 'cand_fidx'); sel = np.sort(np.random.default_rng(1).choice(len(tu), 200_000, replace=False)); toff = tf['cand_off'].to_numpy(); tM = tf['M'].to_numpy()
for name, feeds in (('train sample (200K feeds)', sel), ('holdout', hold)):
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(toff[feeds], tM[feeds])])
    uci = np.nan_to_num(np.asarray(np.load(FE / 'train/tte_uc_imp.npy', mmap_mode='r')[rows])); uii = np.nan_to_num(np.asarray(np.load(FE / 'train/tte_ui_imp.npy', mmap_mode='r')[rows])); iti = np.asarray(np.load(FE / 'train/tte_it_imp.npy', mmap_mode='r')[rows])
    f1 = np.asarray(ld('train', 'cand_f1')[rows]); f2 = np.asarray(ld('train', 'cand_f2')[rows])
    # previous feed of the same user on an earlier MSK day
    d = pl.DataFrame({'f': np.arange(len(tu)), 'u': tu, 't': tts, 'day': (tts + 10800) // 86400}).filter(pl.col('u') > 0)
    q = d.filter(pl.col('f').is_in(feeds)).join(d.select('u', pl.col('t').alias('t2'), pl.col('day').alias('day2')), on='u').filter(pl.col('day2') < pl.col('day')).group_by('f').agg(pl.col('t').first(), pl.col('t2').max())
    g = (q['t'].to_numpy() - q['t2'].to_numpy()) / 86400.0
    P(f'{name}: rows {len(rows)} | feeds with an earlier-day feed of the same user {q.height / len(feeds):.4f} | days since that feed: q10 {np.quantile(g, 0.1):.1f} median {np.median(g):.1f} q90 {np.quantile(g, 0.9):.1f} | rows with user x category earlier-day impressions {np.mean(uci > 0):.4f} | same (user, item) shown on an earlier day {np.mean(uii > 0):.5f} | item has earlier-day impressions {np.mean(iti > 0):.4f} | f1 at floor {np.mean(f1 < 0.018):.4f} | f2 == 0 {np.mean(f2 == 0):.4f}')
(OUT / 'd08_coverage.txt').write_text('\n'.join(L) + '\n')
