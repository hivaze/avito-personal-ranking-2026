"""Per-feed NDCG distribution and segment tables for G3 (val), with lever-response deltas and test (public/private) shares."""
from lib import *

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
lk = g('leak')
keep = ~lk  # leak-free feeds, as in the older err_* reports
S = dict(np.load(OUT / 'out/cand_scores.npz'))
cold_luo = np.load(OUT / 'out/cold_luo.npy')

# ---------------- 1. per-feed NDCG distribution ----------------
nd = g('ndcg_G3'); ndr = g('ndcg_random'); nd1 = g('ndcg_N1'); nd2 = g('ndcg_nf2_oof')
print('=== 1. Per-feed NDCG distribution (G3, all 14,423 val feeds; leak-free in brackets) ===')
print(f'mean {nd.mean():.5f} [{nd[keep].mean():.5f}]  median {np.median(nd):.4f}  sd {nd.std():.4f}  random-order mean {ndr.mean():.4f}')
qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print('quantiles', ' '.join(f'q{int(q*100)}={np.quantile(nd, q):.3f}' for q in qs))
bp = g('bprank_G3')
print(f'P(best positive at rank 0 = NDCG 1 when npos=1): {np.mean(bp == 0):.4f}; nf2_oof {np.mean(g("bprank_nf2_oof") == 0):.4f}')
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.999, 1.001]
h, _ = np.histogram(nd, bins); h2, _ = np.histogram(nd2, bins)
print('NDCG histogram (G3 / nf2_oof feed shares):')
for i in range(len(bins) - 1):
    print(f'  [{bins[i]:.2f},{bins[i+1]:.2f}) {h[i]/n:.4f} / {h2[i]/n:.4f}')
loss = 1 - nd
# loss concentration
srt = np.sort(loss)[::-1]; cs = np.cumsum(srt) / srt.sum()
print('loss concentration: worst 10% of feeds hold {:.3f} of loss, worst 25% {:.3f}, worst 50% {:.3f}'.format(cs[int(0.1*n)-1], cs[int(0.25*n)-1], cs[int(0.5*n)-1]))
# best-positive rank buckets: share of feeds and of loss
rel = bp / np.maximum(M - 1, 1)
bb = np.select([bp == 0, bp <= 2, bp <= 9, bp <= 29, rel <= 0.5], ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..M/2'], 'r>M/2')
print('best-positive rank bucket: feed share | loss share | mean NDCG | (nf2_oof feed share)')
bb2 = np.select([g('bprank_nf2_oof') == 0, g('bprank_nf2_oof') <= 2, g('bprank_nf2_oof') <= 9, g('bprank_nf2_oof') <= 29,
                 g('bprank_nf2_oof') / np.maximum(M - 1, 1) <= 0.5], ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..M/2'], 'r>M/2')
for b in ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..M/2', 'r>M/2']:
    m = bb == b
    print(f'  {b:9s} {m.mean():.4f} | {loss[m].sum()/loss.sum():.4f} | {nd[m].mean():.4f} | ({np.mean(bb2 == b):.4f})')
# lift over random, per M bin
print(f'mean lift over random: G3 {np.mean(nd - ndr):+.4f}, nf2_oof {np.mean(nd2 - ndr):+.4f}')

# ---------------- 2. segment tables ----------------
cell_M = np.digitize(M, np.quantile(M, np.linspace(0, 1, 21)[1:-1]))
npos = g('npos'); cell = cell_M * 4 + np.minimum(npos, 3)
cm = np.bincount(cell[keep], nd[keep], cell.max() + 1) / np.maximum(np.bincount(cell[keep], minlength=cell.max() + 1), 1)
madj = nd - cm[cell]

# test-side observable variables
ft = pl.read_parquet(FLAT / 'test/feeds.parquet')
TS = pl.read_parquet(ROOT / 'eda/errors/segments/test_segments.parquet')
Mt = TS['M'].to_numpy(); fidt = np.repeat(np.arange(len(Mt)), Mt)
cold_t = np.nan_to_num(np.load(FC / 'test/it_hist_all_luo.npy').astype(np.float64), nan=0.0) == 0
TT = {c: TS[c].to_numpy() for c in TS.columns}
TT['cold_share_luo'] = np.bincount(fidt, cold_t, len(Mt)) / Mt
pub = ft['timestamp'].to_numpy() < 1790497920
print(f'\ntest feeds {len(Mt)}: public {pub.sum()} private {(~pub).sum()}')

# label-side per-feed variables
hfid = np.load(FLAT / 'val/hist_fidx.npy').astype(np.int64); hcat = np.load(FLAT / 'val/hist_category_id.npy').astype(np.int64)
hpos = np.load(FLAT / 'val/hist_hpos.npy').astype(np.int64)
cat = V.cat
# min hpos of each (feed, category) in history
hk = hfid * 64 + hcat
mh = {}
uk, inv = np.unique(hk, return_inverse=True)
minh = np.full(len(uk), 10 ** 6); np.minimum.at(minh, inv, hpos)
ck = fid * 64 + cat
idx = np.searchsorted(uk, ck); idx = np.minimum(idx, len(uk) - 1)
cand_cat_minh = np.where(uk[idx] == ck, minh[idx], 10 ** 6)  # 10**6 = category never in history
np.save(OUT / 'out/cand_cat_minh.npy', cand_cat_minh)
# per feed: most recent positive category novelty
pm = np.full(n, 10 ** 6); np.minimum.at(pm, fid[y > 0], cand_cat_minh[y > 0])
nov = np.select([pm < 8, pm < 32, pm < 10 ** 6], ['last8', 'h8-31', 'h>=32'], 'never')
# positive popularity (LUO history count), max over positives
luo = np.nan_to_num(np.load(FC / 'val/it_hist_all_luo.npy').astype(np.float64), nan=0.0)
ppop = np.zeros(n); np.maximum.at(ppop, fid[y > 0], luo[y > 0])
ctr_n = np.nan_to_num(np.load(FC / 'val/it_cand_cnt_train.npy').astype(np.float64), nan=0.0)
pimp = np.zeros(n); np.maximum.at(pimp, fid[y > 0], ctr_n[y > 0])
age = np.load(ROOT / 'eda/errors/synthesis/derived/val/ag_age.npy').astype(np.float64)
print('ag_age units: quantiles', np.quantile(age, [0, .1, .5, .9, 1]))
page = np.full(n, np.inf); np.minimum.at(page, fid[y > 0], age[y > 0])  # youngest positive
# G3 category ranking (LSE) - is the top category a positive category?
z = S['G3'].astype(np.float64)
ukc, invc = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(invc, np.exp(z)))
cfeed = ukc // 64
o = np.lexsort((-lse, cfeed)); first = np.r_[True, cfeed[o][1:] != cfeed[o][:-1]]
topcat = np.full(n, -1); topcat[cfeed[o][first]] = (ukc % 64)[o][first]
poscat_set = np.unique(ck[y > 0])
top_is_pos = np.isin(fid[:0], []) if False else np.isin(np.arange(n) * 64 + topcat, poscat_set)
# G3 category rank of the best positive category (LSE)
crank = np.empty(len(ukc), np.int64); crank[o] = np.arange(len(ukc)) - np.searchsorted(cfeed[o], cfeed[o])
kpos = np.isin(ukc, poscat_set)
pcr = np.full(n, 999); np.minimum.at(pcr, cfeed[kpos], crank[kpos])

SEG = {
    'M': (np.select([M <= 10, M <= 25, M <= 50, M <= 100, M <= 200], ['005-10', '011-25', '026-50', '051-100', '101-200'], '201-300'), True),
    'H': (np.select([g('H') == 0, g('H') < 20, g('H') < 64, g('H') < 192, g('H') < 256], ['0', '1-19', '20-63', '64-191', '192-255'], '256'), True),
    'user': (np.select([g('uid0'), g('n_train_feeds') == 0, g('n_train_feeds') == 1, g('n_train_feeds') <= 4], ['uid0', 'tr0', 'tr1', 'tr2-4'], 'tr5+'), True),
    'cold_share_luo': (np.select([g('cold_share_luo') < .15, g('cold_share_luo') < .3, g('cold_share_luo') < .45], ['<15%', '15-30%', '30-45%'], '>=45%'), True),
    'hour': (np.select([g('hour') < 6, g('hour') < 10, g('hour') < 14, g('hour') < 18], ['h00-05', 'h06-09', 'h10-13', 'h14-17'], 'h18-23'), True),
    'day': (np.where(g('tue'), 'Tue', 'Thu'), False),
    'n_cat': (np.select([g('n_cat') <= 4, g('n_cat') <= 8, g('n_cat') <= 12, g('n_cat') <= 18], ['01-4', '05-8', '09-12', '13-18'], '19+'), True),
    'feed_dom (max_cat_share)': (np.select([g('max_cat_share') < .15, g('max_cat_share') < .3, g('max_cat_share') < .5], ['<15%', '15-30%', '30-50%'], '>=50%'), True),
    'hist_dom (hist_top_cat_share)': (np.select([g('H') == 0, g('hist_top_cat_share') < .25, g('hist_top_cat_share') < .5, g('hist_top_cat_share') < .75], ['H0', '<25%', '25-50%', '50-75%'], '>=75%'), True),
    'npos [L]': (np.select([npos == 1, npos == 2, npos <= 5], ['1', '2', '3-5'], '6+'), False),
    'pos coldness luo [L]': (np.select([g('coldpos'), g('warmpos')], ['all-cold', 'all-warm'], 'mixed'), False),
    'pos f3 cat rank [L]': (np.select([g('pos_f3_catrank') == 0, g('pos_f3_catrank') == 1, g('pos_f3_catrank') == 2, g('pos_f3_catrank') <= 5], ['#1', '#2', '#3', '#4-6'], '#7+'), False),
    'pos cat G3-LSE rank [L]': (np.select([pcr == 0, pcr == 1, pcr == 2, pcr <= 5], ['#1', '#2', '#3', '#4-6'], '#7+'), False),
    'pos cat novelty [L]': (nov, False),
    'pos popularity luo (max) [L]': (np.select([ppop == 0, ppop <= 10, ppop <= 100, ppop <= 1000], ['0', '1-10', '11-100', '101-1000'], '>1000'), False),
    'pos train impressions (max) [L]': (np.select([pimp == 0, pimp <= 10, pimp <= 100, pimp <= 1000], ['0', '1-10', '11-100', '101-1000'], '>1000'), False),
    'youngest pos age [L]': (np.select([page < 1, page < 3, page < 7, page < 14], ['<1d', '1-3d', '3-7d', '7-14d'], '>=14d(cap)'), False),
    'best-pos rank G3 [O]': (bb, False),
}

# lever deltas (per-feed paired)
D = {
    'G3-nf2oof': g('ndcg_G3') - g('ndcg_nf2_oof'),
    'G3-Cd': g('ndcg_G3') - g('ndcg_Cd'),
    'FThalf G3-G3to': g('ndcg_G3') - g('ndcg_G3_to'),
    'm02 F7x_to-F7b_to': g('ndcg_F7x_to') - g('ndcg_F7b_to'),
    'GRU G3-M2': g('ndcg_G3') - g('ndcg_M2'),
    'NEW N1-G3': g('ndcg_N1') - g('ndcg_G3'),
    'a4u_to-F7s12_to': g('ndcg_a4u_to') - g('ndcg_F7_s12_to'),
}
rows = []
L = loss[keep].sum()
for name, (lab, observable) in SEG.items():
    print(f'\n--- {name} ---')
    hdr = f'{"bucket":12s} {"val%":>6s} {"pub%":>6s} {"priv%":>6s} {"G3":>7s} {"rand":>6s} {"loss%":>6s} {"lift":>5s} {"Madj":>16s} ' + ' '.join(f'{k:>18s}' for k in D)
    print(hdr)
    tl = None
    if observable:
        Mt_, Ht = TT['M'], TT['H']
        if name == 'M': tl = np.select([Mt_ <= 10, Mt_ <= 25, Mt_ <= 50, Mt_ <= 100, Mt_ <= 200], ['005-10', '011-25', '026-50', '051-100', '101-200'], '201-300')
        if name == 'H': tl = np.select([Ht == 0, Ht < 20, Ht < 64, Ht < 192, Ht < 256], ['0', '1-19', '20-63', '64-191', '192-255'], '256')
        if name == 'user': tl = np.select([TT['uid0'], TT['n_train_feeds'] == 0, TT['n_train_feeds'] == 1, TT['n_train_feeds'] <= 4], ['uid0', 'tr0', 'tr1', 'tr2-4'], 'tr5+')
        if name == 'cold_share_luo': c = TT['cold_share_luo']; tl = np.select([c < .15, c < .3, c < .45], ['<15%', '15-30%', '30-45%'], '>=45%')
        if name == 'hour': hh = TT['hour']; tl = np.select([hh < 6, hh < 10, hh < 14, hh < 18], ['h00-05', 'h06-09', 'h10-13', 'h14-17'], 'h18-23')
        if name == 'n_cat': c = TT['n_cat']; tl = np.select([c <= 4, c <= 8, c <= 12, c <= 18], ['01-4', '05-8', '09-12', '13-18'], '19+')
        if name == 'feed_dom (max_cat_share)': c = TT['max_cat_share']; tl = np.select([c < .15, c < .3, c < .5], ['<15%', '15-30%', '30-50%'], '>=50%')
        if name == 'hist_dom (hist_top_cat_share)': c = TT['hist_top_cat_share']; tl = np.select([Ht == 0, c < .25, c < .5, c < .75], ['H0', '<25%', '25-50%', '50-75%'], '>=75%')
    for b in sorted(np.unique(lab)):
        m = (lab == b) & keep
        if m.sum() < 30: continue
        sh = m.sum() / keep.sum()
        ps = np.mean(tl[pub] == b) if tl is not None else np.nan
        pv = np.mean(tl[~pub] == b) if tl is not None else np.nan
        r = dict(seg=name, bucket=b, n=int(m.sum()), val_share=sh, pub_share=ps, priv_share=pv, G3=nd[m].mean(), rand=ndr[m].mean(),
                 loss_share=loss[m].sum() / L, madj=madj[m].mean(), madj_se=se(madj[m]))
        for k, d in D.items():
            r[k] = d[m].mean(); r[k + '_se'] = se(d[m])
        rows.append(r)
        print(f'{b:12s} {100*sh:6.1f} {100*ps:6.1f} {100*pv:6.1f} {r["G3"]:7.4f} {r["rand"]:6.3f} {100*r["loss_share"]:6.1f} {r["loss_share"]/sh:5.2f} '
              f'{r["madj"]:+.4f}({r["madj_se"]:.4f}) ' + ' '.join(f'{r[k]:+.4f}({r[k+"_se"]:.4f})' for k in D))
pl.DataFrame(rows).write_csv(OUT / 'segments_G3.csv')
print('\noverall deltas (leak-free):', '  '.join(f'{k} {fmt_d(d[keep])}' for k, d in D.items()))
print('overall deltas (all feeds):', '  '.join(f'{k} {fmt_d(d)}' for k, d in D.items()))

# save label-side per-feed vars for later scripts
np.savez(OUT / 'out/feed_label_vars.npz', nov=nov, ppop=ppop, pimp=pimp, page=page, pcr=pcr, top_is_pos=top_is_pos, bb=bb, madj=madj)
