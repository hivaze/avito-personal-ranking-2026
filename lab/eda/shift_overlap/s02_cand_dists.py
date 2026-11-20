"""Candidate & history distributions across splits: f1..f5, category, emb-missing, event type/source; per-day drift."""
from common import *
import json
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
F = {s: feeds(s) for s in SPLITS}
QS = [0.01, 0.05, .1, .25, .5, .75, .9, .95, .99]
BINS = np.linspace(0, 1, 101)
out = {}
cday = {}
for s in SPLITS:
    f = F[s]
    cday[s] = np.repeat(f['date'].cast(pl.Int32).to_numpy(), f['M'].to_numpy())
hist_f = {}
print('==== f1..f5 quantiles / special values')
for c in ['f1', 'f2', 'f3', 'f4', 'f5']:
    for s in SPLITS:
        a = np.asarray(cand(s, c))
        q = np.quantile(a[::7] if s == 'train' else a, QS)  # train: every 7th candidate (35M) for quantiles
        h = np.histogram(a, BINS)[0]
        hist_f[(c, s)] = h
        lo, hi = a.min(), a.max()
        print(f'{c} {s:5s} mean {a.mean():.4f} std {a.std():.4f} q{QS}={np.round(q,3).tolist()} zero%={np.mean(a==0)*100:.2f} at_min%={np.mean(a==lo)*100:.2f} at_max%={np.mean(a==hi)*100:.2f}')
    for a_, b_ in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        print(f'   {c} JS {a_}-{b_} = {js_div(hist_f[(c,a_)], hist_f[(c,b_)]):.5f}  PSI = {psi(hist_f[(c,a_)], hist_f[(c,b_)]):.4f}')
# per-day means (drift)
print('==== per-day mean of f1..f5 (candidate-weighted)')
rows = []
for s in SPLITS:
    d = cday[s]
    ud, inv = np.unique(d, return_inverse=True)
    cnt = np.bincount(inv)
    r = {'split': [s] * len(ud), 'date': ud, 'n': cnt}
    for c in ['f1', 'f2', 'f3', 'f4', 'f5']:
        a = np.asarray(cand(s, c), dtype=np.float64)
        r[c] = np.bincount(inv, weights=a) / cnt
    cat = np.asarray(cand(s, 'category_id'))
    emb = np.asarray(cand(s, 'emb_idx'))
    r['emb_miss%'] = np.bincount(inv, weights=(emb < 0).astype(float)) / cnt * 100
    r['cat0%'] = np.bincount(inv, weights=(cat == 0).astype(float)) / cnt * 100
    if s != 'test':
        t = np.asarray(cand(s, 'target'), dtype=np.float64)
        r['pos_rate%'] = np.bincount(inv, weights=t) / cnt * 100
    rows.append(pl.DataFrame(r))
dd = pl.concat(rows, how='diagonal').with_columns(pl.col('date').cast(pl.Date))
dd = dd.filter(pl.col('n') > 10000)
print(dd)

# category frequencies
print('==== category_id (candidates)')
cat_cnt = {}
for s in SPLITS:
    a = np.asarray(cand(s, 'category_id')).astype(np.int16)
    cat_cnt[s] = np.bincount(a, minlength=64).astype(float)
hcat_cnt = {}
for s in SPLITS:
    a = np.asarray(hist(s, 'category_id')).astype(np.int16)
    hcat_cnt[s] = np.bincount(a, minlength=64).astype(float)
for a_, b_ in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    print(f'JS cand-cat {a_}-{b_} = {js_div(cat_cnt[a_], cat_cnt[b_]):.5f}  PSI {psi(cat_cnt[a_], cat_cnt[b_]):.4f} | JS hist-cat = {js_div(hcat_cnt[a_], hcat_cnt[b_]):.5f}')
print('JS cand-cat vs hist-cat (same split):', {s: round(js_div(cat_cnt[s], hcat_cnt[s]), 4) for s in SPLITS})
sh = {s: cat_cnt[s] / cat_cnt[s].sum() * 100 for s in SPLITS}
order = np.argsort(-sh['train'])
print('cat | train% val% test% | ratio val/train test/train | hist train% val% test%')
hs = {s: hcat_cnt[s] / hcat_cnt[s].sum() * 100 for s in SPLITS}
for k in order[:25]:
    print(f'{k:3d} | {sh["train"][k]:6.2f} {sh["val"][k]:6.2f} {sh["test"][k]:6.2f} | {sh["val"][k]/max(sh["train"][k],1e-9):5.2f} {sh["test"][k]/max(sh["train"][k],1e-9):5.2f} | {hs["train"][k]:6.2f} {hs["val"][k]:6.2f} {hs["test"][k]:6.2f}')
print('categories present: ', {s: int((cat_cnt[s] > 0).sum()) for s in SPLITS}, ' hist:', {s: int((hcat_cnt[s] > 0).sum()) for s in SPLITS})
# biggest relative movers (val/test vs train) among cats with >=0.2% share
mv = [(k, sh['train'][k], sh['val'][k], sh['test'][k]) for k in range(64) if sh['train'][k] >= 0.2 or sh['test'][k] >= 0.2]
mv.sort(key=lambda x: -abs(np.log((x[3] + 1e-3) / (x[1] + 1e-3))))
print('largest log-ratio movers test vs train:', [(k, round(a, 2), round(b, 2), round(c, 2)) for k, a, b, c in mv[:10]])
# target rate by category train vs val
for s in ['train', 'val']:
    a = np.asarray(cand(s, 'category_id')).astype(np.int16); t = np.asarray(cand(s, 'target')).astype(float)
    pr = np.bincount(a, weights=t, minlength=64) / np.maximum(cat_cnt[s], 1)
    out[f'catrate_{s}'] = pr.tolist()
pt, pv = np.array(out['catrate_train']), np.array(out['catrate_val'])
m = (cat_cnt['val'] >= 2000)
print('category positive rate: corr(train, val) over cats with >=2000 val cands = %.3f (n=%d)' % (np.corrcoef(pt[m], pv[m])[0, 1], m.sum()))
print('top cats by val pos rate:', [(int(k), round(pv[k]*100, 2), round(pt[k]*100, 2)) for k in np.argsort(-np.where(m, pv, -1))[:8]])
print('train overall pos rate %.3f%%, val %.3f%%' % (np.asarray(cand('train','target')).mean()*100, np.asarray(cand('val','target')).mean()*100))

# embedding missing
print('==== embedding missing rates')
for s in SPLITS:
    ce = np.asarray(cand(s, 'emb_idx')); he = np.asarray(hist(s, 'emb_idx'))
    ue = np.asarray(np.load(f'{FLAT}/{s}/user_emb.npy', mmap_mode='r'))
    uz = (np.abs(ue).sum(1) == 0)
    print(f'{s}: cand emb missing {np.mean(ce<0)*100:.3f}%  hist emb missing {np.mean(he<0)*100:.3f}%  user_emb all-zero {uz.mean()*100:.3f}%  cand cat==0 {np.mean(np.asarray(cand(s,"category_id"))==0)*100:.3f}%  hist cat==0 {np.mean(np.asarray(hist(s,"category_id"))==0)*100:.3f}%')
    if s != 'test':
        t = np.asarray(cand(s, 'target'))
        print(f'    pos rate | emb missing {t[ce<0].mean()*100:.3f}%  | emb present {t[ce>=0].mean()*100:.3f}%')
    # history item_id==0
    hi = np.asarray(hist(s, 'item_id'))
    print(f'    hist item_id==0 {np.mean(hi==0)*100:.3f}%')

# event types / sources
print('==== history event_type / event_source mix')
et_cnt = {s: np.bincount(np.asarray(hist(s, 'event_type')).astype(np.int16), minlength=11).astype(float) for s in SPLITS}
es_cnt = {s: np.bincount(np.asarray(hist(s, 'event_source')).astype(np.int16), minlength=31).astype(float) for s in SPLITS}
print('event_type share% (train val test):')
for k in range(11):
    print(k, ' '.join(f'{et_cnt[s][k]/et_cnt[s].sum()*100:7.3f}' for s in SPLITS))
print('event_source share% (train val test):')
for k in range(31):
    if sum(es_cnt[s][k] for s in SPLITS) > 0:
        print(k, ' '.join(f'{es_cnt[s][k]/es_cnt[s].sum()*100:7.3f}' for s in SPLITS))
for a_, b_ in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    print(f'JS event_type {a_}-{b_} {js_div(et_cnt[a_], et_cnt[b_]):.5f}   JS event_source {js_div(es_cnt[a_], es_cnt[b_]):.5f}')
# per-feed event-type mix restricted to non-user0 & hpos<64 (recent)
for s in SPLITS:
    hp = np.asarray(hist(s, 'hpos')); et = np.asarray(hist(s, 'event_type')).astype(np.int16)
    m = hp < 32
    c = np.bincount(et[m], minlength=11).astype(float)
    print(f'{s} recent(hpos<32) event_type share%:', np.round(c / c.sum() * 100, 2).tolist())
json.dump({k: v for k, v in out.items()}, open(f'{OUT}/s02.json', 'w'))
