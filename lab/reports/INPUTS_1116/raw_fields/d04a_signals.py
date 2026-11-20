"""INVENTORY D step 3, part A: build the cheap signal columns (V1, V2, V3, V4, V7, V8) for val and holdout rows and run the LR lens.
Rule and list: d03_rule.txt (written before this script ran). CPU only, read-only; writes x_<name>_<split>.npy + d04a_results.txt here."""
import sys, time
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
t0 = time.time(); L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val(); Hd = load_hold()
P(f'val: feeds {len(V.M)} rows {len(V.y)} | NDCG TO {ndcg_pf(V.score["TO"], V.y, V.off, V.M).mean():.5f} FIN {ndcg_pf(V.score["FIN"], V.y, V.off, V.M).mean():.5f}')
P(f'hold: feeds {len(Hd.M)} rows {len(Hd.y)} | NDCG pool {ndcg_pf(Hd.score["H"], Hd.y, Hd.off, Hd.M).mean():.5f}')
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
FE = R / 'data/feats'
X = {}   # name -> {'val': x, 'hold': x or None}
# ---------------- V1: hour x category / hour x age decile lifts from non-holdout train labels
tf = pl.read_parquet(F / 'train/feeds.parquet'); ts = tf['timestamp'].to_numpy().astype(np.int64)
hb_feed = (((ts + 10800) // 3600) % 24) // 4
inhold = np.zeros(len(ts), bool); inhold[Hd.hold] = True
fi = np.asarray(ld('train', 'cand_fidx')); keep = ~inhold[fi]; hb = hb_feed[fi].astype(np.int64)
y = np.asarray(ld('train', 'cand_target')).astype(np.float64); cat = np.asarray(ld('train', 'cand_category_id')).astype(np.int64)
f4 = np.asarray(ld('train', 'cand_f4')); edges = np.quantile(f4[::50], np.linspace(0, 1, 11)[1:-1]); dec = np.searchsorted(edges, f4).astype(np.int64)
def lift_table(g, ng):
    n = np.bincount((g * 6 + hb)[keep], minlength=ng * 6).reshape(ng, 6).astype(np.float64)
    p = np.bincount((g * 6 + hb)[keep], weights=y[keep], minlength=ng * 6).reshape(ng, 6)
    rg = p.sum(1) / np.maximum(n.sum(1), 1)
    rate = (p + 200 * rg[:, None]) / (n + 200)
    return np.where(rg[:, None] > 0, np.log(np.maximum(rate, 1e-9) / np.maximum(rg[:, None], 1e-9)), 0.0)
Tc = lift_table(cat, 54); Td = lift_table(dec, 10)
P('V1 tables built: hour-bin x category lift sd (weighted by nothing) %.4f, hour-bin x age-decile lift sd %.4f; decile table rows (bins 0-4h..20-24h MSK):' % (Tc[2:].std(), Td.std()))
for r in Td: P('   ' + ' '.join(f'{v:+.3f}' for v in r))
del y, keep
vts = V.feeds['timestamp'].to_numpy().astype(np.int64); vhb = ((((vts + 10800) // 3600) % 24) // 4)[V.fi]
vcat = np.asarray(ld('val', 'cand_category_id')).astype(np.int64); vf4 = np.asarray(ld('val', 'cand_f4'))
hhb = hb_feed[Hd.hold][Hd.fi]; hcat = cat[Hd.rows]; hf4 = f4[Hd.rows]
X['V1a_hourcat'] = dict(val=Tc[vcat, vhb], hold=Tc[hcat, hhb])
X['V1b_hourage'] = dict(val=Td[np.searchsorted(edges, vf4), vhb], hold=Td[dec[Hd.rows], hhb])
P(f'V1 val hour-bin shares {np.bincount(vhb, minlength=6) / len(vhb)} | hold {np.bincount(hhb, minlength=6) / len(hhb)}')
del hb, cat, dec, fi
# ---------------- V2: same user's earlier train feeds
def tte(sp, n, rows=None):
    x = np.load(FE / sp / f'{n}.npy', mmap_mode='r'); return np.nan_to_num(np.asarray(x if rows is None else x[rows], np.float64), nan=0.0)
def v2(sp, rows):
    ui = tte(sp, 'tte_ui_imp', rows); uci = tte(sp, 'tte_uc_imp', rows); ucc = tte(sp, 'tte_uc_ctr', rows); cc = tte(sp, 'tte_c_ctr', rows)
    ucp = np.maximum(ucc * (uci + 20) - 0.0159 * 20, 0.0)
    return (ui > 0).astype(np.float64), np.where(uci > 0, np.log((ucp + 20 * cc) / (uci + 20) / cc), 0.0), ui, uci
a, b, ui, uci = v2('val', None); ah, bh, uih, ucih = v2('train', Hd.rows)
X['V2a_ui_reshown'] = dict(val=a, hold=ah); X['V2b_uc_ctr'] = dict(val=b, hold=bh)
P(f'V2 coverage: ui_reshown val {a.mean():.5f} hold {ah.mean():.5f} | uc history (uc_imp>0) val {np.mean(uci > 0):.4f} hold {np.mean(ucih > 0):.4f}')
# ---------------- V3 / V4: item-level train target / pos statistics (val rows read all of train)
coec = tte('val', 'tte_it_coec'); imp = tte('val', 'tte_it_imp')
X['V3_it_coec'] = dict(val=np.log(np.maximum(coec, 1e-6)), hold=None)
z = np.load(FE / 'tte_tot_it.npz'); keys = z['keys']; vit = np.asarray(ld('val', 'cand_item_id')).astype(np.int64)
i = np.minimum(np.searchsorted(keys, vit), len(keys) - 1); hit = keys[i] == vit
ex = np.where(hit, z['ex'][i], 0.0); im = np.where(hit, z['imp'][i], 0.0); assert np.allclose(im, imp)
X['V4_it_prodpos'] = dict(val=np.log((ex + 20 * 0.0159) / (im + 20)), hold=None)
P(f'V3/V4 coverage: val rows whose item has train impressions {np.mean(im > 0):.4f}; median impressions when > 0: {np.median(im[im > 0]):.0f}')
# ---------------- V7: fresh co-exposure inside val
u_, inv, cnt = np.unique(vit, return_inverse=True, return_counts=True)
X['V7_fresh_cnt'] = dict(val=np.log(cnt[inv].astype(np.float64)), hold=None)
P(f'V7 coverage: val rows whose item is in >= 2 val feeds {np.mean(cnt[inv] >= 2):.4f}')
# ---------------- V8: equality structure of f2 / f1
@nb.njit(parallel=True, cache=True)
def eqfeat(off, M, f2, f1):
    home = np.zeros(len(f2)); tie = np.zeros(len(f2))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        a = np.sort(f2[o:o + m]); best = -1.0; bc = 0; i = 0
        while i < m:
            j = i
            while j + 1 < m and a[j + 1] == a[i]: j += 1
            if a[i] != 0.0 and j - i + 1 > bc: bc = j - i + 1; best = a[i]
            i = j + 1
        for k in range(m):
            if bc > 0 and f2[o + k] == best: home[o + k] = 1.0
            c = 0
            for q in range(m):
                if f1[o + q] == f1[o + k]: c += 1
            tie[o + k] = np.log(c)
    return home, tie
hv, tv = eqfeat(V.off, V.M, np.asarray(ld('val', 'cand_f2')), np.asarray(ld('val', 'cand_f1')))
hh, th_ = eqfeat(Hd.off, Hd.M, np.asarray(ld('train', 'cand_f2')[Hd.rows]), np.asarray(ld('train', 'cand_f1')[Hd.rows]))
X['V8a_f2_home'] = dict(val=hv, hold=hh); X['V8b_f1_tie'] = dict(val=tv, hold=th_)
P(f'V8 coverage: f2_home share val {hv.mean():.4f} hold {hh.mean():.4f} | f1 tie (size>=2) val {np.mean(tv > 0):.4f} hold {np.mean(th_ > 0):.4f}')
P(f'signals built in {time.time() - t0:.0f}s')
for n, d in X.items():
    for sp, x in d.items():
        if x is not None: np.save(OUT / f'x_{n}_{sp}.npy', x.astype(np.float32))
# ---------------- LR reads
P('--- LR reads (rule: LR >= 10.83 and placebo < 3.84) ---')
for n, d in X.items():
    for ctrl in ('TO', 'FIN'):
        P(fmt(n, 'val', ctrl, lr_test(V, d['val'], ctrl)))
    if d.get('hold') is not None:
        P(fmt(n, 'hold', 'H', lr_test(Hd, d['hold'], 'H')))
P(f'control fits: ' + ' | '.join(f'{k}: theta {np.round(v[0], 4).tolist()} ll {v[1]:.1f}' for k, v in list(V._base.items()) + list(Hd._base.items())))
P(f'done {time.time() - t0:.0f}s')
(OUT / 'd04a_results.txt').write_text('\n'.join(L) + '\n')
