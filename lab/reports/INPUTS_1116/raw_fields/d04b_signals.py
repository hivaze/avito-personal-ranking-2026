"""INVENTORY D step 3, part B: the two signals that need a pass over the history arrays (V5 event source of the item across users; V6 history
events without an embedding, content recovered from another split's table). Rule and list: d03_rule.txt. CPU only, read-only."""
import sys, time, json
sys.path.insert(0, '<repo>/reports/INPUTS_1116/raw_fields')
from d03_lib import *
t0 = time.time(); L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
V = load_val(); Hd = load_hold()
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
NI = 111_400_000
fe = {sp: pl.read_parquet(F / sp / 'feeds.parquet') for sp in ('train', 'val', 'test')}
# ================= V5: source mix of the item's e1 events over snapshots of other users
vit = np.asarray(ld('val', 'cand_item_id')); uq = np.unique(vit); lut = np.zeros(NI, bool); lut[uq] = True
n_e1 = np.zeros(len(uq)); n_s1 = np.zeros(len(uq)); vu = np.unique(V.user[V.user > 0]); own = []
for sp in ('train', 'val', 'test'):
    hi = ld(sp, 'hist_item_id'); het = ld(sp, 'hist_event_type'); hes = ld(sp, 'hist_event_source'); hf = ld(sp, 'hist_fidx'); uf = fe[sp]['user_id'].to_numpy()
    for a in range(0, len(hi), 100_000_000):
        it = np.asarray(hi[a:a + 100_000_000]); m = (np.asarray(het[a:a + 100_000_000]) == 1) & lut[it]
        it = it[m]; s1 = (np.asarray(hes[a:a + 100_000_000])[m] == 1); u = uf[np.asarray(hf[a:a + 100_000_000])[m]]
        k = np.searchsorted(uq, it); n_e1 += np.bincount(k, minlength=len(uq)); n_s1 += np.bincount(k, weights=s1, minlength=len(uq))
        mu = np.isin(u, vu)
        own.append(pl.DataFrame({'u': u[mu].astype(np.int64), 'it': it[mu].astype(np.int64), 's1': s1[mu].astype(np.int64)}))
    P(f'V5 pass {sp} done {time.time() - t0:.0f}s')
own = pl.concat(own).group_by(['u', 'it']).agg(pl.len().alias('n'), pl.col('s1').sum().alias('ns1'))
rows = pl.DataFrame({'row': np.arange(len(vit)), 'u': V.user[V.fi].astype(np.int64), 'it': vit.astype(np.int64)}).join(own, on=['u', 'it'], how='inner').filter(pl.col('u') > 0)
k = np.searchsorted(uq, vit); e1 = n_e1[k].copy(); s1c = n_s1[k].copy()
e1[rows['row'].to_numpy()] -= rows['n'].to_numpy(); s1c[rows['row'].to_numpy()] -= rows['ns1'].to_numpy()
assert e1.min() >= 0 and s1c.min() >= 0
P(f'V5: val rows with same-user e1 events removed {rows.height} | rows with n_e1 > 0: {np.mean(e1 > 0):.4f} | mean s1 share where n_e1 > 0: {(s1c[e1 > 0] / e1[e1 > 0]).mean():.4f}')
sh = (s1c + 1) / (e1 + 2); x5a = np.where(e1 > 0, np.log(sh / (1 - sh)), 0.0); x5b = np.log1p(s1c)
np.save(OUT / 'x_V5a_it_s1share_val.npy', x5a.astype(np.float32)); np.save(OUT / 'x_V5b_it_s1cnt_val.npy', x5b.astype(np.float32))
P('--- LR reads V5 ---')
for n, x in (('V5a_it_s1share', x5a), ('V5b_it_s1cnt', x5b)):
    for ctrl in ('TO', 'FIN'): P(fmt(n, 'val', ctrl, lr_test(V, x, ctrl)))
del lut
# ================= V6: history events without an embedding in the own table, content recovered from another split's table
def table(sp):
    d = R / 'data/competition' / sp / 'item_embeddings'; meta = json.loads((d / 'data.meta').read_text())
    return np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
def lut_of(splits):
    l = np.full(NI, -1, np.int32); src = np.zeros(NI, np.int8)
    for q, sp in enumerate(splits):
        for pre in ('cand', 'hist'):
            ii = ld(sp, f'{pre}_item_id'); ee = ld(sp, f'{pre}_emb_idx')
            for a in range(0, len(ii), 100_000_000):
                i_ = np.asarray(ii[a:a + 100_000_000]); e_ = np.asarray(ee[a:a + 100_000_000]); m = e_ >= 0
                l[i_[m]] = e_[m]; src[i_[m]] = q + 1
    return l, src
def unit(a):
    a = a.astype(np.float32); return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-3)
def gather(tab, idx):
    o = np.argsort(idx); out = np.empty((len(idx), 128), np.int8); out[o] = tab[idx[o]]; return out
@nb.njit(parallel=True, cache=True)
def gain(coff, M, hoff, Hn, C, Hv, kind):
    out = np.zeros(len(C)); so = np.zeros(len(C)); sr = np.full(len(C), -2.0)
    for f in nb.prange(len(M)):
        for i in range(coff[f], coff[f] + M[f]):
            mo = 0.0; mr = -2.0
            for j in range(hoff[f], hoff[f] + Hn[f]):
                if kind[j] == 0: continue
                d = 0.0
                for q in range(128): d += C[i, q] * Hv[j, q]
                if kind[j] == 1:
                    if d > mo: mo = d
                elif d > mr: mr = d
            so[i] = mo; sr[i] = mr
            if mr > mo: out[i] = mr - mo
    return out, so, sr
def v6(name, coff, M, hoff, Hn, c_emb, h_emb, h_item, own_tab, rec):
    """rec = list of (lut, table) of the other splits, in priority order."""
    C = unit(gather(own_tab, np.maximum(c_emb, 0))); C[c_emb < 0] = 0
    kind = np.where(h_emb >= 0, 1, 0).astype(np.int8); Hv = np.zeros((len(h_emb), 128), np.float32)
    m = h_emb >= 0; Hv[m] = unit(gather(own_tab, h_emb[m]))
    miss = np.flatnonzero(h_emb < 0); done = np.zeros(len(miss), bool)
    for l, tab in rec:
        e = l[h_item[miss]]; ok = (e >= 0) & ~done
        if ok.any(): Hv[miss[ok]] = unit(gather(tab, e[ok])); kind[miss[ok]] = 2; done |= ok
    x, so, sr = gain(coff, M, hoff, Hn, C, Hv, kind)
    P(f'V6 {name}: history events {len(h_emb)} | without own embedding {np.mean(h_emb < 0):.4f} | recovered {done.sum()} = {done.mean():.4f} of the missing | '
      f'feeds with >= 1 recovered event {np.mean(np.add.reduceat((kind == 2).astype(np.int64), hoff[Hn > 0]) > 0) * np.mean(Hn > 0):.4f} | rows with gain > 0: {np.mean(x > 0):.4f} | mean gain where > 0: {x[x > 0].mean() if (x > 0).any() else 0:.4f}')
    return x
ltr, _ = lut_of(['train']); P(f'lut train built {time.time() - t0:.0f}s')
hoff = fe['val']['hist_off'].to_numpy().astype(np.int64); Hn = fe['val']['H'].to_numpy().astype(np.int64)
x6 = v6('val', V.off, V.M, hoff, Hn, np.asarray(ld('val', 'cand_emb_idx')), np.asarray(ld('val', 'hist_emb_idx')), np.asarray(ld('val', 'hist_item_id')), table('val'), [(ltr, table('train'))])
np.save(OUT / 'x_V6_missemb_gain_val.npy', x6.astype(np.float32)); del ltr
P('--- LR reads V6 val ---')
for ctrl in ('TO', 'FIN'): P(fmt('V6_missemb_gain', 'val', ctrl, lr_test(V, x6, ctrl)))
# holdout rows: train feeds, own table = train, recovery from the val and test tables
lv, _ = lut_of(['val']); lt, _ = lut_of(['test'])
tf = fe['train']; hofft = tf['hist_off'].to_numpy().astype(np.int64)[Hd.hold]; Hh = tf['H'].to_numpy().astype(np.int64)[Hd.hold]
hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(hofft, Hh)]); hoff_l = np.r_[0, np.cumsum(Hh)[:-1]].astype(np.int64)
x6h = v6('hold', Hd.off, Hd.M, hoff_l, Hh, np.asarray(ld('train', 'cand_emb_idx')[Hd.rows]), np.asarray(ld('train', 'hist_emb_idx')[hrows]), np.asarray(ld('train', 'hist_item_id')[hrows]),
         table('train'), [(lv, table('val')), (lt, table('test'))])
np.save(OUT / 'x_V6_missemb_gain_hold.npy', x6h.astype(np.float32))
P('--- LR read V6 hold ---')
P(fmt('V6_missemb_gain', 'hold', 'H', lr_test(Hd, x6h, 'H')))
P(f'done {time.time() - t0:.0f}s')
(OUT / 'd04b_results.txt').write_text('\n'.join(L) + '\n')
