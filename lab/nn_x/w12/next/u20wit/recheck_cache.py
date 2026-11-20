"""u20wit resume check: re-verify a SAMPLE of the caches written by the earlier (cut) run before they are trusted.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/recheck_cache.py
(1) feed lists: cache/sets.npz and cache/sub_meta.npz recomputed from the feeds parquet + the two posteriors (label-free), exact match required.
(2) raw arrays of cache/sub/ against the source memmaps on 500 random feeds (100 per set): candidates, history (first min(H, 256) events), user emb, item table rows, xt.
(3) scores: 100 feeds per set re-scored now with two checkpoints (d128 xf0_s951, d192 kd05w192_s951) against cache/sc (max abs logit difference, within-feed corr).
Labels are read only for the equality check of the cached label array (no metric is computed here)."""
from wit_common import *
import torch
torch.set_num_threads(12)
rng = np.random.default_rng(777)
# (1) feed lists recomputed without the cache
tm = train_meta(); M, xf, pv2 = tm['M'], tm['xf'], tm['pv2']
isH = np.zeros(tm['n'], bool); isH[tm['hold']] = True; ex = (pv2 > 0.9) & ~isH
S = dict(W0S=np.where(ex & (xf == 0) & (M < 50))[0], W0L=np.where(ex & (xf == 0) & (M >= 50))[0], H0=tm['hold'][xf[tm['hold']] == 0], W1S=np.where(ex & (xf == 1) & (M < 50))[0])
w1l = np.where(ex & (xf == 1) & (M >= 50))[0]; S['W1L'] = np.sort(np.random.default_rng(20261115).choice(w1l, 6000, replace=False))
z = np.load(CACHE / 'sets.npz')
for k in SETS: assert (z[k] == S[k]).all(), k
sm = np.load(CACHE / 'sub_meta.npz'); allf = np.unique(np.concatenate([S[k] for k in SETS]))
assert (sm['src_fidx'] == allf).all() and (sm['M'] == M[allf]).all() and (sm['xf'] == xf[allf]).all() and (sm['uid'] == tm['uid'][allf]).all()
assert np.array_equal(sm['pold'], tm['pold'][allf]) and np.array_equal(sm['pv2'], pv2[allf])
for j, k in enumerate(SETS): assert (allf[sm['set_id'] == j] == S[k]).all()
hm = np.load(W5P / 'holdout/holdout_meta.npy'); assert (hm[0] == M[tm['hold']]).all(), 'holdout list differs from the stored holdout meta'
print('(1) feed lists: sets.npz and sub_meta.npz reproduce exactly; holdout list == stored holdout_meta M; sizes ' + ', '.join(f'{k} {len(S[k])}' for k in SETS), flush=True)
# (2) raw arrays, sample of feeds
sub = CACHE / 'sub'; ld = lambda n: np.load(sub / f'{n}.npy', mmap_mode='r')
f = tm['f']; co = tm['cand_off']; ho = f['hist_off'].to_numpy().astype(np.int64); H = f['H'].to_numpy().astype(np.int64)
sM = np.asarray(ld('M')); sco = np.asarray(ld('cand_off')); sho = np.asarray(ld('hist_off')); sH = np.asarray(ld('H')); assert (sH == H[allf]).all() and (sM == M[allf]).all()
samp = np.sort(np.concatenate([rng.choice(np.where(sm['set_id'] == j)[0], 100, replace=False) for j in range(5)]))
mm = lambda name: np.load(FLAT / 'train' / f'{name}.npy', mmap_mode='r')
meta = json.loads((COMP / 'train/item_embeddings/data.meta').read_text()); tab = np.memmap(COMP / 'train/item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
xt_names = json.loads((sub / 'xt_names.json').read_text()); bad = 0; nrow = 0; nh = 0
src = {n: mm(n) for n in ('cand_category_id', 'cand_f1', 'cand_f2', 'cand_f3', 'cand_f4', 'cand_f5', 'cand_target', 'cand_pos', 'cand_emb_idx', 'hist_category_id', 'hist_event_type', 'hist_event_source', 'hist_emb_idx', 'user_emb')}
cch = {n: ld(n) for n in list(src) + ['table', 'xt', 'rows']}
xsrc = {n: np.load(FEAT / 'train' / f'{n[4:] if n.startswith("has:") else n}.npy', mmap_mode='r') for n in xt_names}
for i in samp:
    g = allf[i]; a0, m_ = co[g], M[g]; b0 = sco[i]; hk = min(H[g], 256); nrow += m_; nh += hk
    assert (np.asarray(cch['rows'][b0:b0 + m_]) == np.arange(a0, a0 + m_)).all()
    for n in ('cand_category_id', 'cand_f1', 'cand_f2', 'cand_f3', 'cand_f4', 'cand_f5', 'cand_target', 'cand_pos'):
        bad += int(not np.array_equal(np.asarray(src[n][a0:a0 + m_]), np.asarray(cch[n][b0:b0 + m_]), equal_nan=True))
    for n in ('hist_category_id', 'hist_event_type', 'hist_event_source'):
        bad += int(not np.array_equal(np.asarray(src[n][ho[g]:ho[g] + hk]), np.asarray(cch[n][sho[i]:sho[i] + hk])))
    bad += int(not np.array_equal(np.asarray(src['user_emb'][g]), np.asarray(cch['user_emb'][i])))
    for n, k0, k1, c0 in (('cand_emb_idx', a0, a0 + m_, b0), ('hist_emb_idx', ho[g], ho[g] + hk, sho[i])):
        so = np.asarray(src[n][k0:k1]).astype(np.int64); sc_ = np.asarray(cch[n][c0:c0 + (k1 - k0)]).astype(np.int64)
        bad += int(not ((so < 0) == (sc_ < 0)).all())
        ok = so >= 0
        if ok.any(): bad += int(not np.array_equal(np.asarray(tab[np.sort(so[ok])]), np.asarray(cch['table'][np.sort(sc_[ok])])))  # searchsorted remap is order-preserving
    for c, n in enumerate(xt_names):
        x = np.asarray(xsrc[n][a0:a0 + m_])
        if n.startswith('has:'): x = (~np.isnan(x)).astype(np.float32)
        else:
            x = x.astype(np.float32)
            if n.endswith('_rk'): pass
            elif n.endswith('_z'): x = np.clip(x, -5, 5)
            elif n.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
            else: x = np.clip(x, -10, 10)
        bad += int(not np.array_equal(np.nan_to_num(x, nan=0.0).astype(np.float16), np.asarray(cch['xt'][b0:b0 + m_, c])))
print(f'(2) raw arrays: {len(samp)} feeds ({nrow} candidate rows, {nh} history events, {len(xt_names)} xt columns): mismatching arrays {bad}', flush=True)
assert bad == 0
# (3) re-score the sample and compare with the cached scores
data = load_subsplit(CACHE / 'sub'); p = torch.from_numpy(sm['pold'].astype(np.float32)); dom_tr = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
fid_row = np.repeat(np.arange(data.n), data.M_np); r = np.isin(fid_row, samp)
for v in ('w5_e10_xf0_s951', 'w5_e10_kd05w192_s951'):
    path = CKPT / f'{v}.pt'; st = path.stat(); model, ck = load_model(path); t0 = time.time()
    new = score(model, data, samp, dom=dom_tr); old = np.load(CACHE / 'sc' / f'{v}__tr.npy')
    rr = r & ~np.isnan(old)  # a cache file holds only the sets that were scored (kd05w192: H0, W0S, W0L)
    d = np.abs(new[rr] - old[rr]); a_, b_ = new[rr].astype(np.float64), old[rr].astype(np.float64); fr = fid_row[rr]
    ca = a_ - (np.bincount(fr, weights=a_, minlength=data.n) / np.maximum(np.bincount(fr, minlength=data.n), 1))[fr]; cb = b_ - (np.bincount(fr, weights=b_, minlength=data.n) / np.maximum(np.bincount(fr, minlength=data.n), 1))[fr]
    print(f'(3) {v} (ckpt mtime {time.strftime("%Y-%m-%d %H:%M", time.gmtime(st.st_mtime))}Z): {len(np.unique(fr))} feeds {int(rr.sum())} rows compared, re-scored in {time.time() - t0:.0f}s: max |logit diff| {d.max():.2e}, mean {d.mean():.2e}, centred corr {np.corrcoef(ca, cb)[0, 1]:.8f}', flush=True)
    assert d.max() < 5e-3
print('RECHECK OK')
