"""Build per-candidate pool scores and the per-feed table for the G3 error anatomy (val).
Outputs: out/cand_scores.npz (z-scores per candidate), out/feed_table.parquet, out/member_ndcg.npz."""
from lib import *

(OUT / 'out').mkdir(exist_ok=True)
V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
cache = {}

# ---- NEW members (decide5 gate (a)) ----
NEW = []
for m in new_candidates():
    to = load_member(V, m, 'to'); cache[(m, 'to')] = to; cache[(m, 'ft')] = load_member(V, m, 'ft')
    Mh, oh = np.load(PH / 'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(PH / 'holdout_y.npy')
    sv = V.ndcg(to).mean(); sh = ndcg_per_feed(np.load(PH / f'nn_{m}_holdout.npy'), yh, oh, Mh).mean()
    ok = sv >= 0.39467 and sh >= 0.31799
    log(f'NEW cand {m:24s} to {sv:.5f} hold {sh:.5f} {"PASS" if ok else "fail"}')
    if ok: NEW.append(m)
log('NEW', len(NEW), NEW)

fam = lambda *fs: [m for f in fs for m in FAM[f]]
POOLS = {
    'G3': fam('F10', 'F7x', 'F7x70'),
    'N1': fam('F10', 'F7x', 'F7x70') + NEW,
    'NEW': NEW,
    'PF7x': fam('F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'),
    'Cd': fam('F5cd', 'F6', 'F7'),
    'M2': fam('F7x', 'F7x70'),
    'F7x': FAM['F7x'], 'F7b': FAM['F7b'], 'F7': FAM['F7'], 'F10': FAM['F10'], 'F7x70': FAM['F7x70'],
    'V27': [m for m in NEW if m.startswith('v27')], 'V28': [m for m in NEW if m.startswith('v28')],
    'V29': [m for m in NEW if m.startswith('v29')], 'V30': [m for m in NEW if m.startswith('v30')],
}
S = {}
for k, ms in POOLS.items():
    if not ms: continue
    fin, ft, to = pool_scores(V, ms, cache)
    S[k], S[k + '_ft'], S[k + '_to'] = fin, ft, to
    log(f'pool {k:6s} n={len(ms):2d} final {V.ndcg(fin).mean():.5f} to {V.ndcg(to).mean():.5f} ft {V.ndcg(ft).mean():.5f}')

# check against the saved G3 pool arrays
g3ft = np.load(P / 'ens_G3_ft_val.npy'); g3to = np.load(P / 'ens_G3_to_val.npy')
chk = 0.5 * V.z(g3ft) + 0.5 * V.z(g3to)
log('saved ens_G3 final', V.ndcg(chk).mean(), 'corr with rebuilt', np.corrcoef(chk, S['G3'])[0, 1])

# ---- older references ----
old = {
    'nf2_oof': P / 'nn_final2_val_oof.npy', 'nf2_tr': P / 'nn_final2_trainonly_val.npy',
    'nf3_oof': ROOT / 'eda/errors/synthesis/snap/nn_final3_val_oof.npy', 'nf3_tr': ROOT / 'eda/errors/synthesis/snap/nn_final3_trainonly_val.npy',
    'stk4': P / 'stk2_res4_clean_val.npy',
}
for k, p in old.items():
    S[k] = V.z(np.load(p))
S['v5'] = 0.5 * S['nf2_oof'] + 0.5 * S['stk4']

# a4u / cset e10 pairs (train-only) vs F7 originals s101/s102 (lever-response table)
R6P = ROOT / 'nn_x/r6/preds'
for arm in ['a4u', 'cset']:
    fs = [R6P / f'nn_r6_{arm}_e10_s{s}_trainonly_val.npy' for s in (101, 102)]
    if all(f.exists() for f in fs):
        S[f'{arm}_to'] = V.z(np.mean([V.z(np.load(f)) for f in fs], 0))
S['F7_s12_to'] = V.z(np.mean([cache[(m, 'to')] if (m, 'to') in cache else load_member(V, m, 'to') for m in ['v17_pre2s_s101', 'v17_pre2s_s102']], 0))

np.savez(OUT / 'out/cand_scores.npz', **{k: v.astype(np.float32) for k, v in S.items()})
log('saved cand_scores', list(S))

# ---- member-level per-feed NDCG (final = 0.5 z(ft) + 0.5 z(to)) for oracles ----
mem = POOLS['N1']
MN = np.stack([V.ndcg(0.5 * cache[(m, 'ft')] + 0.5 * cache[(m, 'to')]) for m in mem])
MNto = np.stack([V.ndcg(cache[(m, 'to')]) for m in mem])
np.savez(OUT / 'out/member_ndcg.npz', members=np.array(mem), final=MN, to=MNto, G3n=len(POOLS['G3']))
log('member ndcg', MN.shape)

# ---- per-feed table ----
ft_old = pl.read_parquet(ROOT / 'eda/errors/segments/feed_table.parquet')
assert (ft_old['M'].to_numpy() == M).all()
keep = ['fidx', 'M', 'H', 'uid0', 'n_train_feeds', 'cold_share', 'coldcand_share', 'hour', 'day', 'n_cat', 'max_cat_share', 'hist_n_cat',
        'hist_top_cat_share', 'cand_cat_in_hist_share', 'npos', 'leak', 'n_pos_cat', 'pos_cat', 'pos_cat_share_feed', 'pos_cat_in_hist',
        'pos_cat_hist_share', 'pos_cold_any', 'pos_cold_all', 'pos_f3_pct', 'pos_f1_pct', 'pos_f3_catrank']
T = {c: ft_old[c].to_numpy() for c in keep}
cold_luo = np.nan_to_num(np.load(FC / 'val/it_hist_all_luo.npy').astype(np.float64), nan=0.0) == 0
ncold = np.bincount(fid, y * cold_luo, n)
T['coldpos'] = (V.npos > 0) & (ncold == V.npos)
T['warmpos'] = (V.npos > 0) & (ncold == 0)
T['cold_share_luo'] = np.bincount(fid, cold_luo, n) / M
ts = V.f['timestamp'].to_numpy().astype(np.int64)
T['tue'] = (ts + 10800) // 86400 < 20720
T['ts'] = ts
T['user_id'] = V.f['user_id'].to_numpy()
for k in S:
    T['ndcg_' + k] = V.ndcg(S[k])
# best-positive rank under G3 / N1 / nf2_oof
for k in ['G3', 'N1', 'nf2_oof', 'nf3_oof']:
    r = V.prank(S[k])
    bp = np.full(n, 10 ** 9); np.minimum.at(bp, fid[y > 0], r[y > 0])
    T['bprank_' + k] = bp
# random-order expected NDCG (pessimistic ties irrelevant): E over random permutations, by MC 20 draws
rng = np.random.default_rng(0)
T['ndcg_random'] = np.mean([V.ndcg(rng.random(len(y))) for _ in range(20)], 0)
df = pl.DataFrame(T)
df.write_parquet(OUT / 'out/feed_table.parquet')
np.save(OUT / 'out/cold_luo.npy', cold_luo)
log('feed table', df.shape)
lk = T['leak']
for k in ['G3', 'N1', 'PF7x', 'M2', 'Cd', 'nf2_oof', 'nf3_oof', 'v5']:
    log(f'{k:8s} all {T["ndcg_" + k].mean():.5f}  leak-free {T["ndcg_" + k][~lk].mean():.5f}')
