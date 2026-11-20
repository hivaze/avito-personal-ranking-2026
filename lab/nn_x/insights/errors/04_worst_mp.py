"""Worst-10% feeds and missed positives (rank >= M/2) under G3; member agreement on misses; comparison with found positives."""
from lib import *

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
keep = ~g('leak')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
LV = dict(np.load(OUT / 'out/feed_label_vars.npz', allow_pickle=True))
cat = V.cat
minh = np.load(OUT / 'out/cand_cat_minh.npy')
cold_luo = np.load(OUT / 'out/cold_luo.npy')
cold_tr = np.load(ROOT / 'eda/errors/synthesis/derived/val/cold_item.npy').astype(bool)
age = np.load(ROOT / 'eda/errors/synthesis/derived/val/ag_age.npy').astype(np.float64)
z = S['G3'].astype(np.float64)
r = V.prank(z)
pct = r / np.maximum(M[fid] - 1, 1)

# ---------------- candidate descriptors ----------------
def ld(name, src='feats'):
    p = (FEATS / 'val' / f'{name}.npy') if src == 'feats' else (FC / 'val' / f'{name}.npy')
    return np.nan_to_num(np.load(p).astype(np.float64), nan=0.0)

F = {}
for nm in ['f1', 'f2', 'f3', 'f4', 'f5', 'maxsim32', 'cos_h10', 'seq_cv_invd20', 'pre2_fe', 'ctx_nd_max', 'ctx_hs_k128']:
    F[nm + '_pct'] = V.rk(ld(nm))
F['luo_pop'] = ld('it_hist_all_luo', 'fc'); F['train_imp'] = ld('it_cand_cnt_train', 'fc'); F['it_tr_ctr'] = ld('it_tr_ctr', 'fc')
F['seq_cv_invd20'] = ld('seq_cv_invd20'); F['maxsim32'] = ld('maxsim32')
F['cold_luo'] = cold_luo.astype(float); F['cold_train'] = cold_tr.astype(float); F['age_days'] = age
F['cat_last8'] = (minh < 8).astype(float); F['cat_h8_31'] = ((minh >= 8) & (minh < 32)).astype(float)
F['cat_h32plus'] = ((minh >= 32) & (minh < 10 ** 6)).astype(float); F['cat_never'] = (minh >= 10 ** 6).astype(float)
ck = fid * 64 + cat
uk, inv, cnt = np.unique(ck, return_inverse=True, return_counts=True)
F['cat_share_feed'] = cnt[inv] / M[fid]
# category share in history
hfid = np.load(FLAT / 'val/hist_fidx.npy').astype(np.int64); hcat = np.load(FLAT / 'val/hist_category_id.npy').astype(np.int64)
hitem = np.load(FLAT / 'val/hist_item_id.npy').astype(np.int64); hpos = np.load(FLAT / 'val/hist_hpos.npy').astype(np.int64)
het = np.load(FLAT / 'val/hist_event_type.npy').astype(np.int64)
H = g('H')
hk, hc = np.unique(hfid * 64 + hcat, return_counts=True)
ix = np.minimum(np.searchsorted(hk, ck), len(hk) - 1)
F['cat_hist_share'] = np.where(hk[ix] == ck, hc[ix], 0) / np.maximum(H[fid], 1)
# G3 category (LSE) rank of the candidate's category within the feed
lse = np.log(np.bincount(inv, np.exp(z)))
kf = uk // 64
o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
F['cat_G3rank'] = cr[inv].astype(float)
# item itself in the user's history (re-interaction) and its event type
item = np.load(FLAT / 'val/cand_item_id.npy').astype(np.int64)
key_h = hfid * 200_000_000 + hitem; key_c = fid * 200_000_000 + item
sh = np.argsort(key_h); kh = key_h[sh]
j = np.minimum(np.searchsorted(kh, key_c), len(kh) - 1)
inh = kh[j] == key_c
F['item_in_hist'] = inh.astype(float)
# event types of the category in history: share of contact-like events? report event-type 1 vs others for the category
for et in [1, 2, 3, 4, 5]:
    hk2, hc2 = np.unique((hfid * 64 + hcat)[het == et], return_counts=True)
    ix2 = np.minimum(np.searchsorted(hk2, ck), len(hk2) - 1)
    F[f'cat_hist_et{et}'] = np.where(hk2[ix2] == ck, hc2[ix2], 0).astype(float)

pos = y > 0
MP = pos & (r >= M[fid] / 2) & keep[fid]
TP3 = pos & (r < 3) & keep[fid]
MID = pos & (r >= 3) & (r < M[fid] / 2) & keep[fid]
FA3 = (~pos) & (r < 3) & keep[fid] & (np.bincount(fid, MP, n)[fid] > 0)  # top-3 negatives in feeds that have an MP
ALLN = (~pos) & keep[fid]
print(f'positives (leak-free): {(pos & keep[fid]).sum()}; MP (rank >= M/2): {MP.sum()} ({MP.sum()/(pos & keep[fid]).sum():.4f}); TP3 {TP3.sum()}; '
      f'feeds with an MP: {np.bincount(fid, MP, n)[keep].astype(bool).mean():.4f}')
mpf = np.bincount(fid, MP, n) > 0
nd = g('ndcg_G3')
print(f'NDCG of feeds with an MP {nd[mpf & keep].mean():.4f} vs others {nd[~mpf & keep].mean():.4f}; loss share {(1-nd)[mpf & keep].sum()/(1-nd)[keep].sum():.4f}')
print('\n=== Candidate descriptors: MP vs mid vs TP3 positives vs FA3 negatives vs all negatives (means; medians for counts) ===')
print(f'{"feature":16s} {"MP":>9s} {"mid":>9s} {"TP3":>9s} {"FA3(neg)":>9s} {"allneg":>9s}')
for k, x in F.items():
    if k in ['luo_pop', 'train_imp', 'age_days', 'seq_cv_invd20', 'cat_hist_et1', 'cat_hist_et2', 'cat_hist_et3', 'cat_hist_et4', 'cat_hist_et5']:
        f_ = np.median; tag = ' (med)'
    else:
        f_ = np.mean; tag = ''
    print(f'{k + tag:22s} ' + ' '.join(f'{f_(x[m]):9.4f}' for m in [MP, MID, TP3, FA3, ALLN]))

# MP decomposition by mutually exclusive reason classes
nov = minh >= 32
cls = np.select([nov & cold_luo, nov & ~cold_luo, ~nov & cold_luo], ['novelcat+cold', 'novelcat+warm', 'recentcat+cold'], 'recentcat+warm')
print('\n=== Positive classes: share of positives, MP rate, share of MPs, mean pct rank, act/exp (G3 within-feed softmax) ===')
# calibration: expected positives under softmax of a temperature-fitted G3 z
best = None
for tmp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    e = np.exp(z * tmp); p = e / np.bincount(fid, e, n)[fid] * V.npos[fid]
    ll = np.sum(y * np.log(p / V.npos[fid] + 1e-12))
    if best is None or ll > best[1]: best = (tmp, ll, p)
tmp, _, pexp = best
print(f'(softmax temperature fitted by likelihood: {tmp})')
pk = pos & keep[fid]
for c in ['recentcat+warm', 'recentcat+cold', 'novelcat+warm', 'novelcat+cold']:
    m = cls == c
    a = y[m & keep[fid]].sum(); ex = pexp[m & keep[fid]].sum()
    print(f'  {c:16s} cand share {np.mean(m[keep[fid]]):.4f} pos share {np.sum(pk & m)/pk.sum():.4f} MP rate {np.sum(MP & m)/max(np.sum(pk & m),1):.4f} '
          f'share of MPs {np.sum(MP & m)/MP.sum():.4f} pct rank {pct[pk & m].mean():.4f} act/exp {a/ex:.3f}')

# systematic misses: member agreement
D = np.load(OUT / 'out/member_ndcg.npz'); mem = D['members']
P_ = ROOT / 'preds'
pos_i = np.flatnonzero(pos & keep[fid])
miss_cnt = np.zeros(len(pos_i)); top3_cnt = np.zeros(len(pos_i))
for m in mem:
    zm = 0.5 * V.z(np.load(P_ / f'nn_{m}_n2cpv3_val_oof.npy')) + 0.5 * V.z(np.load(P_ / f'nn_{m}_trainonly_val.npy'))
    rm = V.prank(zm)[pos_i]
    miss_cnt += rm >= M[fid[pos_i]] / 2; top3_cnt += rm < 3
fr = miss_cnt / len(mem)
isMP = MP[pos_i]
print(f'\n=== Member agreement ({len(mem)} members, final = 0.5z(ft)+0.5z(to)) ===')
print(f'G3-MPs missed by >= 90% of members: {np.mean(fr[isMP] >= 0.9):.4f}; by all: {np.mean(fr[isMP] == 1):.4f}; by < 50%: {np.mean(fr[isMP] < 0.5):.4f}')
print(f'positives missed by ALL members: {np.mean(fr == 1):.4f} of positives; missed by some but not all: {np.mean((fr > 0) & (fr < 1)):.4f}; never missed {np.mean(fr == 0):.4f}')
print(f'G3 non-MP positives that >= 50% of members miss: {np.mean(fr[~isMP] >= 0.5):.4f}')
hist = np.histogram(fr[isMP], bins=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.9999, 1.0001])[0] / isMP.sum()
print('distribution of member-miss fraction among G3 MPs:', np.round(hist, 4), '(bins 0-.2,.2-.4,.4-.6,.6-.8,.8-.9,.9-<1,1)')

# ---------------- feed-level: MP feeds vs others; worst 10% ----------------
FV = {'M': M.astype(float), 'H': H.astype(float), 'npos': V.npos, 'uid0': g('uid0').astype(float), 'n_train_feeds>0': (g('n_train_feeds') > 0).astype(float),
      'cold_share_luo': g('cold_share_luo'), 'n_cat': g('n_cat').astype(float), 'max_cat_share': g('max_cat_share'),
      'hist_n_cat': g('hist_n_cat').astype(float), 'hist_top_cat_share': g('hist_top_cat_share'), 'cand_cat_in_hist_share': g('cand_cat_in_hist_share'),
      'coldpos': g('coldpos').astype(float), 'warmpos': g('warmpos').astype(float),
      'pos_cat_novel(>=32)': np.isin(LV['nov'], ['h>=32', 'never']).astype(float), 'pos_cat_never': (LV['nov'] == 'never').astype(float),
      'pos_f3_catrank>0': (g('pos_f3_catrank') > 0).astype(float), 'pos_cat_G3rank>0': (LV['pcr'] > 0).astype(float),
      'pos_cat_hist_share': g('pos_cat_hist_share'), 'tue': g('tue').astype(float), 'evening18-23': (g('hour') >= 18).astype(float)}
# G3 "confidence" features (observable): top-1 margin over #2 (z), entropy of softmax, G3 vs member disagreement
zs = np.sort(z.reshape(-1)) if False else None
top1 = np.full(n, -np.inf); np.maximum.at(top1, fid, z)
e = np.exp(z * tmp); ps = e / np.bincount(fid, e, n)[fid]
ent = -np.bincount(fid, ps * np.log(ps + 1e-12), n) / np.log(M)
FV['softmax_entropy_norm'] = ent
FV['top1_prob'] = np.bincount(fid, np.where(z == top1[fid], ps, 0), n)
madj = LV['madj']
w10 = nd <= np.quantile(nd[keep], 0.1)
wadj = madj <= np.quantile(madj[keep], 0.1)
print('\n=== Feed-level descriptors: worst 10% (raw NDCG), worst 10% (M x npos adjusted), feeds with an MP, rest ===')
print(f'{"var":26s} {"worst10raw":>10s} {"worst10adj":>10s} {"MPfeeds":>10s} {"all":>10s}')
for k, x in FV.items():
    print(f'{k:26s} ' + ' '.join(f'{x[m & keep].mean():10.4f}' for m in [w10, wadj, mpf, np.ones(n, bool)]))
print(f'worst10 raw: mean NDCG {nd[w10 & keep].mean():.4f}, loss share {(1-nd)[w10 & keep].sum()/(1-nd)[keep].sum():.4f}; '
      f'worst10 adj: mean NDCG {nd[wadj & keep].mean():.4f}, loss share {(1-nd)[wadj & keep].sum()/(1-nd)[keep].sum():.4f}')
# predictability of being in worst10adj from observable features (5-fold user-grouped logistic regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
obs_cols = ['M', 'H', 'uid0', 'n_train_feeds>0', 'cold_share_luo', 'n_cat', 'max_cat_share', 'hist_n_cat', 'hist_top_cat_share',
            'cand_cat_in_hist_share', 'tue', 'evening18-23', 'softmax_entropy_norm', 'top1_prob']
X = np.column_stack([FV[c] for c in obs_cols])[keep]; X = (X - X.mean(0)) / (X.std(0) + 1e-9)
X = np.column_stack([X, np.log(M[keep])])
for tgt_name, tgt in [('worst10adj', wadj[keep]), ('has MP', mpf[keep]), ('worst10raw', w10[keep])]:
    oof = np.zeros(keep.sum())
    for tr, te in GroupKFold(5).split(X, tgt, g('user_id')[keep]):
        oof[te] = LogisticRegression(max_iter=2000).fit(X[tr], tgt[tr]).predict_proba(X[te])[:, 1]
    print(f'predictability of {tgt_name} from observable feed features (LR, user-grouped 5-fold): AUC {roc_auc_score(tgt, oof):.3f}')

# ---------------- history views for MP vs TP3 feeds ----------------
print('\n=== Positive item re-interaction (item already in the user history) ===')
for nm, m in [('MP', MP), ('mid', MID), ('TP3', TP3), ('allneg', ALLN)]:
    print(f'  {nm:6s} item_in_hist {F["item_in_hist"][m].mean():.4f}')
print(f'positives with item_in_hist: share {F["item_in_hist"][pk].mean():.4f}, their MP rate {np.mean(MP[pk & inh]) / max(np.mean(pk[pk & inh]),1):.4f} '
      f'vs others {np.sum(MP & ~inh)/np.sum(pk & ~inh):.4f}')
np.save(OUT / 'out/member_miss_frac.npy', np.column_stack([pos_i, fr]))
