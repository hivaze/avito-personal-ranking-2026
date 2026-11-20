# Adversarial validation between the splits. A LightGBM classifier is trained to tell val from test,
# train from val, train from test and public from private test feeds, at feed level and at candidate
# level, with and without features that count how often an item was seen in train.
# How it was run: from lab/eda/shift_overlap, where it imports common.py and featlib.py. The command
# line is not on record; the output is lab/eda/shift_overlap/out/s04.log.
# What it produced: 5-fold AUC per pair. Feed level: val vs test 0.524, train vs val 0.925; with the
# train-overlap features val vs test rises to 0.909. Copy of the log: results/shift_s04_adversarial.log.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/eda/shift_overlap/s04_adversarial.py
"""Adversarial validation: val vs test, test-public vs test-private, train(last days) vs val. LightGBM, 5-fold CV AUC."""
from common import *
from featlib import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import time
t0 = time.time()
rng = np.random.default_rng(42)
counts = {}
counts['train_cand'] = np.bincount(np.asarray(cand('train', 'item_id')), minlength=NID).astype(np.int32)
counts['train_hist'] = np.bincount(np.asarray(hist('train', 'item_id')), minlength=NID).astype(np.int32)
tr_users = feeds('train')['user_id'].unique().to_numpy(); tr_users = tr_users[tr_users != 0]
print('counts', round(time.time() - t0), flush=True)
D = {'val': subset('val'), 'test': subset('test')}
ftr = feeds('train')
# train subsample: 30k random feeds from the last 3 train days (Sep 17-19)
last = np.where(ftr['date'].cast(pl.Int32).to_numpy() >= ftr['date'].cast(pl.Int32).max() - 3)[0]
D['train'] = subset('train', rng.choice(last, 30000, replace=False))
print('subsets', round(time.time() - t0), flush=True)

PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, verbose=-1, num_threads=32)

def adv(X, y, groups=None, rounds=300, name='', cat=None, top=12):
    X = X.to_pandas(); y = np.asarray(y)
    oof = np.zeros(len(y)); imp = np.zeros(X.shape[1])
    kf = GroupKFold(5) if groups is not None else StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in kf.split(X, y, groups):
        ds = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat or 'auto')
        m = lgb.train(PAR, ds, rounds)
        oof[te] = m.predict(X.iloc[te]); imp += m.feature_importance('gain')
    auc = roc_auc_score(y, oof)
    o = np.argsort(-imp); tot = imp.sum()
    print(f'  [{name}] n={len(y)} pos={y.mean():.3f} AUC={auc:.4f}  top gain: ' + ', '.join(f'{X.columns[i]}({imp[i]/tot*100:.0f}%)' for i in o[:top]), flush=True)
    return auc, oof

def single_auc(X, y, name):
    out = []
    for c in X.columns:
        v = X[c].to_numpy().astype(float)
        a = roc_auc_score(y, v); out.append((max(a, 1 - a), c, a))
    out.sort(reverse=True)
    print(f'  [{name}] top single-feature AUCs (|AUC-0.5| sorted): ' + ', '.join(f'{c}={a:.3f}' for _, c, a in out[:12]))

# ---------- feed level
print('\n==== FEED-LEVEL adversarial')
FF = {s: feed_features(D[s], counts, tr_users) for s in ['train', 'val', 'test']}
FFu = {s: feed_features(D[s], None, None, with_uemb=True) for s in ['val', 'test', 'train']}
intrinsic = [c for c in FF['val'].columns if c not in ('hour', 'c_in_train_cand', 'c_in_train_hist', 'c_log_train_hist', 'h_in_train_hist', 'user_in_train')]
overlap = ['c_in_train_cand', 'c_in_train_hist', 'c_log_train_hist', 'h_in_train_hist', 'user_in_train']
def pair(a, b, cols, src=FF, filt=None):
    Xa, Xb = src[a].select(cols), src[b].select(cols)
    X = pl.concat([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    return X, y
for a, b in [('val', 'test'), ('train', 'val'), ('train', 'test')]:
    print(f'-- {a} (0) vs {b} (1)')
    X, y = pair(a, b, intrinsic); adv(X, y, name='intrinsic feed feats'); single_auc(X, y, 'intrinsic')
    X, y = pair(a, b, intrinsic + overlap); adv(X, y, name='+train-overlap feats')
    X, y = pair(a, b, intrinsic + ['hour']); adv(X, y, name='+hour')
    ucols = [c for c in FFu['val'].columns if c.startswith('ue')]
    X, y = pair(a, b, ucols, src=FFu); adv(X, y, name='user_emb 128d only')

# test public vs private
ft = D['test']['f']
ts = ft['timestamp'].to_numpy()
order = np.argsort(ts, kind='stable')
k30 = int(round(0.3 * len(ts)))
bound = ts[order[k30]]
pub = np.zeros(len(ts), bool); pub[order[:k30]] = True
import datetime as dt
print(f'\n==== test public/private: 30% boundary ts={bound} = {dt.datetime.utcfromtimestamp(int(bound)+3*3600)} MSK; feeds at boundary ts: {(ts==bound).sum()}')
Xt = FF['test']
for cols, nm in [(intrinsic, 'intrinsic'), (intrinsic + overlap, '+overlap'), ([c for c in FFu['test'].columns if c.startswith('ue')], 'user_emb')]:
    src = FFu['test'] if nm == 'user_emb' else Xt
    adv(src.select(cols), pub.astype(int), name=f'test pub(1) vs priv(0) {nm}')
single_auc(Xt.select(intrinsic + overlap), pub.astype(int), 'pub vs priv')

# val vs test-private and val vs test-public (intrinsic)
for nm, msk in [('public', pub), ('private', ~pub)]:
    X = pl.concat([FF['val'].select(intrinsic), Xt.filter(pl.Series(msk)).select(intrinsic)]); y = np.r_[np.zeros(len(FF['val'])), np.ones(msk.sum())]
    adv(X, y, name=f'val vs test-{nm} intrinsic')

# ---------- candidate level
print('\n==== CANDIDATE-LEVEL adversarial (feeds as CV groups)')
CF = {s: cand_features(D[s], counts, emb=True, split=s) for s in ['train', 'val', 'test']}
base_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id', 'emb_missing']
emb_cols = [f'e{j}' for j in range(16)] + ['e_norm']
ov_cols = ['log_train_cand', 'log_train_hist']
def cpair(a, b, cols, nmax=600000, ma=None, mb=None):
    A = CF[a]; B = CF[b]; ga = D[a]['cf']; gb = D[b]['cf'] + 10_000_000
    if ma is not None: A, ga = A.filter(pl.Series(ma)), ga[ma]
    if mb is not None: B, gb = B.filter(pl.Series(mb)), gb[mb]
    ia = rng.choice(len(A), min(nmax, len(A)), replace=False); ib = rng.choice(len(B), min(nmax, len(B)), replace=False)
    X = pl.concat([A[ia].select(cols), B[ib].select(cols)]); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    g = np.r_[ga[ia], gb[ib]]
    return X, y, g
for a, b in [('val', 'test'), ('train', 'val'), ('train', 'test')]:
    print(f'-- {a} (0) vs {b} (1)')
    X, y, g = cpair(a, b, base_cols); adv(X, y, g, name='f1..f5+category+emb_missing', cat=['category_id'])
    X, y, g = cpair(a, b, base_cols + ['M']); adv(X, y, g, name='+M', cat=['category_id'])
    X, y, g = cpair(a, b, emb_cols); adv(X, y, g, name='item emb 16 dims+norm')
    X, y, g = cpair(a, b, base_cols + ov_cols); adv(X, y, g, name='+train-overlap counts', cat=['category_id'])
pubc = pub[D['test']['cf']]
X, y, g = cpair('test', 'test', base_cols, ma=~pubc, mb=pubc); adv(X, y, g, name='test priv(0) vs pub(1) cand base', cat=['category_id'])
X, y, g = cpair('test', 'test', base_cols + ov_cols, ma=~pubc, mb=pubc); adv(X, y, g, name='test priv(0) vs pub(1) cand +overlap', cat=['category_id'])
print('done', round(time.time() - t0))
