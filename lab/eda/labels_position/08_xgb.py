"""Train-vs-val experiments with GPU XGBoost rank:ndcg (pairs from top-300 = full list for M<=300), pessimistic-tie exact metric.
(a) train sample -> val; (b) + pos (and rank-from-end) as features, fixed at inference; (c) val-only 5-fold user-grouped CV;
(d) train sample + val folds with group weights; (e) pretrain on train then continue boosting on val folds."""
from xgbexp import *
from sklearn.model_selection import GroupKFold
import sys
nb.set_num_threads(16)
import os
RES = os.environ.get('RES', f'{OUT}/08_xgb.json')
try:
    R = json.load(open(RES))
except Exception:
    R = {}
stages = sys.argv[1].split(',') if len(sys.argv) > 1 else ['a', 'b', 'c', 'd', 'e']
dtr = pl.read_parquet(f'{OUT}/feat_train200k.parquet').with_columns(pl.col('pos').cast(pl.Float32), pl.col('rkend').cast(pl.Float32))
dva = pl.read_parquet(f'{OUT}/feat_val.parquet')
CK = [100, 200, 400, 600]
def save(name, r, preds=None):
    R[name] = r; json.dump(R, open(RES, 'w'), indent=1)
    print(name, json.dumps({k: v for k, v in r.items() if k != 'importance'}), flush=True)
    if preds is not None:
        np.savez(f'{OUT}/pred_{name}.npz', **{str(k): v.astype(np.float32) for k, v in preds.items()})

def run(name, feats, p, rounds=600, overrides=None, d=None):
    if name in R:
        print('skip', name); return
    d = dtr if d is None else d
    t0 = time.time(); bst = xgb.train(p, dmat(d, feats), num_boost_round=rounds); tt = time.time() - t0
    r = {'train_time_s': tt, 'n_train_feeds': int(d['fid'].n_unique())}
    imp = bst.get_score(importance_type='total_gain'); tot = sum(imp.values()); r['importance'] = {k: round(v / tot, 4) for k, v in imp.items()}
    if overrides is None:
        r['ndcg'], preds = evaluate(bst, dva, feats, CK)
        best = max(r['ndcg'], key=r['ndcg'].get); r['detail'], _ = ndcg_detail(preds[best], dva); r['best_ck'] = best
        save(name, r, preds)
    else:
        r['ndcg_by_override'] = {}
        for k, ov in overrides.items():
            r['ndcg_by_override'][k], preds = evaluate(bst, dva, feats, CK, override=ov)
            np.savez(f'{OUT}/pred_{name}__{k.replace("=", "").replace(",", "_")}.npz', **{str(c): v.astype(np.float32) for c, v in preds.items()})
        save(name, r)
    return bst

if 'a' in stages:
    run('a_basic_rank', BASIC, params('rank'))
    run('a_basic_binary', BASIC, params('binary'))
    run('a_rich_rank', RICH, params('rank'))
if 'b' in stages:
    POSV = [0, 1, 2, 5, 10, 20, 50, 100, 200]
    run('b_basic_pos_rank', BASIC + ['pos'], params('rank'), overrides={f'pos={p}': {'pos': p} for p in POSV})
    run('b_basic_pos_rkend_rank', BASIC + ['pos', 'rkend'], params('rank'),
        overrides={**{f'pos={p},rkend=50': {'pos': p, 'rkend': 50} for p in [0, 5, 20, 50]}, 'pos=5,rkend=0': {'pos': 5, 'rkend': 0}, 'pos=5,rkend=2': {'pos': 5, 'rkend': 2}, 'pos=5,rkend=200': {'pos': 5, 'rkend': 200}})
    run('b_rich_pos_rank', RICH + ['pos'], params('rank'), overrides={f'pos={p}': {'pos': p} for p in [0, 5, 20, 50, 100]})
    run('b_rich_pos_rkend_rank', RICH + ['pos', 'rkend'], params('rank'), overrides={f'pos={p},rkend=50': {'pos': p, 'rkend': 50} for p in [0, 5, 20, 50]})

# ---- val CV based stages
gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy(); nF = len(gr)
fid = dva['fid'].to_numpy()
users = dva.group_by('fid', maintain_order=True).agg(pl.col('user_id').first())['user_id'].to_numpy()
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=users))
CKV = [50, 100, 200, 300, 400, 600, 800]

def oof_run(name, feats, p, rounds, extra=None, w_val=1.0, pre=None):
    if name in R:
        print('skip', name); return
    oof = {c: np.zeros(len(dva)) for c in CKV if c <= rounds}
    t0 = time.time()
    for k, (tri, tei) in enumerate(folds):
        trm = np.isin(fid, tri); tem = np.isin(fid, tei)
        dtr_k = dva.filter(pl.Series(trm)); dte = dva.filter(pl.Series(tem))
        gw = None
        if extra is not None:
            ng_extra = extra['fid'].n_unique()
            dtr_k = pl.concat([extra.select(feats + ['fid', 'target']), dtr_k.select(feats + ['fid', 'target']).with_columns(pl.col('fid') + 10_000_000)], how='vertical_relaxed')
            gw = np.concatenate([np.ones(ng_extra), np.full(len(tri), w_val)]).astype(np.float32)
        bst = xgb.train(p, dmat(dtr_k, feats, gweight=gw), num_boost_round=rounds, xgb_model=pre)
        base = pre.num_boosted_rounds() if pre is not None else 0
        dm = dmat(dte, feats, group=False)
        for c in oof:
            oof[c][tem] = bst.predict(dm, iteration_range=(0, base + c))
    r = {'ndcg_oof': {c: float(ndcg_pf(s, t, off, gr).mean()) for c, s in oof.items()}, 'time_s': time.time() - t0}
    best = max(r['ndcg_oof'], key=r['ndcg_oof'].get); r['best_ck'] = best
    r['detail'], _ = ndcg_detail(oof[best], dva)
    save(name, r, oof)

if 'c' in stages:
    oof_run('c_val_basic', BASIC, params('rank', depth=6, mcw=100), 800)
    oof_run('c_val_rich', RICH, params('rank', depth=6, mcw=100), 800)
    oof_run('c_val_rich_binary', RICH, params('binary', depth=6, mcw=100), 800)
if 'd' in stages:
    ex = dtr.select(RICH + ['fid', 'target'])
    for w in [1.0, 5.0, 20.0]:
        oof_run(f'd_train200k+val_w{int(w)}_rich', RICH, params('rank'), 600, extra=ex, w_val=w)
if 'e' in stages:
    pre = xgb.train(params('rank'), dmat(dtr, RICH), num_boost_round=300)
    oof_run('e_pretrain300_finetune_rich', RICH, params('rank', eta=0.02, depth=6, mcw=100), 600, pre=pre)
