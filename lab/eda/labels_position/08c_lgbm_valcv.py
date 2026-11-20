"""(c) 5-fold user-grouped CV on val only; (d) train-sample + val folds (val weight w); (e) pretrain on train then fine-tune on val folds.
All evaluated as out-of-fold NDCG over the full val set (exact metric)."""
from lgbexp import *
from sklearn.model_selection import GroupKFold
import sys
nb.set_num_threads(40)
RES = f'{OUT}/08c_lgbm_valcv.json'
R = json.load(open(RES)) if len(sys.argv) > 1 and sys.argv[1] == 'resume' else {}
dva = pl.read_parquet(f'{OUT}/feat_val.parquet')
gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy(); nF = len(gr)
fid = dva['fid'].to_numpy()
users = dva.group_by('fid', maintain_order=True).agg(pl.col('user_id').first())['user_id'].to_numpy()
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=users))
CK = [50, 100, 200, 300, 500, 800, 1200]

def oof_run(name, feats, params, rounds, extra_train=None, val_weight=1.0, init_from=None):
    if name in R:
        print('skip', name); return
    oof = {c: np.zeros(len(dva)) for c in CK if c <= rounds}
    t0 = time.time()
    for k, (tri, tei) in enumerate(folds):
        trm = np.isin(fid, tri); tem = np.isin(fid, tei)
        dtr = dva.filter(pl.Series(trm)); dte = dva.filter(pl.Series(tem))
        w = None
        if extra_train is not None:
            dtr = pl.concat([extra_train.select(dtr.columns), dtr.with_columns(pl.col('fid') + 10_000_000)], how='vertical_relaxed')
            w = np.concatenate([np.ones(len(extra_train)), np.full(int(trm.sum()), val_weight)])
        X = dtr.select(feats).to_numpy().astype(np.float32)
        ds = lgb.Dataset(X, dtr['target'].to_numpy(), group=groups(dtr) if params['objective'] == 'lambdarank' else None, weight=w,
                         feature_name=feats, categorical_feature=['category_id'])
        bst = lgb.train(params, ds, num_boost_round=rounds, init_model=init_from)
        Xte = dte.select(feats).to_numpy().astype(np.float32)
        base = init_from.current_iteration() if init_from is not None else 0
        for c in oof:
            oof[c][tem] = bst.predict(Xte, num_iteration=base + c, num_threads=40)
    r = {'ndcg_oof': {c: float(ndcg_pf(s, t, off, gr).mean()) for c, s in oof.items()}, 'time_s': time.time() - t0}
    best = max(r['ndcg_oof'], key=r['ndcg_oof'].get); r['best_ck'] = best
    r['detail'], pf = ndcg_detail(oof[best], dva)
    np.savez(f'{OUT}/oof_{name}.npz', **{str(c): s.astype(np.float32) for c, s in oof.items()})
    R[name] = r; print(name, json.dumps(r), flush=True)
    json.dump(R, open(RES, 'w'), indent=1)

oof_run('c_val_basic', BASIC, params_for('rank', small=True), 1200)
oof_run('c_val_rich', RICH, params_for('rank', small=True), 1200)
oof_run('c_val_rich_binary', RICH, params_for('binary', small=True), 1200)
dtr = pl.read_parquet(f'{OUT}/feat_train200k.parquet').drop(['pos', 'prk', 'rkend', 'fidx'])
dtr = dtr.with_columns(pl.lit(0).alias('dummy'), pl.lit(0).cast(pl.Int32).alias('user_id'))
for w in [1.0, 5.0, 20.0]:
    oof_run(f'd_train200k+val_w{int(w)}_rich', RICH, params_for('rank'), 800, extra_train=dtr, val_weight=w)
# (e) pretrain on train sample, fine-tune on val folds
pre, _ = fit(dtr, RICH, params_for('rank'), 300)
oof_run('e_pretrain300_finetune_rich', RICH, params_for('rank', small=True, lr=0.02), 800, init_from=pre)
