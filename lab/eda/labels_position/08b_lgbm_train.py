"""(a) train-sample -> val; (b) + pos feature with pos fixed at inference; objective / truncation variants."""
from lgbexp import *
import sys
nb.set_num_threads(40)
R = {}
RES = f'{OUT}/08b_lgbm_train.json'
dtr = pl.read_parquet(f'{OUT}/feat_train200k.parquet')
dva = pl.read_parquet(f'{OUT}/feat_val.parquet')
CK = [50, 100, 200, 300, 500, 800]
def run(name, feats, params, rounds=800, overrides=None, dtrain=None, weight=None):
    bst, tt = fit(dtrain if dtrain is not None else dtr, feats, params, rounds, weight=weight)
    r = {'train_time_s': tt}
    if overrides is None:
        r['ndcg'], s = evaluate(bst, dva, feats, CK)
        r['detail'], _ = ndcg_detail(s, dva)
        np.savez(f'{OUT}/pred_{name}.npz', **{str(k): v for k, v in evaluate.last_preds.items()})
    else:
        r['ndcg_by_override'] = {}
        for k, ov in overrides.items():
            nd, s = evaluate(bst, dva, feats, CK, override=ov)
            r['ndcg_by_override'][k] = nd
            np.savez(f'{OUT}/pred_{name}_{k.replace("=", "").replace(",", "_")}.npz', **{str(c): v for c, v in evaluate.last_preds.items()})
    imp = bst.feature_importance('gain'); r['importance'] = dict(zip(feats, (imp / imp.sum()).round(4).tolist()))
    R[name] = r; print(name, json.dumps({k: v for k, v in r.items() if k != 'importance'}), flush=True)
    json.dump(R, open(RES, 'w'), indent=1)
    return bst
# (a)
run('a_basic_rank_trunc300', BASIC, params_for('rank'))
run('a_basic_rank_trunc30', BASIC, params_for('rank', trunc=30))
run('a_basic_binary', BASIC, params_for('binary'))
run('a_rich_rank_trunc300', RICH, params_for('rank'))
# (b) pos as feature, fixed at inference
POSV = [0, 1, 2, 5, 10, 20, 50, 100, 200]
dtr = dtr.with_columns(pl.col('pos').cast(pl.Float32), pl.col('rkend').cast(pl.Float32))
run('b_basic_pos_rank', BASIC + ['pos'], params_for('rank'), overrides={f'pos={p}': {'pos': p} for p in POSV})
run('b_basic_pos_rkend_rank', BASIC + ['pos', 'rkend'], params_for('rank'), overrides={f'pos={p},rkend=50': {'pos': p, 'rkend': 50} for p in [0, 5, 20, 50]} | {'pos=5,rkend=0': {'pos': 5, 'rkend': 0}, 'pos=5,rkend=200': {'pos': 5, 'rkend': 200}})
run('b_basic_pos_binary', BASIC + ['pos'], params_for('binary'), overrides={f'pos={p}': {'pos': p} for p in [0, 5, 20, 100]})
run('b_rich_pos_rank', RICH + ['pos'], params_for('rank'), overrides={f'pos={p}': {'pos': p} for p in [0, 5, 20, 50, 100]})
run('b_rich_pos_rkend_rank', RICH + ['pos', 'rkend'], params_for('rank'), overrides={f'pos={p},rkend=50': {'pos': p, 'rkend': 50} for p in [0, 5, 20, 50]})
