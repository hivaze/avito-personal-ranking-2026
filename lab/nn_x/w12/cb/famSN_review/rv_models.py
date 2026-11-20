"""Model-side checks (CPU, <= 6 threads): saved predictions reproduce from the saved models; the truncated seed-1 member equals the long model cut
at its selected tree; a T-tree run equals the first T trees of a 2T-tree run (tiny fit); famN quantisation / one-hot / text / embedding levers are
really inside the fitted models; own-feed share among the nearest earlier rows for CatBoost's online KNN order (emulation)."""
import os, sys, json
sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review'); sys.path.insert(0, '<repo>/nn_x/w12/cb/famN/scripts')
import numpy as np
import fn; from fn import cbx
from catboost import CatBoost, CatBoostRanker
T = cbx.THREADS; assert T <= 6
R = cbx.ROOT; PS = R / 'nn_x/w12/cb/famS/preds'; PN = R / 'nn_x/w12/cb/famN/preds'
what = sys.argv[1:]
if 'repro' in what:
    va = cbx.load('val'); pv = cbx.make_pool(va, cat='native')
    for src, dst in (('s1f_rmse_d8_lr030_s1long', 's1f_rmse_d8_lr030_s1'), ('s4b_qsm_soft_lr030_s0', 's4b_qsm_soft_lr030_s0'), ('s1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s0'), ('s1f_rmse_d8_lr030_s2', 's1f_rmse_d8_lr030_s2')):
        m = CatBoost(); m.load_model(str(PS / f'{src}.cbm')); k = json.load(open(PS / f'{dst}.eval.json'))['extra']['best_iter'] + 1
        p = np.asarray(m.predict(pv, ntree_end=k, thread_count=T), np.float32); s = np.load(PS / f'{dst}_val.npy')
        print(f'repro {dst} from {src}.cbm at {k} trees (model has {m.tree_count_}): max abs diff to the saved val file {float(np.abs(p - s).max()):.3g} (sd of prediction {float(s.std()):.3f})', flush=True)
    for tag in ('Yref_n30k_s0', 'Y_q1024_n30k_s0', 'Rref_n30k_s1'):
        m = CatBoost(); m.load_model(str(PN / f'{tag}.cbm')); k = json.load(open(PN / f'{tag}.eval.json'))['extra']['best_iter'] + 1
        p = np.asarray(m.predict(pv, ntree_end=k, thread_count=T), np.float32); s = np.load(PN / f'{tag}_val.npy')
        print(f'repro {tag} at {k} trees: max abs diff {float(np.abs(p - s).max()):.3g}', flush=True)
if 'levers' in what:
    names = cbx.feature_names()
    for tag in ('Yref_n30k_s0', 'Y_q1024_n30k_s0', 'Y_q1024_n30k_s1'):
        m = CatBoost(); m.load_model(str(PN / f'{tag}.cbm')); b = m.get_borders()
        fl = m.feature_names_
        d = {fl[i]: len(v) for i, v in b.items()}
        print(tag, 'borders: it_hist_et1_luo', d.get('it_hist_et1_luo'), 'pre2_fe_z', d.get('pre2_fe_z'), 'f1', d.get('f1'), 'max over the others', max(v for k_, v in d.items() if k_ not in ('it_hist_et1_luo', 'pre2_fe_z')), flush=True)
    for tag in ('Yref_n30k_s0', 'Y_oh64_n30k_s0', 'Y_txt_s_n30k_s0', 'Y_txt_l_n30k_s0', 'L_lda_all_n30k_s0', 'Y_def_pair32_n6k_s0', 'Y_hall_n30k_s0', 'R_hknn3_n30k_s0'):
        m = CatBoost(); m.load_model(str(PN / f'{tag}.cbm')); ap = m.get_all_params()
        imp = dict(zip(m.feature_names_, m.get_feature_importance(type='PredictionValuesChange', thread_count=T)))
        new = {k_: round(float(v), 2) for k_, v in imp.items() if k_ not in names}
        print(tag, '| one_hot_max_size', ap.get('one_hot_max_size'), '| has_time', ap.get('has_time'), '| simple_ctr', str(ap.get('simple_ctr'))[:60], '| text', str(ap.get('text_processing'))[:200] if 't_cat' in m.feature_names_ else None,
              '| embedding calcers', ap.get('embedding_calcers') or ap.get('embedding_processing'), '| category_id importance', round(float(imp.get('category_id', -1)), 2), '| new columns', new, flush=True)
if 'trunc' in what:
    tr = cbx.load('xm'); rg = np.random.default_rng(5); k1 = np.zeros(tr.n_feeds, bool); k1[rg.choice(tr.n_feeds, 2500, replace=False)] = True
    fit, sel = cbx.trainside_split(tr); pf = cbx.make_pool(tr, feeds=fit & k1, label=tr.teacher('mean11', 'centred')); ps = cbx.make_pool(tr, feeds=sel & k1, label=tr.y)
    ms = []
    for it in (40, 80):
        m = CatBoostRanker(loss_function='RMSE', depth=8, learning_rate=0.3, l2_leaf_reg=5, border_count=254, iterations=it, od_type='Iter', od_wait=it, eval_metric='NDCG:type=Base', use_best_model=False, random_seed=1,
                           allow_writing_files=False, thread_count=T); m.fit(pf, eval_set=ps, verbose=0); ms.append(m)
    a = ms[0].predict(ps, thread_count=T); b = ms[1].predict(ps, ntree_end=40, thread_count=T); c = ms[1].predict(ps, thread_count=T)
    print(f'trunc check (2,500 feeds): 40-tree run vs first 40 trees of the 80-tree run: max abs diff {float(np.abs(a - b).max()):.3g}; vs all 80 trees {float(np.abs(a - c).max()):.3g}', flush=True)
if 'knn' in what:
    # CatBoost online KNN order (has_time=True, feeds in time order, rows shuffled inside the feed): for a sample of fit rows of the n6k subset,
    # share of the 50 / 5 nearest EARLIER rows (32-d PCA of u*e, L2) that come from the row's own feed.
    tr = cbx.load('xm'); fit, sel = fn.subset(6000, 1500); b = fn.Block(tr, fit); P = b.vec_pca('pair', 32)
    perm = np.lexsort((np.random.default_rng(fn.SHUF_SEED).random(b.n), b.gid)); P = P[perm]; g = b.gid[perm]
    t_oof = b.label('oof')[perm].astype(np.float64); y = b.label('y')[perm].astype(np.float64)
    rg = np.random.default_rng(3); q = np.sort(rg.choice(np.arange(b.n // 4, b.n), 1500, replace=False))
    sh50, sh5, own_t, all_t, n_own = [], [], [], [], []
    for i in q:
        d = ((P[:i] - P[i]) ** 2).sum(1)
        if i < 60: continue
        nn = np.argpartition(d, 50)[:50]; nn = nn[np.argsort(d[nn])]
        own = g[nn] == g[i]; sh50.append(own.mean()); sh5.append(own[:5].mean()); all_t.append(t_oof[nn].mean()); n_own.append(int((g[:i] == g[i]).sum()))
    sh50 = np.array(sh50); sh5 = np.array(sh5)
    print(f'online-KNN emulation, n6k fit rows ({b.n} rows, {len(sh50)} sampled queries in the last 3/4 of the pool): mean share of own-feed rows among the 50 nearest earlier rows {sh50.mean():.3f} '
          f'(median {np.median(sh50):.3f}, share of queries with >= 50% own-feed neighbours {float((sh50 >= 0.5).mean()):.3f}); among the 5 nearest {sh5.mean():.3f}; mean own-feed prefix size {np.mean(n_own):.0f} of a prefix of about {int(np.mean(q))} rows', flush=True)
    # same statistic as the tree sees it: corr of the 50-NN mean teacher with the row's own teacher value, online (with own-feed prefix) vs out-of-feed neighbours only
    on, off_, tt = [], [], []
    for i in q[:800]:
        d = ((P[:i] - P[i]) ** 2).sum(1); nn = np.argpartition(d, 50)[:50]; on.append(t_oof[nn].mean())
        d2 = d.copy(); d2[g[:i] == g[i]] = np.inf; nn2 = np.argpartition(d2, 50)[:50]; off_.append(t_oof[nn2].mean()); tt.append(t_oof[i])
    print(f'corr(50-NN teacher mean, own teacher value) over {len(tt)} queries: online prefix incl. own feed {np.corrcoef(on, tt)[0, 1]:.3f}; own feed excluded (what an apply row gets) {np.corrcoef(off_, tt)[0, 1]:.3f}', flush=True)
print('DONE', what)
