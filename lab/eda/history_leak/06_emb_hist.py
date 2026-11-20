"""6) Embedding-based history features (content 'co-visitation') + final LightGBM ablation measuring the
incremental value of history features (category affinity, co-visitation) on top of embedding similarities."""
import time
import json
import numpy as np
import polars as pl
import numba as nb
import lightgbm as lgb
from common import feeds, arr, load_split, eval_score, ndcg_per_feed
from hist_feats import compute_aff, FEATS
from importlib import import_module

nb.set_num_threads(40)
OUT = '<repo>/eda/history_leak/'
ECOLS = ['ue_cos', 'he_cos_last10', 'he_cos_all', 'he_max50', 'he_cos_rec95', 'he_cos_feedclick', 'he_cos_t2t3',
         'cand_emb_norm', 'he_cos_last1']


def emb_table(split):
    meta = json.load(open(f'<repo>/data/competition/{split}/item_embeddings/data.meta'))
    mm = np.memmap(f'<repo>/data/competition/{split}/item_embeddings/data.memmap', dtype=np.int8,
                   mode='r', shape=tuple(meta['shape']))
    return mm


@nb.njit(parallel=True, cache=True)
def emb_kernel(sel, coff, M, hoff, H, uemb, cidx, hidx, htype, hsrc, hpos, E, out_off, out):
    D = E.shape[1]
    for q in nb.prange(len(sel)):
        fi = sel[q]
        h = H[fi]; ho = hoff[fi]
        u = uemb[fi].astype(np.float64)
        un = np.sqrt((u * u).sum()) + 1e-9
        u = u / un
        hv = np.zeros((h, D)); hok = np.zeros(h, np.bool_)
        for j in range(h):
            e = hidx[ho + j]
            if e >= 0:
                v = E[e].astype(np.float64)
                n = np.sqrt((v * v).sum())
                if n > 0:
                    hv[j] = v / n; hok[j] = True
        # profile vectors
        p10 = np.zeros(D); pall = np.zeros(D); prec = np.zeros(D); pfc = np.zeros(D); p23 = np.zeros(D); p1 = np.zeros(D)
        n10 = 0; nall = 0; nfc = 0; n23 = 0; got1 = False
        for j in range(h):
            if not hok[j]:
                continue
            if not got1:
                p1 += hv[j]; got1 = True
            if n10 < 10:
                p10 += hv[j]; n10 += 1
            pall += hv[j]; nall += 1
            prec += hv[j] * 0.95 ** hpos[ho + j]
            if htype[ho + j] == 1 and hsrc[ho + j] == 1:
                pfc += hv[j]; nfc += 1
            if htype[ho + j] == 2 or htype[ho + j] == 3:
                p23 += hv[j]; n23 += 1
        profs = (p10, pall, prec, pfc, p23, p1)
        normed = np.zeros((6, D))
        for k in range(6):
            nn = np.sqrt((profs[k] * profs[k]).sum())
            if nn > 0:
                normed[k] = profs[k] / nn
        co = coff[fi]; m = M[fi]; oo = out_off[q]
        for i in range(m):
            r = oo + i
            e = cidx[co + i]
            if e < 0:
                for k in range(9):
                    out[r, k] = np.nan
                continue
            v = E[e].astype(np.float64)
            n = np.sqrt((v * v).sum())
            out[r, 7] = n
            if n == 0:
                continue
            v = v / n
            out[r, 0] = (u * v).sum()
            out[r, 1] = (normed[0] * v).sum(); out[r, 2] = (normed[1] * v).sum(); out[r, 4] = (normed[2] * v).sum()
            out[r, 5] = (normed[3] * v).sum() if nfc > 0 else np.nan
            out[r, 6] = (normed[4] * v).sum() if n23 > 0 else np.nan
            out[r, 8] = (normed[5] * v).sum() if got1 else np.nan
            mx = -2.0
            cnt = 0
            for j in range(h):
                if hok[j]:
                    s = (hv[j] * v).sum()
                    if s > mx:
                        mx = s
                    cnt += 1
                    if cnt >= 50:
                        break
            out[r, 3] = mx if cnt > 0 else np.nan


def compute_emb(split, sel=None):
    f = feeds(split).sort('fidx')
    coff = f['cand_off'].to_numpy(); M = f['M'].to_numpy(); hoff = f['hist_off'].to_numpy(); H = f['H'].to_numpy()
    if sel is None:
        sel = np.arange(len(f))
    sel = np.asarray(sel, np.int64)
    m = M[sel]
    out_off = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
    out = np.zeros((m.sum(), len(ECOLS)), np.float32)
    t0 = time.time()
    E = np.ascontiguousarray(emb_table(split))
    uemb = np.asarray(arr(split, 'user_emb'))
    cidx = np.asarray(arr(split, 'cand_emb_idx')); hidx = np.asarray(arr(split, 'hist_emb_idx'))
    htype = np.asarray(arr(split, 'hist_event_type')); hsrc = np.asarray(arr(split, 'hist_event_source'))
    hpos = np.asarray(arr(split, 'hist_hpos'))
    print(f'loaded {split} emb table {E.shape} in {time.time()-t0:.0f}s', flush=True)
    emb_kernel(sel, coff, M, hoff, H, uemb, cidx, hidx, htype, hsrc, hpos, E, out_off, out)
    print(f'emb feats {split}: {time.time()-t0:.0f}s', flush=True)
    return out


def main():
    d = load_split('val', cand_cols=('item_id', 'category_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'target'))
    t = d['c_target']; off = d['feeds']['cand_off'].to_numpy(); M = d['feeds']['M'].to_numpy()
    Ev = compute_emb('val')
    np.save(OUT + 'out_emb_val.npy', Ev)
    base = eval_score(d['c_f1'].astype(np.float64), d, 'f1')
    rng = np.random.default_rng(0)
    print('--- single embedding-history features (NaN -> -2) ---')
    for j, n in enumerate(ECOLS):
        x = np.nan_to_num(Ev[:, j].astype(np.float64), nan=-2.0)
        a = ndcg_per_feed(x + 1e-9 * rng.random(len(x)), t, off, M)
        print(f'{n:18s} {a.mean():.4f}  (nan frac {np.isnan(Ev[:, j]).mean():.3f})')
    # ---- LightGBM ablation (same 200k train sample as 04/05)
    ft = feeds('train')
    sel = np.sort(np.random.default_rng(1).choice(len(ft), 200_000, replace=False))
    Etr = compute_emb('train', sel)
    np.save(OUT + 'out_emb_trainsample.npy', Etr)
    Atr, _, cidx = compute_aff('train', sel)
    Ava = np.load(OUT + 'out_aff_val.npy')
    Ctr = np.load(OUT + 'out_covis_trainsample.npy'); Cva = np.load(OUT + 'out_covis_val_train.npy')
    fcols = ['f1', 'f2', 'f3', 'f4', 'f5']
    Btr = np.stack([np.asarray(arr('train', 'cand_' + c)[cidx]) for c in fcols] +
                   [np.asarray(arr('train', 'cand_category_id')[cidx]).astype(np.float32)], 1)
    ytr = np.asarray(arr('train', 'cand_target')[cidx])
    gtr = ft.sort('fidx')['M'].to_numpy()[sel]
    Bva = np.stack([d['c_' + c] for c in fcols] + [d['c_category_id'].astype(np.float32)], 1)
    params = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=300, num_threads=40, verbose=-1)
    from covis_cols import COLS as CCOLS
    configs = [('base f1-5+cat', [Btr], [Bva], fcols + ['cat']),
               ('base+emb', [Btr, Etr], [Bva, Ev], fcols + ['cat'] + ECOLS),
               ('base+emb+aff', [Btr, Etr, Atr], [Bva, Ev, Ava], fcols + ['cat'] + ECOLS + FEATS),
               ('base+emb+covis', [Btr, Etr, Ctr], [Bva, Ev, Cva], fcols + ['cat'] + ECOLS + CCOLS),
               ('base+emb+aff+covis', [Btr, Etr, Atr, Ctr], [Bva, Ev, Ava, Cva], fcols + ['cat'] + ECOLS + FEATS + CCOLS)]
    res = {}
    for name, A, B, names in configs:
        A = np.hstack(A); B = np.hstack(B)
        m = lgb.train(params, lgb.Dataset(A, ytr, group=gtr, categorical_feature=[5]), num_boost_round=500)
        p = m.predict(B)
        res[name] = eval_score(p, d, f'LGB {name}', base)
        np.save(OUT + f'out_lgb_val_{name.replace(" ", "_").replace("+", "_")}.npy', p)
        imp = m.feature_importance('gain'); o = np.argsort(-imp)
        print('   top gain:', [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in o[:12]], flush=True)
    for a, b in [('base+emb', 'base+emb+aff'), ('base+emb', 'base+emb+covis'), ('base+emb', 'base+emb+aff+covis'),
                 ('base+emb+covis', 'base+emb+aff+covis')]:
        diff = res[b] - res[a]
        print(f'{b} vs {a}: {diff.mean():+.4f} (paired SE {diff.std()/np.sqrt(len(diff)):.4f})')


if __name__ == '__main__':
    main()
