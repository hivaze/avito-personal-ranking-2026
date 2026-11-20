# The check that closed the privileged teacher. It splits the new events into those older and those newer than
# the user's first interaction with a candidate of the feed, and measures what each part adds to the teacher's
# score on the 4,815 holdout feeds that have a later feed.
# CPU only, no arguments; it loads 01_build.py by path. Output: ../results/hindsight_pre_post_click.txt.
# Events before the first interaction add -0.00026 +- 0.00036, events after it +0.03883 +- 0.00248: the
# signal is a consequence of the click.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/hindsight/04_prepost.py
"""Diagnostic (holdout pair feeds only): is the hindsight gain click-induced? Split FUT (new non-candidate events of B) into
PRE  = events OLDER than the first post-A interaction with any candidate of A (browsing before the first click; all FUT events if no
       candidate item appears among the new events),
POST = events NEWER than that first candidate interaction (may be induced by the click, e.g. similar-item browsing from the clicked item).
GBM combiner as in 02_eval (same params, 3 repeats, 2-fold user-grouped)."""
import os, sys, json, importlib.util
os.environ.setdefault('OMP_NUM_THREADS', '12'); os.environ.setdefault('NUMBA_NUM_THREADS', '12')
import numpy as np, pandas as pd, numba as nb, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
nb.set_num_threads(12)
ROOT = '<repo>'; HD = f'{ROOT}/nn_x/w12/hindsight'; OUT = f'{HD}/out'; FL = f'{ROOT}/data/flat/train'
sys.path.insert(0, f'{ROOT}/src'); from metrics import ndcg_per_feed
spec = importlib.util.spec_from_file_location('b01', f'{HD}/01_build.py'); b01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b01)
set_feats = b01.set_feats; NF = b01.NF

@nb.njit(parallel=True, cache=True)
def build2(sel_a, sel_b, ks, hoff, H, coff, M, row0, hi, ht, hs, hc, he, ci, cc, ce, table, Fout, meta):
    for p in nb.prange(len(sel_a)):
        a = sel_a[p]; b = sel_b[p]; k = ks[p]; m = M[a]; ca = coff[a]; hb = hoff[b]; r0 = row0[p]
        cemb = np.zeros((m, 128), np.float32); cok = np.zeros(m, np.bool_)
        for i in range(m):
            e = ce[ca + i]
            if e >= 0:
                ss = 0.0
                for d in range(128):
                    v = np.float32(table[e, d]); cemb[i, d] = v; ss += v * v
                if ss > 0:
                    inv = np.float32(1.0 / np.sqrt(ss))
                    for d in range(128): cemb[i, d] *= inv
                    cok[i] = True
        citem = ci[ca:ca + m]; ccat = cc[ca:ca + m]; cs = np.sort(citem)
        pc = -1  # oldest (largest position) new event on a candidate item
        for j in range(k):
            it = hi[hb + j]; q = np.searchsorted(cs, it)
            if q < m and cs[q] == it: pc = j
        ev = np.empty(256, np.int64)
        for which in range(2):
            n = 0
            for j in range(k):
                it = hi[hb + j]; q = np.searchsorted(cs, it)
                if q < m and cs[q] == it: continue
                pre = (pc < 0) or (j > pc)
                if (which == 0 and pre) or (which == 1 and not pre):
                    ev[n] = hb + j; n += 1
            meta[p, which] = n
            set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, which * NF)
        meta[p, 2] = pc

fm = pd.read_parquet(f'{OUT}/feeds_sel.parquet'); F0 = np.load(f'{OUT}/F.npy', mmap_mode='r'); rows0 = np.load(f'{OUT}/rows.npy')
hm = fm.hold.values; fh = fm[hm].reset_index(drop=True); M = fh.M.values.astype(np.int64); n = len(fh)
rmask = np.repeat(hm, fm.M.values); rows = rows0[rmask]; F0 = np.asarray(F0[rmask]); off = np.concatenate([[0], np.cumsum(M)[:-1]])
f = pd.read_parquet(f'{FL}/feeds.parquet'); ld = lambda x: np.load(f'{FL}/{x}.npy')
d = f'{ROOT}/data/competition/train/item_embeddings'; table = np.fromfile(f'{d}/data.memmap', dtype=np.int8).reshape(json.loads(open(f'{d}/data.meta').read())['shape'])
F2 = np.empty((len(rows), 2 * NF), np.float32); meta = np.zeros((n, 3), np.int64)
y_all = ld('cand_target')
build2(fh.a.values, fh.b.values, fh.k.values.astype(np.int64), f.hist_off.values, f.H.values.astype(np.int64), f.cand_off.values, f.M.values.astype(np.int64), off,
       ld('hist_item_id'), ld('hist_event_type'), ld('hist_event_source'), ld('hist_category_id'), ld('hist_emb_idx'),
       ld('cand_item_id'), ld('cand_category_id'), ld('cand_emb_idx'), table, F2, meta)
assert (F2[:, 15] == 0).all() and (F2[:, NF + 15] == 0).all() and (meta[:, 0] + meta[:, 1] == fh.n_FUT.values).all()
y = y_all[rows]; t = np.load(f'{ROOT}/nn_x/w12/kd/out/teacher_mean11.npy', mmap_mode='r')[rows].astype(np.float64)
fid = np.repeat(np.arange(n), M); mu = np.bincount(fid, t, n) / M; sd = np.sqrt(np.bincount(fid, (t - mu[fid]) ** 2, n) / M)
tz = (t - mu[fid]) / (sd[fid] + 1e-6); tc = t - mu[fid]
print(f'holdout pair feeds {n}; feeds with a candidate interaction among the new events {(meta[:,2]>=0).mean():.4f}; mean n_PRE {meta[:,0].mean():.2f} n_POST {meta[:,1].mean():.2f}; '
      f'feeds with n_PRE>=1 {(meta[:,0]>0).mean():.4f}, n_POST>=1 {(meta[:,1]>0).mean():.4f}')
ctx = np.stack([tz, tc, np.repeat(M, M), np.repeat(fh.HA.values, M), np.repeat(np.log1p(fh.gap.values), M), np.repeat(fh.k.values, M)], 1).astype(np.float32)
c = lambda i: F0[:, i * NF:i * NF + 15]
ARMS = {'P': [c(3)], 'P+FUT': [c(3), c(1)], 'P+PRE': [c(3), F2[:, :15]], 'P+POST': [c(3), F2[:, NF:NF + 15]], 'P+PRE+POST': [c(3), F2[:, :15], F2[:, NF:NF + 15]]}
par = dict(objective='binary', verbose=-1, num_threads=12, learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7, lambda_l2=10.0, bagging_fraction=0.8, bagging_freq=1)
res = {a: np.zeros(n) for a in ARMS}; users = fh.user_id.values; uu = np.unique(users)
for rep in range(3):
    half = np.random.default_rng(1000 + rep).permutation(len(uu)) < len(uu) // 2; rf = half[np.searchsorted(uu, users)].astype(int)[fid]
    for a, parts in ARMS.items():
        X = np.concatenate([ctx] + parts, 1); s = np.empty(len(y))
        for fo in (0, 1):
            tr = rf == 1 - fo; te = rf == fo
            b = LogisticRegression(C=1e3, max_iter=200).fit(np.stack([tz[tr], tc[tr]], 1), y[tr]); init = lambda mk: b.decision_function(np.stack([tz[mk], tc[mk]], 1))
            g = lgb.train(dict(seed=7 + rep, **par), lgb.Dataset(X[tr], y[tr], init_score=init(tr)), 200); s[te] = init(te) + g.predict(X[te], raw_score=True)
        res[a] += ndcg_per_feed(s, y, off, M) / 3
nd_t = ndcg_per_feed(t, y, off, M); print(f'teacher {nd_t.mean():.5f}')
for a in ARMS:
    dlt = res[a] - res['P']; print(f'{a:12s} NDCG {res[a].mean():.5f}  delta vs P {dlt.mean():+.5f} +- {dlt.std(ddof=1)/np.sqrt(n):.5f}')
has = meta[:, 2] >= 0
for nm, mk in (('feeds WITH a candidate interaction in new events', has), ('feeds WITHOUT', ~has)):
    print(nm, int(mk.sum()), ' '.join(f'{a} {(res[a]-res["P"])[mk].mean():+.5f}+-{(res[a]-res["P"])[mk].std(ddof=1)/np.sqrt(mk.sum()):.5f}' for a in ARMS if a != 'P'))
