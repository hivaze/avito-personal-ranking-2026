"""u20wit: shared code of the label-clean witness for the U20 experiment (CPU only; read-only on every project file).

Witness feed sets (train split, all OUTSIDE the 30K holdout, exploration-like by the XPOOL v2 posterior
nn_x/w12/xdet/out/prand_v2.npz, p_random > 0.9; user fold = the train.py --xfold hash):
  W0S  fold-0 users, M <  50   natural short witness (never in u20a's training; unlabelled inputs in u20b; labelled in u20c)
  W0L  fold-0 users, M >= 50   natural long witness  (same)
  H0   the fold-0 feeds of the 30,000 holdout (in nobody's training; anchors the CPU path against the stored GPU files)
  W1S  fold-1 users, M <  50   control (in-sample for a fold-0-excluded model AND for a full model)
  W1L  fold-1 users, M >= 50   control, random 6,000-feed subsample (seed 20261115)
SubSplit = nn_x/w5/data.SplitData restricted to these feeds: same gather / batches / scatter code (inherited, not copied), arrays read
through np.load(mmap_mode='r') at the witness rows only, item table compacted to the rows that are used.
Domain token: 'tr' = as in training and as train.py scores the holdout and dump_teacher.py dumps the teacher
  ([1 - p, p, 0] with p = eda/labels_position/out/train_prandom.npz p_random, NaN -> 0, 0 for M < 50);
  'rl' = the val / test convention (no dom in the batch -> model.infer_domain = 1, random-like one-hot)."""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = str(min(12, int(os.environ.get(k, '12') or 12)))  # never more than 12 threads; a lower exported value is kept
sys.path.insert(0, '<repo>/nn_x/w5')   # data.py, model.py (imported, never edited; common.py is NOT imported: it demands a GPU id)
sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path

ROOT = Path('<repo>'); HERE = ROOT / 'nn_x/w12/next/u20wit'; CACHE = HERE / 'cache'; OUT = HERE / 'out'
FLAT = ROOT / 'data/flat'; COMP = ROOT / 'data/competition'; FEAT = ROOT / 'nn_x/w5/feat_cache'
CKPT = ROOT / 'nn_x/w5/ckpt'; W5P = ROOT / 'nn_x/w5/preds'; W5L = ROOT / 'nn_x/w5/logs'
PR_OLD = ROOT / 'eda/labels_position/out/train_prandom.npz'      # holdout identity, domain token, cooldown pool (train.py l.285-306, 432)
PR_V2 = ROOT / 'nn_x/w12/xdet/out/prand_v2.npz'                   # XPOOL v2 posterior (== eda/labels_position/out/train_prandom_v2.npz)
SETS = ('W0S', 'W0L', 'H0', 'W1S', 'W1L')
for d in (CACHE, OUT): d.mkdir(parents=True, exist_ok=True)

se = lambda d: float(np.std(d) / np.sqrt(len(d)))
fmt = lambda d: f'{np.mean(d):+.5f}+-{se(d):.5f}'
lvl = lambda e: f'{np.mean(e):.5f}'


def user_fold(uid, n=None):
    """train.py l.309-310: user-level 5 folds; the user-0 bucket is split by feed index."""
    uid = uid.astype(np.int64); n = len(uid) if n is None else n
    return np.where(uid == 0, np.arange(n) % 5, (uid * 2654435761 % 4294967296) % 5)


def train_meta():
    """Per train feed: M, cand_off, user id, fold, old posterior (as train.py uses it), v2 posterior, holdout index (train.py l.291-292)."""
    f = pl.read_parquet(FLAT / 'train/feeds.parquet'); n = f.height
    M = f['M'].to_numpy().astype(np.int64); uid = f['user_id'].to_numpy().astype(np.int64)
    p1 = np.load(PR_OLD); assert (p1['fidx'] == np.arange(n)).all()
    pold = np.nan_to_num(p1['p_random'].astype(np.float32), nan=0.0); pold[M < 50] = 0.0
    p2 = np.load(PR_V2); assert (p2['fidx'] == np.arange(n)).all()
    pv2 = np.nan_to_num(p2['p_random'].astype(np.float32), nan=0.0)
    cand = np.where((pold > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return dict(f=f, n=n, M=M, uid=uid, xf=user_fold(uid), pold=pold, pv2=pv2, hold=hold,
                cand_off=f['cand_off'].to_numpy().astype(np.int64), ts=f['timestamp'].to_numpy().astype(np.int64))


def witness_sets(tm=None, write=True):
    """Feed-index lists (train feed indices, sorted). Cached in cache/sets.npz. Label-free: uses M, user id, the two posteriors and the holdout list."""
    p = CACHE / 'sets.npz'
    if p.exists():
        z = np.load(p); return {k: z[k] for k in SETS}
    tm = tm or train_meta(); M, xf, pv2 = tm['M'], tm['xf'], tm['pv2']
    isH = np.zeros(tm['n'], bool); isH[tm['hold']] = True
    ex = (pv2 > 0.9) & ~isH
    S = dict(W0S=np.where(ex & (xf == 0) & (M < 50))[0], W0L=np.where(ex & (xf == 0) & (M >= 50))[0], H0=tm['hold'][xf[tm['hold']] == 0],
             W1S=np.where(ex & (xf == 1) & (M < 50))[0])
    w1l = np.where(ex & (xf == 1) & (M >= 50))[0]
    S['W1L'] = np.sort(np.random.default_rng(20261115).choice(w1l, 6000, replace=False))
    if write: np.savez(p, **S)
    return S


class SubSplitBase:
    pass


def build_subsplit(split, feed_ids, cache_dir, max_hist=256, xt_names=None, log=print):
    """Extract the arrays SplitData needs for `feed_ids` (sorted train feed indices) into cache_dir (np.save per array)."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); fid = np.asarray(feed_ids, np.int64); assert (np.diff(fid) > 0).all()
    f = pl.read_parquet(FLAT / split / 'feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64)[fid]; H = f['H'].to_numpy().astype(np.int64)[fid]
    co = f['cand_off'].to_numpy().astype(np.int64)[fid]; ho = f['hist_off'].to_numpy().astype(np.int64)[fid]
    uid = f['user_id'].to_numpy()[fid]
    Hk = np.minimum(H, max_hist)  # history is stored newest-first; gather() reads only the first min(H, max_hist) events
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)])
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, Hk)] + [np.zeros(0, np.int64)])
    mm = lambda name: np.load(FLAT / split / f'{name}.npy', mmap_mode='r')
    sv = lambda name, a: np.save(cache_dir / f'{name}.npy', a)
    sv('src_fidx', fid); sv('rows', rows); sv('M', M); sv('H', H); sv('user_id', uid)
    sv('cand_off', np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)); sv('hist_off', np.concatenate([[0], np.cumsum(Hk)[:-1]]).astype(np.int64))
    sv('user_emb', np.asarray(mm('user_emb')[fid]))
    c_emb = np.asarray(mm('cand_emb_idx')[rows]); h_emb = np.asarray(mm('hist_emb_idx')[hrows])
    used = np.unique(np.concatenate([c_emb[c_emb >= 0], h_emb[h_emb >= 0]]).astype(np.int64))
    meta = json.loads((COMP / split / 'item_embeddings/data.meta').read_text())
    tab = np.memmap(COMP / split / 'item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
    sv('table', np.asarray(tab[used]))
    remap = lambda a: np.where(a >= 0, np.searchsorted(used, np.maximum(a, 0)), -1).astype(np.int32)
    sv('cand_emb_idx', remap(c_emb)); sv('hist_emb_idx', remap(h_emb)); sv('table_src', used)
    log(f'[sub] feeds {len(fid)} cand rows {len(rows)} hist rows {len(hrows)} item rows {len(used)} {time.time() - t0:.0f}s')
    for nme in ('cand_category_id', 'cand_f1', 'cand_f2', 'cand_f3', 'cand_f4', 'cand_f5', 'cand_target', 'cand_pos'):
        sv(nme, np.asarray(mm(nme)[rows]))
    for nme in ('hist_category_id', 'hist_event_type', 'hist_event_source'):
        sv(nme, np.asarray(mm(nme)[hrows]))
    if xt_names:  # same transform as data.SplitData.attach_xt, on the witness rows only; stored fp16 like the GPU pipeline
        cols = []
        for nme in xt_names:
            if nme.startswith('has:'):
                x = (~np.isnan(np.asarray(np.load(FEAT / split / f'{nme[4:]}.npy', mmap_mode='r')[rows]))).astype(np.float32)
            else:
                x = np.asarray(np.load(FEAT / split / f'{nme}.npy', mmap_mode='r')[rows]).astype(np.float32)
            if nme.startswith('has:') or nme.endswith('_rk'): pass
            elif nme.endswith('_z'): x = np.clip(x, -5, 5)
            elif nme.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
            else: x = np.clip(x, -10, 10)
            cols.append(np.nan_to_num(x, nan=0.0).astype(np.float16))
        sv('xt', np.stack(cols, 1)); (cache_dir / 'xt_names.json').write_text(json.dumps(list(xt_names)))
    log(f'[sub] written to {cache_dir} in {time.time() - t0:.0f}s')


def load_subsplit(cache_dir, max_hist=256):
    """A data.SplitData instance (no __init__ call) filled from the cache: gather / batches / scatter are the project's own code."""
    import torch
    from data import SplitData
    cache_dir = Path(cache_dir); ld = lambda n: np.load(cache_dir / f'{n}.npy')
    s = SplitData.__new__(SplitData); dev = torch.device('cpu'); g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    s.split = 'train'; s.device = dev; s.max_hist = max_hist; s.feeds = None
    s.src_fidx = ld('src_fidx'); s.rows = ld('rows')
    s.M_np = ld('M'); s.H_np = ld('H'); s.cand_off_np = ld('cand_off'); s.user_id = ld('user_id'); s.n = len(s.M_np); s.dom = None
    s.M = g(s.M_np); s.H = g(s.H_np); s.cand_off = g(s.cand_off_np); s.hist_off = g(ld('hist_off'))
    s.user_emb = g(ld('user_emb')); s.user0 = g(s.user_id == 0)
    s.c_emb = g(ld('cand_emb_idx')); s.c_item = None; s.h_item = None; s.xt = None; s.n_cand = len(s.c_emb)
    s.c_cat = g(ld('cand_category_id').astype(np.int64)).to(torch.int16)
    s.c_f = torch.stack([g(ld(f'cand_f{i}')) for i in range(1, 6)], dim=1)
    s.has_target = True; s.target_np = ld('cand_target').astype(np.float32); s.c_tgt = g(s.target_np.astype(np.int8))
    s.has_pos = True; s.c_pos = g(ld('cand_pos'))
    s.h_emb = g(ld('hist_emb_idx')); s.h_cat = g(ld('hist_category_id').astype(np.int16)); s.h_et = g(ld('hist_event_type')); s.h_es = g(ld('hist_event_source'))
    s.n_hist = len(s.h_emb); s.table = g(ld('table'))
    if (cache_dir / 'xt.npy').exists():
        s.xt = g(ld('xt')); s.xt_names = json.loads((cache_dir / 'xt_names.json').read_text())
    return s


def load_model(path):
    import torch
    from model import Ranker
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    return m, ck


def score(model, data, feed_ids, dom=None, cand_budget=32768, max_feeds=512, log=None):
    """fp32 CPU forward (eval mode, use_pos False) = common.predict without autocast; returns float32 scores on the sub-split's flat rows
    (rows of feeds not in feed_ids stay NaN). dom: None -> model.infer_domain (random-like), else a [n, 3] float tensor."""
    import torch
    out = torch.full((data.n_cand,), float('nan'), dtype=torch.float32); data.dom = dom; t0 = time.time(); done = 0
    with torch.no_grad():
        bs = data.batches(np.asarray(feed_ids), cand_budget, max_feeds, shuffle=False)
        for i, fb in enumerate(bs):
            b = data.gather(fb); data.scatter(out, b, model(b, use_pos=False)); done += int(data.M_np[fb].sum())
            if log and (i % 50 == 0 or i == len(bs) - 1): log(f'    batch {i + 1}/{len(bs)} rows {done} {time.time() - t0:.0f}s ({done / max(time.time() - t0, 1e-9):.0f} rows/s)')
    data.dom = None
    return out.numpy()


def ndcg_feeds(s, y, off, M):
    from metrics import ndcg_per_feed
    return ndcg_per_feed(np.ascontiguousarray(s, np.float32), y, off, M)
