"""INVENTORY D step 1: raw column inventory (types, cardinality, missing share, time coverage). CPU only, read-only.
Reads data/flat/<split>/ (bit-exact flattening of the raw parquet by src/prep.py) + data/competition item tables' meta.
Output: reports/INPUTS_1116/raw_fields/d01_inventory.txt"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
import numpy as np, polars as pl, json, datetime as dt
from pathlib import Path
R = Path('<repo>'); F = R / 'data/flat'; OUT = R / 'reports/INPUTS_1116/raw_fields'
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
ts = lambda t: dt.datetime.utcfromtimestamp(int(t)).strftime('%m-%d %H:%M:%S')
for sp in ('val', 'test', 'train'):
    P(f'===== {sp} =====')
    f = pl.read_parquet(F / sp / 'feeds.parquet')
    n = len(f); t = f['timestamp'].to_numpy(); fid = f['feed_id'].to_numpy(); uid = f['user_id'].to_numpy()
    M = f['M'].to_numpy(); H = f['H'].to_numpy()
    P(f'feeds {n} | feed_id min {fid.min()} max {fid.max()} unique {len(np.unique(fid))} | corr(feed_id, timestamp) {np.corrcoef(fid, t)[0,1]:.4f} | feed_id sorted by time monotone share {np.mean(np.diff(fid) > 0):.4f}')
    P(f'timestamp UTC {ts(t.min())} .. {ts(t.max())} | distinct {len(np.unique(t))} | share on 128 s grid (t % 128 mode) {np.bincount(t % 128).max() / n:.4f} mode residue {np.bincount(t % 128).argmax()} | feeds sharing a timestamp with another feed {np.mean(np.isin(t, np.unique(t)[np.unique(t, return_counts=True)[1] > 1])):.4f}')
    u, c = np.unique(uid, return_counts=True)
    P(f'user_id min {uid.min()} max {uid.max()} distinct {len(u)} | user 0 feeds {int((uid == 0).sum())} ({np.mean(uid == 0):.4f}) | non-zero users with >1 feed {int((c[u != 0] > 1).sum())} | feeds of such users {int(c[(u != 0) & (c > 1)].sum())}')
    P(f'M min {M.min()} max {M.max()} mean {M.mean():.2f} | H min {H.min()} max {H.max()} mean {H.mean():.2f} H==256 {np.mean(H == 256):.4f} H==0 {np.mean(H == 0):.4f}')
    N = int(M.sum()); NH = int(H.sum())
    for c_ in ('category_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'emb_idx', 'item_id', 'pos', 'target'):
        p = F / sp / f'cand_{c_}.npy'
        if not p.exists(): P(f'cand_{c_}: absent'); continue
        x = np.load(p, mmap_mode='r'); assert len(x) == N
        if sp == 'train' and c_ in ('f1', 'f2', 'f3', 'f4', 'f5', 'item_id', 'emb_idx'):
            xs = np.asarray(x[:: 1]) if c_ in ('item_id', 'emb_idx') else np.asarray(x[::10])
        else:
            xs = np.asarray(x)
        if xs.dtype.kind == 'f':
            uq = np.unique(xs[~np.isnan(xs)])
            P(f'cand_{c_}: {x.dtype} rows {N} (stats on {len(xs)}) | NaN {np.mean(np.isnan(xs)):.5f} | ==0 {np.mean(xs == 0):.5f} | min {np.nanmin(xs):.4f} max {np.nanmax(xs):.4f} mean {np.nanmean(xs):.4f} | distinct {len(uq)}')
        else:
            uq, cn = np.unique(xs, return_counts=True)
            P(f'cand_{c_}: {x.dtype} rows {N} | min {xs.min()} max {xs.max()} | distinct {len(uq)} | ==0 {np.mean(xs == 0):.5f} | ==-1 {np.mean(xs == -1):.5f} | singletons {np.mean(cn == 1):.4f} of distinct')
        del xs
    for c_ in ('item_id', 'category_id', 'event_type', 'event_source', 'emb_idx'):
        x = np.load(F / sp / f'hist_{c_}.npy', mmap_mode='r'); assert len(x) == NH
        xs = np.asarray(x)
        uq, cn = np.unique(xs, return_counts=True)
        extra = ''
        if len(uq) <= 64: extra = ' | shares ' + ' '.join(f'{a}:{b / NH:.4f}' for a, b in zip(uq, cn))
        P(f'hist_{c_}: {x.dtype} rows {NH} | min {xs.min()} max {xs.max()} | distinct {len(uq)} | ==0 {np.mean(xs == 0):.5f} | ==-1 {np.mean(xs == -1):.5f}{extra}')
        del xs
    meta = json.loads((R / f'data/competition/{sp}/item_embeddings/data.meta').read_text())
    ce = np.asarray(np.load(F / sp / 'cand_emb_idx.npy', mmap_mode='r')); he = np.asarray(np.load(F / sp / 'hist_emb_idx.npy', mmap_mode='r'))
    ref = np.zeros(meta['shape'][0], np.int8); ref[ce[ce >= 0]] |= 1; ref[he[he >= 0]] |= 2
    P(f'item table rows {meta["shape"][0]} | referenced by cand only {np.mean(ref == 1):.4f} hist only {np.mean(ref == 2):.4f} both {np.mean(ref == 3):.4f} unreferenced {np.mean(ref == 0):.5f}')
    del ce, he, ref
(OUT / 'd01_inventory.txt').write_text('\n'.join(L) + '\n')
