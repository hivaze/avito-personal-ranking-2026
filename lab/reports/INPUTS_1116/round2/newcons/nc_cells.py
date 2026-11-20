"""newcons step 1: FINE CONTENT CELLS (label-free; PRELIST.txt, written 2026-11-16T04:56:50Z).
cell = (existing 4096 spherical k-means cluster, data/clu_work/item_cl_4096_<split>.npy) x (sub-k-means inside the cluster).
  k2(cluster) = clip(round(n_train_rows / 64), 8, 1024); spherical k-means on L2-normalised train item-table rows of the cluster (fit sample: at most 32 * k2 rows),
  8 Lloyd iterations, init = random rows, seed 20261116; empty sub-clusters keep their previous centroid.
Every item-table row of train / val / test is assigned through ITS OWN split's content vector and the same centroids.
Writes (exclusive create) under reports/INPUTS_1116/round2/newcons/out/cells/:
  item_cell_<split>.npy  int32 per item-table row;  hist_cell_<split>.npy  int32 per history event (-1 = no vector);  cand_cell_<split>.npy  int32 per candidate row (-1 = no vector)
  cell_base.npy (first cell id of each of the 4096 clusters, length 4097), sub_cent.npy (float32 [n_cells, 128])
CPU only, 12 threads. No label array is opened."""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
sys.dont_write_bytecode = True
import numpy as np
import torch
torch.set_num_threads(12)
from pathlib import Path

ROOT = Path('<repo>'); COMP = ROOT / 'data/competition'; FLAT = ROOT / 'data/flat'; CW = ROOT / 'data/clu_work'
OUT = ROOT / 'reports/INPUTS_1116/round2/newcons/out/cells'; os.makedirs(OUT, exist_ok=True)
T0 = time.time()
LOGF = open(ROOT / 'reports/INPUTS_1116/round2/newcons/out/cells.log', 'x')


def P(*a):
    s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


def save_new(path, x):
    with open(path, 'xb') as f:
        np.save(f, x)


def table(split):
    d = COMP / split / 'item_embeddings'; meta = json.loads((d / 'data.meta').read_text())
    return np.fromfile(d / 'data.memmap', dtype=np.int8).reshape(meta['shape'])


def norm(x):
    x = torch.from_numpy(np.ascontiguousarray(x)).float()
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


K1 = 4096; TARGET = 64; ITERS = 8
g = torch.Generator().manual_seed(20261116)
tab = table('train'); cl = np.load(CW / 'item_cl_4096_train.npy').astype(np.int64)
assert len(cl) == len(tab) and cl.min() >= 0 and cl.max() == K1 - 1, (cl.min(), cl.max())
P(f'train item table {tab.shape}, clusters min {cl.min()} max {cl.max()}')
order = np.argsort(cl, kind='stable'); cnt = np.bincount(cl, minlength=K1); st = np.concatenate([[0], np.cumsum(cnt)])
k2 = np.clip(np.round(cnt / TARGET).astype(np.int64), 8, 1024); base = np.concatenate([[0], np.cumsum(k2)]).astype(np.int64); NC = int(base[-1])
P(f'cluster sizes (train rows): min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()}; k2 min {k2.min()} median {int(np.median(k2))} max {k2.max()}; cells {NC}')
cent = np.zeros((NC, 128), np.float32); cell_tr = np.full(len(tab), -1, np.int32); cosown = np.zeros(K1)
for c in range(K1):
    rows = order[st[c]:st[c + 1]]; X = norm(tab[rows]); n = len(rows); k = int(k2[c])
    ns = min(n, 32 * k); S = X[torch.randperm(n, generator=g)[:ns]]
    C = S[torch.randperm(ns, generator=g)[:k]].clone()
    if len(C) < k:      # fewer rows than sub-clusters (tiny cluster): pad with copies; duplicates never win an argmax tie against the first copy
        C = torch.cat([C, C[torch.randint(0, len(C), (k - len(C),), generator=g)]])
    for it in range(ITERS):
        a = (S @ C.T).argmax(1)
        N = torch.zeros(k, 128).index_add_(0, a, S); has = torch.bincount(a, minlength=k) > 0
        C[has] = N[has] / N[has].norm(dim=1, keepdim=True).clamp_min(1e-6)
    sim = X @ C.T; mx, a = sim.max(1)
    cell_tr[rows] = (base[c] + a.numpy()).astype(np.int32); cent[base[c]:base[c + 1]] = C.numpy(); cosown[c] = float(mx.mean())
    if c % 512 == 0:
        P(f'cluster {c}: n {n}, k2 {k}, mean cos to own sub-centroid {cosown[c]:.3f}')
assert cell_tr.min() >= 0
sz = np.bincount(cell_tr, minlength=NC)
P(f'train rows assigned; mean cos to own sub-centroid (row-weighted) {np.average(cosown, weights=cnt):.4f}; cell size (train rows): empty {int((sz == 0).sum())}, p5 {np.percentile(sz, 5):.0f} median {np.median(sz):.0f} p95 {np.percentile(sz, 95):.0f} max {sz.max()}')
save_new(OUT / 'cell_base.npy', base); save_new(OUT / 'sub_cent.npy', cent); save_new(OUT / 'item_cell_train.npy', cell_tr)
CELL = {'train': cell_tr}
del tab
centT = torch.from_numpy(cent)
for sp in ('val', 'test'):
    tb = table(sp); c1 = np.load(CW / f'item_cl_4096_{sp}.npy').astype(np.int64); assert len(c1) == len(tb) and c1.min() >= 0
    o = np.argsort(c1, kind='stable'); cn = np.bincount(c1, minlength=K1); s2 = np.concatenate([[0], np.cumsum(cn)]); out = np.full(len(tb), -1, np.int32)
    for c in range(K1):
        if cn[c] == 0: continue
        rows = o[s2[c]:s2[c + 1]]; a = (norm(tb[rows]) @ centT[base[c]:base[c + 1]].T).argmax(1)
        out[rows] = (base[c] + a.numpy()).astype(np.int32)
    assert out.min() >= 0
    save_new(OUT / f'item_cell_{sp}.npy', out); CELL[sp] = out
    P(f'{sp}: item table {tb.shape} assigned; distinct cells used {len(np.unique(out))}')
for sp in ('train', 'val', 'test'):
    for kind in ('hist', 'cand'):
        ei = np.load(FLAT / sp / f'{kind}_emb_idx.npy'); out = np.where(ei >= 0, CELL[sp][np.maximum(ei, 0)], -1).astype(np.int32)
        save_new(OUT / f'{kind}_cell_{sp}.npy', out)
        P(f'{sp} {kind}: rows {len(out)}, without a cell {np.mean(out < 0):.5f}')
P('DONE')
