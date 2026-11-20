"""RESCUE r02: P3 catalogue density in the view D5 (PREDECLARED.txt).
kn1_5 / kn20_5 = highest / mean-top-20 cosine between the candidate's embedding and the INDEX items of its own 4096-cluster (own item skipped);
index of a window = per cluster the <= 1500 most-shown items with >= 5 impressions in non-holdout train feeds OF THE WINDOW, an embedding and a cluster
(nn_x/w12/cb/famB/b3_build_c.py block c, label-free columns kn_cos1 / kn_cos20; there the index was built on all 15 train days for every row).
Label-free: no label array is opened. The index of a row is built from candidate logs of the window only, so its size is the same function of
5 source days for every row of every split.
CPU only. Output: out/arr/<set>__{kn1_5,kn20_5}.npy, log out/r02_density.log."""
import json
from rs_common import *
P = Log(OUT / 'r02_density.log')
V = Views(P)
a = V.A['train']
MIN_IMP, CAP, K = 5, 1500, 20
CLU = ROOT / 'data/clu_work'
EMB = {s: ROOT / f'data/competition/{s}/item_embeddings' for s in SPLITS}


def emb_table(split):
    mt = json.loads((EMB[split] / 'data.meta').read_text()); assert mt['dtype'] == 'int8' and mt['shape'][1] == 128
    return np.memmap(EMB[split] / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(mt['shape']))


@nb.njit(cache=True)
def accum_cnt(item, fidx, fs12, cnt):
    for r in range(len(item)):
        s = fs12[fidx[r]]
        if s >= 0:
            cnt[item[r], s] += 1


@nb.njit(parallel=True, cache=True)
def gather_i8(table, idx, out):
    for i in nb.prange(len(idx)):
        e = idx[i]
        if e >= 0:
            for d in range(128):
                out[i, d] = table[e, d]
        else:
            for d in range(128):
                out[i, d] = 0


@nb.njit(parallel=True, fastmath=True, cache=True)
def knq(temb, tcl, titem, cptr, iemb, inorm, iitem, K, out):
    for r in nb.prange(len(tcl)):
        c = tcl[r]
        if c < 0 or cptr[c + 1] == cptr[c]:
            out[r, 0] = np.nan; out[r, 1] = np.nan
            continue
        v = np.empty(128, np.float32); nv = np.float32(0.0)
        for q in range(128):
            v[q] = temb[r, q]; nv += v[q] * v[q]
        nv = np.sqrt(nv)
        top = np.full(K, -2.0, np.float32); nn_ = 0
        for j in range(cptr[c], cptr[c + 1]):
            if iitem[j] == titem[r]:
                continue
            d = np.float32(0.0)
            for q in range(128):
                d += v[q] * iemb[j, q]
            cs = d / (nv * inorm[j]) if nv > 0 else np.float32(0.0)
            nn_ += 1
            if cs > top[K - 1]:
                i = K - 1
                while i > 0 and top[i - 1] < cs:
                    top[i] = top[i - 1]; i -= 1
                top[i] = cs
        if nn_ == 0:
            out[r, 0] = np.nan; out[r, 1] = np.nan
        else:
            m = min(nn_, K); s = 0.0
            for i in range(m):
                s += top[i]
            out[r, 0] = top[0]; out[r, 1] = s / m


item = np.load(FLAT / 'train/cand_item_id.npy'); fidx = np.load(FLAT / 'train/cand_fidx.npy')
ci = {'train': item, 'val': np.load(FLAT / 'val/cand_item_id.npy'), 'test': np.load(FLAT / 'test/cand_item_id.npy')}
NITEM = int(max(int(x.max()) for x in ci.values())) + 1
fs12 = np.where(V.ish, -1, S12[a['slot']]).astype(np.int64)
cnt = np.zeros((NITEM, 12), np.int32); accum_cnt(item, fidx, fs12, cnt); del fidx
ei = np.load(FLAT / 'train/cand_emb_idx.npy'); cl = np.load(CLU / 'cand_cl_4096_train.npy')
emb_of = np.full(NITEM, -1, np.int32); emb_of[item] = ei
cl_of = np.full(NITEM, -1, np.int32); cl_of[item] = cl
P(f'train items with an embedding {int((emb_of >= 0).sum())}, with a cluster {int((cl_of >= 0).sum())}; impressions counted {int(cnt.sum(dtype=np.int64))}')
ET = {s: emb_table(s) for s in SPLITS}
# query sets: item, cluster, embedding of every row
R = {q: V.rows(q) for q in V.Q}; WK = {q: V.row_wkey(q) for q in V.Q}; QD = {}
for q, (sp, sel, cut) in V.Q.items():
    rows = R[q][0]
    eidx = np.asarray((ei if sp == 'train' else np.load(FLAT / sp / 'cand_emb_idx.npy', mmap_mode='r'))[rows]).astype(np.int64)
    tcl = np.asarray((cl if sp == 'train' else np.load(CLU / f'cand_cl_4096_{sp}.npy', mmap_mode='r'))[rows]).astype(np.int64)
    QD[q] = (np.asarray(ci[sp][rows]).astype(np.int64), tcl, eidx)
    P(f'{q}: rows {len(rows)}, without an embedding {np.mean(eidx < 0):.4f}, without a cluster {np.mean(tcl < 0):.4f}')
del ei, cl, item
res = {q: np.full((len(R[q][0]), 2), np.nan, np.float32) for q in V.Q}
for k in V.wkeys:
    t1 = time.time()
    cols = np.array([j for j in range(12) if (k >> j) & 1], np.int64); cw = cnt[:, cols].sum(1)
    cand = np.nonzero((cw >= MIN_IMP) & (emb_of >= 0) & (cl_of >= 0))[0]
    o = np.lexsort((-cw[cand], cl_of[cand])); cand = cand[o]; cc = cl_of[cand].astype(np.int64)
    start = np.searchsorted(cc, np.arange(4097)); rank = np.arange(len(cand)) - start[cc]
    keep = rank < CAP; idx_item = cand[keep]; icl = cc[keep]; cptr = np.searchsorted(icl, np.arange(4097)).astype(np.int64)
    i8 = np.empty((len(idx_item), 128), np.int8); gather_i8(ET['train'], np.ascontiguousarray(emb_of[idx_item]).astype(np.int64), i8)
    iemb = i8.astype(np.float32); inorm = np.sqrt((iemb ** 2).sum(1)).astype(np.float32); del i8
    P(f'window key {k} (slots {DENSE[cols].tolist()}): items with >= {MIN_IMP} impressions, an embedding and a cluster {len(cand)}; index {len(idx_item)} items, clusters at the cap {int((np.diff(cptr) == CAP).sum())}, non-empty clusters {int((np.diff(cptr) > 0).sum())}; they cover {cw[idx_item].sum() / max(cw.sum(), 1):.3f} of the window impressions ({time.time() - t1:.0f}s)')
    for q, (sp, sel, cut) in V.Q.items():
        m = np.nonzero(WK[q] == k)[0]
        if len(m) == 0:
            continue
        t2 = time.time(); it, tcl, eidx = QD[q]
        t8 = np.empty((len(m), 128), np.int8); gather_i8(ET[sp], np.ascontiguousarray(eidx[m]), t8)
        o_ = np.empty((len(m), 2), np.float32)
        knq(t8, np.ascontiguousarray(tcl[m]), np.ascontiguousarray(it[m]), cptr, iemb, inorm, idx_item.astype(np.int64), K, o_)
        res[q][m] = o_
        P(f'   {q}: {len(m)} rows, NaN {np.isnan(o_[:, 0]).mean():.4f}, mean kn1 {np.nanmean(o_[:, 0]):.4f}, mean kn20 {np.nanmean(o_[:, 1]):.4f} ({time.time() - t2:.0f}s)')
    del iemb, inorm
for q in V.Q:
    put(q, 'kn1_5', res[q][:, 0]); put(q, 'kn20_5', res[q][:, 1])
    v = WK[q] > 0
    P(f'{q}: with a view {int(v.sum())}; NaN among them {np.isnan(res[q][v, 0]).mean():.4f}; kn1 mean {np.nanmean(res[q][v, 0]):.4f} sd {np.nanstd(res[q][v, 0]):.4f}; kn20 mean {np.nanmean(res[q][v, 1]):.4f}')
P('DONE')
