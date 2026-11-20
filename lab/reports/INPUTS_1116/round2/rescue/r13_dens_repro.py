"""RESCUE r13: G-code check of the density kernel of r02_density.py. The same kernel (functions copied verbatim from r02) with the index built on ALL dense
train days (the record's index uses all 15 days; slots 9, 10, 14 hold 76 feeds) must reproduce the stored columns of record
nn_x/w12/cb/famB/feat/c_val.npy (kn_cos20 = column 4, kn_cos1 = column 5) on the natural val rows. Label-free.
CPU only, read-only. Output out/r13_dens_repro.txt (exclusive create)."""
import json
from rs_common import *
P = Log(OUT / 'r13_dens_repro.txt')
V = Views(None)
a = V.A['train']
MIN_IMP, CAP, K = 5, 1500, 20
CLU = ROOT / 'data/clu_work'
EMB = {s: ROOT / f'data/competition/{s}/item_embeddings' for s in SPLITS}


def emb_table(split):
    mt = json.loads((EMB[split] / 'data.meta').read_text()); assert mt['dtype'] == 'int8' and mt['shape'][1] == 128
    return np.memmap(EMB[split] / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(mt['shape']))


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


item = np.load(FLAT / 'train/cand_item_id.npy')
NITEM = int(max(int(item.max()), int(np.load(FLAT / 'val/cand_item_id.npy').max()))) + 1
rowok = np.repeat(~V.ish, a['M'])                       # the record: every non-holdout train row, all 15 days
cw = np.bincount(item[rowok], minlength=NITEM); del rowok
ei = np.load(FLAT / 'train/cand_emb_idx.npy'); cl = np.load(CLU / 'cand_cl_4096_train.npy')
emb_of = np.full(NITEM, -1, np.int32); emb_of[item] = ei; cl_of = np.full(NITEM, -1, np.int32); cl_of[item] = cl
del ei, cl, item
cand = np.nonzero((cw >= MIN_IMP) & (emb_of >= 0) & (cl_of >= 0))[0]
o = np.lexsort((-cw[cand], cl_of[cand])); cand = cand[o]; cc = cl_of[cand].astype(np.int64)
start = np.searchsorted(cc, np.arange(4097)); rank = np.arange(len(cand)) - start[cc]
keep = rank < CAP; idx_item = cand[keep]; icl = cc[keep]; cptr = np.searchsorted(icl, np.arange(4097)).astype(np.int64)
i8 = np.empty((len(idx_item), 128), np.int8); gather_i8(emb_table('train'), np.ascontiguousarray(emb_of[idx_item]).astype(np.int64), i8)
iemb = i8.astype(np.float32); inorm = np.sqrt((iemb ** 2).sum(1)).astype(np.float32); del i8
P(f'all-days index: {len(cand)} eligible items, index {len(idx_item)} items, clusters at the cap {int((np.diff(cptr) == CAP).sum())}')
iv = np.load(FLAT / 'val/cand_item_id.npy').astype(np.int64); ev = np.load(FLAT / 'val/cand_emb_idx.npy').astype(np.int64); cv = np.load(CLU / 'cand_cl_4096_val.npy').astype(np.int64)
t8 = np.empty((len(iv), 128), np.int8); gather_i8(emb_table('val'), ev, t8)
out = np.empty((len(iv), 2), np.float32); knq(t8, cv, iv, cptr, iemb, inorm, idx_item.astype(np.int64), K, out)
st = np.load(ROOT / 'nn_x/w12/cb/famB/feat/c_val.npy')
for nm, mine, rec in (('kn_cos1', out[:, 0], st[:, 5]), ('kn_cos20', out[:, 1], st[:, 4])):
    ok = np.isfinite(mine) & np.isfinite(rec); d = np.abs(mine[ok] - rec[ok])
    P(f'{nm}: NaN pattern equal {bool((np.isnan(mine) == np.isnan(rec)).all())}; max abs diff {d.max():.2e}, rows differing by > 1e-4: {int((d > 1e-4).sum())} of {int(ok.sum())}; corr {np.corrcoef(mine[ok], rec[ok])[0, 1]:.6f}')
P('DONE')
