"""B3 block (c): label-aware NEAREST-TRAIN-ITEM statistics (hand-made, leave-user-fold-out, time-aware, holdout-free).

Index (a SAMPLE of train items, stated in the report): per 4096-cluster the <= CAP most-shown train items with >= MIN_IMP impressions in
non-holdout train rows and an embedding. For a candidate row the neighbours are the K = 20 index items of ITS OWN 4096-cluster with the highest
cosine to the candidate's embedding (the candidate's own item is skipped, so its own rows never enter).
Neighbour labels: clicks P and expected clicks E (position prior) accumulated per (user fold, day); a row of a feed with fold k on day d reads
folds != k and days < d (never the row's feed, never a feed of the same user, never a same-day or later feed); val / test read all 15 days of the
4 other folds. Columns: kn_coec5, kn_coec20, kn_wcoec20 (cosine-weighted), kn_ex20 (support = expected clicks of the 20), kn_cos20 (mean cosine,
label-free), kn_cos1 (highest cosine, label-free).  Files feat/c_<set>.npy, feat/c_names.json.   usage: python -u b3_build_c.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
import numba as nb

MIN_IMP, CAP, K, A = 5, 1500, 20, 2.0
DAY0, ND, NFOLD = 20702, 15, 5
NAMES = ["kn_coec5", "kn_coec20", "kn_wcoec20", "kn_ex20", "kn_cos20", "kn_cos1"]


@nb.njit(parallel=True, cache=True)
def accumulate(item, y, ex, row_part, pos_of, PE):
    """PE [n_index, 75, 2]; rows are processed in parallel chunks by item hash to avoid races."""
    nth = 64
    for t in nb.prange(nth):
        for r in range(len(item)):
            it = item[r]
            if it % nth != t:
                continue
            p = row_part[r]
            if p < 0:
                continue
            j = pos_of[it]
            if j >= 0:
                PE[j, p, 0] += y[r]; PE[j, p, 1] += ex[r]


@nb.njit(parallel=True, fastmath=True, cache=True)
def query(temb, tcl, titem, tfold, tcut, tgid, cptr, iemb, inorm, iitem, C, TOT, out):
    n = len(tcl)
    for r in nb.prange(n):
        c = tcl[r]
        if c < 0 or cptr[c + 1] == cptr[c]:
            for z in range(6):
                out[r, z] = np.nan
            continue
        v = np.empty(128, np.float32); nv = 0.0
        for q in range(128):
            v[q] = temb[r, q]; nv += v[q] * v[q]
        nv = np.sqrt(nv)
        top_c = np.full(K, -2.0, np.float32); top_i = np.full(K, -1, np.int64)
        for j in range(cptr[c], cptr[c + 1]):
            if iitem[j] == titem[r]:
                continue
            d = np.float32(0.0)
            for q in range(128):
                d += v[q] * iemb[j, q]
            cs = d / (nv * inorm[j]) if nv > 0 else np.float32(0.0)
            if cs > top_c[K - 1]:
                i = K - 1
                while i > 0 and top_c[i - 1] < cs:
                    top_c[i] = top_c[i - 1]; top_i[i] = top_i[i - 1]; i -= 1
                top_c[i] = cs; top_i[i] = j
        f = tgid[r]; k = tfold[f]; dcut = tcut[f]
        p5 = 0.0; e5 = 0.0; p20 = 0.0; e20 = 0.0; wp = 0.0; we = 0.0; sc = 0.0; nn = 0
        for i in range(K):
            j = top_i[i]
            if j < 0:
                break
            P = TOT[j, dcut, 0] - C[j, k, dcut, 0]; E = TOT[j, dcut, 1] - C[j, k, dcut, 1]
            w = max(top_c[i], 0.0)
            p20 += P; e20 += E; wp += w * P; we += w * E; sc += top_c[i]; nn += 1
            if i < 5:
                p5 += P; e5 += E
        if nn == 0:
            for z in range(6):
                out[r, z] = np.nan
            continue
        out[r, 4] = sc / nn; out[r, 5] = top_c[0]
        if dcut == 0:
            out[r, 0] = np.nan; out[r, 1] = np.nan; out[r, 2] = np.nan; out[r, 3] = np.nan
        else:
            out[r, 0] = (p5 + A) / (e5 + A); out[r, 1] = (p20 + A) / (e20 + A); out[r, 2] = (wp + A) / (we + A); out[r, 3] = e20


def main():
    t0 = time.time()
    f = cbx.feeds("train"); uid, ts, Mtr = f["uid"], f["ts"], f["M"]
    S = cbx.train_sets(); n_feeds = len(Mtr)
    hold = np.zeros(n_feeds, bool); hold[S["hold"]] = True
    day = ((ts + 3 * 3600) // 86400 - DAY0).astype(np.int64)
    fold = cbx.xfold_of(uid, np.arange(n_feeds))[0].astype(np.int64)
    part = np.where(hold, -1, fold * ND + day).astype(np.int16)
    row_part = np.repeat(part, Mtr)
    item = np.load(cbx.FLAT / "train/cand_item_id.npy"); nmax = int(item.max()) + 1
    cnt = np.bincount(item[row_part >= 0], minlength=nmax)
    ei = np.load(cbx.FLAT / "train/cand_emb_idx.npy"); cl = np.load(CLU / "cand_cl_4096_train.npy")
    emb_of = np.full(nmax, -1, np.int32); emb_of[item] = ei
    cl_of = np.full(nmax, -1, np.int16); cl_of[item] = cl
    del ei, cl
    cand = np.where((cnt >= MIN_IMP) & (emb_of >= 0) & (cl_of >= 0))[0]
    log(f"items with >= {MIN_IMP} non-holdout impressions, an embedding and a cluster: {len(cand)} of {int((cnt > 0).sum())} train items ({time.time() - t0:.0f}s)")
    # per cluster keep the CAP most-shown
    o = np.lexsort((-cnt[cand], cl_of[cand])); cand = cand[o]; cc = cl_of[cand].astype(np.int64)
    start = np.searchsorted(cc, np.arange(4097)); rank = np.arange(len(cand)) - start[cc]
    keep = rank < CAP; idx_item = cand[keep]; icl = cc[keep]
    cptr = np.searchsorted(icl, np.arange(4097)).astype(np.int64)
    n_idx = len(idx_item)
    log(f"index: {n_idx} items (cap {CAP} per cluster; clusters at the cap {int((np.diff(cptr) == CAP).sum())}); they cover {cnt[idx_item].sum() / cnt.sum():.3f} of non-holdout train impressions")
    pos_of = np.full(nmax, -1, np.int32); pos_of[idx_item] = np.arange(n_idx, dtype=np.int32)
    pc = np.load(FEAT / "pos_prior.npy")
    y = np.load(cbx.FLAT / "train/cand_target.npy").astype(np.float32)
    ex = pc[np.load(cbx.FLAT / "train/cand_pos.npy").astype(np.int64)].astype(np.float32)
    PE = np.zeros((n_idx, NFOLD * ND, 2), np.float32)
    accumulate(item, y, ex, row_part, pos_of, PE)
    log(f"accumulated: clicks {PE[..., 0].sum():.0f} expected {PE[..., 1].sum():.0f} ({time.time() - t0:.0f}s)")
    del item, y, ex, row_part
    PE = PE.reshape(n_idx, NFOLD, ND, 2)
    C = np.zeros((n_idx, NFOLD, ND + 1, 2), np.float32); np.cumsum(PE, axis=2, out=C[:, :, 1:]); del PE
    TOT = C.sum(1)
    iemb8 = np.empty((n_idx, 128), np.int8); cbx._gather_rows_i8(cbx.emb_table("train"), np.ascontiguousarray(emb_of[idx_item]), iemb8)
    iemb = iemb8.astype(np.float32); inorm = np.sqrt((iemb ** 2).sum(1)).astype(np.float32); del iemb8
    log(f"index ready ({time.time() - t0:.0f}s) rss {cbx.rss_gb()[0]:.1f} GB")
    json.dump(NAMES, open(FEAT / "c_names.json", "w"))
    for name in (("hold", "xm") if LAG else ("val", "hold", "test", "xm")):
        d = cbx.load(name); t1 = time.time()
        k = cbx.xfold_of(d.uid_feed, d.fidx)[0].astype(np.int64)
        cut = ((d.ts_feed + 3 * 3600) // 86400 - DAY0).astype(np.int64) if d.split == "train" else np.full(d.n_feeds, ND, np.int64)
        if LAG:
            cut = np.maximum(cut - lag_days(d.fidx) + 1, 0)           # click statistics of the neighbours: days <= own day - L only (the index itself is unchanged)
        # val / test item ids live in the same id space as train (tte_it_* are joined on them); an item of val / test that is an index item is skipped as its own neighbour
        out = np.empty((d.n_rows, 6), np.float32)
        query(np.ascontiguousarray(d.emb_cand_i8), clu(d, 4096).astype(np.int64), np.asarray(d.item_id).astype(np.int64), k, cut, np.asarray(d.gid).astype(np.int64),
              cptr, iemb, inorm, idx_item.astype(np.int64), C, TOT, out)
        np.save(FEAT / f"{PRE}c_{name}.npy", out)
        log(f"[{name}] written {out.shape}; NaN share {np.isnan(out).mean(0).round(4)}; mean top-1 cosine {np.nanmean(out[:, 5]):.3f} ({time.time() - t1:.0f}s)")
    log(f"DONE block c ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
