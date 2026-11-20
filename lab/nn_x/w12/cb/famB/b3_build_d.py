"""B3 block (d): NEAR-DUPLICATE statistics, a sharper form of block (c) (same index, same leak construction as b3_build_c.py: see below).
Columns: dup_n99 / n97 / n95 / n90 = number of index items of the candidate's 4096-cluster with cosine above the threshold (label-free; own item skipped);
kn_cos2 / 5 / 10 = cosine of the 2nd / 5th / 10th nearest index item (label-free); dup_coec97 / dup_coec95 = clicks over expected clicks of the items with
cosine > 0.97 / 0.95 (other user folds, earlier days only), dup_ex97 / dup_ex95 = their expected clicks (support). Files feat/d_<set>.npy, feat/d_names.json.
Motivated by the TRAIN-side per-column screen (b3_uni.py: kn_cos1 / kn_cos20 were the strongest new columns on xm), not by a witness read.
--- docstring of the block (c) builder, whose index and label construction are reused unchanged ---
B3 block (c): label-aware NEAREST-TRAIN-ITEM statistics (hand-made, leave-user-fold-out, time-aware, holdout-free).

Index (a SAMPLE of train items, stated in the report): per 4096-cluster the <= CAP most-shown train items with >= MIN_IMP impressions in
non-holdout train rows and an embedding. For a candidate row the neighbours are the K = 20 index items of ITS OWN 4096-cluster with the highest
cosine to the candidate's embedding (the candidate's own item is skipped, so its own rows never enter).
Neighbour labels: clicks P and expected clicks E (position prior) accumulated per (user fold, day); a row of a feed with fold k on day d reads
folds != k and days < d (never the row's feed, never a feed of the same user, never a same-day or later feed); val / test read all 15 days of the
4 other folds. Columns: kn_coec5, kn_coec20, kn_wcoec20 (cosine-weighted), kn_ex20 (support = expected clicks of the 20), kn_cos20 (mean cosine,
label-free), kn_cos1 (highest cosine, label-free).  Files feat/c_<set>.npy, feat/c_names.json.   usage: python -u b3_build_d.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
import numba as nb

MIN_IMP, CAP, K, A = 5, 1500, 20, 2.0
DAY0, ND, NFOLD = 20702, 15, 5
NAMES = ["dup_n99", "dup_n97", "dup_n95", "dup_n90", "kn_cos2", "kn_cos5", "kn_cos10", "dup_coec97", "dup_ex97", "dup_coec95", "dup_ex95"]
NCOL = len(NAMES)


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
            for z in range(NCOL):
                out[r, z] = np.nan
            continue
        v = np.empty(128, np.float32); nv = 0.0
        for q in range(128):
            v[q] = temb[r, q]; nv += v[q] * v[q]
        nv = np.sqrt(nv)
        f = tgid[r]; k = tfold[f]; dcut = tcut[f]
        top_c = np.full(10, -2.0, np.float32)
        n99 = 0.0; n97 = 0.0; n95 = 0.0; n90 = 0.0; p97 = 0.0; e97 = 0.0; p95 = 0.0; e95 = 0.0
        for j in range(cptr[c], cptr[c + 1]):
            if iitem[j] == titem[r]:
                continue
            d = np.float32(0.0)
            for q in range(128):
                d += v[q] * iemb[j, q]
            cs = d / (nv * inorm[j]) if nv > 0 else np.float32(0.0)
            if cs > 0.90:
                n90 += 1.0
                if cs > 0.95:
                    n95 += 1.0
                    P = TOT[j, dcut, 0] - C[j, k, dcut, 0]; E = TOT[j, dcut, 1] - C[j, k, dcut, 1]
                    p95 += P; e95 += E
                    if cs > 0.97:
                        n97 += 1.0; p97 += P; e97 += E
                        if cs > 0.99:
                            n99 += 1.0
            if cs > top_c[9]:
                i = 9
                while i > 0 and top_c[i - 1] < cs:
                    top_c[i] = top_c[i - 1]; i -= 1
                top_c[i] = cs
        out[r, 0] = n99; out[r, 1] = n97; out[r, 2] = n95; out[r, 3] = n90
        out[r, 4] = top_c[1] if top_c[1] > -2.0 else np.nan
        out[r, 5] = top_c[4] if top_c[4] > -2.0 else np.nan
        out[r, 6] = top_c[9] if top_c[9] > -2.0 else np.nan
        if dcut == 0:
            out[r, 7] = np.nan; out[r, 8] = np.nan; out[r, 9] = np.nan; out[r, 10] = np.nan
        else:
            out[r, 7] = (p97 + A) / (e97 + A); out[r, 8] = e97; out[r, 9] = (p95 + A) / (e95 + A); out[r, 10] = e95


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
    json.dump(NAMES, open(FEAT / "d_names.json", "w"))
    for name in ("val", "hold", "test", "xm"):
        d = cbx.load(name); t1 = time.time()
        k = cbx.xfold_of(d.uid_feed, d.fidx)[0].astype(np.int64)
        cut = ((d.ts_feed + 3 * 3600) // 86400 - DAY0).astype(np.int64) if d.split == "train" else np.full(d.n_feeds, ND, np.int64)
        # val / test item ids live in the same id space as train (tte_it_* are joined on them); an item of val / test that is an index item is skipped as its own neighbour
        out = np.empty((d.n_rows, NCOL), np.float32)
        query(np.ascontiguousarray(d.emb_cand_i8), clu(d, 4096).astype(np.int64), np.asarray(d.item_id).astype(np.int64), k, cut, np.asarray(d.gid).astype(np.int64),
              cptr, iemb, inorm, idx_item.astype(np.int64), C, TOT, out)
        np.save(FEAT / f"d_{name}.npy", out)
        log(f"[{name}] written {out.shape}; NaN share {np.isnan(out).mean(0).round(4)}; mean near-duplicates (cos > 0.97) {np.nanmean(out[:, 1]):.2f} ({time.time() - t1:.0f}s)")
    log(f"DONE block d ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
