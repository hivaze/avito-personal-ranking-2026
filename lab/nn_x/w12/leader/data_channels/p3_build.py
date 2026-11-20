"""P3 build: content near-neighbour pairs among validation candidate rows (for a validation-label kNN prior), and the
label-free coverage of the same prior on test rows (test row -> validation rows).

Pairs: rows q, r of the same category with cos(e_q, e_r) >= 0.90 (unit item vectors of the val table), q != r, different
feeds. Stored as out/p3_pairs_val.npz (q, r, cos float16). For test: per test row the number of val rows with cos >= 0.95
and >= 0.98 (same category) and the number of val feeds with the exact item (label-free; out/p3_test_cov.npz).
faiss range search, exact inner product, 12 threads.
"""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
import faiss
faiss.omp_set_num_threads(12)
TAU = 0.90


def unit(split):
    tab = np.memmap(f"{ROOT}/artifacts/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r").reshape(-1, 128)
    idx = np.load(f"{FLAT}/{split}/cand_emb_idx.npy")
    e = np.asarray(tab[np.maximum(idx, 0)], np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    e[idx < 0] = 0
    return e, idx >= 0


v = Split("val", hist=False); t = Split("test", hist=False)
ev, okv = unit("val"); et, okt = unit("test")
log("embeddings loaded", ev.shape, et.shape)
Q, R, Cs = [], [], []
cnt95 = np.zeros(t.N, np.int32); cnt98 = np.zeros(t.N, np.int32)
for c in np.unique(v.cat):
    rv = np.flatnonzero((v.cat == c) & okv)
    if len(rv) < 2:
        continue
    index = faiss.IndexFlatIP(128); index.add(ev[rv])
    # val -> val
    lims, D, I = index.range_search(ev[rv], TAU)
    q = np.repeat(np.arange(len(rv)), np.diff(lims).astype(np.int64))
    qa, ra = rv[q], rv[I.astype(np.int64)]
    keep = v.fid[qa] != v.fid[ra]
    Q.append(qa[keep].astype(np.int32)); R.append(ra[keep].astype(np.int32)); Cs.append(D[keep].astype(np.float16))
    # test -> val (coverage only)
    rt = np.flatnonzero((t.cat == c) & okt)
    if len(rt):
        lims, D, I = index.range_search(et[rt], 0.95)
        qq = np.repeat(np.arange(len(rt)), np.diff(lims).astype(np.int64))
        cnt95[rt] = np.bincount(qq, minlength=len(rt))
        cnt98[rt] = np.bincount(qq[D >= 0.98], minlength=len(rt))
    log(f"cat {c:2d} val rows {len(rv):7d} pairs kept {keep.sum():10d} test rows {len(rt):7d}")
Q = np.concatenate(Q); R = np.concatenate(R); Cs = np.concatenate(Cs)
np.savez(f"{OUT}/out/p3_pairs_val.npz", q=Q, r=R, c=Cs)
# exact-item matches test -> val feeds (label-free)
vi = pl.DataFrame({"item": v.item}).group_by("item").len()
ex = pl.DataFrame({"item": t.item}).join(vi, on="item", how="left")["len"].fill_null(0).to_numpy()
np.savez(f"{OUT}/out/p3_test_cov.npz", n95=cnt95, n98=cnt98, nex=ex)
pub = t.public[t.fid]
for nm, a in (("val-row neighbours >=0.95", cnt95), ("val-row neighbours >=0.98", cnt98), ("exact item in val", ex)):
    print(f"test rows with {nm}: public {(a[pub] > 0).mean():.4f} private {(a[~pub] > 0).mean():.4f}")
log("pairs", len(Q), "per val row", len(Q) / v.N)
