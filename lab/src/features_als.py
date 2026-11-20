"""ALS item-item profile features for val/test (collaborative signal that the content embeddings lack).

Item factors: GPU CG-ALS (12 it, event-weighted confidence) on deduplicated train+val+test histories (+ train positives),
with every (user, own val/test candidate) pair removed so no later-snapshot outcome leaks in.
Not built for train rows: the factors saw the same users' later snapshots, which would leak on train.
Features (prefix als_): exp-decay profile cosines (tau 2/5/10), last-10 mean, type-1-only profile, same-category profile,
coverage flag, and within-feed rank/z of the main profile score.
"""
import os, sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

# leak-free factors (pairs of a user with their own val/test candidates removed, src/als_clean_pairs.py); override via ALS_FACTORS
ALS = os.environ.get("ALS_FACTORS", "<repo>/eda/collab_stats/out/alsclean_f64_i12_evt_a5_emb_sub.npz")


@nb.njit(parallel=True, fastmath=True, cache=True)
def _als(cand_off, M, hist_off, H, crow, hrow, het, hcat, ccat, V, out):
    """out: 0 exp2, 1 exp5, 2 exp10, 3 last10, 4 et1_exp5, 5 samecat_exp10, 6 has"""
    D = V.shape[1]
    for f in nb.prange(len(M)):
        p2 = np.zeros(D, np.float32); p5 = np.zeros(D, np.float32); p10 = np.zeros(D, np.float32)
        l10 = np.zeros(D, np.float32); e1 = np.zeros(D, np.float32)
        pc = np.zeros((54, D), np.float32)
        ho = hist_off[f]
        for j in range(H[f]):
            r = hrow[ho + j]
            if r < 0:
                continue
            v = V[r]
            w2 = np.exp(-j / 2.0); w5 = np.exp(-j / 5.0); w10 = np.exp(-j / 10.0)
            p2 += w2 * v; p5 += w5 * v; p10 += w10 * v
            if j < 10:
                l10 += v
            if het[ho + j] == 1:
                e1 += w5 * v
            c = hcat[ho + j]
            if 0 <= c < 54:
                pc[c] += w10 * v
        n2 = np.sqrt((p2 * p2).sum()); n5 = np.sqrt((p5 * p5).sum()); n10 = np.sqrt((p10 * p10).sum())
        nl = np.sqrt((l10 * l10).sum()); ne = np.sqrt((e1 * e1).sum())
        co = cand_off[f]
        for i in range(M[f]):
            r = crow[co + i]
            if r < 0:
                for k in range(6):
                    out[co + i, k] = np.nan
                out[co + i, 6] = 0.0
                continue
            v = V[r]
            out[co + i, 0] = (v * p2).sum() / n2 if n2 > 0 else np.nan
            out[co + i, 1] = (v * p5).sum() / n5 if n5 > 0 else np.nan
            out[co + i, 2] = (v * p10).sum() / n10 if n10 > 0 else np.nan
            out[co + i, 3] = (v * l10).sum() / nl if nl > 0 else np.nan
            out[co + i, 4] = (v * e1).sum() / ne if ne > 0 else np.nan
            pcc = pc[ccat[co + i]]
            npc = np.sqrt((pcc * pcc).sum())
            out[co + i, 5] = (v * pcc).sum() / npc if npc > 0 else np.nan
            out[co + i, 6] = 1.0


NAMES = ["als_exp2", "als_exp5", "als_exp10", "als_last10", "als_et1_exp5", "als_samecat_exp10", "als_has"]


def main():
    nb.set_num_threads(16)
    z = np.load(ALS)
    ids, V = z["item_id"], z["ein"].astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    order = np.argsort(ids); ids, V = ids[order], np.ascontiguousarray(V[order])
    from stack_val import feed_rank_z

    def rows_of(items):
        i = np.minimum(np.searchsorted(ids, items), len(ids) - 1)
        return np.where(ids[i] == items, i, -1).astype(np.int64)

    for split in ["val", "test"]:
        f, M, off = feed_meta(split)
        H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
        crow = rows_of(np.load(FLAT / split / "cand_item_id.npy").astype(np.int64))
        hrow = rows_of(np.load(FLAT / split / "hist_item_id.npy").astype(np.int64))
        out = np.empty((len(crow), len(NAMES)), np.float32)
        _als(off, M.astype(np.int64), ho, H, crow, hrow, np.load(FLAT / split / "hist_event_type.npy").astype(np.int64),
             np.load(FLAT / split / "hist_category_id.npy").astype(np.int64),
             np.load(FLAT / split / "cand_category_id.npy").astype(np.int64), V, out)
        for k, n in enumerate(NAMES):
            np.save(FEATS / split / f"{n}.npy", out[:, k])
        x = np.nan_to_num(out[:, 1], nan=-1.0).astype(np.float32)
        rz = feed_rank_z(off, M, x)
        np.save(FEATS / split / "als_exp5_rk.npy", rz[:, 0]); np.save(FEATS / split / "als_exp5_z.npy", rz[:, 1])
        print(split, "coverage", out[:, 6].mean(), flush=True)


if __name__ == "__main__":
    main()
