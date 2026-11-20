"""Rocchio-style debiased preference directions from train labels (dense, embedding-level collaborative signal).

For a group g (user-embedding cluster, or the candidate's own item cluster, or 'global') accumulate over train candidate rows
  Sp[g] = sum y * v      (v = unit item embedding, y = target)
  Se[g] = sum ex * v     (ex = position-model expected clicks, see tables.py)
The direction d_g = Sp/P - Se/E points from "what this group would click by position alone" to "what it actually clicked".
Per train calendar day, so train rows use strictly earlier days.
Outputs data/clu_work/roc_{name}.npz: Sp, Se [D, G, 128], P, E [D, G]
"""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *


@nb.njit(parallel=True, cache=True)
def accum(r0, r1, grp, cemb, table, y, ex, G, nth):
    Sp = np.zeros((nth, G, 128)); Se = np.zeros((nth, G, 128))
    P = np.zeros((nth, G)); E = np.zeros((nth, G))
    n = r1 - r0
    for t in nb.prange(nth):
        for r in range(r0 + t * n // nth, r0 + (t + 1) * n // nth):
            g = grp[r]; e = cemb[r]
            if g < 0 or e < 0:
                continue
            v = table[e].astype(np.float64)
            v /= np.sqrt((v * v).sum()) + 1e-9
            yy = y[r]; ee = ex[r]
            P[t, g] += yy; E[t, g] += ee
            if yy > 0:
                Sp[t, g] += yy * v
            Se[t, g] += ee * v
    return Sp.sum(0), Se.sum(0), P.sum(0), E.sum(0)


def main():
    nb.set_num_threads(24)
    fe = feeds("train")
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float64)
    ex = np.load(WORK / "train_ex.npy").astype(np.float64)
    cemb = np.load(FLAT / "train/cand_emb_idx.npy")
    d = ROOT / "competition" / "train" / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    table = np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])
    dayf = day_index(fe["ts"])
    # train rows are sorted by (timestamp, feed) -> day ranges of rows
    rstart = np.searchsorted(dayf[fidx], np.arange(D + 1))
    groups = {"g": np.zeros(len(fidx), np.int64)}
    for Ku in (64, 256, 1024):
        groups[f"u{Ku}"] = user_cl("train", Ku)[fidx]
    for K in (256, 1024):
        groups[f"i{K}"] = np.asarray(cand_cl("train", K)).astype(np.int64)
    for name, grp in groups.items():
        G = int(grp.max()) + 1
        Sp = np.zeros((D, G, 128), np.float32); Se = np.zeros((D, G, 128), np.float32)
        P = np.zeros((D, G)); E = np.zeros((D, G))
        for dd in range(D):
            r0, r1 = rstart[dd], rstart[dd + 1]
            if r1 <= r0:
                continue
            a, b, c, e = accum(r0, r1, grp, cemb, table, y, ex, G, 24)
            Sp[dd] = a; Se[dd] = b; P[dd] = c; E[dd] = e
        np.savez(WORK / f"roc_{name}.npz", Sp=Sp, Se=Se, P=P, E=E)
        log("roc", name, G)


if __name__ == "__main__":
    main()
