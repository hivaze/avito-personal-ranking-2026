"""Extra targets for 'dense all-action' training: for each corpus position t, the items of the first R events j>t
with event_type 1 & event_source 1 whose item is new w.r.t. events <= t (distinct items). Output tgt_fe_multi [T,R]
int32 (-1 padded); column 0 equals tgt_fe."""
import numpy as np, numba as nb
from pathlib import Path
PT = Path("<repo>/data/pretrain")
nb.set_num_threads(32)
R = 4


@nb.njit(parallel=True, cache=True)
def multi(off, item, et, es, out):
    for s in nb.prange(len(off) - 1):
        a, b = off[s], off[s + 1]
        L = b - a
        fo = np.empty(L, np.int32)
        for j in range(L):
            fo[j] = j
            x = item[a + j]
            for i in range(j):
                if item[a + i] == x:
                    fo[j] = i
                    break
        for t in range(L):
            n = 0
            for r in range(out.shape[1]):
                out[a + t, r] = -1
            for j in range(t + 1, L):
                if et[a + j] == 1 and es[a + j] == 1 and fo[j] > t:
                    x = item[a + j]
                    dup = False
                    for r in range(n):
                        if out[a + t, r] == x:
                            dup = True
                    if not dup:
                        out[a + t, n] = x
                        n += 1
                        if n == out.shape[1]:
                            break


if __name__ == "__main__":
    off = np.load(PT / "seq_off.npy"); item = np.load(PT / "tok_item.npy"); et = np.load(PT / "tok_et.npy"); es = np.load(PT / "tok_es.npy")
    out = np.empty((len(item), R), np.int32)
    multi(off, item, et, es, out)
    assert (out[:, 0] == np.load(PT / "tgt_fe.npy")).all()
    print("valid per column", (out >= 0).mean(0))
    np.save(PT / "tgt_fe_multi.npy", out)
