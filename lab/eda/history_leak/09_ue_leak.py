"""9) Does the user embedding of a LATER feed of the same user encode the clicks of an earlier feed?
Train triples (P -> A -> B) of consecutive feeds of one user. For A's candidates compare NDCG of cos(ue_X, item_emb)
for X in {A (own), P (previous), B (next)}. If ue is a snapshot built from history, ue_B 'knows' A's clicks."""
import json
import numpy as np
import polars as pl
from common import feeds, arr, ndcg_per_feed


def main():
    f = feeds('train').filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp', 'fidx'])
    u = f['user_id'].to_numpy(); fi = f['fidx'].to_numpy(); ts = f['timestamp'].to_numpy()
    ok = np.where((u[1:-1] == u[:-2]) & (u[1:-1] == u[2:]))[0] + 1   # index of A with a previous and next feed
    rng = np.random.default_rng(0)
    ok = np.sort(rng.choice(ok, min(40000, len(ok)), replace=False))
    P, A, B = fi[ok - 1], fi[ok], fi[ok + 1]
    print('triples', len(A), 'median gap P->A (h)', np.median(ts[ok] - ts[ok - 1]) / 3600, 'A->B', np.median(ts[ok + 1] - ts[ok]) / 3600)
    F = feeds('train').sort('fidx')
    coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    meta = json.load(open('<repo>/data/competition/train/item_embeddings/data.meta'))
    E = np.memmap('<repo>/data/competition/train/item_embeddings/data.memmap', dtype=np.int8, mode='r',
                  shape=tuple(meta['shape']))
    UE = arr('train', 'user_emb')
    cidx = np.concatenate([np.arange(coff[a], coff[a] + M[a]) for a in A])
    m = M[A]
    t = np.asarray(arr('train', 'cand_target')[cidx]).astype(np.float64)
    ei = np.asarray(arr('train', 'cand_emb_idx')[cidx])
    order = np.argsort(ei)
    V = np.zeros((len(ei), 128), np.float32)
    good = ei[order] >= 0
    V[order[good]] = np.asarray(E[ei[order][good]]).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    off = np.concatenate([[0], np.cumsum(m)[:-1]])
    res = {}
    for name, X in [('own A', A), ('previous P', P), ('next B', B)]:
        U = np.asarray(UE[np.sort(X)])[np.argsort(np.argsort(X))].astype(np.float32)
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
        s = (V * np.repeat(U, m, axis=0)).sum(1)
        res[name] = ndcg_per_feed(s, t, off, m)
        print(f'cos(ue_{name}, cand) NDCG on A = {res[name].mean():.4f}')
    d = res['next B'] - res['own A']
    print(f'next - own: {d.mean():+.4f} (SE {d.std()/np.sqrt(len(d)):.4f});  own - previous: {(res["own A"]-res["previous P"]).mean():+.4f}')


if __name__ == '__main__':
    main()
