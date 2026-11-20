"""B3 offline: 2-level residual k-means semantic IDs of the frozen content embedding (CPU only, no GPU).

  level 1: spherical k-means, K1=256, on L2-normalized embeddings  c1 = argmax_k <e, C1_k>  (|C1_k| = 1)
  level 2: Euclidean k-means, K2=64, on the residuals r = e - C1_{c1}  c2 = argmin_k |r - C2_k|^2
Fit on 2M rows sampled from the train item table (k-means++ init on 50K rows, 25 Lloyd iterations per level);
assign every row of the train/val/test item tables in 1M-row chunks.
Output: nn_x/feat/sid/{split}_codes.npy  int16 [n_items, 2] (c1, c2) aligned with the split's item table rows.

  OMP_NUM_THREADS=24 ../../.venv/bin/python sid.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(24)
COMP = Path("<repo>/data/competition")
OUT = Path(__file__).resolve().parent / "sid"
OUT.mkdir(exist_ok=True)
K1, K2, N_FIT, N_INIT, ITERS = 256, 64, 2_000_000, 50_000, 25


def load_table(split):
    d = COMP / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def normalize(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


def kmeanspp(X, k, g, spherical):
    """k-means++ (D^2 sampling) on X [n,d]."""
    n = X.shape[0]
    C = [X[torch.randint(n, (1,), generator=g)].squeeze(0)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum()
        i = torch.multinomial(p, 1, generator=g)
        c = X[i].squeeze(0)
        C.append(c)
        d2 = torch.minimum(d2, ((X - c) ** 2).sum(1))
    C = torch.stack(C)
    return normalize(C) if spherical else C


def assign(X, C, spherical, chunk=262_144):
    out = torch.empty(X.shape[0], dtype=torch.long)
    cn = (C ** 2).sum(1)
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        out[i:i + chunk] = (x @ C.T).argmax(1) if spherical else (cn[None] - 2 * x @ C.T).argmin(1)
    return out


def lloyd(X, C, spherical, g, name):
    k = C.shape[0]
    for it in range(ITERS):
        a = assign(X, C, spherical)
        S = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k).float()
        empty = cnt == 0
        newC = S / cnt.clamp_min(1)[:, None]
        if empty.any():  # re-seed empty clusters with random points
            newC[empty] = X[torch.randint(X.shape[0], (int(empty.sum()),), generator=g)]
        newC = normalize(newC) if spherical else newC
        shift = (newC - C).norm(dim=1).max().item()
        C = newC
        if it % 5 == 0 or it == ITERS - 1:
            # objective: mean cosine (spherical) / mean squared distance
            obj = (X * C[a]).sum(1).mean().item() if spherical else ((X - C[a]) ** 2).sum(1).mean().item()
            print(f"[{name}] iter {it} obj {obj:.4f} max shift {shift:.4f} empty {int(empty.sum())} "
                  f"size min {int(cnt.min())} max {int(cnt.max())}", flush=True)
    return C


def main():
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    tab = load_table("train")
    n = tab.shape[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, N_FIT, replace=False))
    X = torch.from_numpy(np.asarray(tab[idx]).astype(np.float32))
    nz = X.norm(dim=1) > 0
    print(f"train table {tab.shape}, fit rows {N_FIT}, zero rows in sample {int((~nz).sum())}, "
          f"{time.time() - t0:.0f}s", flush=True)
    X = normalize(X[nz])
    init = X[torch.randperm(X.shape[0], generator=g)[:N_INIT]]
    C1 = lloyd(X, kmeanspp(init, K1, g, True), True, g, "L1")
    a1 = assign(X, C1, True)
    R = X - C1[a1]
    init = R[torch.randperm(R.shape[0], generator=g)[:N_INIT]]
    C2 = lloyd(R, kmeanspp(init, K2, g, False), False, g, "L2")
    r2 = ((R - C2[assign(R, C2, False)]) ** 2).sum(1).mean().item()
    print(f"residual energy: after L1 {(R ** 2).sum(1).mean().item():.4f}, after L2 {r2:.4f}; "
          f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"C1": C1, "C2": C2}, OUT / "centroids.pt")
    for split in ("train", "val", "test"):
        tab = load_table(split)
        n = tab.shape[0]
        codes = np.zeros((n, 2), np.int16)
        for i in range(0, n, 1_000_000):
            x = normalize(torch.from_numpy(np.asarray(tab[i:i + 1_000_000]).astype(np.float32)))
            c1 = assign(x, C1, True)
            c2 = assign(x - C1[c1], C2, False)
            codes[i:i + len(c1), 0] = c1.numpy()
            codes[i:i + len(c1), 1] = c2.numpy()
        np.save(OUT / f"{split}_codes.npy", codes)
        u1 = np.bincount(codes[:, 0], minlength=K1)
        pair = codes[:, 0].astype(np.int64) * K2 + codes[:, 1]
        up = np.bincount(pair, minlength=K1 * K2)
        print(f"{split}: {n} rows; c1 used {int((u1 > 0).sum())}/{K1}, pairs used {int((up > 0).sum())}/{K1 * K2}, "
              f"pair size median {int(np.median(up[up > 0]))} max {int(up.max())}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
