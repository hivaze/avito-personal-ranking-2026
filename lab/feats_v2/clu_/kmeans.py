"""Spherical k-means on item and user embeddings (torch, single GPU: run with CUDA_VISIBLE_DEVICES=N).

Item centroids: K in (256, 1024, 4096), fitted on a mixed sample of candidate and history occurrences from all splits
(occurrence-weighted, so frequent items count more). Every row of every split's item table gets a cluster id per K and
its cosine to the assigned centroid. Missing embeddings (emb_idx = -1) are handled downstream (cluster = -1).
User centroids: K in (64, 256, 1024), fitted on non-zero user vectors (train sample + all val + all test).
Also fits a PCA (top-16) on the normalized item sample and stores the projection matrix.

Outputs in data/clu_work/:
  item_cent_{K}.npy                 float32 [K,128] unit-norm centroids
  item_cl_{K}_{split}.npy           int16  [n_table_rows]  cluster per item-table row
  item_ccos_{K}_{split}.npy         float16 [n_table_rows] cosine to own centroid
  user_cent_{K}.npy, user_cl_{K}_{split}.npy (int16 per feed, -1 for zero vectors), user_ccos_{K}_{split}.npy
  pca_mean.npy, pca_comp.npy        (16 x 128)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
WORK = ROOT / "clu_work"
WORK.mkdir(exist_ok=True)
dev = torch.device("cuda:0")
rng = np.random.default_rng(0)
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)


def item_table(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def to_unit(x_int8):
    x = torch.as_tensor(np.ascontiguousarray(x_int8), device=dev).float()
    n = x.norm(dim=1, keepdim=True)
    return x / n.clamp_min(1e-6), (n[:, 0] > 0)


def spherical_kmeans(X, K, iters=25, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = X.shape[0]
    C = X[torch.randperm(n, device=dev, generator=g)[:K]].clone()
    for it in range(iters):
        sums = torch.zeros_like(C)
        cnt = torch.zeros(K, device=dev)
        obj = 0.0
        for s in range(0, n, 262144):
            xb = X[s:s + 262144]
            sim = xb @ C.T
            mv, a = sim.max(1)
            obj += mv.sum().item()
            sums.index_add_(0, a, xb)
            cnt.index_add_(0, a, torch.ones_like(mv))
        empty = cnt == 0
        newC = sums / sums.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if empty.any():
            ne = int(empty.sum())
            newC[empty] = X[torch.randint(0, n, (ne,), device=dev, generator=g)]
        C = newC
        if it % 5 == 0 or it == iters - 1:
            log(f"  K={K} it={it} mean_cos={obj/n:.4f} empty={int(empty.sum())} min/max cnt={int(cnt.min())}/{int(cnt.max())}")
    return C


def assign(Cs, table, chunk=2_000_000):
    """Cs: list of centroid tensors. Returns per K (cluster int16, cos float16)."""
    n = table.shape[0]
    outs = [(np.empty(n, np.int16), np.empty(n, np.float16)) for _ in Cs]
    for s in range(0, n, chunk):
        xb, ok = to_unit(table[s:s + chunk])
        for (cl, cc), C in zip(outs, Cs):
            mv, a = (xb @ C.T).max(1)
            a = torch.where(ok, a, torch.full_like(a, -1))
            cl[s:s + chunk] = a.to(torch.int16).cpu().numpy()
            cc[s:s + chunk] = mv.half().cpu().numpy()
    return outs


def main():
    # ---------------- item sample (occurrence-weighted mix)
    parts = []
    plan = {"train": (1_000_000, 300_000), "val": (500_000, 150_000), "test": (700_000, 250_000)}
    for split, (nc, nh) in plan.items():
        tab = item_table(split)
        ce = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")
        he = np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r")
        ic = ce[np.sort(rng.integers(0, len(ce), nc))]
        ih = he[np.sort(rng.integers(0, len(he), nh))]
        idx = np.concatenate([ic, ih])
        idx = np.sort(idx[idx >= 0])
        parts.append(np.asarray(tab[idx]))
        log("sampled", split, len(idx))
    X, ok = to_unit(np.concatenate(parts))
    X = X[ok]
    log("item sample", X.shape)
    # PCA
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    ev, evec = torch.linalg.eigh(cov)
    comp = evec[:, -16:].flip(1).T.contiguous()
    log("PCA var ratio top16", (ev.flip(0)[:16] / ev.sum()).cpu().numpy().round(4))
    np.save(WORK / "pca_mean.npy", mu.cpu().numpy()); np.save(WORK / "pca_comp.npy", comp.cpu().numpy())
    del Xc
    Ks = [256, 1024, 4096]
    Cs = []
    for K in Ks:
        C = spherical_kmeans(X, K, iters=30 if K < 4096 else 25, seed=K)
        np.save(WORK / f"item_cent_{K}.npy", C.cpu().numpy())
        Cs.append(C)
    del X
    torch.cuda.empty_cache()
    for split in ["val", "test", "train"]:
        tab = item_table(split)
        outs = assign(Cs, tab)
        for K, (cl, cc) in zip(Ks, outs):
            np.save(WORK / f"item_cl_{K}_{split}.npy", cl)
            np.save(WORK / f"item_ccos_{K}_{split}.npy", cc)
        log("assigned items", split, tab.shape[0])

    # ---------------- users
    ue_tr = np.load(FLAT / "train" / "user_emb.npy", mmap_mode="r")
    samp = np.sort(rng.choice(len(ue_tr), 1_000_000, replace=False))
    U = np.concatenate([np.asarray(ue_tr[samp]), np.load(FLAT / "val" / "user_emb.npy"), np.load(FLAT / "test" / "user_emb.npy")])
    U, ok = to_unit(U)
    U = U[ok]
    log("user sample", U.shape)
    Ku = [64, 256, 1024]
    UCs = []
    for K in Ku:
        C = spherical_kmeans(U, K, iters=30, seed=K + 1)
        np.save(WORK / f"user_cent_{K}.npy", C.cpu().numpy())
        UCs.append(C)
    for split in ["train", "val", "test"]:
        ue = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
        outs = assign(UCs, ue)
        for K, (cl, cc) in zip(Ku, outs):
            np.save(WORK / f"user_cl_{K}_{split}.npy", cl)
            np.save(WORK / f"user_ccos_{K}_{split}.npy", cc)
        log("assigned users", split, ue.shape[0])


if __name__ == "__main__":
    main()
