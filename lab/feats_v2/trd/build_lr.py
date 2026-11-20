"""trd_lr_: long-run embedding-neighbourhood popularity and the time-local trend ratio (GPU, CUDA_VISIBLE_DEVICES=N).

Long-run pool L = NLR train history events with hpos < 10 (emb available), sampled uniformly by hash over all train days
(fixed set, identical for every split). For candidate c (unit emb v):
  trd_lr_kde   = 1e4 / |L| * sum_{e in L, user(e) != user(c)} exp(20 (cos(v,e) - 1))     (density per 10K events)
  trd_lr_n90   = 1e4 / |L| * #{e in L: cos >= 0.90}
  trd_lr_max   = max cos(v, e), e in L
  trd_lr_tr_sday / _s6h = log((kde_w_per10k + 0.02) / (trd_lr_kde + 0.02))   trend: today's (6h) population density vs long run,
                 where kde_w_per10k = trd_kde_<w> (per 1K window feeds) rescaled to per 10K events of the split's pool.
"""
import sys, time, os
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *
import torch

NLR = 200_000
CH = 2048


def unit(x):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-6), 0).astype(np.float32)


def lr_pool():
    p = CACHE / "lr_pool.npz"
    if p.exists():
        z = np.load(p); return z["v"], z["u"]
    f = feeds_df("train")
    hp = np.load(FLAT / "train/hist_hpos.npy", mmap_mode="r"); he = np.load(FLAT / "train/hist_emb_idx.npy", mmap_mode="r")
    hf = np.load(FLAT / "train/hist_fidx.npy", mmap_mode="r")
    cand = np.flatnonzero((np.asarray(hp) < 10) & (np.asarray(he) >= 0))
    h = (cand.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(33)
    sel = np.sort(cand[np.argsort(h)[:NLR]])
    v = unit(load_item_emb("train")[np.asarray(he[sel])])
    u = f["user_id"].to_numpy().astype(np.int64)[np.asarray(hf[sel])]
    np.savez(p, v=v, u=u)
    return v, u


def build(split):
    dev = torch.device("cuda")
    t0 = time.time()
    lv, lu = lr_pool()
    LV = torch.from_numpy(lv).to(dev, torch.float16); LU = torch.from_numpy(lu).to(dev)
    f = feeds_df(split)
    uid = f["user_id"].to_numpy().astype(np.int64)
    fidx = np.load(FLAT / split / "cand_fidx.npy"); cemb = np.load(FLAT / split / "cand_emb_idx.npy")
    tbl = load_item_emb(split)
    N = len(fidx)
    kde = np.full(N, np.nan, np.float32); n90 = np.full(N, np.nan, np.float32); mx = np.full(N, np.nan, np.float32)
    for s in range(0, N, CH):
        e = min(N, s + CH)
        qv = torch.from_numpy(unit(tbl[np.maximum(cemb[s:e], 0)])).to(dev, torch.float16)
        qu = torch.from_numpy(uid[fidx[s:e]]).to(dev)[:, None]
        S = (qv @ LV.T).float()
        m = ((LU[None] != qu) | (qu == 0)).float()
        kde[s:e] = ((torch.exp(20 * (S - 1)) * m).sum(1) * (1e4 / NLR)).cpu().numpy()
        n90[s:e] = (((S >= 0.9).float() * m).sum(1) * (1e4 / NLR)).cpu().numpy()
        mx[s:e] = S.max(1).values.cpu().numpy()
        if (s // CH) % 200 == 0:
            print(split, s, N, f"{time.time()-t0:.0f}s", flush=True)
    miss = cemb < 0
    for a in (kde, n90, mx):
        a[miss] = np.nan
    d = FEATS / split
    np.save(d / "trd_lr_kde.npy", kde); np.save(d / "trd_lr_n90.npy", n90); np.save(d / "trd_lr_max.npy", mx)
    # events per pool feed in this split (hpos<10, emb available) to rescale per-1K-feeds densities to per-10K-events
    hp = np.load(FLAT / split / "hist_hpos.npy"); he = np.load(FLAT / split / "hist_emb_idx.npy"); hf = np.load(FLAT / split / "hist_fidx.npy")
    pm = pool_mask(split, f)
    epf = float(((hp < 10) & (he >= 0) & pm[hf]).sum()) / pm.sum()
    for w in ["sday", "s6h"]:
        p = d / f"trd_kde_{w}.npy"
        if p.exists():
            kw = np.load(p) * 10.0 / epf
            np.save(d / f"trd_lr_tr_{w}.npy", np.log((kw + 0.02) / (kde + 0.02)).astype(np.float32))
    print(split, "done", f"{time.time()-t0:.0f}s", "epf", epf, flush=True)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
    for sp in sys.argv[1].split(","):
        build(sp)
