"""Label-free, own-feed candidate-candidate features for a split, given the base score s (for softmax masses).
Per row: cos-weighted softmax mass of the other candidates, mass of near-copies (cos>=0.9), same-category mass,
mean cos to the top-5 other candidates by score. Saves ../out/pair_<split>.npz."""
import sys, json, ph, numpy as np
split = sys.argv[1]
D = ph.Data(split)
FL = "<repo>/data/flat/" + split
cemb = np.load(f"{FL}/cand_emb_idx.npy", mmap_mode="r"); cat = np.load(f"{FL}/cand_category_id.npy", mmap_mode="r")
meta = json.load(open(f"<repo>/artifacts/data/competition/{split}/item_embeddings/data.meta"))
items = np.memmap(f"<repo>/artifacts/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
N = len(D.fid)
out = {k: np.zeros(N, np.float32) for k in ("simmass", "dupmass", "catmass", "top5cos", "simmass_hi")}
s = D.s
for f in range(D.n):
    o, m = int(D.off[f]), int(D.M[f])
    v = s[o:o + m]; q = np.exp(v - v.max()); q /= q.sum()
    ce = np.asarray(cemb[o:o + m]); ok = ce >= 0
    X = np.zeros((m, 128), np.float32)
    if ok.any(): X[ok] = items[ce[ok]].astype(np.float32)
    C = (X @ X.T) / 65536.0
    np.fill_diagonal(C, 0.0)
    C[~ok] = 0; C[:, ~ok] = 0
    out["simmass"][o:o + m] = (np.clip(C, 0, None) * q[None, :]).sum(1)
    out["simmass_hi"][o:o + m] = ((C >= 0.8) * q[None, :]).sum(1)
    out["dupmass"][o:o + m] = ((C >= 0.9) * q[None, :]).sum(1)
    c = np.asarray(cat[o:o + m]); same = (c[:, None] == c[None, :]).astype(np.float32); np.fill_diagonal(same, 0)
    out["catmass"][o:o + m] = (same * q[None, :]).sum(1)
    top = np.argsort(-v)[:6]
    Ct = C[:, top]
    # mean over top items excluding self (self has C=0 on diag -> exclude explicitly)
    msk = np.ones_like(Ct, bool); idx = np.nonzero(top[None, :] == np.arange(m)[:, None])
    msk[idx] = False
    cnt = np.minimum(msk.sum(1), 5)
    srt = np.where(msk, Ct, np.nan)
    out["top5cos"][o:o + m] = np.nan_to_num(np.nanmean(srt[:, :], 1))
np.savez(f"../out/pair_{split}.npz", **out)
print(split, {k: (float(v.mean()), float(v.std())) for k, v in out.items()})
