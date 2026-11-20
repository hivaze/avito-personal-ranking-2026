"""(5b) robustness of the 'retrieval' test: cosine of the user vector / history pools to the feed's candidate centroid AFTER removing the
global item mean (averaged vectors share the common direction, which inflates raw cosines of pooled vectors). VAL, label-free."""
import json, numpy as np, polars as pl

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128).astype(np.float32) / 256
m = np.load(OUT + "cache/item_mean_vec.npy").astype(np.float32)
feeds = pl.read_parquet(V + "feeds.parquet"); F = len(feeds)
ce = np.load(V + "cand_emb_idx.npy"); cf = np.load(V + "cand_fidx.npy").astype(np.int64)
he = np.load(V + "hist_emb_idx.npy"); hf = np.load(V + "hist_fidx.npy").astype(np.int64); hp = np.load(V + "hist_hpos.npy")
u = np.load(V + "user_emb.npy").astype(np.float32) / 256
ok = ce >= 0; hok = he >= 0


def seg_mean(X, g, w=None):
    w = np.ones(len(g)) if w is None else w
    S = np.stack([np.bincount(g, weights=X[:, d] * w, minlength=F) for d in range(128)], 1)
    n = np.bincount(g, weights=w, minlength=F)
    return S / np.maximum(n, 1e-9)[:, None], n


C, nc = seg_mean(E[ce[ok]], cf[ok])
Ha, nh = seg_mean(E[he[hok]], hf[hok])
H5, _ = seg_mean(E[he[hok]], hf[hok], np.exp(-hp[hok] / 5.0))
l20 = hok & (hp < 20)
H20, _ = seg_mean(E[he[l20]], hf[l20])
valid = (np.abs(u).sum(1) > 0) & (nh > 0) & (nc > 0)


def mc(a, b, center):
    if center:
        a = a - m; b = b - m
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a[valid] * b[valid]).sum(1).mean())


rng = np.random.default_rng(0); perm = rng.permutation(F)
res = {}
for center in (False, True):
    k = "centered" if center else "raw"
    res[k] = {"user~cand_centroid": mc(u, C, center), "hist_all~cand_centroid": mc(Ha, C, center), "hist_last20~cand_centroid": mc(H20, C, center),
              "hist_rec5~cand_centroid": mc(H5, C, center), "user~hist_last20": mc(u, H20, center), "user~hist_rec5": mc(u, H5, center),
              "shuffled_user~cand_centroid": mc(u[perm], C, center), "shuffled_hist_all~cand_centroid": mc(Ha[perm], C, center)}
res["n_feeds"] = int(valid.sum())
print(json.dumps(res, indent=1))
json.dump(res, open(OUT + "s11_centroid_centered.json", "w"), indent=1)
