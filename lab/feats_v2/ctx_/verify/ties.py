import sys
sys.path.insert(0, "<repo>/feats_v2/ctx_")
import numpy as np
from build_ctx import load_meta, load_item_emb, FLAT, FEATS
split = "val"
feeds, M, H, co, ho = load_meta(split)
T = load_item_emb(split)
cemb = np.load(FLAT / split / "cand_emb_idx.npy"); cu = np.load(FEATS / split / "cos_u.npy"); st = np.load(FEATS / split / "ctx_nd_top5u.npy")
rng = np.random.default_rng(7); fs = rng.choice(np.where((H > 0) & (M > 1))[0], 300, replace=False)
nbad = 0
for fi in fs:
    m = M[fi]; ci = cemb[co[fi]:co[fi] + m]; ok = ci >= 0
    if ok.sum() < 2: continue
    V = np.asarray(T[np.where(ok, ci, 0)]).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-6; V[~ok] = 0
    c = cu[co[fi]:co[fi] + m].astype(np.float64); cus = np.where(ok & ~np.isnan(c), c, -9)
    SS = V @ V.T; order = np.argsort(-cus, kind="stable")
    for i in np.where(ok)[0]:
        top = [j for j in order if j != i and cus[j] > -5][:5]; v = SS[i, top].mean()
        if abs(v - st[co[fi] + i]) > 1e-4:
            nbad += 1
            if nbad <= 3:
                k6 = [j for j in order if j != i][:7]
                print("feed", fi, "M", m, "i", i, "ref", round(v, 4), "stored", round(float(st[co[fi] + i]), 4), "top cos_u", np.round(cus[k6], 5), "emb idx", ci[k6])
print("rows differing", nbad)
