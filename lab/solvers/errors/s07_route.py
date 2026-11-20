"""Observable-segment routing of the FT/TO weight in v11 (nested over nn/finetune.py user folds).
score = w * z(FT5) + (1-w) * z(TO5); w in {0.25, 0.5, 0.75} chosen per segment on 4 folds, applied to the 5th."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
fold_of = np.zeros(n, int)
for k, (_, ho) in enumerate(folds):
    fold_of[ho] = k
for fam, (a, b) in {"nn_final5": ("nn_final5_val_oof", "nn_final5_trainonly_val"), "nn_final5cd12": ("nn_final5cd12_val_oof", "nn_final5cd12_trainonly_val"),
                    "nn_final4": ("nn_final4_val_oof", "nn_final4_trainonly_val")}.items():
    za, zb = fz(np.load(f"{ROOT}/preds/{a}.npy"), fid, M), fz(np.load(f"{ROOT}/preds/{b}.npy"), fid, M)
    W = [0.0, 0.25, 0.5, 0.75, 1.0]
    nd = {w: ndcg_per_feed(w * za + (1 - w) * zb, y, off, M) for w in W}
    base = nd[0.5]
    print(f"{fam}: " + "  ".join(f"w_FT={w}: {nd[w].mean():.5f} ({(nd[w] - base).mean():+.5f}+-{(nd[w] - base).std() / np.sqrt(n):.5f})" for w in W))
    segs = {"M<50|M>=50": np.where(M < 50, 0, 1), "coldshare<.3|>=.3": np.where(pf["coldshare"].to_numpy() < 0.3, 0, 1),
            "user new|train|u0": np.where(pf["u0"].to_numpy(), 2, np.where(pf["cold_user"].to_numpy(), 0, 1)),
            "H<129|>=129": np.where(pf["H"].to_numpy() < 129, 0, 1)}
    for sn, lab in segs.items():
        msg = []
        for g in np.unique(lab):
            m = lab == g
            msg.append(f"seg{g}(" + ",".join(f"{w}:{(nd[w] - base)[m].mean():+.4f}" for w in (0.25, 0.75, 1.0)) + ")")
        # nested routing over {0.25,0.5,0.75}
        Wc = [0.25, 0.5, 0.75]
        D = np.stack([nd[w] - base for w in Wc])
        nest = np.zeros(n)
        for k in range(5):
            tr = fold_of != k
            for g in np.unique(lab):
                m = lab == g
                j = int(np.argmax(D[:, tr & m].mean(1)))
                nest[(fold_of == k) & m] = D[j, (fold_of == k) & m]
        print(f"   {sn:20s} {' '.join(msg)} | nested routed {nest.mean():+.5f}+-{nest.std() / np.sqrt(n):.5f}")
