"""Within-feed pairwise AUC by pair type (cold/warm positive x cold/warm negative, cold = cold_item flag), leak-free val,
for nn_final2_trainonly, nn_final3_trainonly, nn_final3_val_oof. Shows where the train-stage co-visitation/LUO inputs moved errors."""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from common import *
from qt import leak_mask

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn


D = Path("<repo>/eda/errors/synthesis")
P = Path("<repo>/preds")
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy"); kf = ~leak_mask()
cat = np.load(FLAT / "val/cand_category_id.npy")
cold = np.load(D / "derived/val/cold_item.npy").astype(bool)
pi, ni = [], []
for g in np.nonzero(kf)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p)))
pi = np.concatenate(pi); ni = np.concatenate(ni); same = cat[pi] == cat[ni]
preds = {"f2_tr": np.load(pp("nn_final2_trainonly_val.npy")), "f3_tr": np.load(pp("nn_final3_trainonly_val.npy")),
         "f3_oof": np.load(pp("nn_final3_val_oof.npy"))}
print(f"pairs {len(pi):,}")
for lab, m in [("cold+ vs warm-", cold[pi] & ~cold[ni]), ("warm+ vs cold-", ~cold[pi] & cold[ni]), ("warm+ vs warm-", ~cold[pi] & ~cold[ni]),
               ("cold+ vs cold-", cold[pi] & cold[ni])]:
    row = [f"{lab:15s} share {m.mean():.3f}"]
    for k, s in preds.items():
        d = s[pi[m]] - s[ni[m]]; ds = s[pi[m & same]] - s[ni[m & same]]
        row.append(f"{k} AUC {((d > 0) + 0.5 * (d == 0)).mean():.4f} (same-cat {((ds > 0) + 0.5 * (ds == 0)).mean():.4f})")
    print(" | ".join(row))
