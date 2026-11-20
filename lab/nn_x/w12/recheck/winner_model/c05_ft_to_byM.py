"""winner_model recheck, check 5: where does the val fine-tune (FT) differ from train-only (TO)? W8 and C3 pools by feed size.
Paired per feed; FT = out-of-fold. CPU, existing prediction files only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
P = "<repo>/preds"
fv, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
se = lambda d: d.std() / np.sqrt(len(d))
bins = [(5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 400)]
for tag in ("W8kd", "Q2a8s3"):
    ft, to = np.load(f"{P}/ens_{tag}_ft_val.npy"), np.load(f"{P}/ens_{tag}_to_val.npy")
    eft, eto, emix = ev(ft), ev(to), ev(0.5 * z(ft) + 0.5 * z(to))
    print(f"== {tag}: FT {eft.mean():.5f} TO {eto.mean():.5f} mix {emix.mean():.5f}; FT-TO {(eft-eto).mean():+.5f}+-{se(eft-eto):.5f}; mix-TO {(emix-eto).mean():+.5f}+-{se(emix-eto):.5f}")
    for lo, hi in bins:
        m = (M >= lo) & (M < hi)
        d, d2 = (eft - eto)[m], (emix - eto)[m]
        print(f"   M [{lo:3d},{hi:3d}) n {m.sum():5d} share {m.mean():.3f}: TO {eto[m].mean():.4f} | FT-TO {d.mean():+.5f}+-{se(d):.5f} | mix-TO {d2.mean():+.5f}+-{se(d2):.5f} | loss share (1-NDCG, mix) {(1-emix[m]).sum()/(1-emix).sum():.3f}")
