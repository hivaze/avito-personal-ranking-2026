"""Compare fine-tune reruns (out_ft/<path>path__<v>_n2cpv3_*) with the saved preds/nn_<v>_n2cpv3_* (val OOF and test),
and the nn-path rerun with the r6-path rerun. Fine-tuning is not bit-deterministic on the GPU (atomics in backward), so
the yardstick is symmetry: a test-side pipeline difference would make the test agreement worse than the val agreement."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>/src")
from common import feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

R = "<repo>"
O = Path(__file__).resolve().parent / "out_ft"
meta = {sp: feed_meta(sp) for sp in ("val", "test")}
yv = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)


def fz(sp, x):
    _, M, off = meta[sp]
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]


def pc(sp, a, b):  # mean per-feed Pearson of per-feed z-scores
    _, M, off = meta[sp]
    return float((np.add.reduceat((fz(sp, a) * fz(sp, b)).astype(np.float64), off) / M).mean())


for v in ("r6_cd_m02_r101", "v17_pre2s_s101"):
    srcs = {"saved": {"val": np.load(f"{R}/preds/nn_{v}_n2cpv3_val_oof.npy"), "test": np.load(f"{R}/preds/nn_{v}_n2cpv3_test.npy")}}
    for p in ("nn", "r6"):
        fv, ft = O / f"nn_{p}path__{v}_n2cpv3_val_oof.npy", O / f"nn_{p}path__{v}_n2cpv3_test.npy"
        if fv.exists() and ft.exists():
            srcs[f"{p}path"] = {"val": np.load(fv), "test": np.load(ft)}
    _, M, off = meta["val"]
    print(f"== {v}: sources {list(srcs)}")
    for k, s in srcs.items():
        print(f"  {k:7s} val OOF ndcg {ndcg_per_feed(s['val'], yv, off, M).mean():.5f}")
    ks = list(srcs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = srcs[ks[i]], srcs[ks[j]]
            dn = ndcg_per_feed(a['val'], yv, off, M) - ndcg_per_feed(b['val'], yv, off, M)
            print(f"  {ks[i]:7s} vs {ks[j]:7s}: 1 - per-feed Pearson val {1 - pc('val', a['val'], b['val']):.2e} "
                  f"test {1 - pc('test', a['test'], b['test']):.2e} | bitexact val {np.array_equal(a['val'], b['val'])} "
                  f"test {np.array_equal(a['test'], b['test'])} | val ndcg delta {dn.mean():+.5f} (SE {dn.std() / np.sqrt(len(dn)):.5f})")
