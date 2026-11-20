"""Validation (information): B9 holdout cv_invd20 (union index) vs on-disk train seq_cv_invd20 (feat_covis per-day index)."""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa
X = np.load(FEATD / "covis_hold.npy")
old = np.asarray(np.load(ROOT / "data/feats/train/seq_cv_invd20.npy", mmap_mode="r")[HOLD_ROWS])
a = X[:, 10].astype(np.float64); o = old.astype(np.float64)
print(f"holdout cv_invd20 (union B9) vs disk train seq_cv_invd20: corr {np.corrcoef(a, o)[0,1]:.5f}; nonzero {np.mean(a>0):.4f} vs {np.mean(o>0):.4f}; "
      f"max abs diff {np.abs(a-o).max():.4g}; share |diff|>1e-4 {np.mean(np.abs(a-o)>1e-4):.5f}")
