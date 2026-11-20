"""Cache per-feed z-scored member scores: cache/Z_<split>_<half>.npy  (members x rows, float32), MEMBERS order."""
from lib import *
import time

C = OUT / "cache"
C.mkdir(exist_ok=True)
for split, halves in [("val", ["to", "ft"]), ("holdout", ["to"]), ("test", ["to", "ft"])]:
    S = Split(split)
    for half in halves:
        f = C / f"Z_{split}_{half}.npy"
        if f.exists():
            continue
        t = time.time()
        Z = np.stack([S.load(m, half) for m in MEMBERS]).astype(np.float32)
        np.save(f, Z)
        print(split, half, Z.shape, f"{time.time() - t:.0f}s", flush=True)
np.save(C / "members.npy", np.array(MEMBERS))
