"""Nested LIN read of C05 and C11 from the covis_b9.py columns (one LIN per item, all primary columns)."""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa: F401,F403
from covis_b9 import NAMES

LOG = open(HERE / "logs/eval_covis.log", "a")
log = log_to(LOG)
log("==== eval C05 / C11", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
sanity(log)
Xv = np.load(FEATD / "covis_val.npy").astype(np.float64); Xh = np.load(FEATD / "covis_hold.npy").astype(np.float64)
ix = {n: i for i, n in enumerate(NAMES)}
log(f"INFO coverage: val rows with B9 invd20 > 0 {np.mean(Xv[:, ix['cv_invd20']] > 0):.4f}, swing_max > 0 "
    f"{np.mean(Xv[:, ix['swing_max']] > 0):.4f}; holdout {np.mean(Xh[:, ix['cv_invd20']] > 0):.4f} / {np.mean(Xh[:, ix['swing_max']] > 0):.4f}")
ITEMS = {"C05": ["cv_iuf", "cv_idf", "cv_bm25", "cv_invd20_idfdamp", "swing_max"],
         "C11": ["cvx_g1", "cvx_g2", "cvx_g3", "cvx_intent", "cvx_tw"]}
for item in sys.argv[1:] if len(sys.argv) > 1 else ITEMS:
    cols = ITEMS[item]
    res = evaluate(item, cols, [Xv[:, ix[c]] for c in cols], [Xh[:, ix[c]] for c in cols], log)
    with open(HERE / "logs/results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
log("DONE")
