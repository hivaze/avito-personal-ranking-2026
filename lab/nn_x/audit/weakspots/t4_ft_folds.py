"""T4: the FT branch's val OOF comes from ONE fold model per feed, its test prediction is the MEAN of 5 fold models.
How different are the fold models on test? (per-feed Spearman between folds, and fold-vs-mean), compared with the
FT-vs-TO difference. If fold models are near-identical, the OOF = test assumption holds; if not, the FT test branch is
a bagged (stronger, smoother) version of what val measured and the 0.5/0.5 weight is tuned on the wrong object."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta
from presub_gate import per_feed_spearman
P = "<repo>/preds"
f, M, off = feed_meta("test"); M = M.astype(np.int64)
members = ["r6_cd_m02_r101", "r6_cd_m02_r104", "r6_cd_m02_p70k_r102", "v23_gru_pre2s_s501", "v30_f7_m02_s901", "v17_pre2s_s103"]
out = []
def say(s): print(s, flush=True); out.append(s)
for v in members:
    fp = f"{P}/nn_{v}_n2cpv3_test_folds.npy"
    if not os.path.exists(fp): say(f"{v}: no folds file"); continue
    F = np.load(fp).astype(np.float32)  # [5, n_cand] per-feed z of each fold's test prediction
    to = np.load(f"{P}/nn_{v}_trainonly_test.npy").astype(np.float32)
    ft = np.load(f"{P}/nn_{v}_n2cpv3_test.npy").astype(np.float32)
    mean = F.mean(0)
    r_ff = [per_feed_spearman(off, M, F[i], F[j]).mean() for i in range(5) for j in range(i + 1, 5)]
    r_fm = [per_feed_spearman(off, M, F[i], mean).mean() for i in range(5)]
    r_to_ft = per_feed_spearman(off, M, to, ft).mean()
    r_to_f = [per_feed_spearman(off, M, to, F[i]).mean() for i in range(5)]
    say(f"{v:26s} fold-fold rho {np.mean(r_ff):.4f} (min {np.min(r_ff):.4f}) | fold-mean rho {np.mean(r_fm):.4f} | TO-FT(mean) rho {r_to_ft:.4f} | TO-fold rho {np.mean(r_to_f):.4f}")
open("<repo>/nn_x/audit/weakspots/t4_ft_folds.txt", "w").write("\n".join(out) + "\n")
