"""hz2n: stale-val predictions under scenario 'both_pre2full' = nn_x/w5/stale_pred.py (scenario both_pre2both) + the NEW head replaced too on newly cold val rows
(solvers/out/stale/cf/pre2full_{pre2_new,pre2_new_z,pre2_new_s}.npy, built 11-11 by solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2full).
stale_pred.py itself is imported UNCHANGED; this wrapper only adds three entries to its override table and redirects its output directory, so the two scenarios
never share a file name. GPU script (as stale_pred.py): NOT run by the builder (CPU-only task); the first variant's "base check vs stored" line validates the path.

  cd <repo>/nn_x/w5 && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../.venv/bin/python ../w12/stale/hz2n/stale_pred_full.py <variant> [<variant> ...]
  -> nn_x/w12/stale/hz2n/preds_stale_full/nn_<v>_staleval.npy (+ stale_pred.log there). Read it with the same tool as the both_pre2both files.
Why: arms trained on an F directory (both encoder heads without item ID on rows made cold) never see "fe head without ID + new head with ID", which is what
both_pre2both feeds on newly cold val rows; both_pre2full is the scenario in which val rows look like cold test rows in BOTH heads.
"""
import sys
from pathlib import Path

sys.path.insert(0, "<repo>/nn_x/w5")
import stale_pred as SP  # noqa: E402

FULL = {n: SP.CF / f"pre2full_{n}.npy" for n in ("pre2_new", "pre2_new_z", "pre2_new_s")}
assert not set(FULL) & set(SP.OVR)
SP.OVR.update(FULL)
SP.OUT = Path("<repo>/nn_x/w12/stale/hz2n/preds_stale_full")

if __name__ == "__main__":
    SP.main(sys.argv[1:])
