"""OPTIONAL (review artefact, NOT run by the reviewer: GPU script): stale-val predictions under scenario 'both_pre2lt3' = both_pre2full with the no-ID encoder score (both heads) also on val rows
whose 'both' counter is 1 or 2 and that carry an item-ID contribution (impossible at test: P(ID | counter in {1,2}) = 0 exactly on val / public / private). Same pattern as
nn_x/w12/stale/hz2n/stale_pred_full.py: nn_x/w5/stale_pred.py imported UNCHANGED, six override entries replaced, output directory redirected.
  cd <repo>/nn_x/w5 && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../.venv/bin/python ../w12/stale/hz2n_review/stale_pred_lt3.py <variant> [<variant> ...]
  -> nn_x/w12/stale/hz2n_review/preds_stale_lt3/nn_<v>_staleval.npy
Files: nn_x/w12/stale/hz2n_review/cf_lt3/lt3_pre2_{fe,fe_z,fe_s,new,new_z,new_s}.npy (build_cf_lt3.py; self-check against the witness files in build_cf_lt3.json)."""
import sys
from pathlib import Path

sys.path.insert(0, "<repo>/nn_x/w5")
import stale_pred as SP  # noqa: E402

LT3 = Path("<repo>/nn_x/w12/stale/hz2n_review/cf_lt3")
SP.OVR.update({n: LT3 / f"lt3_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s", "pre2_new", "pre2_new_z", "pre2_new_s")})
SP.OUT = Path("<repo>/nn_x/w12/stale/hz2n_review/preds_stale_lt3")

if __name__ == "__main__":
    SP.main(sys.argv[1:])
