"""exp12: cross-recipe member-equal pools on the n2cpv3 basis (thin wrapper).
The implementation lives in <repo>/solvers/CPU1_xrecipe_member_equal_pool_v3.py;
outputs go to <repo>/solvers/out/CPU1_xrecipe_member_equal_pool_v3/ (never preds/ or subs/)."""
import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = "<repo>/solvers/CPU1_xrecipe_member_equal_pool_v3.py"
    runpy.run_path(sys.argv[0], run_name="__main__")
