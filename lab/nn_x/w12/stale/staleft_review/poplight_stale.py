"""staleft_review: how much does the stale view cost the popularity-light students (P=0 runs of the study)? CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft_review")
import numpy as np
import rederive as R
V, P = R.V, R.P
R.lines.clear()
for nm, mem in (("pop-light d128 x4 (abl_e10_pop1kd05t25_s951-954)", R.P1), ("W8", R.W8)):
    ftf = V.ndcg(R.zm([R.ld(m, "p0", "val_oof") for m in mem])); fts = V.ndcg(R.zm([R.ld(m, "p0", "val_oof_stale") for m in mem]))
    tof = V.ndcg(R.zm([R.ld(m, "p0", "val_e0") for m in mem])); tos = V.ndcg(R.zm([R.ld(m, "p0", "val_e0_stale") for m in mem]))
    P(f"{nm}: FT pool (standard fine-tune, 1 seed) stale - fresh {R.row(fts - ftf)}")
    P(f"{nm}: TO pool stale - fresh {R.row(tos - tof)}")
(R.OUT / "poplight_stale.txt").write_text("\n".join(R.lines) + "\n")
