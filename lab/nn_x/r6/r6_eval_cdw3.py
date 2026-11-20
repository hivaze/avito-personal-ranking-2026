"""Read-out of the r6 WAVE-3 cooldown-LENGTH arms vs the m02 reference (m02 = the default cooldown since 11-12 10:50).
CPU only, read-only. Same code and the same pre-declared stage rules as r6_eval_cdw2.py; only the reference changes from
c0 to m02 (r6_cd_m02_r<root>, F7 v17_pre2s_s<root>@77500 re-cooled 2 epochs with --cooldown_mix 0.02, seed 811).

Arms (nn_x/r6/cd_lane.sh, all with --cooldown_mix 0.02, seed 811, holdout-selected):
  cf4_p60k : SNAP=60000, --epochs 4  -> 40% cooldown at matched total compute (~99.3K vs ~97.2K steps)
  e3       : SNAP=77500, --epochs 3  -> 3-epoch cooldown, total ~107K steps (+10% compute)
  s2c2     : SNAP=77500, --epochs 4 --cooldown_frac 0.5 -> 2 more stable epochs, then the 2-epoch cooldown (~117K = 12 ep;
             added 12:51 under the same rules, before its runs started)

PRE-DECLARED RULES (frozen 2026-11-12 12:46 UTC before any wave-3 run started; CURRENT_STATUS Log):
  Stage 1 (roots 101-104): an arm CONTINUES iff its pooled member delta vs m02 >= 0 on val AND >= 0 on holdout.
  Stage 2 (roots 105-107 added; m02 105-107 exist): PASS-CD iff member val > 2 SE; member holdout >= 0; 7-member
      ensemble vs 7-member m02 ensemble holdout >= 0 and val >= -1 SE; no val slice < -2 SE. Plus the pmix veto
      (src/pmix.py, private-mix member delta < -1 SE blocks adoption).
  Adoption: a PASS-CD arm becomes the default re-cool for F7 roots and the cooldown of new 10-epoch launches
      (cf4 -> --cooldown_frac 0.4; e3 -> re-cools only, since it needs more total compute).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw3.py [stage1|stage2 arm ...]
"""
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r6_eval_cdw2 as W  # noqa: E402

W.WAVE2 = ["cf4_p60k", "e3", "s2c2"]
W.C = lambda r: f"r6:r6_cd_m02_r{r}"  # noqa: E731  the reference replaces c0 everywhere in r6_eval_cdw2


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    os.environ.setdefault("OUT", f"{W.R.R6}/logs/r6_eval_cd_w3{'' if mode == 'stage1' else '_stage2'}.txt")
    sys.argv = [sys.argv[0], mode] + (sys.argv[2:] or W.WAVE2)
    W.main()
    out = os.environ["OUT"]
    txt = open(out).read()
    open(out, "w").write("REFERENCE = m02 (r6_cd_m02_r<root>); every 'c0' label below means the m02 reference\n"
                         + re.sub(r"\bc0\b", "c0[=m02]", txt))


if __name__ == "__main__":
    main()
