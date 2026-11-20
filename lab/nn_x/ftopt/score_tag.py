"""Score a post-freeze set-A run (out/<tag>/) like confirm.py: TUNE / CONFIRM / ALL pool deltas vs n2cpv3.
Usage: python score_tag.py <tag> [<tag> ...]"""
import json
import sys

import numpy as np

from confirm import evaluate
from ftlib import FT_DIR, SET_A, Val

v = Val()
for tag in sys.argv[1:]:
    ft = [np.load(FT_DIR / "out" / tag / f"nn_{m}_{tag}_val_oof.npy").astype(np.float64) for m in SET_A]
    res = {k: evaluate(v, SET_A, ft, mk, f"set A {tag} {k}") for k, mk in
           (("all", np.ones(v.n, bool)), ("tune", v.tune), ("confirm", v.confirm))}
    json.dump(res, open(FT_DIR / f"score_{tag}.json", "w"), indent=1)
