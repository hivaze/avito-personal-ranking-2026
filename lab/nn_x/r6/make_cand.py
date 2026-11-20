raise SystemExit("disabled in the nn_x/r6 copy: writes into subs/ (shared paths)")
"""Write a candidate submission subs/cand_<name>.parquet from a --test spec in the presub_gate format
("w*name+w*name", each preds/<name>.npy z-scored within feed by the gate's own spec_score), so check B reproduces
it exactly.

  python nn/make_cand.py <name> "<test spec>"
"""
import sys

sys.path.insert(0, "<repo>/src")
from common import feed_meta  # noqa: E402
from make_sub import make_sub  # noqa: E402
from presub_gate import spec_score  # noqa: E402

name, spec = sys.argv[1], sys.argv[2]
_, M, off = feed_meta("test")
out = f"<repo>/subs/cand_{name}.parquet"
make_sub(spec_score(spec, off, M.astype("int64")), out)
print("wrote", out)
