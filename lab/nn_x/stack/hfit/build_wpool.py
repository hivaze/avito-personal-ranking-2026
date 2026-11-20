"""Weighted family z-mean pool builder (same spec as build_pool.py, family-weighted, separate ft / to weights):
  S_h = sum_f w_h,f * mean_{m in f} z(pred_m,h), h in {ft, to};  final = 0.5*z(S_ft) + 0.5*z(S_to)  (per-feed z).
Member-equal on a half = weights proportional to family sizes.
Usage: build_wpool.py TAG OUT.parquet FTSPEC TOSPEC     SPEC = FAM1:w1,FAM2:w2,...
  -> preds/ens_TAG_{ft,to}_{val,test}.npy (NEW names only; refuses to overwrite) + subs/OUT (must be cand_S*_*.parquet)
Family membership = nn_x/stack/hfit/cache/families.json."""
import os, sys, json
sys.path.insert(0, "<repo>/src")
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub

P = Path("<repo>/preds"); FAMS = json.load(open("<repo>/nn_x/stack/hfit/cache/families.json"))
tag, out = sys.argv[1], sys.argv[2]
assert out.startswith("cand_S") and out.endswith(".parquet"), out
assert not (Path("<repo>/subs") / out).exists(), "refusing to overwrite " + out


def parse(spec):
    W = {k: float(v) for k, v in (t.split(":") for t in spec.split(","))}
    W = {k: v for k, v in W.items() if v > 0}; tot = sum(W.values()); return {k: v / tot for k, v in W.items()}


WS = {"ft": parse(sys.argv[3]), "to": parse(sys.argv[4])}
for h in WS: print(h, "weights", {k: round(v, 4) for k, v in WS[h].items()}, "members", sum(len(FAMS[f]) for f in WS[h]))
for split, ftn, ton in [("val", "n2cpv3_val_oof", "trainonly_val"), ("test", "n2cpv3_test", "trainonly_test")]:
    f, M, o = feed_meta(split)
    z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
    for h, suf in [("ft", ftn), ("to", ton)]:
        fn = P / f"ens_{tag}_{h}_{split}.npy"
        assert not fn.exists(), "refusing to overwrite " + str(fn)
        S = sum(w * np.mean([z(np.load(P / f"nn_{m}_{suf}.npy")) for m in FAMS[fam]], 0) for fam, w in WS[h].items())
        np.save(fn, S.astype(np.float32))
ft_, Mt, ot = feed_meta("test"); zt = lambda n: feed_rank_z(ot, Mt, np.load(P / f"{n}.npy"))[:, 1]
make_sub((0.5 * zt(f"ens_{tag}_ft_test") + 0.5 * zt(f"ens_{tag}_to_test")).astype(np.float32), f"<repo>/subs/{out}")
print("written subs/" + out)
mem = sorted({m for h in WS for fam in WS[h] for m in FAMS[fam]})
print("members (%d):" % len(mem), ",".join(mem))
