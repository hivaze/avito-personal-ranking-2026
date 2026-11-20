"""Hybrid try file (2026-11-17, lead; owner try mode): fine-tuned branch of one built file + train-only branch of another.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/kd/hyb/hybrid.py <FT tag> <TO tag> <out tag> [build]
Final = 0.5 ens_<FT tag>_ft + 0.5 ens_<TO tag>_to (the same expression the gate and build_r4.py use; each branch is already a per-feed z pool).
Read vs FD16P20kd and FD16P20_m04_02kd on val / pmix-public / pmix-private / M halves; 'build' writes subs/cand_<out tag>_r4.parquet (refuses to overwrite)."""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from make_sub import make_sub
import pmix
from pathlib import Path
P = Path("<repo>/preds"); S = Path("<repo>/subs")
ftag, ttag, out = sys.argv[1:4]; build = "build" in sys.argv[4:]
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy"); W = pmix.weights(); big = Mv >= 50
ld = lambda t, br, sp: np.load(P / f"ens_{t}_{br}_{sp}.npy").astype(np.float64)
nd = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
hv = 0.5 * ld(ftag, "ft", "val") + 0.5 * ld(ttag, "to", "val")
e = nd(hv); print(f"hybrid FT={ftag} TO={ttag}: final val {e.mean():.5f}")
for ref in ("FD16P20kd", "FD16P20_m04_02kd"):
    r = nd(0.5 * ld(ref, "ft", "val") + 0.5 * ld(ref, "to", "val")); d = e - r
    f = lambda x, w=None: "%+.5f +- %.5f" % (pmix.wmean_se(x, w) if w is not None else (x.mean(), x.std(ddof=1) / np.sqrt(len(x))))
    print(f"  vs {ref}: val {f(d)} | pmix-public {f(d, W['public'])} | pmix-private {f(d, W['private'])} | M<50 {f(d[~big])} | M>=50 {f(d[big])}")
if build:
    o = S / f"cand_{out}_r4.parquet"; assert not o.exists(), o
    ht = 0.5 * ld(ftag, "ft", "test") + 0.5 * ld(ttag, "to", "test")
    make_sub(ht.astype(np.float32), str(o)); print("written", o)
