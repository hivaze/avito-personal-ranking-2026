"""b3_abc_s0 (my model file) re-predicted at fixed tree counts, for like-for-like reads against a seed replicate that stops at fewer trees.
Writes preds/b3trunc_<N>_{val,hold}.npy and prints the nats line.   usage: python -u b3_trunc.py 300 400"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker
m = CatBoostRanker(); m.load_model(str(PREDS / "b3_abc_s0.cbm"))
extra = {}
for b in ("a", "b", "c"):
    for j, n in enumerate(json.load(open(FEAT / f"{b}_names.json"))):
        extra[n] = (lambda d, b=b, j=j: np.load(FEAT / f"{b}_{d.name}.npy", mmap_mode="r")[:, j])
pv, ph = (cbx.make_pool(cbx.load(s), cat="native", extra=extra) for s in ("val", "hold"))
for n in [int(x) for x in sys.argv[1:]]:
    a = np.asarray(m.predict(pv, ntree_end=n, thread_count=cbx.THREADS), np.float32); b = np.asarray(m.predict(ph, ntree_end=n, thread_count=cbx.THREADS), np.float32)
    np.save(PREDS / f"b3trunc_{n}_val.npy", a); np.save(PREDS / f"b3trunc_{n}_hold.npy", b)
    if n == 495:
        print("check vs the arm's own predictions (495 trees):", float(np.abs(a - np.load(PREDS / "b3_abc_s0_val.npy")).max()), float(np.abs(b - np.load(PREDS / "b3_abc_s0_hold.npy")).max()))
    nats_line(f"b3trunc_{n}", a, b)
