"""B0 (infra ts_lr020_s0.cbm, read-only) re-predicted at fixed tree counts, for like-for-like reads of arms that stop at fewer trees.
Writes preds/b0trunc_<N>_{val,hold}.npy and prints the nats line for each N.   usage: python -u b0_trunc.py 100 200 300 400 509"""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker
m = CatBoostRanker(); m.load_model(str(cbx.ROOT / "nn_x/w12/cb/infra/out/ts_lr020_s0.cbm"))
pv, ph = cbx.make_pool(cbx.load("val")), cbx.make_pool(cbx.load("hold"))
for n in [int(x) for x in sys.argv[1:]]:
    a = np.asarray(m.predict(pv, ntree_end=n, thread_count=cbx.THREADS), np.float32); b = np.asarray(m.predict(ph, ntree_end=n, thread_count=cbx.THREADS), np.float32)
    np.save(PREDS / f"b0trunc_{n}_val.npy", a); np.save(PREDS / f"b0trunc_{n}_hold.npy", b)
    if n == 510:
        print("check vs infra predictions (510 trees = best_iter 509 + 1):", float(np.abs(a - np.load(B0V)).max()), float(np.abs(b - np.load(B0H)).max()))
    nats_line(f"b0trunc_{n}", a, b)
