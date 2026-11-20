"""famB_review: own replication of the study's 'default native CTR (15 borders) is blind near a 1% click rate' fact (synthetic, CPU, 4 threads)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from catboost import CatBoostClassifier, Pool
rg = np.random.default_rng(42); K = 200; rate = np.linspace(0.004, 0.03, K)
def make(n):
    k = rg.integers(0, K, n); return k, (rg.random(n) < rate[k]).astype(np.int8)
ktr, ytr = make(600000); kev, yev = make(200000)
import pandas as pd
P = lambda k, y: Pool(pd.DataFrame({"key": k.astype(np.int64)}), label=y, cat_features=["key"], thread_count=4)
ll = lambda p, y: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
print("constant-rate Logloss %.5f oracle %.5f" % (ll(np.full(len(yev), ytr.mean()), yev), ll(rate[kev], yev)))
C255 = ["Borders:CtrBorderCount=255:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1", "Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1"]
for tag, kw in (("default CTR", {}), ("CtrBorderCount=255", dict(simple_ctr=C255, combinations_ctr=C255)), ("one_hot (one_hot_max_size=255)", dict(one_hot_max_size=255))):
    m = CatBoostClassifier(loss_function="Logloss", iterations=200, learning_rate=0.1, depth=6, random_seed=0, verbose=0, allow_writing_files=False, thread_count=4, task_type="CPU", **kw)
    m.fit(P(ktr, ytr))
    p = m.predict_proba(P(np.arange(K), np.zeros(K, np.int8)), thread_count=4)[:, 1]
    pe = m.predict_proba(P(kev, yev), thread_count=4)[:, 1]
    print(f"{tag:32s} eval Logloss {ll(pe, yev):.5f} | distinct predicted levels over the 200 keys {len(np.unique(np.round(p, 7)))} | corr(pred, true rate) {np.corrcoef(p, rate)[0, 1]:.3f}", flush=True)
