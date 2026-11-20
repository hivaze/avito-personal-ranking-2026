"""Timing table: seconds per boosting iteration of CatBoost on CPU (12 threads) on the 134,118-feed exploration set
(19,741,607 rows x 177 features, category_id native categorical, border_count 254, pool quantized once and reused).
One line per configuration is appended to out/timing.jsonl as soon as it is measured.
usage: CUDA_VISIBLE_DEVICES="" python cb_timing.py [--iters 10] [--only substring]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from cbx import log

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--only", default="", help="comma-separated substrings of config names to run")
ap.add_argument("--cat", default="native", choices=["native", "numeric"], help="category_id as a categorical feature (record) or as a number")
ap.add_argument("--out", default="timing.jsonl")
ap.add_argument("--extra_cat", default="", help="comma list among item_id,user_id: extra native categorical id features (CTR cost)")
a = ap.parse_args()
from catboost import CatBoostRanker, CatBoostRegressor

OUT = cbx.HERE / "out" / a.out
t0 = time.time()
tr = cbx.load("xm")
mem = {"start": cbx.rss_gb()[0]}
XC = [c for c in a.extra_cat.split(",") if c]
pool = cbx.make_pool(tr, label=tr.y, cat=a.cat, extra={c: None for c in XC} or None, extra_cat=XC or None)
SUF = ("" if a.cat == "native" else "_catnum") + "".join("+" + c for c in XC)
mem["pool_raw"] = cbx.rss_gb()[0]
t_pool = time.time() - t0
log(f"pool {pool.num_row()} x {pool.num_col()} built in {t_pool:.0f}s rss {cbx.rss_gb()}")
tq = time.time()
cbx.quantize(pool, border_count=254)  # Pool.quantize ignores thread_count: pinned to 12 cores
t_quant = time.time() - tq
mem["pool_quantized"] = cbx.rss_gb()[0]
log(f"quantized in {t_quant:.0f}s rss {cbx.rss_gb()}")


class Clock:
    def __init__(self): self.t = []

    def after_iteration(self, info):
        self.t.append(time.time())
        return True


def run(name, cls, n_it, **kw):
    if a.only and not any((name == o[:-1]) if o.endswith("$") else (o in name) for o in a.only.split(",")):
        return
    name = name + SUF
    c = Clock()
    m = cls(iterations=n_it, learning_rate=0.08, l2_leaf_reg=5, random_seed=0, thread_count=cbx.THREADS,
            allow_writing_files=False, **kw)
    t1 = time.time()
    try:
        m.fit(pool, verbose=0, callbacks=[c])
    except Exception as e:  # noqa: BLE001
        log(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}")
        with open(OUT, "a") as f:
            f.write(json.dumps(dict(name=name, error=str(e)[:300])) + "\n")
        return
    wall = time.time() - t1
    d = np.diff(np.r_[t1, c.t])
    spi = float(np.median(d[2:])) if len(d) > 3 else float(d[-1])
    r = dict(name=name, iters=n_it, s_per_iter=spi, s_per_iter_mean=float(d[2:].mean()) if len(d) > 3 else spi, first_iter_s=float(d[0]),
             setup_s=float(wall - d.sum()), rows=int(pool.num_row()), rows_per_s=float(pool.num_row() / spi),
             rss_gb=cbx.rss_gb()[0], peak_rss_gb=cbx.rss_gb()[1], params={k: str(v) for k, v in kw.items()})
    log(f"{name:34s} {spi:7.2f} s/iter  {r['rows_per_s'] / 1e6:6.2f} M rows/s  first {d[0]:.1f}s  rss {r['rss_gb']:.1f} GB (peak {r['peak_rss_gb']:.1f})")
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    del m


with open(OUT, "a") as f:
    f.write(json.dumps(dict(name="_pool", t_pool_s=t_pool, t_quantize_s=t_quant, mem_gb=mem, rows=int(pool.num_row()),
                            load=os.getloadavg(), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
n = a.iters
# the 3 x 3 table
for depth in (8, 6, 10):
    ni = n if depth < 10 else max(6, n - 3)
    run(f"YetiRank_d{depth}", CatBoostRanker, ni, loss_function="YetiRank", depth=depth)
    run(f"QuerySoftMax_d{depth}", CatBoostRanker, ni, loss_function="QuerySoftMax", depth=depth)
    run(f"RMSE_d{depth}", CatBoostRegressor, ni, loss_function="RMSE", depth=depth)
# speed levers at depth 8 (YetiRank unless stated)
run("YetiRank_d8_rsm0.3", CatBoostRanker, n, loss_function="YetiRank", depth=8, rsm=0.3)
run("YetiRank_d8_rsm0.1", CatBoostRanker, n, loss_function="YetiRank", depth=8, rsm=0.1)
run("YetiRank_d8_bern0.5", CatBoostRanker, n, loss_function="YetiRank", depth=8, bootstrap_type="Bernoulli", subsample=0.5)
run("YetiRank_d8_bern0.5_group", CatBoostRanker, n, loss_function="YetiRank", depth=8, bootstrap_type="Bernoulli", subsample=0.5,
    sampling_unit="Group")
run("YetiRank_d8_perm3", CatBoostRanker, n, loss_function="YetiRank:permutations=3", depth=8)
run("YetiRank_d8_rsm0.3_bern0.5", CatBoostRanker, n, loss_function="YetiRank", depth=8, rsm=0.3, bootstrap_type="Bernoulli", subsample=0.5)
run("RMSE_d8_rsm0.3", CatBoostRegressor, n, loss_function="RMSE", depth=8, rsm=0.3)
run("RMSE_d8_mvs", CatBoostRegressor, n, loss_function="RMSE", depth=8, bootstrap_type="MVS", subsample=0.5)
run("YetiRankPairwise_d8", CatBoostRanker, max(4, n // 2), loss_function="YetiRankPairwise", depth=8)
run("Logloss_like_RMSE_d8_lossguide64", CatBoostRegressor, n, loss_function="RMSE", grow_policy="Lossguide", num_leaves=64, depth=16)
log(f"DONE {time.time() - t0:.0f}s")
