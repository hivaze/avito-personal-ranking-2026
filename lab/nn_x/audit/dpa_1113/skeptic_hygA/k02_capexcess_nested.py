"""Independent re-run of the decisive number for P1 HYG-A mechanism (1): nested residual of the information the
+-10 cap destroys, on the G3 pool.  s' = b + w * g, g = sum over the 4 mass columns of log(max(raw,10)/10)
("cap excess", 0 for uncapped rows) or per column; w chosen on a grid by NDCG on 4 user-grouped folds, scored on the
5th (GroupKFold by user over val feeds); holdout uses the w fit on FULL val (project hygiene).  Also a within-feed
z of log1p(raw) (uncapped log representation) as a second residual.  CPU only.
"""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k01_capgroup import feed_z, ROOT, PIPE, MASS, ndcg_per_feed, feed_meta, b, yv, offv, Mv, Rv, bh, yh, offh, Mh, Rh  # noqa
import polars as pl

f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
user = f["user_id"].to_numpy()
nF = len(Mv)
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
GRID = np.round(np.concatenate([-np.geomspace(0.01, 0.6, 12)[::-1], [0.0], np.geomspace(0.01, 0.6, 12)]), 4)


def nested(name, gv, gh):
    base_v = ndcg_per_feed(b, yv, offv, Mv); base_h = ndcg_per_feed(bh, yh, offh, Mh)
    per_w = np.stack([ndcg_per_feed(b + w * gv, yv, offv, Mv) for w in GRID])  # [W, nF]
    oof = np.empty(nF); ws = []
    for trf, vaf in folds:
        wi = int(np.argmax(per_w[:, trf].mean(1))); ws.append(GRID[wi]); oof[vaf] = per_w[wi, vaf]
    wf = GRID[int(np.argmax(per_w.mean(1)))]
    dv = oof - base_v; dh = ndcg_per_feed(bh + wf * gh, yh, offh, Mh) - base_h
    se = dv.std() / np.sqrt(nF); seh = dh.std() / np.sqrt(len(dh))
    print(f"  {name:34s} fold w {ws} full w {wf:+.4f} | val {dv.mean():+.5f} ({se:.5f}) z {dv.mean()/se:+.2f} | "
          f"hold {dh.mean():+.5f} ({seh:.5f}) z {dh.mean()/seh:+.2f} | in-sample best val {per_w.mean(1).max()-base_v.mean():+.5f}")


print(f"base val {ndcg_per_feed(b, yv, offv, Mv).mean():.5f} hold {ndcg_per_feed(bh, yh, offh, Mh).mean():.5f}")
exc = lambda R: sum(np.log(np.maximum(R[c], 10.0) / 10.0) for c in MASS)
nested("capexcess_sum4", exc(Rv), exc(Rh))
for c in MASS:
    nested(f"capexcess_{c}", np.log(np.maximum(Rv[c], 10.0) / 10.0), np.log(np.maximum(Rh[c], 10.0) / 10.0))
lz = lambda R, off, M: sum(feed_z(np.log1p(np.maximum(R[c], 0)), off, M) for c in MASS) / 4
lc = lambda R, off, M: sum(feed_z(np.clip(R[c], -10, 10), off, M) for c in MASS) / 4
nested("z_log1p_uncapped_mean4", lz(Rv, offv, Mv), lz(Rh, offh, Mh))
nested("z_capped_linear_mean4 (current repr)", lc(Rv, offv, Mv), lc(Rh, offh, Mh))
nested("z_log1p minus z_capped (repr change)", lz(Rv, offv, Mv) - lc(Rv, offv, Mv), lz(Rh, offh, Mh) - lc(Rh, offh, Mh))
rng = np.random.default_rng(0)
nested("noise control", rng.standard_normal(len(yv)), rng.standard_normal(len(yh)))
