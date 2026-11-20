"""famB shared helpers (family B of nn_x/w12/cb/PREDECLARED.txt, ROUND-2 PLAN). CPU only; imports cbx (read-only infra).
  clu(ds, K)        candidate cluster id per row (data/clu_work/cand_cl_K_<split>.npy), int32, -1 = no embedding
  ShufDS(ds, seed)  view of a train set with rows SHUFFLED inside each feed (feeds stay in file order = time order) for has_time CTRs
  TimeCat(parts)    union of train sets with feeds in TIME order (file order of data/flat/train), for has_time CTRs on xm + xs
  b0_model(seed)    the B0 recipe (infra ts_lr020_s0): YetiRank d8 lr 0.2 l2 5 border 254, 520 trees
"""
import os, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
os.environ.setdefault("CBX_CACHE_EXTRA", "<repo>/nn_x/w12/cb/famB/cache")
import numpy as np
import cbx  # noqa: E402
from cbx import log  # noqa: E402

HERE = cbx.ROOT / "nn_x/w12/cb/famB"
PREDS = HERE / "preds"
FEAT = HERE / "feat"
CLU = cbx.ROOT / "data/clu_work"
B0V = cbx.ROOT / "nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy"
B0H = cbx.ROOT / "nn_x/w12/cb/infra/out/ts_lr020_s0_hold.npy"
ITERS = 520
# Native target statistics with the finest CPU resolution. Measured (out/ctr_border_check.txt): with the DEFAULT 15 uniform borders the statistic
# (clicks + prior) / (count + 1) is cut at multiples of 1/15 = 0.067, so keys whose click rates lie between 0.4% and 3% are indistinguishable
# (synthetic: eval Logloss = constant-rate model, 2 predicted levels); with 255 borders (width 0.0039) corr(prediction, true rate) = 0.95.
CTR255 = ["Borders:CtrBorderCount=255:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1",
          "Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1"]


def clu(ds, K):
    """candidate cluster id (K in 256 / 1024 / 4096) per row of a set (plain cbx DS, ShufDS or TimeCat)."""
    if hasattr(ds, "take"):
        return ds.take(lambda d: clu(d, K))
    a = np.load(CLU / f"cand_cl_{K}_{ds.split}.npy", mmap_mode="r")
    return np.asarray(a if ds.identity else a[ds.rows], np.int32)


class TimeCat:
    """Union of train sets with feeds sorted by flat feed index (= time order: train feeds.parquet is sorted by timestamp).
    Row arrays are materialised in RAM. Used as the base of ShufDS for the has_time arms on xm + xs."""

    def __init__(self, parts):
        assert all(p.split == "train" and p.name != "hold" for p in parts)
        self.parts = parts; self.name = "+".join(p.name for p in parts) + "(time)"; self.split = "train"; self.identity = False
        self.names = parts[0].names
        fidx = np.concatenate([p.fidx for p in parts]); M = np.concatenate([p.M for p in parts])
        src_off = np.concatenate([p.off + b for p, b in zip(parts, np.r_[0, np.cumsum([p.n_rows for p in parts])[:-1]])])
        o = np.argsort(fidx, kind="stable")
        self.fidx = fidx[o]; self.M = M[o]; self.off = np.r_[0, np.cumsum(self.M)[:-1]].astype(np.int64)
        assert len(np.unique(self.fidx)) == len(self.fidx) and not np.isin(self.fidx, cbx.train_sets()["hold"]).any()
        self.n_feeds = len(self.M); self.n_rows = int(self.M.sum())
        self._src = cbx._rows_of(src_off[o], self.M)          # row index into the plain concatenation of the parts
        cat1 = lambda a: np.concatenate([np.asarray(getattr(p, a)) for p in parts])[o]
        self.uid_feed = cat1("uid_feed"); self.ts_feed = cat1("ts_feed"); self.xfold = cat1("xfold"); self.xsub = cat1("xsub")
        assert (np.diff(self.ts_feed) >= 0).all(), "feeds are not in time order"
        self.gid = np.repeat(np.arange(self.n_feeds, dtype=np.int32), self.M)
        self.rows = np.concatenate([p.rows for p in parts])[self._src]

    def take(self, f):
        return np.concatenate([np.asarray(f(p)) for p in self.parts])[self._src]

    X = property(lambda s: s.take(lambda p: p.X)); y = property(lambda s: s.take(lambda p: p.y)); item_id = property(lambda s: s.take(lambda p: p.item_id))
    uid = property(lambda s: s.uid_feed[s.gid])


class ShufDS:
    """Rows shuffled inside each feed (fixed seed); feeds keep the base order (file order = time order). Every per-row array a
    pool needs is served through the permutation, so cbx.make_pool / cbx.run_arm work unchanged."""

    def __init__(self, base, seed=0):
        assert base.split == "train"
        self.base = base; self.name = base.name + f"(shuf{seed})"; self.split = "train"; self.identity = False
        for a in ("fidx", "M", "off", "n_feeds", "n_rows", "uid_feed", "ts_feed", "xfold", "xsub", "names", "gid"):
            setattr(self, a, getattr(base, a))
        assert (np.diff(self.ts_feed) >= 0).all(), "feeds are not in time order"
        rg = np.random.default_rng(1000 + seed)
        key = rg.random(self.n_rows)
        self.perm = np.lexsort((key, np.asarray(self.gid)))    # stable inside-feed shuffle: sort by (feed, random key)
        assert (np.asarray(self.gid)[self.perm] == np.asarray(self.gid)).all()
        self._X = None

    def take(self, f):
        return np.asarray(f(self.base))[self.perm]

    @property
    def X(self):
        if self._X is None:
            self._X = np.asarray(self.base.X)[self.perm]
        return self._X

    y = property(lambda s: s.take(lambda b: b.y)); item_id = property(lambda s: s.take(lambda b: b.item_id))
    uid = property(lambda s: s.uid_feed[s.gid])


def b0_params(seed=0, iterations=ITERS, **kw):
    p = dict(loss_function="YetiRank", eval_metric="NDCG:type=Base", iterations=iterations, learning_rate=0.2, depth=8, l2_leaf_reg=5,
             border_count=254, random_seed=seed, metric_period=25, od_type="Iter", od_wait=100, use_best_model=False,
             allow_writing_files=False, thread_count=cbx.THREADS)
    p.update(kw)
    return p


def b0_model(seed=0, **kw):
    from catboost import CatBoostRanker
    return CatBoostRanker(**b0_params(seed, **kw))


def nats_line(tag, pv, ph):
    sys.path.insert(0, str(cbx.ROOT / "nn_x/w12/cb"))
    from nats import screen
    return screen(pv, ph, tag=tag)


# ---- test-lag variant of the hand-made statistics (lead's notice of 02:19:49Z in PREDECLARED.txt, read by me at 05:10Z) ----
LAG = int(os.environ.get("FB_LAG", "0"))          # FB_LAG=1: train-split rows read only sources that are L days old, L in 6..9 per feed (the test rows' lag)
PRE = "lag_" if LAG else ""


def lag_days(fidx):
    """L in {6, 7, 8, 9} per train feed (fixed draw): the newest source DAY a row may read is its own day minus L (test: 6 to 9 days after the last train day)."""
    return np.random.default_rng(20261115).integers(6, 10, 3151208)[np.asarray(fidx)]
