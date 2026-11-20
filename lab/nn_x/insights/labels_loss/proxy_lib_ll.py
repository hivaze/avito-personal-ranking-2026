"""Shared helpers for the CPU GBDT loss proxy: LightGBM with a custom listwise softmax-CE objective (same loss as
nn/model.py listwise_loss: target y/K, softmax over the whole feed), per-feed weights, optional fixed train-only
logit offsets (init_score; not added at prediction)."""
import sys, time
import numpy as np
import lightgbm as lgb
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

D = "<repo>/nn_x/insights/labels_loss/out/proxy"

def load(name):
    X = np.load(f"{D}/{name}_X.npy", mmap_mode="r"); y = np.load(f"{D}/{name}_y.npy"); M = np.load(f"{D}/{name}_M.npy")
    return X, y, M

def offs(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)

def subset(X, y, M, feeds):
    o = offs(M)
    r = np.concatenate([np.arange(o[i], o[i] + M[i]) for i in feeds])
    return np.asarray(X[r]), y[r], M[feeds], r

class SoftmaxCE:
    """grad/hess of sum_f w_f * CE_f, CE_f = -(1/K_f) sum_i y_i log softmax(s_f)_i; dynamic weights optional."""
    def __init__(self, y, M, w=None, dyn=None):
        self.y = y.astype(np.float64); self.M = M; self.o = offs(M)
        self.fid = np.repeat(np.arange(len(M)), M)
        self.K = np.add.reduceat(self.y, self.o); self.K[self.K == 0] = np.inf
        self.w = np.ones(len(M)) if w is None else w.astype(np.float64)
        self.w = self.w / self.w.mean()
        self.dyn = dyn  # callable(scores, self) -> per-feed multiplicative weights (detached)
    def __call__(self, preds, ds):
        s = np.asarray(preds, dtype=np.float64)
        mx = np.maximum.reduceat(s, self.o)
        e = np.exp(s - np.repeat(mx, self.M)); Z = np.add.reduceat(e, self.o)
        p = e / np.repeat(Z, self.M)
        w = self.w
        if self.dyn is not None:
            w = w * self.dyn(s, self); w = w / w.mean()
        wr = np.repeat(w, self.M)
        tgt = self.y / np.repeat(self.K, self.M)
        g = wr * (p - tgt)
        h = np.maximum(wr * p * (1 - p), 1e-7)
        return g, h

PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, num_threads=12, verbose=-1, max_bin=127)

def train(X, y, M, w=None, init=None, dyn=None, rounds=400, seed=0, params=None):
    obj = SoftmaxCE(y, M, w, dyn)
    prm = dict(PARAMS); prm.update(params or {}); prm.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                                                             objective=obj)
    ds = lgb.Dataset(X, label=y, init_score=init, free_raw_data=False, params={"max_bin": prm["max_bin"]})
    return lgb.train(prm, ds, num_boost_round=rounds)

def ndcg_curve(bst, X, y, M, steps):
    o = offs(M)
    out = {}
    for k in steps:
        s = bst.predict(X, raw_score=True, num_iteration=k)
        out[k] = ndcg_per_feed(s, y, o, M)
    return out
