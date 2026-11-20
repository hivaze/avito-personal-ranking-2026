"""CPU probe (kd_literature): is a STORED full-feed teacher logit a "consistent teacher" for a random SUB-feed?

Beyer et al. 2022 (arXiv 2106.05237, Table 3): a fixed / precomputed teacher target paired with an augmented student
view ("fixed teacher") is clearly worse than a teacher re-evaluated on the same view ("consistent teaching") and
overfits at long schedules. The running kwsub5 / kwsub2 arms renormalise STORED full-feed logits over a random sub-feed.
That is consistent teaching only if the ranker's logits do not depend on the candidate set. They can: set attention,
within-feed z / rank features, feed category share, log M.

Measure on VAL (CPU, eval mode, no labels enter any input; labels only keep positives as in cd_arms.augment_pool_batch
and score NDCG): for a model (or the mean logit of several = teacher proxy)
  full  = logits on the full feed, restricted to the kept rows
  sub   = logits re-evaluated on the sub-feed (cmask restricted, exactly like augment_pool_batch)
  -> within-feed Spearman / Pearson(full, sub), top-1 agreement, KL(softmax(full|kept) || softmax(sub)) in nats,
     NDCG on the sub-feed of both scorings (paired).
Noise floors on the same kept rows: seed-vs-seed (two different checkpoints, both full-feed).
Usage: CUDA_VISIBLE_DEVICES="" python subfeed_probe.py <tag> <aug_cand> <variant> [<variant> ...]
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

torch.set_num_threads(12)
sys.path.insert(0, "<repo>/nn_x/w5")
sys.path.insert(0, "<repo>/src")
from data import SplitData  # noqa: E402  (CPU-safe; nn_x/w5/common.py is NOT imported: it demands a GPU id)
from model import Ranker  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

tag, aug, variants = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
CK = "<repo>/nn_x/w5/ckpt"
OUT = "<repo>/nn_x/w12/recheck/kd_literature/out"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cpu")
va = SplitData("val", dev)
n, M, off = va.n, va.M_np, va.cand_off_np
y = va.target_np
feed_of = np.repeat(np.arange(n), M)

# sub-feed mask, same law as cd_arms.augment_pool_batch: q ~ U(aug, 1) per feed, positives always kept
rng = np.random.default_rng(20261114)
q = aug + (1 - aug) * rng.random(n)
keep = (rng.random(len(y)) < q[feed_of]) | (y > 0)
keep_t = torch.from_numpy(keep)


@torch.no_grad()
def run(model, sub):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    model.eval()
    for fb in va.batches(np.arange(n), 16384, 256, shuffle=False):
        b = va.gather(fb)
        if sub:
            new = b["cmask"] & keep_t[b["ci"]]
            new = torch.where(new.any(1, keepdim=True), new, b["cmask"])
            b["cmask"] = new
            b["y"] = b["y"] * new
        lg = model(b, use_pos=False)
        va.scatter(out, b, lg)
    return out.numpy()


xt = "unset"
full, sub = [], []
for v in variants:
    ck = torch.load(f"{CK}/{v}.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    if cfg.get("xt_names") != xt:
        va.xt = None
        if cfg.get("xt_names"):
            va.attach_xt(cfg["xt_names"])
        xt = cfg.get("xt_names")
    m = Ranker(**cfg)
    m.load_state_dict(ck["state_dict"])
    f_, s_ = run(m, False), run(m, True)
    full.append(f_)
    sub.append(s_)
    nd = ndcg_per_feed(f_, y, off, M).mean()
    print(f"{v}: full-feed val NDCG {nd:.5f} (CPU fp32 replay)", flush=True)
    del m
full, sub = np.array(full), np.array(sub)
np.save(f"{OUT}/subfeed_{tag}_full.npy", full.astype(np.float32))
np.save(f"{OUT}/subfeed_{tag}_sub.npy", sub.astype(np.float32))
np.save(f"{OUT}/subfeed_{tag}_keep.npy", keep)

# ---------------------------------------------------------------- per-feed statistics on the kept rows
kidx = np.where(keep)[0]
Mk = np.bincount(feed_of[kidx], minlength=n).astype(np.int64)
offk = np.concatenate([[0], np.cumsum(Mk)[:-1]])
yk = y[kidx]
fk = feed_of[kidx]
ok = Mk >= 5  # feeds with at least 5 kept candidates


def rankdata_feed(x):
    order = np.lexsort((x, fk))
    r = np.empty(len(x))
    r[order] = np.arange(len(x)) - offk[fk[order]]
    return r


def feed_corr(a, b):
    def c(u):
        s = np.add.reduceat(u, offk)
        return u - (s / Mk)[fk]
    a, b = c(a.astype(np.float64)), c(b.astype(np.float64))
    num = np.add.reduceat(a * b, offk)
    den = np.sqrt(np.add.reduceat(a * a, offk) * np.add.reduceat(b * b, offk))
    return num / np.maximum(den, 1e-12)


def logsoftmax_feed(x):
    x = x.astype(np.float64)
    mx = np.maximum.reduceat(x, offk)[fk]
    lse = np.log(np.add.reduceat(np.exp(x - mx), offk))[fk] + mx
    return x - lse


def kl_feed(a, b):
    la, lb = logsoftmax_feed(a), logsoftmax_feed(b)
    return np.add.reduceat(np.exp(la) * (la - lb), offk)


def top1_feed(a, b):
    def am(x):
        mx = np.maximum.reduceat(x, offk)[fk]
        first = np.full(n, -1, np.int64)
        idx = np.where(x == mx)[0]
        first[fk[idx[::-1]]] = idx[::-1]
        return first
    return am(a) == am(b)


def stats(name, a, b):
    a, b = a[kidx], b[kidx]
    sp = feed_corr(rankdata_feed(a), rankdata_feed(b))[ok]
    pe = feed_corr(a, b)[ok]
    kl = kl_feed(a, b)[ok]
    t1 = top1_feed(a, b)[ok]
    na, nb = ndcg_per_feed(a.astype(np.float32), yk, offk, Mk), ndcg_per_feed(b.astype(np.float32), yk, offk, Mk)
    d = (nb - na)
    se = lambda u: u.std() / np.sqrt(len(u))
    print(f" {name:<46s} Spearman {sp.mean():.4f}  Pearson {pe.mean():.4f}  top-1 agree {t1.mean():.3f}  "
          f"KL {kl.mean():.4f} nats (median {np.median(kl):.4f})  NDCG(sub-feed) A {na.mean():.5f} B {nb.mean():.5f} "
          f"B-A {d.mean():+.5f}+-{se(d):.5f}", flush=True)


print(f"== {tag}: aug_cand {aug}; kept share {keep.mean():.3f}; feeds with >= 5 kept {ok.sum()} of {n}")
K = len(variants)
for i in range(K):
    stats(f"member {variants[i][-12:]}: full (A) vs re-scored sub (B)", full[i], sub[i])
if K > 1:
    stats(f"MEAN of {K}: full (A) vs re-scored sub (B)", full.mean(0), sub.mean(0))
    stats("noise floor: member 0 full (A) vs member 1 full (B)", full[0], full[1])
    stats("noise floor: member 0 sub (A) vs member 1 sub (B)", sub[0], sub[1])
    if K >= 4:
        stats("noise floor: mean(0,1) full (A) vs mean(2,3) full (B)", full[:2].mean(0), full[2:4].mean(0))
        stats("cross: mean(0,1) FULL (A) vs mean(2,3) SUB (B)", full[:2].mean(0), sub[2:4].mean(0))
        stats("cross: mean(0,1) SUB (A) vs mean(2,3) SUB (B)", sub[:2].mean(0), sub[2:4].mean(0))
