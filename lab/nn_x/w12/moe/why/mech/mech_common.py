"""w12/moe/why/mech: shared loaders for the mechanism analysis (read-only on checkpoints / data).
  load_val(dev, xt_names)    SplitData('val') + attach_xt (as train.py)
  load_hold(dev, xt_names)   a SplitData-like object holding ONLY the 30,000 holdout feeds of the train split (same feed selection as
                             train.py --holdout 30000: p_random > 0.9 and M >= 50, rng 12345), with the train-time domain mixture
                             [1-p, p, 0]; item table compacted to the rows the holdout uses (indices remapped, exact).
  build(ckpt)                Ranker(**cfg) from model_moe.py (= w5 model when the MoE flags are off) in eval mode.
  predict(model, data, ...)  bf16 autocast forward over every feed (same batching as common.predict), flat scores.
  ndcg_fast                  numba copy of pc_common._ndcg (pessimistic ties).
"""
import os
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl
import torch

ROOT = Path('<repo>')
sys.path.insert(0, str(ROOT / 'lab/nn_x/w12/moe'))
sys.path.insert(1, str(ROOT / 'lab/nn_x/w5'))
torch.set_num_threads(12)

import data as w5data  # noqa: E402  (lab/nn_x/w5/data.py)
from data import SplitData, FLAT, feat_path, load_item_table  # noqa: E402
from model_moe import Ranker  # noqa: E402

MOE = ROOT / 'lab/nn_x/w12/moe/out'
W5 = ROOT / 'nn_x/w5'
OUT = ROOT / 'lab/nn_x/w12/moe/why/mech/out'


def ckpt_path(variant, suf=''):
    p = MOE / 'ckpt' / f'{variant}{suf}.pt'
    return p if p.exists() else W5 / 'ckpt' / f'{variant}{suf}.pt'


def pred_paths(variant, suf=''):
    if (MOE / 'preds' / f'nn_{variant}{suf}_trainonly_val.npy').exists():
        d = MOE / 'preds'
    else:
        d = W5 / 'preds'
    return d / f'nn_{variant}{suf}_trainonly_val.npy', d / f'holdout/nn_{variant}{suf}_holdout.npy'


def build(variant, dev, suf=''):
    ck = torch.load(ckpt_path(variant, suf), map_location='cpu', weights_only=False)
    m = Ranker(**ck['cfg'])
    m.load_state_dict(ck['state_dict'])
    return m.to(dev).eval(), ck['cfg']


def _xt_transform(n, x):
    """= SplitData.attach_xt column transform."""
    if n.startswith('has:') or n.endswith('_rk'):
        pass
    elif n.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif n.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def load_val(dev, xt_names):
    va = SplitData('val', dev)
    va.attach_xt(xt_names)
    va.c_pos = None
    return va


def holdout_feeds():
    f = pl.read_parquet(FLAT / 'train' / 'feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return f, hold, p


def trainfit_feeds(n=30000, seed=777):
    """n TRAINING feeds with the holdout's selection rule (p_random > 0.9, M >= 50) and NOT in the holdout (they were trained on)."""
    f, hold, p = holdout_feeds()
    M = f['M'].to_numpy().astype(np.int64)
    cand = np.setdiff1d(np.where((p > 0.9) & (M >= 50))[0], hold)
    return f, np.sort(np.random.default_rng(seed).choice(cand, n, replace=False)), p


def load_hold(dev, xt_names, max_hist=256, feeds=None):
    t0 = time.time()
    f, hold, prand = holdout_feeds() if feeds is None else feeds
    M_all = f['M'].to_numpy().astype(np.int64)
    H_all = f['H'].to_numpy().astype(np.int64)
    co_all = f['cand_off'].to_numpy().astype(np.int64)
    ho_all = f['hist_off'].to_numpy().astype(np.int64)
    M, H, co, ho = M_all[hold], H_all[hold], co_all[hold], ho_all[hold]
    Hc = np.minimum(H, max_hist)
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)])
    hrow = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, Hc)])
    new_co = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    new_ho = np.concatenate([[0], np.cumsum(Hc)[:-1]]).astype(np.int64)
    mm = lambda name: np.load(FLAT / 'train' / f'{name}.npy', mmap_mode='r')
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    d = SplitData.__new__(SplitData)
    d.split, d.device, d.max_hist = 'train_holdout', dev, max_hist
    d.feeds = None
    d.n = len(hold)
    d.hold = hold
    d.crow = crow
    d.M_np, d.H_np, d.cand_off_np = M, H, new_co
    d.user_id = f['user_id'].to_numpy()[hold]
    d.M, d.H, d.cand_off, d.hist_off = g(M), g(H), g(new_co), g(new_ho)
    d.user_emb = g(np.asarray(mm('user_emb')[hold]))
    d.user0 = g(d.user_id == 0)
    c_emb = np.asarray(mm('cand_emb_idx')[crow])
    h_emb = np.asarray(mm('hist_emb_idx')[hrow])
    used = np.unique(np.concatenate([c_emb[c_emb >= 0], h_emb[h_emb >= 0]]))
    remap = lambda e: np.where(e >= 0, np.searchsorted(used, np.maximum(e, 0)), -1).astype(np.int32)
    d.c_emb, d.h_emb = g(remap(c_emb)), g(remap(h_emb))
    d.c_item = d.h_item = None
    d.xt = None
    d.n_cand = len(crow)
    d.c_cat = g(np.asarray(mm('cand_category_id')[crow]).astype(np.int64)).to(torch.int16)
    d.c_f = torch.stack([g(np.asarray(mm(f'cand_f{i}')[crow])) for i in range(1, 6)], 1)
    d.has_target = True
    d.target_np = np.asarray(mm('cand_target')[crow]).astype(np.float32)
    d.c_tgt = g(d.target_np.astype(np.int8))
    d.has_pos = True
    d.c_pos = g(np.asarray(mm('cand_pos')[crow]))
    d.h_cat = g(np.asarray(mm('hist_category_id')[hrow]).astype(np.int16))
    d.h_et = g(np.asarray(mm('hist_event_type')[hrow]))
    d.h_es = g(np.asarray(mm('hist_event_source')[hrow]))
    d.n_hist = len(hrow)
    dd = ROOT / 'data/competition/train/item_embeddings'
    import json
    meta = json.loads((dd / 'data.meta').read_text())
    tab = np.memmap(dd / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
    d.table = g(np.asarray(tab[used]))
    d.hist_ext = 0
    p = torch.from_numpy(prand[hold]).to(dev)
    d.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
    cols = []
    for n in xt_names:
        if n.startswith('has:'):
            x = (~np.isnan(np.asarray(np.load(feat_path('train', n[4:]), mmap_mode='r')[crow]))).astype(np.float32)
        else:
            x = np.asarray(np.load(feat_path('train', n), mmap_mode='r')[crow]).astype(np.float32)
        cols.append(torch.from_numpy(_xt_transform(n, x)).to(dev, torch.float16))
    d.xt = torch.stack(cols, 1)
    torch.cuda.synchronize(dev)
    print(f'[hold] {d.n} feeds, {d.n_cand} cands, {d.n_hist} hist, table {tuple(d.table.shape)}, {time.time() - t0:.0f}s', flush=True)
    return d


@torch.no_grad()
def predict(model, data, pre=None, post=None, tf32=False, cand_budget=65536, max_feeds=1024):
    """flat float32 scores; pre(b) / post(b, logits) are called around each batch forward (instrumentation)."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        out = torch.zeros(data.n_cand, dtype=torch.float32, device=data.device)
        for fb in data.batches(np.arange(data.n), cand_budget, max_feeds, shuffle=False):
            b = data.gather(fb)
            if pre is not None:
                pre(b)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(b, use_pos=False)
            if post is not None:
                post(b, lg)
            data.scatter(out, b, lg)
        return out.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_fast(s, y, off, M):
    return _ndcg(np.ascontiguousarray(s, np.float32), np.ascontiguousarray(y, np.float32), np.asarray(off, np.int64), np.asarray(M, np.int64))


def split_ndcg(data, s):
    return ndcg_fast(s, data.target_np, data.cand_off_np, data.M_np)


se = lambda d: float(np.std(d) / np.sqrt(len(d)))
