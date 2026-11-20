"""C5: frozen-teacher logits over the whole train split (flat candidate order), fp16.

The teacher is a read-only checkpoint (default <repo>/nn/ckpt/s2_v2_feed_k9.pt, whose stages excluded
the same 30K holdout). Predicted with no tr.dom, i.e. the random-like domain (the teacher's unbiased view).
Usage: CUDA_VISIBLE_DEVICES=N python teacher.py [--ckpt PATH] [--out PATH]
"""
import argparse
import time

import numpy as np
import torch

from common import predict
from data import SplitData
from model import Ranker

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="<repo>/nn/ckpt/s2_v2_feed_k9.pt")
p.add_argument("--out", default="<repo>/nnx/optim/teacher/s2_v2_feed_k9_train.npy")
p.add_argument("--split", default="train")
a = p.parse_args()
t0 = time.time()
dev = torch.device("cuda:0")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
model = Ranker(**ck["cfg"]).to(dev)
model.load_state_dict(ck["state_dict"])
data = SplitData(a.split, dev, max_hist=ck["args"]["max_hist"])
assert data.dom is None and not ck["cfg"].get("xt_names") and not ck["cfg"].get("id_buckets")
z = predict(model, data)
print(f"teacher logits: n={len(z)} mean {z.mean():.3f} std {z.std():.3f} min {z.min():.2f} max {z.max():.2f}")
np.save(a.out, z.astype(np.float16))
if a.split == "val":
    from common import eval_ndcg
    print("val ndcg (fp32 / fp16):", eval_ndcg(z, data), eval_ndcg(z.astype(np.float16).astype(np.float32), data))
print(f"saved {a.out} in {(time.time() - t0) / 60:.1f} min")
