"""Predict val and test with a saved train-only checkpoint.

  CUDA_VISIBLE_DEVICES=N python nn/predict.py --variant nopos
writes preds/nn_<variant>_trainonly_{val,test}.npy (flat candidate order of data/flat/<split>/).
"""
import argparse

import numpy as np
import torch

from common import CKPT, PREDS, eval_ndcg, predict
from data import SplitData
from model import Ranker

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    a = p.parse_args()
    dev = torch.device("cuda:0")
    ck = torch.load(CKPT / f"{a.variant}.pt", map_location=dev, weights_only=False)
    model = Ranker(**ck["cfg"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    for split in ["val", "test"]:
        d = SplitData(split, dev, max_hist=ck["args"]["max_hist"])
        if ck["cfg"].get("xt_names"):
            d.attach_xt(ck["cfg"]["xt_names"])
        if ck["cfg"].get("id_buckets"):
            d.load_item_ids()
        if ck["cfg"].get("sid"):
            d.load_codes()
        pr = predict(model, d)
        np.save(PREDS / f"nn_{a.variant}_trainonly_{split}.npy", pr.astype(np.float32))
        if d.has_target:
            print(f"{a.variant} val ndcg {eval_ndcg(pr, d):.5f}")
        del d
        torch.cuda.empty_cache()
