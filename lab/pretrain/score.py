"""Re-score val/test candidates from a saved checkpoint (e.g. after adding new export features).
Usage: CUDA_VISIBLE_DEVICES=N python pretrain/score.py --tag v1"""
import argparse, sys, numpy as np, torch
sys.path.insert(0, "<repo>/pretrain")
sys.path.insert(0, "<repo>/src")
from seqenc import SeqEncoder
from train import counts_excluding_test, Items, FeedSplit, infer, PT, PREDS
from metrics import ndcg

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
dev = torch.device("cuda:0")
ck = torch.load(PT / f"ckpt_{a.tag}.pt", map_location="cpu"); ca = ck["args"]
items = Items(dev, ca["min_cnt"], counts_excluding_test() if ca.get("exclude_test") else None)
heads = ["fe", "new"] if ca["w_new"] > 0 else ["fe"]
model = SeqEncoder(items.n_id, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads, id_bias=bool(ca.get("id_bias", 0)))
model.load_state_dict(ck["model"]); model.to(dev)
for split in ["val", "test"]:
    fs = FeedSplit(split, dev)
    sc = infer(model, items, fs, heads)
    for k, v in sc.items():
        np.save(PREDS / f"pre_{a.tag}_{k}_{split}.npy", v)
    if fs.y is not None:
        print(" ".join(f"{k} {ndcg(v, fs.y, fs.off_c, fs.M):.4f}" for k, v in sc.items()), flush=True)
