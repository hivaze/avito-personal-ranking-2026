"""Independent train-only inference of one r6 checkpoint (root not checked by bughunt) through the nn/ code path;
outputs go to this directory only. Compare with preds/nn_<v>_trainonly_{val,test}.npy."""
import sys, os
sys.path.insert(0, "<repo>/nn")
import numpy as np, torch
from common import predict, eval_ndcg
from data import SplitData
from model import Ranker

v = sys.argv[1]
OUT = "<repo>/nn_x/pubdiag/verify_diag/infer"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0")
ck = torch.load(f"<repo>/nn_x/r6/ckpt/{v}.pt", map_location=dev, weights_only=False)
model = Ranker(**ck["cfg"]).to(dev)
model.load_state_dict(ck["state_dict"])
for split in ["val", "test"]:
    d = SplitData(split, dev, max_hist=ck["args"]["max_hist"])
    if ck["cfg"].get("xt_names"):
        d.attach_xt(ck["cfg"]["xt_names"])
    if ck["cfg"].get("id_buckets"):
        d.load_item_ids()
    pr = predict(model, d).astype(np.float32)
    np.save(f"{OUT}/{v}_trainonly_{split}.npy", pr)
    saved = np.load(f"<repo>/preds/nn_{v}_trainonly_{split}.npy")
    print(split, "n", len(pr), "max|diff| vs saved", float(np.abs(pr - saved).max()), "n_diff", int((pr != saved).sum()), flush=True)
    if d.has_target:
        print(f"{v} val ndcg {eval_ndcg(pr, d):.5f}", flush=True)
    del d
    torch.cuda.empty_cache()
