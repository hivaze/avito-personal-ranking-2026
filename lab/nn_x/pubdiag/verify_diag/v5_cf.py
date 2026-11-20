"""SHIFT-2 replay (val only, train-only members): base vs 'notest' it_hist_*_luo (no test-split snapshots, the
information horizon of test feeds). Outputs to infer_cf/ in this directory. GPU: CUDA_VISIBLE_DEVICES=<one id>."""
import sys, os, time
sys.path.insert(0, "<repo>/nn")
import numpy as np, torch
import data as D
from common import predict, eval_ndcg
from model import Ranker

CF = "/tmp/scratchpad/review/shift/out/cf"
OUT = "<repo>/nn_x/pubdiag/verify_diag/infer_cf"
os.makedirs(OUT, exist_ok=True)
_orig = D.feat_path
MODE = {"cf": False}


def feat_path_cf(split, name):
    if MODE["cf"] and split == "val" and name.startswith("it_hist") and name.endswith("_luo"):
        return f"{CF}/notest_{name}.npy"
    return _orig(split, name)


D.feat_path = feat_path_cf
roots = range(101, 108)
models = ([("nn/ckpt", m) for m in ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85",
                                     "v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)] + [f"v17_pre2s_s{r}" for r in roots]]
          + [("nn_x/r6/ckpt", f"r6_cd_{k}_r{r}") for k in ["c0", "m02", "m02_p70k"] for r in roots])
dev = torch.device("cuda:0")
cache = {}
t0 = time.time()
for d_, v in models:
    ck = torch.load(f"<repo>/{d_}/{v}.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    key = (ck["args"]["max_hist"], tuple(cfg.get("xt_names") or ()), bool(cfg.get("id_buckets")))
    model = Ranker(**cfg).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    for mode in ["base", "cf"]:
        fn = f"{OUT}/{v}_{mode}_val.npy"
        if os.path.exists(fn):
            continue
        if (key, mode) not in cache:
            for k_ in list(cache):
                if k_[0] != key:
                    del cache[k_]
            torch.cuda.empty_cache()
            MODE["cf"] = mode == "cf"
            d = D.SplitData("val", dev, max_hist=key[0], verbose=False)
            if key[1]:
                d.attach_xt(list(key[1]))
            if key[2]:
                d.load_item_ids()
            cache[(key, mode)] = d
        d = cache[(key, mode)]
        pr = predict(model, d).astype(np.float32)
        np.save(fn, pr)
        print(f"[{time.time()-t0:7.1f}s] {v} {mode} val ndcg {eval_ndcg(pr, d):.5f}", flush=True)
    del model
print("DONE", flush=True)
