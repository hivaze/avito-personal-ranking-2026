"""Ranker arm B of the matched-data test: the UNMODIFIED nn/train.py, outputs redirected into nn_x/llm/baseline/.

  cd nn_x/llm && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 ../../.venv/bin/python run_nn_baseline.py --variant v ...
  (all arguments are passed verbatim to nn/train.py's parser)

How the redirect works without editing nn/: nn/common.py is imported first and its CKPT/LOGS/PREDS paths are
replaced; nn/train.py then binds those names at import time (`from common import CKPT, LOGS, PREDS, ...`).
Nothing is written to nn/ckpt, nn/logs or preds/ (asserted), and bytecode writing is disabled (no nn/__pycache__ change).
After training, the holdout predictions of the holdout-selected checkpoint are written in the preds/holdout layout
(holdout feeds sorted, candidates in flat order) to baseline/preds/holdout/nn_<variant>_holdout.npy, using the
read-only FeedStore for the 30K holdout feeds (no second full train-split load).
Outputs: baseline/ckpt/<v>.pt, baseline/logs/<v>.{log,json}, baseline/preds/nn_<v>_trainonly_{val,test}.npy,
baseline/preds/holdout/nn_<v>_holdout.npy
"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "baseline"
NN = "<repo>/nn"
sys.path.insert(0, NN)
import common  # noqa: E402  (GPU guard: CUDA_VISIBLE_DEVICES must be 4..7)

for sub in ("ckpt", "logs", "preds/holdout"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
common.CKPT, common.LOGS, common.PREDS = OUT / "ckpt", OUT / "logs", OUT / "preds"
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(12)  # nn/common.py sets 16; this task's cap is 12
import train  # noqa: E402  (nn/train.py, unmodified)

assert train.CKPT == OUT / "ckpt" and train.LOGS == OUT / "logs" and train.PREDS == OUT / "preds"
sys.path.insert(1, str(HERE))
import data_llm as D  # noqa: E402


def holdout_preds(variant, dev="cuda:0", out=None, limit=0):
    """Holdout predictions of baseline/ckpt/<variant>.pt. `limit` > 0 (tests only) scores just the first holdout feeds."""
    dev = torch.device(dev)
    out = OUT if out is None else Path(out)
    ck = torch.load(out / "ckpt" / f"{variant}.pt", map_location=dev, weights_only=False)
    cfg, args = ck["cfg"], ck["args"]
    M_all, _ = D.train_meta()
    pr = D.load_prandom(M_all)
    hold = D.holdout_feeds(pr, M_all, args["holdout"])
    if limit:
        hold = hold[:limit]
    st = D.FeedStore("train", hold, dev, args["max_hist"], cfg.get("xt_names"))
    if cfg.get("domain"):
        st.set_domain(pr)  # as nn/train.py: tr.dom = [1-p, p, 0] for every train feed, holdout included
    if cfg.get("id_buckets"):
        raise NotImplementedError("item-ID residual not used by the F7 recipe")
    m = train.Ranker(**cfg).to(dev)
    m.load_state_dict(ck["state_dict"])
    s = common.predict(m, st)  # flat order of the store == holdout feeds sorted, candidates in train flat order
    nd = D.ndcg(s, st.target_np, st.cand_off_np, st.M_np)
    (out / "preds" / "holdout").mkdir(parents=True, exist_ok=True)
    np.save(out / "preds" / "holdout" / f"nn_{variant}_holdout.npy", s.astype(np.float32))
    np.save(out / "preds" / "holdout" / "holdout_meta.npy", np.stack([st.M_np, st.cand_off_np]))
    np.save(out / "preds" / "holdout" / "holdout_y.npy", st.target_np)
    print(f"[baseline] {variant}: holdout NDCG of the selected checkpoint {nd:.5f} (compare with the BEST line of the "
          f"log; batch composition differs, so expect agreement to ~1e-4)", flush=True)
    return nd


if __name__ == "__main__":
    argv = sys.argv[1:]
    variant = argv[argv.index("--variant") + 1]
    sys.argv = ["nn/train.py"] + argv
    train.main()
    holdout_preds(variant)
