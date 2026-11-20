"""Reviewer check: how closely does the saved holdout-selected checkpoint reproduce the saved holdout predictions?
16 holdout feeds; fp32 trunk + TF32 off (the I3 final path); alone vs in one batch vs saved file. Read-only.
  CUDA_VISIBLE_DEVICES=<gpu> python gpu_holdout_repro.py <tag>"""
import sys

import numpy as np
import torch

sys.path.insert(0, "<repo>/nn_x/llm")
import data_llm as D  # noqa: E402
from model_llm import INPUT_KEYS, fp32_from_state  # noqa: E402
from qwen35_branch import Qwen35Ranker  # noqa: E402
from model_llm import LLMRanker  # noqa: E402

TAG = sys.argv[1]
LLM = "<repo>/nn_x/llm"
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
CK = torch.load(f"{LLM}/ckpt/{TAG}.pt", map_location="cpu", weights_only=False)
cls = Qwen35Ranker if CK["cfg"]["backbone"].startswith("qwen3.5") else LLMRanker
m = fp32_from_state(cls, CK["cfg"], CK["state"]).to(dev).eval()
names = D.xt_config("cvnd_pre2s")[0]
M_all, _ = D.train_meta()
pr = D.load_prandom(M_all)
hold = D.holdout_feeds(pr, M_all)
pick = np.sort(np.random.default_rng(7).choice(len(hold), 16, replace=False))
st = D.FeedStore("train", hold[pick], dev, 256, names, verbose=False)
st.set_domain(pr)
ph = np.load(f"{LLM}/preds/holdout/{TAG}_holdout.npy")
Mh = M_all[hold]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
inp = lambda b: {k: v for k, v in b.items() if k in INPUT_KEYS}  # noqa: E731
with torch.no_grad():
    sb = m(inp(st.gather(np.arange(st.n)))).float().cpu().numpy()
    d_saved, d_alone, d_alone_saved, mag = [], [], [], []
    for i, hi in enumerate(pick):
        M = int(Mh[hi])
        sa = m(inp(st.gather([i]))).float().cpu().numpy()[0, :M]
        sv = ph[offh[hi]:offh[hi] + M]
        d_saved.append(np.abs(sb[i, :M] - sv).max())
        d_alone.append(np.abs(sb[i, :M] - sa).max())
        d_alone_saved.append(np.abs(sa - sv).max())
        mag.append(np.abs(sv).max())
print(f"{TAG}: 16 holdout feeds (M {Mh[pick].min()}-{Mh[pick].max()}, H {st.H_np.min()}-{st.H_np.max()}): "
      f"max|batch16 - saved| {max(d_saved):.2e} (median {np.median(d_saved):.2e}); max|alone - batch16| {max(d_alone):.2e}; "
      f"max|alone - saved| {max(d_alone_saved):.2e}; max|score| {max(mag):.2f}; ckpt step {CK['step']}")
