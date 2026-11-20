"""B5 diagnostics: per-member val NDCG, mean pairwise member correlation, ensemble (mean logit) NDCG.
  CUDA_VISIBLE_DEVICES=N $PY tabm_diag.py feat_tabm_s23"""
import sys
import numpy as np
import torch
from common import CKPT, eval_ndcg
from data import SplitData
from model import Ranker
dev = torch.device("cuda:0")
va = SplitData("val", dev)
for name in sys.argv[1:]:
    ck = torch.load(CKPT / f"{name}.pt", map_location=dev, weights_only=False)
    m = Ranker(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.keep_members = True
    k = m.tabm
    out = torch.zeros(k, va.n_cand, device=dev)
    with torch.no_grad():
        for fb in va.batches(np.arange(va.n), 65536, 1024, shuffle=False):
            b = va.gather(fb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(b, use_pos=False)
            ml = m.member_logits  # [B,k,M]
            for i in range(k):
                va.scatter(out[i], b, ml[:, i])
    P = out.cpu().numpy()
    nd = [eval_ndcg(P[i], va) for i in range(k)]
    C = np.corrcoef(P)
    print(name, "member ndcg", np.round(nd, 5).tolist(), "mean", round(float(np.mean(nd)), 5),
          "ensemble", round(eval_ndcg(P.mean(0), va), 5), "mean pairwise corr", round(float(C[np.triu_indices(k, 1)].mean()), 4),
          "|r1-1|", round(float((m.tabm_r1.abs() - 1).abs().mean()), 3), flush=True)
