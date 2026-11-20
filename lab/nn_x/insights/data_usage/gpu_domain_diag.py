"""GPU DIAGNOSTIC (not run by the CPU analyst; ~5 min per root on one GPU, read-only, writes only into this folder).

Question: how much of the snapshot's production-train advantage over the pool-only reference (L_ref - L_snap = 0.0265
CE/logM on production feeds vs 0.0131 on val, see findings.txt sec. 3) is carried by the TRAINING domain token, which is
switched to 'random-like' at inference? Re-score a fixed 300K sample of the production feeds with the snapshot under the
INFERENCE domain (tr.dom = None, as for val/test) and compare with the stored L_snap (training domain).
If adv_prod(inference token) ~ adv_val, the token already absorbs the production-specific (imitation/examination) part and a
fixed examination offset (findings E1) has little left to remove; if it stays ~0.026, the production-specific fit lives in
the shared trunk and transfers to val as bias.

  cd <repo>/nn_x/r6 && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../.venv/bin/python \
      ../insights/data_usage/gpu_domain_diag.py --roots 101,102
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

R6 = Path("<repo>/nn_x/r6")
sys.path.insert(0, str(R6))
from cd_arms4 import per_feed_ce  # noqa: E402
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

OUT = Path("<repo>/nn_x/insights/data_usage/out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="101,102")
    ap.add_argument("--n", type=int, default=300000)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    prod = np.load(R6 / "rho/prod_fidx.npy")
    sel = np.sort(np.random.default_rng(5).choice(len(prod), a.n, replace=False))
    feeds = prod[sel]
    tr = SplitData("train", dev, max_hist=256)
    xt_cur = None
    for r in [int(x) for x in a.roots.split(",")]:
        ck = torch.load(f"<repo>/nn/ckpt/v17_pre2s_s{r}@77500.pt", map_location=dev, weights_only=False)
        m = Ranker(**ck["cfg"]).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
        if ck["cfg"].get("xt_names") != xt_cur:
            tr.xt = None
            if ck["cfg"].get("xt_names"):
                tr.attach_xt(ck["cfg"]["xt_names"])
            xt_cur = ck["cfg"].get("xt_names")
        tr.dom = None  # inference domain (random-like), as for val/test
        L_inf = per_feed_ce(m, tr, feeds)
        L_trn = np.load(R6 / f"rho/L_snap_r{r}.npy")[sel]
        L_ref = np.load(R6 / "rho/L_ref_r6_rhoref_pool_s1.npy")[sel]
        np.savez(OUT / f"gpu_domain_diag_r{r}.npz", fidx=feeds, L_inf=L_inf, L_trn=L_trn, L_ref=L_ref)
        print(f"r{r}: adv_prod training token {np.nanmean(L_ref - L_trn):.4f}  inference token {np.nanmean(L_ref - L_inf):.4f}"
              f"  (val adv 0.0131)", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
