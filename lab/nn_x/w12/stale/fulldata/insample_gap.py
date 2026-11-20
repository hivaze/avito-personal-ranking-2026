"""PREDECLARED.txt R7: the in-sample gap G_s of an ORIGINAL final member (calibration of the A2 bars; no twin involved).
Executed UNCHANGED harness code on the CPU through cpu_run.py (cuda -> cpu, fp32), which runs this file as __main__:

  CUDA_VISIBLE_DEVICES="" ... .venv/bin/python cpu_run.py <nn_x/w5 | nn_x/abl> insample_gap.py runs/<tag> <member> <out json>

Loads the member's @last checkpoint exactly as <harness>/holdout_eval.py does, predicts (a) 20,000 exploration-like TRAINING
feeds drawn by the holdout rule from the non-holdout feeds (rng 777; in sample for the member and for its teacher) and
(b) 2,000 holdout feeds (rng 778; path check against the member's saved GPU file). Reads only; writes <out json>.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import ROOT, predict  # noqa: E402
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

v, out_json = sys.argv[1], sys.argv[2]
R = "<repo>"
H = f"{R}/nn_x/{'w5' if v.startswith('w5_') else 'abl'}"
t0 = time.time()
dev = torch.device("cuda:0")  # cpu under cpu_run.py
tr = SplitData("train", dev)
pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[tr.M_np < 50] = 0.0
cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
ins = np.sort(np.random.default_rng(777).choice(np.setdiff1d(cand, hold), 20000, replace=False))
hsub_i = np.sort(np.random.default_rng(778).choice(30000, 2000, replace=False))
hsub = hold[hsub_i]
ck = torch.load(f"{H}/ckpt/{v}@last.pt", map_location=dev, weights_only=False)
a, cfg = ck["args"], ck["cfg"]
assert a["holdout"] == 30000 and not a.get("holdout_in_train") and a["train_feeds"] == 0, "an ORIGINAL member is required"
if cfg.get("xt_names"):
    tr.attach_xt(cfg["xt_names"])
prt = torch.from_numpy(pr).to(dev)
tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
m = Ranker(**cfg).to(dev)
m.load_state_dict(ck["state_dict"])
print(f"[insample_gap] {v}@last loaded ({(time.time() - t0) / 60:.1f} min); predicting {len(ins)} in-sample + {len(hsub)} holdout feeds", flush=True)
s = predict(m, tr, np.sort(np.concatenate([ins, hsub])))
print(f"[insample_gap] predicted ({(time.time() - t0) / 60:.1f} min)", flush=True)


def rows_of(f):
    return np.concatenate([np.arange(o, o + k) for o, k in zip(tr.cand_off_np[f], tr.M_np[f])])


def per_feed(sc, f, rows):
    M = tr.M_np[f].astype(np.int64)
    return ndcg_per_feed(np.asarray(sc, np.float32), np.asarray(tr.target_np[rows], np.float32), np.concatenate([[0], np.cumsum(M)[:-1]]), M)


def ms(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


r_in, r_h, r_hs = rows_of(ins), rows_of(hold), rows_of(hsub)
nd_in = per_feed(s[r_in], ins, r_in)
saved = np.load(f"{H}/preds/holdout/nn_{v}@last_holdout.npy")
assert saved.shape == (len(r_h),)
nd_oos = per_feed(saved, hold, r_h)
# path check on the 2,000 holdout feeds: this CPU prediction against the member's saved GPU file
hoff = np.concatenate([[0], np.cumsum(tr.M_np[hold])])
sv_sub = np.concatenate([saved[hoff[i]:hoff[i + 1]] for i in hsub_i])
mine = s[r_hs]
fid = np.repeat(np.arange(len(hsub)), tr.M_np[hsub])
cen = lambda x: x - (np.bincount(fid, weights=x) / tr.M_np[hsub])[fid]  # noqa: E731
corr = float(np.corrcoef(cen(mine.astype(np.float64)), cen(sv_sub.astype(np.float64)))[0, 1])
nd_cpu, nd_gpu = per_feed(mine, hsub, r_hs), per_feed(sv_sub, hsub, r_hs)
# the member's teacher on the same feeds (reference)
kd = np.load(a["kd_file"], mmap_mode="r") if a.get("kd_file") else None
t_in = per_feed(np.asarray(kd[r_in]), ins, r_in) if kd is not None else None
t_oos = per_feed(np.asarray(kd[r_h]), hold, r_h) if kd is not None else None
G = nd_in.mean() - nd_oos.mean()
se = float(np.sqrt(ms(nd_in)[1] ** 2 + ms(nd_oos)[1] ** 2))
res = dict(member=v, checkpoint=f"{v}@last", n_in=len(ins), n_oos=len(hold), ndcg_in=ms(nd_in), ndcg_oos_saved_file=ms(nd_oos), G_s=float(G), G_s_se=se,
           path_check=dict(n=len(hsub), corr_centred=corr, ndcg_cpu=ms(nd_cpu), ndcg_gpu_file=ms(nd_gpu), diff=ms(nd_cpu - nd_gpu),
                           ok=bool(corr > 0.999 and abs(float((nd_cpu - nd_gpu).mean())) < 0.002)),
           teacher=None if kd is None else dict(file=os.path.basename(a["kd_file"]), ndcg_in=ms(t_in), ndcg_oos=ms(t_oos), G_t=float(t_in.mean() - t_oos.mean()),
                                                student_minus_teacher_in=ms(nd_in - t_in), student_minus_teacher_oos=ms(nd_oos - t_oos)),
           minutes=(time.time() - t0) / 60)
json.dump(res, open(out_json, "w"), indent=1)
print("[insample_gap] " + json.dumps(res), flush=True)
