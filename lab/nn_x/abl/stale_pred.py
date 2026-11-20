"""Stale-val predictions on the GPU (the project's stale-val scenario 'both_pre2both' of
solvers/CPU2_stale_replay_pre2s_and_arms.py, used by r6_eval_cdw4.py for the hcap kill rule). Read-only on every
checkpoint / feature file; writes nn_x/w5/preds/stale/nn_<v>_staleval.npy (+ a .log line per variant).

  cd <repo>/nn_x/w5 && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../.venv/bin/python stale_pred.py \
      w5_cd_m02ref_r101@last w5_cd_truncp_r101@last ...   (names in nn_x/w5/ckpt; 'nn:<v>' reads nn/ckpt/<v>.pt)
nn_x/w5 copy of nn_x/r6/stale_pred.py (paths only; the counterfactual inputs are read from nn_x/r6/stale_cf, read-only).

Scenario (val inputs only; private-like coverage, CPU2 docstring): 'both' = review_shift popularity / co-vis at
private-like coverage for it_hist_{all,et1..et4}_luo and seq_cv_{invd20,norm,pop}; 'pre2both' = pre2_fe / pre2_fe_z /
pre2_fe_s with the no-ID score on newly cold rows (has:pre2_fe follows the overridden pre2_fe). Every other input,
the val histories and the model are unchanged; inference exactly as train.py's val predict (common.predict, bf16).
Check printed for the first variant: the same code WITHOUT overrides reproduces its stored train-only val preds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

import data as Dm
from common import CKPT, PREDS, ROOT, eval_ndcg, predict
from model import Ranker

# reviewer 16:45 UTC: durable md5-identical copy in nn_x/r6/stale_cf (stale_cf/README.txt); the session scratchpad is
# the fallback (it lives in /tmp and may be cleaned before the hcap stale-val read-out)
_CF_DUR = Path("<repo>/nn_x/r6/stale_cf")
_CF_TMP = Path("/tmp/scratchpad/review/shift/out/cf")
CF_BOTH = _CF_DUR if (_CF_DUR / "both_seq_cv_pop.npy").exists() else _CF_TMP
CF = ROOT / "solvers/out/stale/cf"
OVR = {**{n: CF_BOTH / f"both_{n}.npy" for n in ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
                                                 "it_hist_et3_luo", "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_norm",
                                                 "seq_cv_pop")},
       **{n: CF / f"pre2both_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s")}}
OUT = PREDS / "stale"
_feat_path = Dm.feat_path
ACTIVE = {"on": False}


def feat_path(split, name):
    if ACTIVE["on"] and split == "val" and name in OVR:
        return OVR[name]
    return _feat_path(split, name)


Dm.feat_path = feat_path  # SplitData.attach_xt resolves feature files through this module-level name


def ck_path(v):
    return ROOT / "nn/ckpt" / f"{v[3:]}.pt" if v.startswith("nn:") else CKPT / f"{v}.pt"


def main(variants):
    for p in OVR.values():
        assert p.exists(), f"missing counterfactual file {p}"
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True  # as train.py, whose final val predict wrote the stored preds
    va = Dm.SplitData("val", dev)
    log = open(OUT / "stale_pred.log", "a")
    for i, v in enumerate(variants):
        out = OUT / f"nn_{v.replace('nn:', '')}_staleval.npy"
        if out.exists():
            print(f"{v}: exists", flush=True)
            continue
        t0 = time.time()
        ck = torch.load(ck_path(v), map_location=dev, weights_only=False)
        m = Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        xt = ck["cfg"].get("xt_names") or []
        used = sorted(n for n in OVR if n in xt or f"has:{n}" in xt)
        msg = ""
        if i == 0:  # base reproduction check (no overrides) against the stored train-only val preds
            ACTIVE["on"] = False
            va.xt = None
            va.attach_xt(xt)
            pb = predict(m, va)
            ref = (ROOT / "preds" if v.startswith("nn:") else PREDS) / f"nn_{v.replace('nn:', '')}_trainonly_val.npy"
            if ref.exists():
                r = np.load(ref)
                msg = f"; base check vs stored: corr {np.corrcoef(r, pb)[0, 1]:.7f} max|d| {np.abs(r - pb).max():.2e}"
        ACTIVE["on"] = True
        va.xt = None
        va.attach_xt(xt)
        ps = predict(m, va)
        ACTIVE["on"] = False
        np.save(out, ps.astype(np.float32))
        s = (f"{time.strftime('%FT%TZ', time.gmtime())} {v}: stale-val ndcg {eval_ndcg(ps, va):.5f}, overridden inputs "
             f"used {used}{msg}, {time.time() - t0:.0f}s")
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
