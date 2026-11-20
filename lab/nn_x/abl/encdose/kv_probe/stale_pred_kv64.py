"""ENCODER-DOSE (C): stale-val predictions for --kv64 checkpoints (ledger nn_x/abl/ledger/ledger.md section 10.C): nn_x/abl/stale_pred.py's
scenario 'both_pre2both' (its module is imported, so the SAME feat_path override and counterfactual files act), with the model rebuilt by
kv_adapter.load_ranker_kv64 (Ranker(**cfg) fails on cfg['kv64']) and the 64-d states attached to the val SplitData (kv_data.attach_kv64).
The kv states have no private-like counterfactual (as the time arm's tm_* inputs): under the stale scenario they keep their val values and
only the popularity / pre2 overrides act. Writes nn_x/abl/preds/stale/nn_<v>_staleval.npy (+ a .log line), read by abl_eval's stale block.

  cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=<4|5> OMP_NUM_THREADS=12 ../../.venv/bin/python encdose/kv_probe/stale_pred_kv64.py \
      abl_kv64_s23 abl_kv64_s23@last abl_kv64_s24 abl_kv64_s24@last
The first variant also gets the base-reproduction check (no overrides) against its stored train-only val preds.
"""
import os
import sys
import time

import numpy as np
import torch

ABL = "<repo>/nn_x/abl"
sys.path.insert(0, ABL)
sys.path.insert(1, f"{ABL}/encdose/kv_probe")
import stale_pred as SP  # noqa: E402  (installs the feat_path override on nn_x/abl/data; OVR / CF paths / OUT)
import data as Dm  # noqa: E402
from common import PREDS, ROOT, eval_ndcg, predict  # noqa: E402
from kv_adapter import load_ranker_kv64  # noqa: E402
from kv_data import attach_kv64  # noqa: E402


def main(variants):
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("4", "5"), "ENCODER-DOSE: GPUs N only"
    for p in SP.OVR.values():
        assert p.exists(), f"missing counterfactual file {p}"
    SP.OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    va = Dm.SplitData("val", dev)
    attach_kv64(va, ROOT / "data/feats/val")
    log = open(SP.OUT / "stale_pred.log", "a")
    for i, v in enumerate(variants):
        out = SP.OUT / f"nn_{v}_staleval.npy"
        if out.exists():
            print(f"{v}: exists", flush=True)
            continue
        t0 = time.time()
        ck = torch.load(SP.ck_path(v), map_location=dev, weights_only=False)
        assert ck["cfg"].get("kv64"), f"{v}: not a --kv64 checkpoint (use stale_pred.py)"
        m = load_ranker_kv64(ck, dev)
        xt = ck["cfg"].get("xt_names") or []
        used = sorted(n for n in SP.OVR if n in xt or f"has:{n}" in xt)
        msg = ""
        if i == 0:
            SP.ACTIVE["on"] = False
            va.xt = None
            va.attach_xt(xt)
            pb = predict(m, va)
            ref = PREDS / f"nn_{v}_trainonly_val.npy"
            if ref.exists():
                r = np.load(ref)
                msg = f"; base check vs stored: corr {np.corrcoef(r, pb)[0, 1]:.7f} max|d| {np.abs(r - pb).max():.2e}"
        SP.ACTIVE["on"] = True
        va.xt = None
        va.attach_xt(xt)
        ps = predict(m, va)
        SP.ACTIVE["on"] = False
        np.save(out, ps.astype(np.float32))
        s = (f"{time.strftime('%FT%TZ', time.gmtime())} {v} [kv64]: stale-val ndcg {eval_ndcg(ps, va):.5f}, overridden inputs used {used}"
             f"{msg}, {time.time() - t0:.0f}s")
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
