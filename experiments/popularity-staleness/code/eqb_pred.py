# Inference with lag-equalised inputs: the 5 item-occurrence inputs of validation rows are replaced by
# copies adjusted to the test lag (half dose), every checkpoint and every other input unchanged. The
# same script scores the stored inputs as the same-path reference.
# Run through eqb_launch.sh at 23:14 UTC on 2026-11-15 on GPU N:
# eqb_pred.py val base,B val16 <16 all-data wide members>
# Produced preds/gpu_<member>_<arm>_val.npy (not copied); eqb_read.py then gave file-level validation
# -0.00024 +- 0.00017 (../results/eqb_read_val.txt): closed.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/eqb_1115/eqb_pred.py
#
"""EQB file read (ledger nn_x/w5/logs/E_kd_predeclared.txt block 2026-11-15T23:08:02Z item Q6; details nn_x/w12/eqb_1115/PREDECLARED.txt).
GPU inference only, on the stored files' own path (nn_x/w5/common.predict: bf16 autocast, tf32 on). NEW file modelled on nn_x/w5/stale_pred.py
(the feat_path switch); read-only on every checkpoint / feature / prediction file of the project; reads no label.

  CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/eqb_1115/eqb_pred.py <val|test> <arm,arm> <tag> <member> [<member> ...]
     arm 'base' = stored inputs (same-path reference; compared with the stored preds/nn_<v>_trainonly_<split>.npy)
     arm 'B'    = the 5 it_hist_*_luo inputs replaced by the half-dose lag-equalised copies
                  (val: reports/TRICKS_1115/web_shift_probe/cf/B_<name>.npy, test: .../cf_test/B_<name>.npy)
  -> nn_x/w12/eqb_1115/preds/gpu_<member>_<arm>_<split>.npy (never overwrites), one log line per file in logs/pred_<tag>.log (stdout)
"""
import os
import sys
import time
from pathlib import Path

HERE = Path("<repo>/nn_x/w12/eqb_1115")
split, arms, tag = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
variants = sys.argv[4:]
assert split in ("val", "test") and all(a in ("base", "B") for a in arms) and variants
(HERE / "logs" / f"pid_{tag}.txt").write_text(str(os.getpid()))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "<repo>/nn_x/w5")
import data as Dm  # noqa: E402
from common import CKPT, ROOT, predict  # noqa: E402
from model import Ranker  # noqa: E402

OUT = HERE / "preds"
NAMES = ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo")
CFD = {"val": ROOT / "reports/TRICKS_1115/web_shift_probe/cf", "test": ROOT / "reports/TRICKS_1115/web_shift_probe/cf_test"}
_fp = Dm.feat_path
ARM = {"v": None}


def feat_path(sp, name):
    if ARM["v"] and name in NAMES:
        return CFD[sp] / f"{ARM['v']}_{name}.npy"
    return _fp(sp, name)


Dm.feat_path = feat_path  # SplitData.attach_xt resolves feature files through this module-level name
T0 = time.time()


def lg(*a):
    print(f"{time.strftime('%FT%TZ', time.gmtime())} [{time.time() - T0:5.0f}s]", *a, flush=True)


def main():
    dev = torch.device("cuda:0")
    torch.zeros(1, device=dev)  # makes the process visible in nvidia-smi at once (the launcher holds the GPU lock until then)
    torch.backends.cuda.matmul.allow_tf32 = True  # as train.py, whose final predict wrote the stored files
    for a in arms:
        if a != "base":
            for n in NAMES:
                assert (CFD[split] / f"{a}_{n}.npy").exists(), f"missing {a}_{n}"
    d = Dm.SplitData(split, dev)
    lg(f"split {split}: {d.n} feeds, {d.n_cand} rows; arms {arms}; {len(variants)} members; GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    xts = {}
    for v in variants:
        ck = torch.load(CKPT / f"{v}.pt", map_location=dev, weights_only=False)
        assert ck["args"]["max_hist"] == 256 and not ck["cfg"].get("id_buckets"), "member needs another data setup"
        xt = tuple(ck["cfg"].get("xt_names") or [])
        assert all(n in xt for n in NAMES)
        if xt not in xts:
            dd = {}
            for a in arms:
                ARM["v"] = None if a == "base" else a
                d.xt = None
                d.attach_xt(list(xt))
                dd[a] = d.xt
            ARM["v"] = None
            xts[xt] = dd
            if "base" in dd:
                for a in arms:
                    if a != "base":
                        ch = (dd[a] != dd["base"])
                        lg(f"xt attached ({len(xt)} columns): arm {a} changes {float(ch.any(1).float().mean()):.4f} of rows, columns changed "
                           f"{[xt[i] for i in torch.nonzero(ch.any(0)).flatten().tolist()]}")
        m = Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        for a in arms:
            o = OUT / f"gpu_{v}_{a}_{split}.npy"
            if o.exists():
                lg(f"{v} {a}: exists")
                continue
            d.xt = xts[xt][a]
            t1 = time.time()
            p = predict(m, d).astype(np.float32)
            msg = ""
            if a == "base":
                r = np.load(ROOT / "preds" / f"nn_{v}_trainonly_{split}.npy")
                msg = (f" | vs stored preds/nn_{v}_trainonly_{split}.npy: corr {np.corrcoef(r, p)[0, 1]:.8f}, max|d| {np.abs(r - p).max():.3e}, "
                       f"identical {bool(np.array_equal(r, p))}")
            np.save(o, p)
            lg(f"{v} {a} {split}: saved {o.name} ({time.time() - t1:.0f}s){msg}")
        del m
        torch.cuda.empty_cache()
    lg("done")


if __name__ == "__main__":
    main()
