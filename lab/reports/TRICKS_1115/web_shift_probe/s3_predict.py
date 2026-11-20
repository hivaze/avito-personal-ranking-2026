"""web_shift probe s3: CPU fp32 re-prediction of stored held-out students on VAL with the lag-equalised item-occurrence inputs
(arms A, B from s2_build.py) and with the stored inputs (baseline, same CPU path). No training; read-only on every project file.
Modelled on nn_x/w12/stale/veto/stale_cpu.py. Output: web_shift_probe/preds/cpu_<variant>_{base,A,B}.npy
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 ... .venv/bin/python s3_predict.py <variant> [...]
"""
import os, sys, time
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
import torch
torch.set_num_threads(12)
ROOT = Path("<repo>")
HERE = ROOT / "reports/TRICKS_1115/web_shift_probe"
OUT = HERE / "preds"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "nn_x/w5")); sys.path.insert(0, str(ROOT / "src"))
import data as Dm  # noqa: E402
from model import Ranker  # noqa: E402
from metrics import ndcg  # noqa: E402
NAMES = ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo")
_fp = Dm.feat_path
ARM = {"v": None}
def feat_path(split, name):
    if ARM["v"] and split == "val" and name in NAMES:
        return HERE / "cf" / f"{ARM['v']}_{name}.npy"
    return _fp(split, name)
Dm.feat_path = feat_path
T0 = time.time(); lg = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

@torch.inference_mode()
def predict(m, va):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    for fb in va.batches(np.arange(va.n), 65536, 1024, shuffle=False):
        b = va.gather(fb); va.scatter(out, b, m(b, use_pos=False))
    return out.numpy().astype(np.float32)

def main(variants):
    va = Dm.SplitData("val", torch.device("cpu"))
    y, M, off = va.target_np, va.M_np, va.cand_off_np
    xts = {}
    for v in variants:
        ck = torch.load(ROOT / "nn_x/w5/ckpt" / f"{v}.pt", map_location="cpu", weights_only=False)
        xt = tuple(ck["cfg"].get("xt_names") or [])
        assert all(n in xt for n in NAMES)
        if xt not in xts:
            d = {}
            for arm in (None, "A", "B"):
                ARM["v"] = arm; va.xt = None; va.attach_xt(list(xt)); d[arm or "base"] = va.xt
            ARM["v"] = None; xts[xt] = d
            lg(f"xt attached {len(xt)} cols; rows changed A {float((d['A'] != d['base']).any(1).float().mean()):.3f}, B {float((d['B'] != d['base']).any(1).float().mean()):.3f}")
        m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
        for arm in ("base", "A", "B"):
            o = OUT / f"cpu_{v}_{arm}.npy"
            if o.exists():
                continue
            va.xt = xts[xt][arm]; t1 = time.time(); p = predict(m, va); np.save(o, p)
            lg(f"{v} {arm}: val ndcg {ndcg(p, y, off, M):.5f} ({time.time()-t1:.0f}s)")
        del m

if __name__ == "__main__":
    main(sys.argv[1:])
