"""SYNTHESIS check DOMTOK: CPU fp32 re-prediction of stored all-data wide students on VAL with the inference domain token changed (rule: PREDECLARED.txt next to this file).
No training; read-only on every project file. Modelled on reports/TRICKS_1115/web_shift_probe/s3_predict.py. Output: domtok/preds/cpu_<variant>_{base,prod,mix}.npy
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 ... .venv/bin/python domtok_predict.py <with_base 0|1> <variant> [...]
"""
import os, sys, time
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
import torch
torch.set_num_threads(12)
ROOT = Path("<repo>")
HERE = ROOT / "reports/TRICKS_1115/synth_checks/domtok"
OUT = HERE / "preds"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "nn_x/w5")); sys.path.insert(0, str(ROOT / "src"))
import data as Dm  # noqa: E402
from model import Ranker  # noqa: E402
from metrics import ndcg  # noqa: E402
T0 = time.time(); lg = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

@torch.inference_mode()
def predict(m, va):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    for fb in va.batches(np.arange(va.n), 65536, 1024, shuffle=False):
        b = va.gather(fb); va.scatter(out, b, m(b, use_pos=False))
    return out.numpy().astype(np.float32)

def main(with_base, variants):
    va = Dm.SplitData("val", torch.device("cpu"))
    y, M, off = va.target_np, va.M_np, va.cand_off_np
    xt_now = None
    for i, v in enumerate(variants):
        ck = torch.load(ROOT / "nn_x/w5/ckpt" / f"{v}.pt", map_location="cpu", weights_only=False)
        xt = tuple(ck["cfg"].get("xt_names") or [])
        if xt != xt_now:
            va.xt = None; va.attach_xt(list(xt)); xt_now = xt
        m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
        assert m.domain and m.infer_domain == 1, "expects a model with the domain token and the random-like inference default"
        W0 = m.dom.weight.detach().clone()
        arms = (["base"] if (with_base and i < 2) else []) + ["prod", "mix"]
        if os.environ.get("DOMTOK_ARMS"):  # e.g. DOMTOK_ARMS=base: the stored-token CPU pass only (added 22:1xZ so that the reference pool is CPU fp32 for all 16 members)
            arms = os.environ["DOMTOK_ARMS"].split(",")
        for arm in arms:
            o = OUT / f"cpu_{v}_{arm}.npy"
            if o.exists():
                continue
            with torch.no_grad():
                m.dom.weight.copy_(W0); m.infer_domain = 1
                if arm == "prod": m.infer_domain = 0
                if arm == "mix": m.dom.weight[1] = 0.5 * (W0[0] + W0[1])
            t1 = time.time(); p = predict(m, va); np.save(o, p)
            msg = ""
            if arm == "base":
                r = np.load(ROOT / "preds" / f"nn_{v}_trainonly_val.npy"); msg = f" | vs stored: corr {np.corrcoef(r, p)[0, 1]:.7f} max|d| {np.abs(r - p).max():.2e} stored ndcg {ndcg(r, y, off, M):.5f}"
            lg(f"{v} {arm}: val ndcg {ndcg(p, y, off, M):.5f} ({time.time()-t1:.0f}s){msg}")
        del m

if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:])
