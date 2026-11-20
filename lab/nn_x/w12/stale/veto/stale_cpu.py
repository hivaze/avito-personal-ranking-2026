"""veto: CPU fp32 stale-val re-prediction of stored checkpoints (no training, read-only on every project file).

CPU twin of nn_x/w5/stale_pred.py (scenario 'both_pre2both': private-like coverage for it_hist_{all,et1..et4}_luo and
seq_cv_{invd20,norm,pop} from nn_x/r6/stale_cf, pre2_fe / pre2_fe_z / pre2_fe_s from solvers/out/stale/cf; val inputs only).
Differences: device cpu, fp32 (the stored files are bf16 autocast on a GPU) -> compare CPU with CPU only. nn_x/w5/common.py is
NOT imported (it refuses to load without a GPU id); data.py / model.py are imported read-only (PYTHONDONTWRITEBYTECODE=1).

  cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
     OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 .venv/bin/python nn_x/w12/stale/veto/stale_cpu.py \
     [--fresh] <variant> [<variant> ...]
  variant = name in nn_x/w5/ckpt or nn_x/abl/ckpt (prefix w5_ / abl_), optional '@last'.
  Output: nn_x/w12/stale/veto/preds/cpu_<variant>_staleval.npy; with --fresh (or for the first variant of every distinct xt
  list) also cpu_<variant>_val.npy = the same path WITHOUT overrides, checked against the stored train-only val file.
"""
import os
import sys
import time
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: set CUDA_VISIBLE_DEVICES=\"\""
import numpy as np
import torch

torch.set_num_threads(12)
ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/stale/veto"
OUT = HERE / "preds"
sys.path.insert(0, str(ROOT / "nn_x/w5"))
sys.path.insert(0, str(ROOT / "src"))
import data as Dm  # noqa: E402  (nn_x/w5/data.py, read-only)
from model import Ranker  # noqa: E402  (nn_x/w5/model.py == nn_x/abl/model.py)
from metrics import ndcg  # noqa: E402

CF_BOTH = ROOT / "nn_x/r6/stale_cf"
CF = ROOT / "solvers/out/stale/cf"
OVR = {**{n: CF_BOTH / f"both_{n}.npy" for n in ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
                                                 "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop")},
       **{n: CF / f"pre2both_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s")}}
_feat_path = Dm.feat_path
ACTIVE = {"on": False}


def feat_path(split, name):
    if ACTIVE["on"] and split == "val" and name in OVR:
        return OVR[name]
    return _feat_path(split, name)


Dm.feat_path = feat_path
T0 = time.time()
lg = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def ck_path(v):
    d = ROOT / ("nn_x/abl/ckpt" if v.startswith("abl_") else "nn_x/w5/ckpt")
    return d / f"{v}.pt"


def stored_val(v):
    for d in (ROOT / "nn_x/w5/preds", ROOT / "nn_x/abl/preds", ROOT / "preds"):
        p = d / f"nn_{v}_trainonly_val.npy"
        if p.exists():
            return p
    return None


@torch.inference_mode()
def predict(m, va):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    for fb in va.batches(np.arange(va.n), 65536, 1024, shuffle=False):
        b = va.gather(fb)
        va.scatter(out, b, m(b, use_pos=False))
    return out.numpy().astype(np.float32)


def feed_corr(a, b, M, off):
    fid = np.repeat(np.arange(len(M)), M)
    def z(s):
        s = s.astype(np.float64); mu = np.bincount(fid, s, len(M)) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
        return (s - mu[fid]) / sd[fid]
    return float((np.bincount(fid, z(a) * z(b), len(M)) / M).mean())


def main(argv):
    fresh_all = "--fresh" in argv
    variants = [a for a in argv if not a.startswith("--")]
    for p in OVR.values():
        assert p.exists(), f"missing counterfactual file {p}"
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    va = Dm.SplitData("val", dev)
    y, M, off = va.target_np, va.M_np, va.cand_off_np
    log = open(OUT / "stale_cpu.log", "a")
    xts = {}  # xt name tuple -> (fresh xt, stale xt, used overrides)
    seen = set()
    for v in variants:
        o_s = OUT / f"cpu_{v}_staleval.npy"
        o_f = OUT / f"cpu_{v}_val.npy"
        ck = torch.load(ck_path(v), map_location="cpu", weights_only=False)
        xt = tuple(ck["cfg"].get("xt_names") or [])
        want_fresh = (fresh_all or xt not in seen) and not o_f.exists()
        seen.add(xt)
        if o_s.exists() and not want_fresh:
            lg(f"{v}: exists"); continue
        if xt not in xts:
            ACTIVE["on"] = False; va.xt = None; va.attach_xt(list(xt)); a = va.xt
            ACTIVE["on"] = True; va.xt = None; va.attach_xt(list(xt)); b = va.xt
            ACTIVE["on"] = False
            used = sorted(n for n in OVR if n in xt or f"has:{n}" in xt)
            xts[xt] = (a, b, used)
            lg(f"xt attached: {len(xt)} columns, overridden {used}; rows changed {float((a != b).any(1).float().mean()):.3f}")
        a, b, used = xts[xt]
        m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
        t1 = time.time(); msg = ""
        if want_fresh:
            va.xt = a; pf = predict(m, va); np.save(o_f, pf)
            sp = stored_val(v)
            if sp is not None:
                r = np.load(sp).astype(np.float32)
                msg = (f"; fresh CPU ndcg {ndcg(pf, y, off, M):.5f} vs stored {ndcg(r, y, off, M):.5f}, "
                       f"within-feed corr {feed_corr(pf, r, M, off):.6f}")
        if not o_s.exists():
            va.xt = b; ps = predict(m, va); np.save(o_s, ps)
        else:
            ps = np.load(o_s)
        s = (f"{time.strftime('%FT%TZ', time.gmtime())} {v}: stale-val ndcg {ndcg(ps, y, off, M):.5f}, overrides {len(used)}"
             f"{msg}, {time.time() - t1:.0f}s")
        lg(s); log.write(s + "\n"); log.flush()
        del m


if __name__ == "__main__":
    main(sys.argv[1:])
