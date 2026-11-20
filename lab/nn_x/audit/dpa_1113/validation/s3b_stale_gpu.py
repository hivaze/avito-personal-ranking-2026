"""S3b (inference-only probe, GPU N, one process): stale-val predictions ('both_pre2both' scenario of the project's
stale_pred.py, + the pop1 input xp_imp_c50 at the same coverage cut) for the TRAIN-ONLY members of every pool with a
public score. Read-only on every checkpoint / feature; imports the code dir's own stale_pred / data / model modules and
writes ONLY to <scratch>/dpa/validation/out/stale/.

  cd <scratch>/dpa/validation && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 \
      <repo>/.venv/bin/python s3b_stale_gpu.py <codedir: w5|r6|abl> <variant> ...   ('nn:<v>' = nn/ckpt/<v>.pt)
For every variant: (1) base prediction without overrides, compared with the stored preds/nn_<v>_trainonly_val.npy;
(2) scenario 'both' (+ xp cut <20713); (3) for models that read xp_imp_c50 also 'both' + xp cut <20714 (public-like).
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
code = sys.argv[1]
CODEDIR = Path("<repo>/nn_x") / code
sys.path.insert(0, str(CODEDIR))
os.chdir(CODEDIR)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import stale_pred as SP  # noqa: E402  (module-level: installs the feat_path override hook; writes nothing)

OUT = HERE / "out/stale"
OUT.mkdir(parents=True, exist_ok=True)
XP = {"cut20713": HERE / "out/cf/xp_imp_c50_cut20713_val.npy", "cut20714": HERE / "out/cf/xp_imp_c50_cut20714_val.npy"}
for p in list(SP.OVR.values()) + list(XP.values()):
    assert Path(p).exists(), p
STORED = Path("<repo>/preds")


def run(variants):
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    va = SP.Dm.SplitData("val", dev)
    log = open(OUT / "stale_gpu.log", "a")
    base_ovr = dict(SP.OVR)
    for v in variants:
        name = v.replace("nn:", "")
        t0 = time.time()
        ck = torch.load(SP.ck_path(v), map_location=dev, weights_only=False)
        m = SP.Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        xt = ck["cfg"].get("xt_names") or []
        # base
        SP.ACTIVE["on"] = False; va.xt = None; va.attach_xt(xt)
        pb = SP.predict(m, va)
        ref = STORED / f"nn_{name}_trainonly_val.npy"
        chk = "no stored ref"
        if ref.exists():
            r = np.load(ref)
            chk = f"base vs stored corr {np.corrcoef(r, pb)[0, 1]:.7f} max|d| {np.abs(r - pb).max():.2e}"
        np.save(OUT / f"nn_{name}_base.npy", pb.astype(np.float32))
        scen = {"both": "cut20713"}
        if "xp_imp_c50" in xt:
            scen["both_xp14"] = "cut20714"
        msgs = []
        for sname, cut in scen.items():
            SP.OVR.clear(); SP.OVR.update(base_ovr); SP.OVR["xp_imp_c50"] = XP[cut]
            used = sorted(n for n in SP.OVR if n in xt or f"has:{n}" in xt)
            SP.ACTIVE["on"] = True; va.xt = None; va.attach_xt(xt)
            ps = SP.predict(m, va)
            SP.ACTIVE["on"] = False
            np.save(OUT / f"nn_{name}_{sname}.npy", ps.astype(np.float32))
            msgs.append(f"{sname}: ndcg {SP.eval_ndcg(ps, va):.5f} (base {SP.eval_ndcg(pb, va):.5f}) overrides {used}")
        s = f"{time.strftime('%FT%TZ', time.gmtime())} [{code}] {v}: {chk}; " + "; ".join(msgs) + f"; {time.time() - t0:.0f}s"
        print(s, flush=True); log.write(s + "\n"); log.flush()
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run(sys.argv[2:])
