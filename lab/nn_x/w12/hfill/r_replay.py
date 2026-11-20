"""hfill Stage A replay (GPU, inference only): the 20 FD16P20 all-data members re-scored with masked history vectors filled.

Copy of the predict path: hdata_base.py = verbatim copy of lab/nn_x/w5/data.py (SplitData / gather / attach_xt), model_w5.py /
model_abl.py = verbatim copies of lab/nn_x/{w5,abl}/model.py (each member uses its own harness's model code), predict() below =
lab/nn_x/w5/common.py predict (TF32 on as in train.py, bf16 autocast, cand_budget 65536, max_feeds 1024). Checkpoints:
<harness>/ckpt/<member>.pt (== <member>@last.pt; the stored trainonly preds came from it: m1_verify_members.py).
Arms (the members are never changed):
  A0 unpatched                      -> must reproduce artifacts/predictions/nn_<m>_trainonly_<split>.npy to ~1e-5
  A1 model path filled              history events with emb_idx < 0 whose item has a row in an EARLIER split's table get that
                                    row (hfill_lib.load_fill: val <- train; test <- val, then train): h_emb re-pointed, the rows
                                    appended to the split's table, so he_raw / hemb_ok / hidx (all model inputs) see the vector
  A2 A1 + the 20 history-similarity ctx_* xt columns rebuilt from the filled history (a2_build_ctx.py, feats/fill/<split>/)
Output: preds/<arm>/nn_<member>_trainonly_<split>.npy (float32, flat cand order). Never writes outside lab/nn_x/w12/hfill/.
Usage: CUDA_VISIBLE_DEVICES=<6|7> python r_replay.py <split> <arms comma list>
"""
import importlib.util
import json
import os
import sys
import time
import tomllib
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("6", "7"):
    raise RuntimeError("hfill: set CUDA_VISIBLE_DEVICES to 6 or 7")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(12, int(os.environ.get("OMP_NUM_THREADS", "12") or 12)))
HF = Path(__file__).resolve().parent
sys.path.insert(0, str(HF))
import hdata_base  # noqa: E402
from hfill_lib import load_fill  # noqa: E402

ROOT = Path("<repo>")
STORED = ROOT / "artifacts" / "predictions"
CTX_FILLED = ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
              "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk",
              "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
T0 = time.time()


def log(*a):
    s = f"[{time.time() - T0:6.0f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODELS = {"w5": load_module("model_w5", HF / "model_w5.py"), "abl": load_module("model_abl", HF / "model_abl.py")}


class FillSplitData(hdata_base.SplitData):
    """hdata_base.SplitData + the hfill patch (fill=True) + an xt override directory (xt_dir: the CTX_FILLED columns are read
    from there instead of the shared feat_cache; same transform as attach_xt)."""

    def __init__(self, split, device, fill=False, xt_dir=None):
        super().__init__(split, device, max_hist=256)
        self.xt_dir = Path(xt_dir) if xt_dir else None
        self.fill_info = None
        if fill:
            filled, ext, info, feeds_rec = load_fill(split, len(self.table))
            h_old = self.h_emb
            self.h_emb = torch.from_numpy(filled).to(device)
            assert self.h_emb.dtype == h_old.dtype and self.h_emb.shape == h_old.shape
            self.table = torch.cat([self.table, torch.from_numpy(ext).to(device)], 0)
            self.fill_info = info
            log(f"[fill] {split}: {json.dumps(info)}; table now {tuple(self.table.shape)}")

    def attach_xt(self, names):
        cols = []
        n_over = 0
        for n in names:
            src = None
            if self.xt_dir is not None and n in CTX_FILLED:
                src = self.xt_dir / f"{n}.npy"
                n_over += 1
            if n.startswith("has:"):
                x = (~np.isnan(np.load(hdata_base.feat_path(self.split, n[4:])))).astype(np.float32)
            else:
                x = np.load(src if src is not None else hdata_base.feat_path(self.split, n)).astype(np.float32)
            if n.startswith("has:") or n.endswith("_rk"):
                pass
            elif n.endswith("_z"):
                x = np.clip(x, -5, 5)
            elif n.startswith(("seq_cv", "it_hist")):
                x = np.log1p(np.maximum(x, 0))
            else:
                x = np.clip(x, -10, 10)
            cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(self.device, torch.float16))
            del x
        self.xt = torch.stack(cols, 1)
        return n_over


@torch.no_grad()
def predict(model, data, feed_ids=None, cand_budget=65536, max_feeds=1024):
    """lab/nn_x/w5/common.py predict, verbatim."""
    model.eval()
    feed_ids = np.arange(data.n) if feed_ids is None else feed_ids
    out = torch.zeros(data.n_cand, dtype=torch.float32, device=data.device)
    for fb in data.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = data.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    model.train()
    return out.cpu().numpy()


def members():
    cfg = tomllib.loads((ROOT / "configs/ensembles/fd16p20.toml").read_text())
    return cfg["members"]["full_view"] + cfg["members"]["popularity_light"]


def main():
    split = sys.argv[1]
    arms = sys.argv[2].split(",")
    assert split in ("val", "test") and set(arms) <= {"A0", "A1", "A2"}
    torch.backends.cuda.matmul.allow_tf32 = True  # train.py main() sets this before the stored val / test predict
    dev = torch.device("cuda:0")
    for arm in arms:
        data = FillSplitData(split, dev, fill=arm in ("A1", "A2"),
                             xt_dir=HF / "feats" / "fill" / split if arm == "A2" else None)
        od = HF / "preds" / arm
        od.mkdir(parents=True, exist_ok=True)
        cur_xt = None
        for m in members():
            h = "w5" if m.startswith("w5_") else "abl"
            ck = torch.load(ROOT / "lab/nn_x" / h / "ckpt" / f"{m}.pt", map_location=dev, weights_only=False)
            xt = ck["cfg"].get("xt_names")
            assert not ck["cfg"].get("id_buckets") and not ck["cfg"].get("hist_ext")
            if xt != cur_xt:
                n_over = data.attach_xt(xt)
                cur_xt = xt
                log(f"{arm} {split}: attached {len(xt)} xt columns ({n_over} from {data.xt_dir})")
            model = MODELS[h].Ranker(**ck["cfg"]).to(dev)
            model.load_state_dict(ck["state_dict"])
            p = predict(model, data)
            np.save(od / f"nn_{m}_trainonly_{split}.npy", p.astype(np.float32))
            ref = np.load(STORED / f"nn_{m}_trainonly_{split}.npy")
            d = np.abs(p - ref)
            log(f"{arm} {split} {m}: max|d| vs stored {d.max():.3g}, mean|d| {d.mean():.3g}, rows changed {(d > 0).mean():.4f}")
            del model, ck
            torch.cuda.empty_cache()
        del data
        torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
