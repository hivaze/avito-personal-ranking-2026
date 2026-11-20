"""hfill Stage A, step 1 (CPU): which checkpoint each stored train-only member prediction came from, and the stored pool.
For every FD16P20 member: (a) artifacts/predictions/nn_<m>_trainonly_val.npy == harness preds copy == @last copy (bytes);
(b) ckpt/<m>.pt and ckpt/<m>@last.pt hold equal state dicts (args keep='last' => the selected checkpoint is the last step);
(c) the train-only pool rebuilt from the 20 stored member files == artifacts/predictions/ens_FD16P20kd_to_val.npy (bytes)."""
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<repo>")
from avito_ranker.ensemble.build import feed_zscore  # noqa: E402
from avito_ranker.ensemble.feeds import load_feed_index  # noqa: E402

ROOT = Path("<repo>")
P = ROOT / "artifacts" / "predictions"
cfg = tomllib.loads((ROOT / "configs/ensembles/fd16p20.toml").read_text())
FULL, LIGHT = cfg["members"]["full_view"], cfg["members"]["popularity_light"]
W = cfg["weights"]["popularity_light"]


def harness(m):
    return ROOT / "lab/nn_x" / ("w5" if m.startswith("w5_") else "abl")


ok_all = True
for m in FULL + LIGHT:
    h = harness(m)
    for split in ("val", "test"):
        a = np.load(P / f"nn_{m}_trainonly_{split}.npy")
        b = np.load(h / "preds" / f"nn_{m}_trainonly_{split}.npy")
        c = np.load(h / "preds" / f"nn_{m}@last_trainonly_{split}.npy")
        same_ab, same_ac = bool(np.array_equal(a, b)), bool(np.array_equal(a, c))
        if not (same_ab and same_ac):
            ok_all = False
        print(f"{m} {split}: artifacts==harness {same_ab}, ==@last {same_ac}, maxdiff@last {np.abs(a - c).max():.3g}")
    k1 = torch.load(h / "ckpt" / f"{m}.pt", map_location="cpu", weights_only=False)
    k2 = torch.load(h / "ckpt" / f"{m}@last.pt", map_location="cpu", weights_only=False)
    eq = all(torch.equal(k1["state_dict"][k], k2["state_dict"][k]) for k in k1["state_dict"]) and k1["state_dict"].keys() == k2["state_dict"].keys()
    print(f"  ckpt {m}.pt == @last.pt: {eq}; keep={k1['args']['keep']} epochs={k1['args']['epochs']} last step={k2.get('step')}")
    ok_all &= eq

fi = load_feed_index("val")


def pool(members):
    return np.mean([feed_zscore(fi, np.load(P / f"nn_{m}_trainonly_val.npy")) for m in members], 0)


to = (1 - W) * feed_zscore(fi, pool(FULL)) + W * feed_zscore(fi, pool(LIGHT))
ref = np.load(P / "ens_FD16P20kd_to_val.npy")
print("pool rebuilt == ens_FD16P20kd_to_val:", np.array_equal(to.astype(np.float32), ref), "maxdiff", np.abs(to - ref).max())
print("ALL OK" if ok_all else "MISMATCH")
