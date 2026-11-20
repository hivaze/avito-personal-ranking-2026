"""W25 G2: ratio-weighted LAE scores (the winner's predictions *= log1p(bayesian_ratio_C)), from build_lae + build_te outputs.
Columns: w25_grp_logratio (from build_te; S2 Bayesian group ratio, time-respecting), w25_grp_t1share (label-free type-1 share of the
group over train snapshot histories), products rdlae_bin / dan_bin / rdlae_int3 x log1p(ratio), rdlae_bin x log1p(t1share)."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, "<repo>/src")
ROOT = Path("<repo>"); CLU = ROOT / "data/clu_work"; WD = ROOT / "nn_x/w25/data"

t1share = np.load(WD / "grp_t1share_cl4096.npy")
for split in ("train", "val", "test"):
    g = np.load(CLU / f"cand_cl_4096_{split}.npy", mmap_mode="r")
    if split == "train":
        g = g[np.load(WD / "train/rows.npy")]
    g = np.asarray(g).astype(np.int64)
    ts = np.where(g >= 0, t1share[np.maximum(g, 0)], np.nan).astype(np.float32)
    lr = np.load(WD / f"{split}/w25_grp_logratio.npy")
    cols = {"w25_grp_t1share": ts}
    for n in ("w25_rdlae_bin", "w25_dan_bin", "w25_rdlae_int3"):
        cols[n + "_xlr"] = (np.load(WD / f"{split}/{n}.npy") * lr).astype(np.float32)
    cols["w25_rdlae_bin_xt1"] = (np.load(WD / f"{split}/w25_rdlae_bin.npy") * np.log1p(ts)).astype(np.float32)
    for n, v in cols.items():
        np.save(WD / f"{split}/{n}.npy", v)
    print(split, {n: (float(np.nanmean(v)), float(np.isnan(v).mean())) for n, v in cols.items()}, flush=True)
gp = WD / "groups.json"
G = json.loads(gp.read_text())
G["G2"] = ["w25_grp_logratio", "w25_grp_t1share", "w25_rdlae_bin_xlr", "w25_dan_bin_xlr", "w25_rdlae_int3_xlr", "w25_rdlae_bin_xt1"]
gp.write_text(json.dumps(G, indent=1))
print({k: len(v) for k, v in G.items()})
