"""hfill Stage A, INFORMATION ONLY (not part of the rule; the verdict is r_read.py's): test-side checks and magnitudes.
  * test: A0 pool == ens_FD16P20kd_to_test (bytes); A1 / A2 feeds with no recovered event bit-identical; per-feed rank
    correlation of the A1 / A2 train-only pool with the stored one; the pools are saved to pools/ens_FD16P20kd_<arm>_to_{val,test}.npy
  * val final score with the STORED fine-tuned branch: 0.5 z(ft) + 0.5 z(to arm) vs the stored final (the fine-tuned members
    were not replayed, so this is not the arm's full recipe).
"""
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>")
sys.path.insert(0, "<repo>/lab/src")
from avito_ranker.ensemble.build import feed_zscore  # noqa: E402
from avito_ranker.ensemble.feeds import load_feed_index  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from pmix import weights, wmean_se  # noqa: E402

ROOT = Path("<repo>")
HF = Path(__file__).resolve().parent
STORED = ROOT / "artifacts" / "predictions"
cfg = tomllib.loads((ROOT / "configs/ensembles/fd16p20.toml").read_text())
FULL, LIGHT = cfg["members"]["full_view"], cfg["members"]["popularity_light"]
W = cfg["weights"]["popularity_light"]
(HF / "pools").mkdir(exist_ok=True)


def pool(folder, split, fi):
    def p(ms):
        return np.mean([feed_zscore(fi, np.load(folder / f"nn_{m}_trainonly_{split}.npy")) for m in ms], 0)
    return ((1 - W) * feed_zscore(fi, p(FULL)) + W * feed_zscore(fi, p(LIGHT))).astype(np.float32)


for split in ("val", "test"):
    fi = load_feed_index(split)
    ref = np.load(STORED / f"ens_FD16P20kd_to_{split}.npy")
    fz = np.load(HF / "out" / f"fill_map_{split}.npz")
    touched = np.zeros(fi.n_feeds, bool)
    touched[fz["feeds_rec"]] = True
    rt = np.repeat(touched, fi.sizes)
    for arm in ("A0", "A1", "A2"):
        pl_ = pool(HF / "preds" / arm, split, fi)
        out = HF / "pools" / f"ens_FD16P20kd_{arm}_to_{split}.npy"
        if not out.exists():
            np.save(out, pl_)
        same = np.array_equal(pl_.view(np.uint32), ref.view(np.uint32))
        unt = int((pl_[~rt].view(np.uint32) != ref[~rt].view(np.uint32)).sum())
        # per-feed Spearman-like: correlation of within-feed ranks (feed_rank_z col 0) over touched rows
        from avito_ranker.ensemble.zscore import feed_rank_z
        ra = feed_rank_z(fi.offsets, fi.sizes, pl_)[:, 0]
        rb = feed_rank_z(fi.offsets, fi.sizes, ref)[:, 0]
        cr = np.corrcoef(ra[rt], rb[rt])[0, 1] if arm != "A0" else 1.0
        print(f"{split} {arm}: pool == stored {same}; untouched rows not bit-identical {unt}; touched feeds {touched.sum()} of "
              f"{fi.n_feeds} ({touched.mean():.3f}); within-feed rank corr on touched rows {cr:.5f}; "
              f"mean |z diff| on touched rows {np.abs(pl_[rt] - ref[rt]).mean():.4f}")

# val final score with the stored fine-tuned branch (info)
fi = load_feed_index("val")
y = np.load(ROOT / "data/flat/val/cand_target.npy")
ft = np.load(STORED / "ens_FD16P20kd_ft_val.npy")
ref_to = np.load(STORED / "ens_FD16P20kd_to_val.npy")
fin_ref = 0.5 * feed_zscore(fi, ft) + 0.5 * feed_zscore(fi, ref_to)
nd_ref = ndcg_per_feed(fin_ref.astype(np.float32), y, fi.offsets, fi.sizes)
Wm = weights()
print(f"val final (stored ft + stored to) {nd_ref.mean():.5f} (recipe record 0.40271)")
for arm in ("A1", "A2"):
    to = np.load(HF / "pools" / f"ens_FD16P20kd_{arm}_to_val.npy")
    fin = 0.5 * feed_zscore(fi, ft) + 0.5 * feed_zscore(fi, to)
    nd = ndcg_per_feed(fin.astype(np.float32), y, fi.offsets, fi.sizes)
    d = nd - nd_ref
    m, s = wmean_se(d)
    mp, sp = wmean_se(d, Wm["private"])
    print(f"  INFO final with stored ft + {arm} to: val {nd.mean():.5f} delta {m:+.5f} +- {s:.5f} (z {m / s:+.2f}) "
          f"pmix-priv {mp:+.5f} +- {sp:.5f}")
