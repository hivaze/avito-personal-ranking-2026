"""S3 robustness (leave-one-hedge-out, leave-one-family-out) and test-side file for the lead (solvers/out/errors/cand_S3_*.npy)."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz, PRIV_BOUND
import numpy as np
from scipy.stats import spearmanr

v12 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
v15 = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
v13 = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr2_mix1"]


def build(split, members):
    f, M, off, fid = meta(split)
    sfx = ("trainonly_val", "n2cpf_val_oof") if split == "val" else ("trainonly_test", "n2cpf_test")
    zt = np.mean([fz(np.load(f"{ROOT}/preds/nn_{m}_{sfx[0]}.npy"), fid, M) for m in members], 0)
    zf = np.mean([fz(np.load(f"{ROOT}/preds/nn_{m}_{sfx[1]}.npy"), fid, M) for m in members], 0)
    return 0.5 * fz(zt, fid, M) + 0.5 * fz(zf, fid, M)


f, M, off, fid = meta("val")
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
base = ndcg_per_feed(build("val", v12), y, off, M)
full = v12 + v15 + v13
for lab, mem in [("S3", full)] + [(f"S3 - {h}", [m for m in full if m != h]) for h in v13] + [("S3 - v15", v12 + v13), ("S3 - v12", v15 + v13)]:
    d = ndcg_per_feed(build("val", mem), y, off, M) - base
    print(f"{lab:22s} n={len(mem):2d}  d {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}")
zt = build("test", full)
zv = build("val", full)
np.save(f"<repo>/solvers/out/errors/cand_S3_val.npy", zv.astype(np.float32))
np.save(f"<repo>/solvers/out/errors/cand_S3_test.npy", zt.astype(np.float32))
ft, Mt, offt, fidt = meta("test")
v11t = 0.5 * fz(np.load(f"{ROOT}/preds/nn_final5_test.npy"), fidt, Mt) + 0.5 * fz(np.load(f"{ROOT}/preds/nn_final5_trainonly_test.npy"), fidt, Mt)
print("test z-corr S3 vs v11:", np.corrcoef(zt, v11t)[0, 1], " global Spearman:", spearmanr(zt, v11t).correlation)
print("finite:", np.isfinite(zt).all(), "len", len(zt), "test cands", Mt.sum())
