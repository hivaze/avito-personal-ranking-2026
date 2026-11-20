"""Step 4 power (CPU; labels are read ONLY on the two REFERENCE re-cool pools of the harness, i.e. the harness's own A/A null
w5_{k,l}cd_ref2 vs w5_{k,l}cd_ref, 16 vs 16; no extension arm exists, nothing about the extension's effect is read).
Gives the SE of the harness contrast (16-pool, train-only, member-equal per-feed z-mean, plain per-feed SE as in
nn_x/w12/kd/arm_read8.py) on all feeds and on the bucket of feeds the extension reaches (capped-only k0 arrays), and the power
of a 16-re-cool test for the effect sizes the refill replay put on the table.
  python p_power.py > out_p_power.txt
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, "<repo>/lab/src")
from metrics import ndcg_per_feed  # noqa: E402  (pure numpy)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import DATA_OUT, Flat  # noqa: E402

W = "<repo>/lab/nn_x/w5/preds"
F = Flat()
hold, _ = F.holdout()  # sorted, = the holdout_meta order
fv = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
Mv, ov = fv["M"].to_numpy().astype(np.int64), fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float32)
Mh, oh = np.load(f"{W}/holdout/holdout_meta.npy").astype(np.int64)
yh = np.load(f"{W}/holdout/holdout_y.npy").astype(np.float32)
assert (Mh == F.M["train"][hold]).all()


def z(off, M, x):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, weights=x, minlength=len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x.astype(np.float64) ** 2, minlength=len(M)) / M - mu ** 2, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


def pool(names):
    zv = np.mean([z(ov, Mv, np.load(f"{W}/nn_{v}_trainonly_val.npy").astype(np.float64)) for v in names], 0)
    zh = np.mean([z(oh, Mh, np.load(f"{W}/holdout/nn_{v}_holdout.npy").astype(np.float64)) for v in names], 0)
    return ndcg_per_feed(zv, yv, ov, Mv), ndcg_per_feed(zh, yh, oh, Mh)


mem = lambda arm: [f"w5_{st}cd_{arm}_s{x}" for st in "kl" for x in range(951, 959)]
rv, rh = pool(mem("ref"))
r2v, r2h = pool(mem("ref2"))
dv, dh = r2v - rv, r2h - rh
se = lambda d: float(d.std() / np.sqrt(len(d)))
ext_v = np.load(f"{DATA_OUT}/val/ext_n.npy") > 0
ext_v &= np.load(f"{DATA_OUT}/val/ext_capped.npy")
ext_t = (np.load(f"{DATA_OUT}/train/ext_n.npy") > 0) & np.load(f"{DATA_OUT}/train/ext_capped.npy")
ext_h = ext_t[hold]
print("## Harness A/A null: 16-pool ref2 - ref (train-only re-cool references), plain per-feed SE as arm_read8.py")
print(f"val all ({len(dv)}): {dv.mean():+.5f} +- {se(dv):.5f} | bucket = extended, capped-only k0 ({ext_v.sum()}): {dv[ext_v].mean():+.5f} +- {se(dv[ext_v]):.5f} "
      f"| rest: {dv[~ext_v].mean():+.5f} +- {se(dv[~ext_v]):.5f}")
print(f"holdout all ({len(dh)}): {dh.mean():+.5f} +- {se(dh):.5f} | bucket ({ext_h.sum()}): {dh[ext_h].mean():+.5f} +- {se(dh[ext_h]):.5f} "
      f"| rest: {dh[~ext_h].mean():+.5f} +- {se(dh[~ext_h]):.5f}")
print(f"per-feed sd of the A/A difference: val all {dv.std():.4f}, bucket {dv[ext_v].std():.4f}; holdout all {dh.std():.4f}, bucket {dh[ext_h].std():.4f}")
print(f"reference pool level on the bucket: val {rv[ext_v].mean():.4f} (rest {rv[~ext_v].mean():.4f}); holdout {rh[ext_h].mean():.4f} (rest {rh[~ext_h].mean():.4f})")

fv_, fh_ = ext_v.mean(), ext_h.mean()
sv_all, sh_all, sv_b, sh_b = se(dv), se(dh), se(dv[ext_v]), se(dh[ext_h])
print("\n## Power of a 16-re-cool test (arm vs ref, same harness), per-feed effect on the bucket = delta; 0 elsewhere")
print("| assumed per-bucket-feed effect (val / holdout) | source | pool delta val / holdout | z val / holdout (all feeds) | pooled z all feeds | "
      "P(pooled z >= 2.5) | P(>= 1.96) | bucket z val / holdout | pooled bucket z | P(bucket pooled z >= 2.5) | P(>= 1.96) | P(>= 1.645) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (ev, eh, src) in ((0.00269, 0.00117, "replay proxy c - b, design 1 (val / holdout read sets)"),
                      (0.00173, 0.00173, "replay proxy, pooled holdout + val"),
                      (0.00087, 0.00087, "untrained d1 - a, pooled (= the re-cool's start point)"),
                      (0.000865, 0.000865, "half of the pooled proxy"),
                      (0.0, 0.0, "no effect")):
    pdv, pdh = fv_ * ev, fh_ * eh
    zv, zh = pdv / sv_all, pdh / sh_all
    zp = (zv + zh) / np.sqrt(2)
    bzv, bzh = ev / sv_b, eh / sh_b
    bzp = (bzv + bzh) / np.sqrt(2)
    print(f"| {ev:+.5f} / {eh:+.5f} | {src} | {pdv:+.5f} / {pdh:+.5f} | {zv:.2f} / {zh:.2f} | {zp:.2f} | {1 - norm.cdf(2.5 - zp):.3f} | {1 - norm.cdf(1.96 - zp):.3f} | "
          f"{bzv:.2f} / {bzh:.2f} | {bzp:.2f} | {1 - norm.cdf(2.5 - bzp):.3f} | {1 - norm.cdf(1.96 - bzp):.3f} | {1 - norm.cdf(1.645 - bzp):.3f} |")
print("(power ignores the requirement that both deltas be >= 0, i.e. it is an upper bound; bucket SEs from the A/A null above)")
print(f"\nbucket shares: val {fv_:.4f}, holdout {fh_:.4f}; test {np.mean((np.load(f'{DATA_OUT}/test/ext_n.npy') > 0) & np.load(f'{DATA_OUT}/test/ext_capped.npy')):.4f}")
