"""why/fit F3 (CPU, read-only on stored predictions): per-feed loss components / teacher agreement on VAL and HOLDOUT for every model
of fwd_train.model_list(). Teacher on val = mean raw logit of the 11 teacher members' train-only val preds (= how teacher_mean11.npy was
built on train: dump_teacher.py sums raw predict() outputs); the same mean over their stored holdout preds is checked against
teacher_mean11.npy at the holdout rows. Output: <OUT>/vh/<name>.npz (per-feed stats) + <OUT>/vh/teacher.npz; log on stdout."""
import os
import sys

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import Meta, stats_dict, ndcg_fast, feed_rank_z  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

R = Path("<repo>")
OUT = "/tmp/scratchpad/why/fit"
os.makedirs(f"{OUT}/vh", exist_ok=True)
m = Meta()
TM = [("preds", f"r6_cd_m02_r{r}") for r in range(101, 108)] + [("nn_x/w5/preds", f"w5_e10_base_s{s}") for s in range(951, 955)]
tv = np.zeros(m.n["val"], np.float64)
th = np.zeros(m.n["hold"], np.float64)
for d, v in TM:
    tv += np.load(R / d / f"nn_{v}_trainonly_val.npy").astype(np.float64)
    th += np.load(R / d / "holdout" / f"nn_{v}_holdout.npy").astype(np.float64)
tv = (tv / len(TM)).astype(np.float32)
th = (th / len(TM)).astype(np.float32)
# check against teacher_mean11.npy at the holdout rows (feed-centred comparison)
meta = np.load(f"{OUT}/meta.npz")
t_file = meta["hold_t"]
oh, Mh = m.oh, m.Mh
fid = np.repeat(np.arange(len(Mh)), Mh)
cen = lambda x: x - (np.bincount(fid, x.astype(np.float64)) / Mh)[fid]
print(f"teacher check on holdout rows: max |centred(mean of 11 stored holdout preds) - centred(teacher_mean11)| "
      f"{np.abs(cen(th) - cen(t_file)).max():.3e}; corr {np.corrcoef(cen(th), cen(t_file))[0, 1]:.6f}", flush=True)
print(f"teacher NDCG raw-mean: val {ndcg_fast(tv, m.yv, m.ov, m.Mv).mean():.5f} holdout {ndcg_fast(t_file, m.yh, m.oh, m.Mh).mean():.5f}; "
      f"z-pool val {ndcg_fast(np.mean([m.z('val', np.load(R / d / f'nn_{v}_trainonly_val.npy')) for d, v in TM], 0), m.yv, m.ov, m.Mv).mean():.5f}",
      flush=True)
np.savez(f"{OUT}/vh/teacher.npz", val=tv, hold=t_file,
         **{f"val_{k}": x for k, x in stats_dict(tv, tv, m.yv, m.ov, m.Mv).items()},
         **{f"hold_{k}": x for k, x in stats_dict(t_file, t_file, m.yh, m.oh, m.Mh).items()})
sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("fwdl", "<repo>/lab/nn_x/w12/moe/why/fit/fwd_list.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)
SUF = sys.argv[1] if len(sys.argv) > 1 else ""  # "@last": the last checkpoints
for v, val_path, hold_path in fl.pred_list(SUF):
    dst = f"{OUT}/vh/{v}.npz"
    if os.path.exists(dst):
        continue
    sv = np.load(val_path).astype(np.float32)
    sh = np.load(hold_path).astype(np.float32)
    dv = stats_dict(sv, tv, m.yv, m.ov, m.Mv)
    dh = stats_dict(sh, t_file, m.yh, m.oh, m.Mh)
    np.savez(dst, **{f"val_{k}": x for k, x in dv.items()}, **{f"hold_{k}": x for k, x in dh.items()})
    print(f"{v}: val nd {dv['nd'].mean():.5f} kl {np.nanmean(dv['kl']):.4f} r {np.nanmean(dv['r']):.4f} | hold nd {dh['nd'].mean():.5f} "
          f"kl {np.nanmean(dh['kl']):.4f} r {np.nanmean(dh['r']):.4f}", flush=True)
print("DONE")
