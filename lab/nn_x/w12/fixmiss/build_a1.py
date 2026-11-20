"""Try file A1 (2026-11-18, lead; workflow a workflow run training stream item 1, measured in training/m6_msplit_robust.py):
FD16P20@m0.4_0.2 with the popularity-light part of the FINE-TUNED branch taken from the pop-light students' TRAIN-ONLY predictions on
feeds with M >= 50 (threshold 50 fixed before the scan). Everything else identical to the current best file. Writes
preds/ens_FD16P20_m04_02_A1kd_{ft,to}_{val,test}.npy and subs/cand_FD16P20_m04_02_A1kd_r4.parquet (refuses to overwrite).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/fixmiss/build_a1.py"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np
from pathlib import Path
from common import feed_meta, FLAT
from stack_val import feed_rank_z
from make_sub import make_sub
from metrics import ndcg_per_feed
P = Path("<repo>/preds"); S = Path("<repo>/subs"); TAG = "FD16P20_m04_02_A1kd"
base = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
pop = [f"abl_fd10_pop1kd05w192_s{s}" for s in range(951, 955)]
out = S / f"cand_{TAG}_r4.parquet"; assert not out.exists() and not list(P.glob(f"ens_{TAG}_*")), "exists"
res = {}
for sp, (ftsuf, tosuf) in {"val": ("n2cpv3s3_val_oof", "trainonly_val"), "test": ("n2cpv3s3_test", "trainonly_test")}.items():
    f, M, o = feed_meta(sp); MR = np.repeat(M, M)
    Z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    pl = lambda ms, suf: np.mean([Z(np.load(P / f"nn_{m}_{suf}.npy")) for m in ms], 0)
    Fb, Fp, Tb, Tp = pl(base, ftsuf), pl(pop, ftsuf), pl(base, tosuf), pl(pop, tosuf)
    w = np.where(MR >= 50, 0.2, 0.4).astype(np.float32)
    ft = Z((1 - w) * Z(Fb) + w * np.where(MR >= 50, Z(Tp), Z(Fp))); to = Z((1 - w) * Z(Tb) + w * Z(Tp))
    np.save(P / f"ens_{TAG}_ft_{sp}.npy", ft.astype(np.float32)); np.save(P / f"ens_{TAG}_to_{sp}.npy", to.astype(np.float32))
    res[sp] = (0.5 * ft + 0.5 * to, M, o)
s, M, o = res["val"]; yv = np.load(FLAT / "val/cand_target.npy")
cur = 0.5 * np.load(P / "ens_FD16P20_m04_02kd_ft_val.npy") + 0.5 * np.load(P / "ens_FD16P20_m04_02kd_to_val.npy")
e, c = ndcg_per_feed(s.astype(np.float32), yv, o, M), ndcg_per_feed(cur.astype(np.float32), yv, o, M); d = e - c
print(f"A1 val {e.mean():.5f} vs current best {c.mean():.5f}: {d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f}")
st = 0.5 * np.load(P / "ens_FD16P20_m04_02kd_ft_test.npy") + 0.5 * np.load(P / "ens_FD16P20_m04_02kd_to_test.npy")
print(f"test corr to current best {np.corrcoef(res['test'][0], st)[0, 1]:.6f}")
make_sub(res["test"][0].astype(np.float32), str(out)); print("written", out)
