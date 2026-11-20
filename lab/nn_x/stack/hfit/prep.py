"""Cache family z-means (holdout train-only; val/test ft+to) + holdout user ids + val slices for STUDY A.
  OMP_NUM_THREADS=12 python nn_x/stack/hfit/prep.py"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "12"); os.environ.setdefault("NUMBA_NUM_THREADS", "12")
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta
from stack_val import feed_rank_z

ROOT = "<repo>"; P = f"{ROOT}/preds"; C = f"{ROOT}/nn_x/stack/hfit/cache"
FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"],
    "F7": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
}
# new from-scratch m02 members: include those with complete ft + to + holdout files
import glob, re
for tag, pat in [("V27", "v27_gru_m02_s"), ("V28", "v28_lr1_m02_s"), ("V29", "v29_d192_m02_s"), ("V30", "v30_f7_m02_s")]:
    ms = sorted({re.match(r"nn_(.*)_trainonly_val\.npy", os.path.basename(p)).group(1) for p in glob.glob(f"{P}/nn_{pat}*_trainonly_val.npy")})
    ok = [m for m in ms if all(os.path.exists(f"{P}/nn_{m}_{s}.npy") for s in ["trainonly_val", "trainonly_test", "n2cpv3_val_oof", "n2cpv3_test"])
          and os.path.exists(f"{P}/holdout/nn_{m}_holdout.npy")]
    if ok: FAM[tag] = ok
for f, ms in FAM.items():
    for m in ms:
        for s in ["trainonly_val", "trainonly_test", "n2cpv3_val_oof", "n2cpv3_test"]:
            assert os.path.exists(f"{P}/nn_{m}_{s}.npy"), (m, s)
        assert os.path.exists(f"{P}/holdout/nn_{m}_holdout.npy"), m
print({f: len(m) for f, m in FAM.items()}, flush=True)
json.dump(FAM, open(f"{C}/families.json", "w"), indent=1)

# holdout meta + users
Mh, offh = np.load(f"{P}/holdout/holdout_meta.npy")
tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M", "user_id"])
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
trM = tr["M"].to_numpy(); pr[trM < 50] = 0.0
cand = np.where((pr > 0.9) & (trM >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
assert (trM[hold] == Mh).all()
np.save(f"{C}/hold_user.npy", tr["user_id"].to_numpy()[hold]); np.save(f"{C}/hold_ts.npy", tr["timestamp"].to_numpy()[hold])

def zm(ms, path, off, M):
    acc = None
    for m in ms:
        x = np.nan_to_num(np.load(path(m)).astype(np.float32))
        z = feed_rank_z(off, M, x)[:, 1].astype(np.float64)
        acc = z if acc is None else acc + z
    return (acc / len(ms)).astype(np.float32)

for f, ms in FAM.items():
    np.save(f"{C}/hold_{f}.npy", zm(ms, lambda m: f"{P}/holdout/nn_{m}_holdout.npy", offh, Mh))
    for split, ftn, ton in [("val", "n2cpv3_val_oof", "trainonly_val"), ("test", "n2cpv3_test", "trainonly_test")]:
        _, M, off = feed_meta(split)
        np.save(f"{C}/{split}_ft_{f}.npy", zm(ms, lambda m: f"{P}/nn_{m}_{ftn}.npy", off, M))
        np.save(f"{C}/{split}_to_{f}.npy", zm(ms, lambda m: f"{P}/nn_{m}_{ton}.npy", off, M))
    print("done", f, flush=True)

# val slices
fv, M, off = feed_meta("val")
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
fid = np.repeat(np.arange(len(M)), M)
day = (fv["timestamp"].to_numpy() + 10800) // 86400
luo = np.load(f"{ROOT}/nn_x/r6/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
cold = np.nan_to_num(np.asarray(luo, dtype=np.float64), nan=0.0) == 0
npos = np.bincount(fid, y, len(M)); ncold = np.bincount(fid, y * cold, len(M))
sl = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day < 20720, "Thu": day >= 20720,
      "coldpos": (npos > 0) & (ncold == npos), "warmpos": (npos > 0) & (ncold == 0)}
np.savez(f"{C}/val_slices.npz", **sl)
print({k: int(v.sum()) for k, v in sl.items()})
