"""Pre-declared cross-recipe member ensembles (no training): val paired vs v11 (slices) and holdout (train-only halves).
Sets (equal member weight inside each half; final = 0.5 z(FT half) + 0.5 z(TO half)):
  S0 = nn_final5 (6 v12 members) = v11; S1 = S0 + 6 v15 cooldown members (= nn_final5cd12);
  S2 = S0 + 3 non-collapsed v13 hedges (lr1_mix0, lr1_mix1, lr2_mix1); S3 = S1 + the same 3 hedges;
  F3 = equal FAMILY weight z-mean of (S0 mix, v15 mix, v13 mix)."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
v12 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
v15 = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
v13 = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr2_mix1"]
cache = {}


def zv(m, kind):
    k = (m, kind)
    if k not in cache:
        nm = f"nn_{m}_trainonly_val" if kind == "TO" else f"nn_{m}_n2cpf_val_oof"
        cache[k] = fz(np.load(f"{ROOT}/preds/{nm}.npy"), fid, M)
    return cache[k]


def mix(members):
    zt = np.mean([zv(m, "TO") for m in members], 0)
    zf = np.mean([zv(m, "FT") for m in members], 0)
    return 0.5 * fz(zt, fid, M) + 0.5 * fz(zf, fid, M)


sets = {"S0 (=v11)": v12, "S1 (=cd12)": v12 + v15, "S2 (+v13)": v12 + v13, "S3 (cd12+v13)": v12 + v15 + v13}
Zs = {k: mix(v) for k, v in sets.items()}
Zs["F3 family-equal"] = (fz(mix(v12), fid, M) + fz(mix(v15), fid, M) + fz(mix(v13), fid, M)) / 3
base = ndcg_per_feed(Zs["S0 (=v11)"], y, off, M)
day = pf["day"].to_numpy()
pc = pf["poscold"].to_numpy()
SL = {"all": np.ones(n, bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1, "allcold": pc == "all_cold", "allwarm": pc == "all_warm"}
for k, z in Zs.items():
    nd = ndcg_per_feed(z, y, off, M)
    d = nd - base
    print(f"{k:18s} val {nd.mean():.5f}  " + "  ".join(f"{s} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}" for s, m in SL.items()))
# holdout (train-only halves only)
D = f"{ROOT}/preds/holdout"
Mh, offh = np.load(f"{D}/holdout_meta.npy")
yh = np.load(f"{D}/holdout_y.npy").astype(np.float64)
fidh = np.repeat(np.arange(len(Mh)), Mh)
tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
prr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
trM = tr["M"].to_numpy()
prr[trM < 50] = 0.0
cand = np.where((prr > 0.9) & (trM >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
assert (trM[hold] == Mh).all()
tsh = tr["timestamp"].to_numpy()[hold]
late = tsh >= np.quantile(tsh, 0.8)
hz = lambda m: fz(np.load(f"{D}/nn_{m}_holdout.npy"), fidh, Mh)  # noqa: E731
hb = None
for k, mem in sets.items():
    z = np.mean([hz(m) for m in mem], 0)
    nd = ndcg_per_feed(z, yh, offh, Mh)
    if hb is None:
        hb = nd
    d = nd - hb
    print(f"holdout TO-half {k:14s} {nd.mean():.5f}  d {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}  late20% {d[late].mean():+.5f}+-{d[late].std() / np.sqrt(late.sum()):.5f}")
z = (fz(np.mean([hz(m) for m in v12], 0), fidh, Mh) + fz(np.mean([hz(m) for m in v15], 0), fidh, Mh) + fz(np.mean([hz(m) for m in v13], 0), fidh, Mh)) / 3
nd = ndcg_per_feed(z, yh, offh, Mh)
print(f"holdout TO-half F3 family-equal {nd.mean():.5f}  d {(nd - hb).mean():+.5f}+-{(nd - hb).std() / np.sqrt(len(nd)):.5f}")
