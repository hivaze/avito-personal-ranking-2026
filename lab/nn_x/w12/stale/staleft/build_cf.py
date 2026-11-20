"""staleft: private-like ('both') counterfactual VAL columns for the fine-tune's n2c_pre_v3 extras that have no file yet.
CPU only, label-free (no target file is opened), read-only on every existing file; writes nn_x/w12/stale/staleft/cf/*.npy.

Construction = scratchpad review/shift/s08_cf_feats.py ('both' = no test-split snapshots + train snapshots of Sep 17-19 removed) and
solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2both (newly cold rows read the no-ID score), applied to:
  A  it_hist_et5..10_luo (and all / et1..4 again as the reproduction check against nn_x/r6/stale_cf/both_*.npy)
  B  it_hist_all_luo_rk / _zsl / _gap (src/features_luo.py), seq_cv_invd20_rk (src/features.py::_rel_feats)
  C  pre_fe, pre_fe_z, pre_id_contrib_v2, pre_idmax32, pre_idmax32_has (pre family, configs v3A->v3, v4A->v4)
Every bar of DECLARATION.txt is checked and written to logs/build_cf.json; a column that misses its bar is NOT written.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/stale/staleft/build_cf.py
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import numba as nb
import numpy as np
import polars as pl

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
FE = f"{ROOT}/nn/feat_cache/val"
HERE = f"{ROOT}/nn_x/w12/stale/staleft"
CF = f"{HERE}/cf"
CF_BOTH = f"{ROOT}/nn_x/r6/stale_cf"
CF_PRE2 = f"{ROOT}/solvers/out/stale/cf"
os.makedirs(CF, exist_ok=True)
t0 = time.time()
chk = {}


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


# ---- verbatim copies of the builders' numba kernels (no import from src/: numba cache=True would write into src/__pycache__)
@nb.njit(parallel=True)
def feed_rank_z(off, M, x):  # src/stack_val.py
    out = np.empty((len(x), 2), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        r = np.argsort(np.argsort(v))
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k, 0] = r[k] / (m - 1) if m > 1 else 0.5
            out[o + k, 1] = (v[k] - mu) / sd
    return out


@nb.njit(parallel=True)
def rel_rank(cand_off, M, x):  # src/features.py::_rel_feats, rank part (average ties)
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        co = cand_off[f]
        m = M[f]
        v = x[co:co + m]
        order = np.argsort(v)
        r = np.empty(m, np.float32)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        for k in range(m):
            out[co + k] = r[k] / (m - 1) if m > 1 else 0.5
    return out


def affine(x, ref):  # pretrain/assemble.py
    mx, mr = (x > -2) & np.isfinite(x), (ref > -2) & np.isfinite(ref)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    return (lambda v: np.where(v > -2, a * v + b, v).astype(np.float32)), (a, b)


fva = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
M = fva["M"].to_numpy().astype(np.int64)
off = fva["cand_off"].to_numpy().astype(np.int64)
fid = np.repeat(np.arange(len(M)), M)
item = np.load(f"{FLAT}/val/cand_item_id.npy")

# ------------------------------------------------------------------------------------------------ A: it_hist both
NI = 111_400_000
MSK = 3 * 3600


def dayof(ts):
    return (np.asarray(ts, np.int64) + MSK) // 86400


ETS = ["all"] + list(range(1, 11))
names = {"all": "it_hist_all_luo", **{e: f"it_hist_et{e}_luo" for e in range(1, 11)}}
if not all(os.path.exists(f"{CF}/both_{names[e]}.npy") for e in range(5, 11)):
    D17 = dayof(np.int64(1788642048)) + 11  # Sep 6 + 11 = Sep 17
    ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
    trday = dayof(ftr["timestamp"].to_numpy())
    hf = np.load(f"{FLAT}/train/hist_fidx.npy")
    late = trday[hf] >= D17
    del hf
    hi = np.load(f"{FLAT}/train/hist_item_id.npy")[late]
    he = np.load(f"{FLAT}/train/hist_event_type.npy")[late]
    del late
    assert int(item.max()) < NI and int(hi.max()) < NI
    late_cnt = {"all": np.bincount(hi, minlength=NI)[item].astype(np.float32)}
    for e in range(1, 11):
        late_cnt[e] = np.bincount(hi[he == e], minlength=NI)[item].astype(np.float32)
    del hi, he
    log("train late (Sep 17-19) history counts at the val candidates done")
    fte = pl.read_parquet(f"{FLAT}/test/feeds.parquet")
    thi = np.load(f"{FLAT}/test/hist_item_id.npy")
    the = np.load(f"{FLAT}/test/hist_event_type.npy")
    thf = np.load(f"{FLAT}/test/hist_fidx.npy")
    test_cnt = {"all": np.bincount(thi, minlength=NI)[item].astype(np.float32)}
    for e in range(1, 11):
        test_cnt[e] = np.bincount(thi[the == e], minlength=NI)[item].astype(np.float32)
    uid = fva["user_id"].to_numpy()[fid]
    ts = fva["timestamp"].to_numpy().astype(np.int64)[fid]
    tu = fte["user_id"].to_numpy()[thf]
    tts = fte["timestamp"].to_numpy().astype(np.int64)[thf]
    H = pl.DataFrame({"user": tu, "item": thi, "th": tts, "et": the}).filter(pl.col("user") != 0)
    C = pl.DataFrame({"row": np.arange(len(item)), "user": uid, "item": item, "tc": ts}).filter(pl.col("user") != 0)
    j = (C.join(H, on=["user", "item"]).filter(pl.col("th") > pl.col("tc")).group_by("row")
         .agg([pl.len().alias("n")] + [(pl.col("et") == e).sum().alias(f"n{e}") for e in range(1, 11)]))
    r = j["row"].to_numpy()
    ov = {e: np.zeros(len(item), np.float32) for e in ETS}
    ov["all"][r] = j["n"].to_numpy()
    for e in range(1, 11):
        ov[e][r] = j[f"n{e}"].to_numpy()
    log("LUO overlap rows", len(r))
    rep = {}
    built = {}
    for e in ETS:
        n = names[e]
        x = np.load(f"{FE}/{n}.npy").astype(np.float32)
        notest = np.maximum(x - test_cnt[e] + ov[e], 0)
        both = np.maximum(notest - late_cnt[e], 0).astype(np.float32)
        built[n] = both
        ref = f"{CF_BOTH}/both_{n}.npy"
        if os.path.exists(ref):
            d = np.abs(np.load(ref).astype(np.float64) - both)
            rep[n] = float(d.max())
        log(n, "zero share fresh / both", round(float((x == 0).mean()), 4), round(float((both == 0).mean()), 4),
            "repro max|d| vs existing", rep.get(n, "n/a"))
    chk["A_repro_max_abs_existing_both"] = rep
    chk["A_pass"] = bool(len(rep) == 5 and max(rep.values()) == 0.0)
    if chk["A_pass"]:
        for e in range(5, 11):
            np.save(f"{CF}/both_{names[e]}.npy", built[names[e]])
    log("A pass", chk["A_pass"])
else:
    chk["A_pass"] = "files existed (earlier run)"

# ------------------------------------------------------------------------------------------------ B: derived
x0 = np.load(f"{FE}/it_hist_all_luo.npy").astype(np.float32)
xb = np.load(f"{CF_BOTH}/both_it_hist_all_luo.npy").astype(np.float32)


def luo_derived(x):
    rz = feed_rank_z(off, M, x)
    mx = np.maximum.reduceat(x, off)
    z = rz[:, 1]
    return {"it_hist_all_luo_rk": rz[:, 0], "it_hist_all_luo_zsl": (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32),
            "it_hist_all_luo_gap": (x - np.repeat(mx, M)).astype(np.float32)}


d0, db = luo_derived(x0), luo_derived(xb)
chk["B_repro"] = {}
for n in d0:
    live = np.load(f"{FE}/{n}.npy").astype(np.float32)
    e = float(np.nanmax(np.abs(live - d0[n])))
    chk["B_repro"][n] = e
    ok = e <= 1e-5
    if n == "it_hist_all_luo_rk" and not ok:  # argsort ties are implementation-defined: report the share of rows that differ
        chk["B_repro"][n + "_rows_differ"] = float(np.mean(live != d0[n]))
    log(n, "fresh reproduction max|d|", e, "->", "WRITE" if ok else "KEEP FRESH")
    if ok:
        np.save(f"{CF}/both_{n}.npy", db[n])
s0 = np.load(f"{FE}/seq_cv_invd20.npy")
sb = np.load(f"{CF_BOTH}/both_seq_cv_invd20.npy")
live = np.load(f"{FE}/seq_cv_invd20_rk.npy").astype(np.float32)
r0 = rel_rank(off, M, s0.astype(live.dtype if s0.dtype == live.dtype else s0.dtype))
e = float(np.nanmax(np.abs(live - r0)))
chk["B_repro"]["seq_cv_invd20_rk"] = e
log("seq_cv_invd20_rk fresh reproduction max|d|", e, "->", "WRITE" if e <= 1e-5 else "KEEP FRESH")
if e <= 1e-5:
    np.save(f"{CF}/both_seq_cv_invd20_rk.npy", rel_rank(off, M, sb.astype(s0.dtype)))

# ------------------------------------------------------------------------------------------------ C: pre family
nc = np.load(f"{CF_PRE2}/newly_cold.npy")
assert nc.shape == x0.shape and bool((nc == ((xb == 0) & (x0 > 0))).all())
P = f"{ROOT}/preds"
L = lambda n: np.load(f"{FE}/{n}.npy").astype(np.float32)  # noqa: E731
fe_l, fez_l, noid_l, idc, idm, has = L("pre_fe"), L("pre_fe_z"), L("pre_fe_noid"), L("pre_id_contrib_v2"), L("pre_idmax32"), L("pre_idmax32_has")
fes, noids, zs, fes2, zs2 = [], [], [], [], []
for t, c in [("v3A", "v3"), ("v4A", "v4")]:
    L0 = lambda k: np.load(f"{P}/pre_{t}_{k}_val.npy").astype(np.float32)  # noqa: E731
    R = lambda k: np.load(f"{P}/pre_{c}_{k}_val.npy").astype(np.float32)  # noqa: E731
    ffe, _ = affine(L0("fe"), R("fe"))
    fno, _ = affine(L0("fe_noid"), R("fe_noid"))
    f_, n_ = ffe(L0("fe")), fno(L0("fe_noid"))
    fes.append(f_); noids.append(n_); zs.append(feed_rank_z(off, M, f_)[:, 1])
    f2 = np.where(nc, n_, f_).astype(np.float32)
    fes2.append(f2); zs2.append(feed_rank_z(off, M, f2)[:, 1])
c_ = {"C_repro_pre_fe_max_abs": float(np.abs(np.mean(fes, 0) - fe_l).max()),
      "C_repro_pre_fe_noid_max_abs": float(np.abs(np.mean(noids, 0) - noid_l).max()),
      "C_repro_pre_fe_z_max_abs": float(np.abs(np.mean(zs, 0) - fez_l).max()),
      "C_P(idc==0|luo==0)_val": float((idc[x0 == 0] == 0).mean()),
      "C_P(idc==0|luo>0)_val": float((idc[x0 > 0] == 0).mean()),
      "C_P(idmax32 NaN|luo==0)_val": float(np.isnan(idm[x0 == 0]).mean()),
      "C_P(idmax32 NaN|luo>0)_val": float(np.isnan(idm[x0 > 0]).mean()),
      "C_newly_cold_share": float(nc.mean()),
      "C_newly_cold_with_id_share": float((idc[nc] != 0).mean()),
      "C_|fe - noid| on idc==0 rows max": float(np.abs(fe_l - noid_l)[idc == 0].max())}
tfe = np.load(f"{ROOT}/nn/feat_cache/test/it_hist_all_luo.npy")
tidc = np.load(f"{ROOT}/nn/feat_cache/test/pre_id_contrib_v2.npy")
c_["C_P(idc==0|luo==0)_test"] = float((tidc[tfe == 0] == 0).mean())
c_["C_P(idc==0|luo>0)_test"] = float((tidc[tfe > 0] == 0).mean())
chk.update(c_)
chk["C_pass"] = bool(c_["C_repro_pre_fe_max_abs"] <= 1e-4 and c_["C_repro_pre_fe_z_max_abs"] <= 1e-4 and c_["C_P(idc==0|luo==0)_val"] >= 0.95)
for k_, v_ in c_.items():
    log(k_, v_)
if chk["C_pass"]:
    fe2 = np.mean(fes2, 0).astype(np.float32)
    np.save(f"{CF}/both_pre_fe.npy", fe2)
    np.save(f"{CF}/both_pre_fe_z.npy", np.mean(zs2, 0).astype(np.float32))
    np.save(f"{CF}/both_pre_id_contrib_v2.npy", np.where(nc, 0.0, idc).astype(np.float32))
    np.save(f"{CF}/both_pre_idmax32.npy", np.where(nc, np.nan, idm).astype(np.float32))
    np.save(f"{CF}/both_pre_idmax32_has.npy", np.where(nc, 0.0, has).astype(np.float32))
    chk["C_rows_changed_fe"] = int((fe2 != fe_l).sum())
log("C pass", chk["C_pass"])
os.makedirs(f"{HERE}/logs", exist_ok=True)
json.dump(chk, open(f"{HERE}/logs/build_cf.json", "w"), indent=1)
log("written:", sorted(os.listdir(CF)))
