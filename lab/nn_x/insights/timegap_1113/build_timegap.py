"""TIMEGAP-1113 builder: per-item train-day masks (candidate / history appearances), then per-row gap columns for val,
the 30K holdout (earlier train days only) and a 300K train sample (earlier train days only). CPU, 12 threads, read-only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
FE = HERE / "feats"; FE.mkdir(exist_ok=True)
T0 = time.time()
LOG = open(HERE / "build.log", "a")


def log(*a):
    s = f"[{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


DAY0 = 20702; ND = 15; NITEM = 111_400_000

# ---------------------------------------------------------------- train feeds, holdout, train sample
ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp", "user_id", "cand_off", "hist_off", "H"])
Mt = ft["M"].to_numpy().astype(np.int64); cot = ft["cand_off"].to_numpy().astype(np.int64)
hot = ft["hist_off"].to_numpy().astype(np.int64); Ht = ft["H"].to_numpy().astype(np.int64)
tst = ft["timestamp"].to_numpy().astype(np.int64); dayt = ((tst + 10800) // 86400 - DAY0).astype(np.int64)
assert dayt.min() == 0 and dayt.max() == ND - 1
pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mt < 50] = 0.0
HOLD = np.sort(np.random.default_rng(12345).choice(np.where((pr > 0.9) & (Mt >= 50))[0], 30000, replace=False))
Mh_ref, _ = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert np.array_equal(Mt[HOLD], Mh_ref.astype(np.int64)), "holdout feed ids do not reproduce holdout_meta"
notin = np.ones(len(Mt), bool); notin[HOLD] = False
TRS = np.sort(np.random.default_rng(7).choice(np.where(notin)[0], 300_000, replace=False))
log(f"train feeds {len(Mt)}, holdout 30000, train sample {len(TRS)}; day counts {np.bincount(dayt, minlength=ND).tolist()}")
# timestamp resolution (plain-data fact)
log(f"feed timestamp resolution: train unique ts {len(np.unique(tst))}, all divisible by 128: {bool((tst % 128 == 0).all())}, "
    f"by 64: {bool((tst % 64 == 0).all())}; min gap between distinct ts {np.diff(np.unique(tst)).min()} s")

# ---------------------------------------------------------------- per-item day masks (holdout feeds excluded as a source)
is_hold_feed = np.zeros(len(Mt), bool); is_hold_feed[HOLD] = True


@nb.njit(cache=False)
def set_bits(mask, items, fidx, dayf, holdf):
    for i in range(len(items)):
        f = fidx[i]
        if holdf[f]:
            continue
        it = items[i]
        if it < 0:
            continue
        mask[it] |= np.int16(1 << dayf[f])


mask_c = np.zeros(NITEM, np.int16)
ci = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r"); cf = np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r")
assert int(np.asarray(ci).max()) < NITEM
CH = 50_000_000
for s in range(0, len(ci), CH):
    set_bits(mask_c, np.asarray(ci[s:s + CH]), np.asarray(cf[s:s + CH]).astype(np.int64), dayt, is_hold_feed)
log(f"candidate mask: items seen as candidate {np.count_nonzero(mask_c)}; popcount mean "
    f"{np.unpackbits(mask_c[mask_c != 0].view(np.uint8)).sum() / max(np.count_nonzero(mask_c), 1):.2f}")
del ci, cf
mask_h = np.zeros(NITEM, np.int16)
hi = np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r"); hf = np.load(FLAT / "train/hist_fidx.npy", mmap_mode="r")
assert int(np.asarray(hi).max()) < NITEM
for s in range(0, len(hi), CH):
    set_bits(mask_h, np.asarray(hi[s:s + CH]), np.asarray(hf[s:s + CH]).astype(np.int64), dayt, is_hold_feed)
log(f"history mask: items seen as history item {np.count_nonzero(mask_h)}; union {np.count_nonzero(mask_c | mask_h)}")
del hi, hf
np.save(FE / "mask_c.npy", mask_c); np.save(FE / "mask_h.npy", mask_h)


# ---------------------------------------------------------------- per-row columns
@nb.njit(cache=False)
def row_cols(items, D, mask_c, mask_h):
    n = len(items)
    gap_last = np.full(n, np.nan); gap_any = np.full(n, np.nan); gap_first = np.full(n, np.nan)
    never = np.ones(n); nact = np.zeros(n); nact_any = np.zeros(n); gap_first_any = np.full(n, np.nan)
    for i in range(n):
        it = items[i]; d = D[i]
        lim = (1 << min(d, 15)) - 1  # days < d
        mc = mask_c[it] & lim; ma = (mask_c[it] | mask_h[it]) & lim
        if mc != 0:
            never[i] = 0.0
            lo = -1; hi = -1; c = 0
            for k in range(15):
                if (mc >> k) & 1:
                    c += 1
                    if lo < 0:
                        lo = k
                    hi = k
            gap_last[i] = d - hi; gap_first[i] = d - lo; nact[i] = c
        if ma != 0:
            lo = -1; hi = -1; c = 0
            for k in range(15):
                if (ma >> k) & 1:
                    c += 1
                    if lo < 0:
                        lo = k
                    hi = k
            gap_any[i] = d - hi; gap_first_any[i] = d - lo; nact_any[i] = c
    return gap_last, gap_any, gap_first, never, nact, nact_any, gap_first_any


def build_split(name, split, feeds_idx, f, Mall, coff):
    M = Mall[feeds_idx]; co = coff[feeds_idx]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)])
    loc_off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    item = np.asarray(np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")[rows]).astype(np.int64)
    f4 = np.asarray(np.load(FLAT / split / "cand_f4.npy", mmap_mode="r")[rows]).astype(np.float32)
    y = np.asarray(np.load(FLAT / split / "cand_target.npy", mmap_mode="r")[rows]).astype(np.float64)
    ts = f["timestamp"].to_numpy().astype(np.int64)[feeds_idx]
    day = ((ts + 10800) // 86400 - DAY0).astype(np.int64)
    fid = np.repeat(np.arange(len(M)), M)
    D = day[fid]
    gl, ga, gf, nv, na, naa, gfa = row_cols(item, D, mask_c, mask_h)
    d = dict(M=M, loc_off=loc_off, rows=rows, item=item, f4=f4, y=y, day=day, fid=fid, gap_last=gl, gap_last_any=ga, gap_first=gf,
             never_seen=nv, n_days_active=na, n_days_active_any=naa, gap_first_any=gfa)
    np.savez(FE / f"{name}.npz", **d)
    ok = ~np.isnan(gl)
    log(f"{name}: {len(M)} feeds, {len(rows)} rows, feed days {np.unique(day).tolist()}; never_seen {nv.mean():.3f}; "
        f"gap_last mean {np.nanmean(gl):.2f} [{np.nanmin(gl) if ok.any() else np.nan:.0f}..{np.nanmax(gl) if ok.any() else np.nan:.0f}]; "
        f"gap_last_any NaN {np.isnan(ga).mean():.3f} mean {np.nanmean(ga):.2f}; gap_first mean {np.nanmean(gf):.2f}; "
        f"n_days_active mean {na.mean():.2f}; n_days_active_any mean {naa.mean():.2f}")
    return d


fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); cov = fv["cand_off"].to_numpy().astype(np.int64)
V = build_split("val", "val", np.arange(len(Mv)), fv, Mv, cov)
yh_ref = np.load(ROOT / "preds/holdout/holdout_y.npy").astype(np.float64)
Hd = build_split("hold", "train", HOLD, ft, Mt, cot)
assert np.array_equal(Hd["y"], yh_ref), "holdout rows != holdout_y"
log("holdout rows reproduce holdout_y")
Tr = build_split("trs", "train", TRS, ft, Mt, cot)

# ---------------------------------------------------------------- plain-data measurements
ok = ~np.isnan(V["gap_first"])
r1 = spearmanr(V["gap_first"][ok], V["f4"][ok])[0]
r2 = spearmanr(V["gap_first_any"][~np.isnan(V["gap_first_any"])], V["f4"][~np.isnan(V["gap_first_any"])])[0]
r3 = spearmanr(V["n_days_active"], V["f4"])[0]
log(f"val Spearman(gap_first, f4) = {r1:+.4f} (n {ok.sum()}); Spearman(gap_first_any, f4) = {r2:+.4f}; Spearman(n_days_active, f4) = {r3:+.4f}")
okh = ~np.isnan(Hd["gap_first"])
log(f"hold Spearman(gap_first, f4) = {spearmanr(Hd['gap_first'][okh], Hd['f4'][okh])[0]:+.4f} (n {okh.sum()})")
# f4 vs gap_first: mean f4 by gap_first on val
for g in range(2, 19):
    m = V["gap_first"] == g
    if m.sum() > 1000:
        log(f"  val gap_first={g:2d}: n {m.sum():7d}  mean f4 {V['f4'][m].mean():.4f}  CTR {V['y'][m].mean():.4f}")
m = np.isnan(V["gap_first"]); log(f"  val never seen as candidate: n {m.sum():7d}  mean f4 {V['f4'][m].mean():.4f}  CTR {V['y'][m].mean():.4f}")
# gap distributions by split (raw, the parity break)
for nm, D in [("trs", Tr), ("hold", Hd), ("val", V)]:
    g = D["gap_last"]; h = np.bincount(np.nan_to_num(g, nan=99).astype(int).clip(0, 99), minlength=100)
    log(f"{nm} gap_last histogram (NaN=99): " + " ".join(f"{k}:{h[k]/len(g):.3f}" for k in list(range(1, 20)) + [99] if h[k] > 0))
log(f"DONE build {time.time()-T0:.0f}s")
