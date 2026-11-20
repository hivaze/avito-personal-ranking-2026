"""ENRICH-1113 builder: own-history, region/local-hour and f4-age columns for val, the 30K holdout and a 300K train sample;
train tables (C2 / R2 / U2) by day. Read-only on the project; outputs under ./feats and ./out. CPU, 12 threads."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json
from pathlib import Path
import numpy as np, polars as pl, numba as nb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
FE = HERE / "feats"; FE.mkdir(exist_ok=True); OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
T0 = time.time()
LOG = open(HERE / "logs/build.log", "a")


def log(*a):
    s = f"[{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ---------------------------------------------------------------- constants
K2 = np.load(ROOT / "nn_x/codes/features/out/k2_item_comp.npy")
k2_order = np.argsort(K2[:, 0]); K2_ITEMS = K2[k2_order, 0]; K2_COMP = K2[k2_order, 1].astype(np.int64)
NCOMP = int(K2_COMP.max()) + 1
# table (b) of nn_x/codes/features/04_k2_geo.log: best shift of the component's MSK-hour activity profile vs the global one.
# A profile shifted EARLIER by s hours (negative) means the region is s hours AHEAD of Moscow -> local hour = MSK hour - shift.
SHIFT = {69: -5, 154: -4, 189: -4, 0: -4, 7: -2, 87: -2, 173: -2, 54: -2, 103: -2, 144: -1, 51: -1, 37: -1, 22: -1, 67: -1,
         227: -1, 143: +1, 46: +1}
SHIFT_ARR = np.zeros(NCOMP, np.int64)
for c, s in SHIFT.items():
    SHIFT_ARR[c] = s
KNOTS = np.load(ROOT / "data/feats/life_f4_age_knots.npy"); KK = 64; CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = x.astype(np.float64); u = x * KK; k = np.clip(np.floor(u).astype(int), 1, KK - 2); w = u - k
    a = KNOTS[k] * (1 - w) + KNOTS[k + 1] * w
    return np.where(x < CAP, a, np.nan)


@nb.njit(cache=False)
def own_hist(cand_cat, cand_off, M, hist_cat, hist_src, hist_et, hist_hpos, hist_off, H):
    n = len(cand_cat)
    minh = np.full(n, 999, np.int64); s2 = np.zeros(n, np.int64); s2r = np.zeros(n, np.int64); s4 = np.zeros(n, np.int64)
    mh = np.empty(64, np.int64); c2 = np.empty(64, np.int64); c2r = np.empty(64, np.int64); c4 = np.empty(64, np.int64)
    for f in range(len(M)):
        for c in range(64):
            mh[c] = 999; c2[c] = 0; c2r[c] = 0; c4[c] = 0
        ho = hist_off[f]
        for j in range(H[f]):
            c = hist_cat[ho + j]
            if c < 0 or c >= 64:
                continue
            hp = hist_hpos[ho + j]
            if hp < mh[c]:
                mh[c] = hp
            if hist_et[ho + j] == 1 and hist_src[ho + j] == 2:
                c2[c] += 1
                if hp < 32:
                    c2r[c] += 1
            if hist_src[ho + j] == 4:
                c4[c] += 1
        co = cand_off[f]
        for i in range(M[f]):
            c = cand_cat[co + i]
            if c < 0 or c >= 64:
                continue
            minh[co + i] = mh[c]; s2[co + i] = c2[c]; s2r[co + i] = c2r[c]; s4[co + i] = c4[c]
    return minh, s2, s2r, s4


@nb.njit(cache=False)
def feed_mode_home(f2, cand_off, M):
    """per feed: mode of f2 over candidates (ties -> larger value); per row: home flag (f2 == mode)"""
    nF = len(M); mode = np.zeros(nF); home = np.zeros(len(f2), np.float64)
    for f in range(nF):
        o = cand_off[f]; m = M[f]
        v = np.sort(f2[o:o + m].astype(np.float64))
        best = v[0]; bc = 0; i = 0
        while i < m:
            j = i
            while j + 1 < m and v[j + 1] == v[i]:
                j += 1
            c = j - i + 1
            if c >= bc:
                bc = c; best = v[i]
            i = j + 1
        mode[f] = best
        for i in range(m):
            if f2[o + i] == best:
                home[o + i] = 1.0
    return mode, home


@nb.njit(cache=False)
def feed_comp_vote(comp, home, cand_off, M, ncomp):
    """per feed: majority K2 component among home-level candidates with a known component (-1 if none)"""
    nF = len(M); out = np.full(nF, -1, np.int64); cnt = np.zeros(ncomp, np.int64)
    for f in range(nF):
        o = cand_off[f]; m = M[f]; best = -1; bc = 0
        for i in range(m):
            c = comp[o + i]
            if c >= 0 and home[o + i] > 0:
                cnt[c] += 1
                if cnt[c] > bc:
                    bc = cnt[c]; best = c
        for i in range(m):
            c = comp[o + i]
            if c >= 0:
                cnt[c] = 0
        out[f] = best
    return out


def item_comp(items):
    pos = np.searchsorted(K2_ITEMS, items); pos = np.clip(pos, 0, len(K2_ITEMS) - 1)
    ok = K2_ITEMS[pos] == items
    return np.where(ok, K2_COMP[pos], -1).astype(np.int64)


def build_split(name, split, feeds_idx, f, Mall, coff, hoff, Hall):
    """feeds_idx: indices into the split's feeds (sorted). Returns dict of per-row / per-feed arrays, saves to feats/."""
    M = Mall[feeds_idx]; co = coff[feeds_idx]; ho = hoff[feeds_idx]; H = Hall[feeds_idx]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)])
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, H)]) if H.sum() > 0 else np.zeros(0, np.int64)
    loc_off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    loc_hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    ld = lambda n, dt=None: np.load(FLAT / split / f"{n}.npy", mmap_mode="r")  # noqa: E731
    cat = np.asarray(ld("cand_category_id")[rows]).astype(np.int64)
    f2 = np.asarray(ld("cand_f2")[rows]).astype(np.float32)
    f4 = np.asarray(ld("cand_f4")[rows]).astype(np.float32)
    item = np.asarray(ld("cand_item_id")[rows]).astype(np.int64)
    y = np.asarray(ld("cand_target")[rows]).astype(np.float64)
    hcat = np.asarray(ld("hist_category_id")[hrows]).astype(np.int64)
    hsrc = np.asarray(ld("hist_event_source")[hrows]).astype(np.int64)
    het = np.asarray(ld("hist_event_type")[hrows]).astype(np.int64)
    hhp = np.asarray(ld("hist_hpos")[hrows]).astype(np.int64)
    log(f"{name}: {len(M)} feeds, {len(rows)} rows, {len(hrows)} history events loaded")
    minh, s2, s2r, s4 = own_hist(cat, loc_off, M, hcat, hsrc, het, hhp, loc_hoff, H)
    mode, home = feed_mode_home(f2, loc_off, M)
    comp = item_comp(item)
    hcomp = feed_comp_vote(comp, home, loc_off, M, NCOMP)
    shift = np.where(hcomp >= 0, SHIFT_ARR[np.maximum(hcomp, 0)], 0)
    ts = f["timestamp"].to_numpy().astype(np.int64)[feeds_idx]
    msk_hour = ((ts + 10800) // 3600) % 24
    local_hour = (msk_hour - shift) % 24
    day = (ts + 10800) // 86400
    user = f["user_id"].to_numpy().astype(np.int64)[feeds_idx]
    age = age_hours(f4)
    fid = np.repeat(np.arange(len(M)), M)
    d = dict(M=M, rows=rows, cat=cat, f2=f2, f4=f4, item=item, y=y, minh=minh, s2=s2, s2r=s2r, s4=s4, mode=mode, home=home,
             comp=comp, hcomp=hcomp, shift=shift, msk_hour=msk_hour, local_hour=local_hour, day=day, user=user, age=age, fid=fid,
             loc_off=loc_off)
    np.savez(FE / f"{name}.npz", **d)
    log(f"{name}: home share {home.mean():.3f}; feeds with a home component {np.mean(hcomp >= 0):.3f}; "
        f"feeds with non-zero shift {np.mean(shift != 0):.3f}; shift mix {dict(zip(*np.unique(shift, return_counts=True)))}; "
        f"rows with same-cat history {np.mean(minh < 999):.3f}; s2 same-cat > 0 {np.mean(s2 > 0):.3f}; s4 > 0 {np.mean(s4 > 0):.3f}; "
        f"age NaN {np.mean(np.isnan(age)):.3f}; user0 feeds {np.mean(user == 0):.3f}")
    return d


# ---------------------------------------------------------------- val
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); cov = fv["cand_off"].to_numpy().astype(np.int64)
hov = fv["hist_off"].to_numpy().astype(np.int64); Hv = fv["H"].to_numpy().astype(np.int64)
V = build_split("val", "val", np.arange(len(Mv)), fv, Mv, cov, hov, Hv)

# ---------------------------------------------------------------- holdout + train sample
ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp", "user_id", "cand_off", "hist_off", "H"])
Mt = ft["M"].to_numpy().astype(np.int64); cot = ft["cand_off"].to_numpy().astype(np.int64)
hot = ft["hist_off"].to_numpy().astype(np.int64); Ht = ft["H"].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mt < 50] = 0.0
HOLD = np.sort(np.random.default_rng(12345).choice(np.where((pr > 0.9) & (Mt >= 50))[0], 30000, replace=False))
Mh_ref, _ = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert np.array_equal(Mt[HOLD], Mh_ref.astype(np.int64)), "holdout feed ids do not reproduce holdout_meta"
Hd = build_split("hold", "train", HOLD, ft, Mt, cot, hot, Ht)
yh_ref = np.load(ROOT / "preds/holdout/holdout_y.npy").astype(np.float64)
assert np.array_equal(Hd["y"], yh_ref), "holdout rows != holdout_y"
log("holdout rows reproduce holdout_y")

notin = np.ones(len(Mt), bool); notin[HOLD] = False
pool = np.where(notin)[0]
TRS = np.sort(np.random.default_rng(7).choice(pool, 300_000, replace=False))
Tr = build_split("trs", "train", TRS, ft, Mt, cot, hot, Ht)
Tr_prand = pr[TRS]
np.save(FE / "trs_prand.npy", Tr_prand)

# ---------------------------------------------------------------- tables by day (train sample)
DAY0 = 20702


def hbin(minh):
    b = np.full(len(minh), 8, np.int64)
    edges = [(0, 1, 0), (2, 3, 1), (4, 7, 2), (8, 15, 3), (16, 31, 4), (32, 63, 5), (64, 127, 6), (128, 255, 7)]
    for lo, hi, k in edges:
        b[(minh >= lo) & (minh <= hi)] = k
    return b


def lhbin(lh):
    return np.where(lh < 6, 0, np.where(lh < 11, 1, np.where(lh < 18, 2, 3)))


NB = 9; NH = 4; ND = 15
fidT = Tr["fid"]; dayT = (Tr["day"] - DAY0)[fidT]; catT = Tr["cat"]; yT = Tr["y"]
assert dayT.min() >= 0 and dayT.max() < ND
binT = hbin(Tr["minh"]); lhT = lhbin(Tr["local_hour"])[fidT]; u0T = (Tr["user"] == 0).astype(np.int64)[fidT]


def counts(keys, nk):
    idx = (dayT * 64 + catT) * nk + keys
    n = np.bincount(idx, minlength=ND * 64 * nk).reshape(ND, 64, nk).astype(np.float64)
    p = np.bincount(idx, weights=yT, minlength=ND * 64 * nk).reshape(ND, 64, nk)
    return n, p


tabs = {}
for nm, keys, nk in [("C2", binT, NB), ("R2", lhT, NH), ("U2", u0T, 2)]:
    n, p = counts(keys, nk); tabs[nm] = (n, p)
    np.savez(OUT / f"tab_{nm}.npz", n=n, p=p)
    log(f"table {nm}: cells {n.shape}, rows {n.sum():.0f}, positives {p.sum():.0f}")
log(f"DONE build {time.time()-T0:.0f}s")
