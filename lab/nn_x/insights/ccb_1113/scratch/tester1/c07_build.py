"""C07 item trend / lifecycle at a gap-matched as-of (copy of feats_v2/life_/ts_train_stream.py's dated-interaction stream).
Dated interaction = train history event with hpos < 8 not present in the same user's previous train feed (or undatable),
dated by the feed time t (new_flags copied verbatim). Candidate appearances = train candidate rows (item, MSK day).
As-of A: val/test rows = TR_END (Sep 20 00:00 MSK); holdout rows = feed-day start (MSK) - 5 d. State = events with t < A.
Columns: it_n7 = # dated interactions in [A-7d, A); it_trend = log((dec24 + .5)/(dec72 + .5)) (decayed counts at A, half-lives
24h/72h); it_slope7 = OLS slope of the 7 daily counts in [A-7d, A) / their mean (NaN if none); it_q95gap = (A - 0.95 quantile
of the item's dated interaction times) in days (NaN if none); exhaust = c_ndays (distinct MSK days as a candidate before A) /
age_days(f4) (f4 decoded with data/feats/life_f4_age_knots.npy as scratchpad/ext/xcheck193/build_user.py; age floored at 1 h).
Outputs out/c07_<col>_{val,test,hold}.npy."""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl, numba as nb
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
sys.path.insert(0, str(HERE))
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
TR_END = 1789862400; MSK = 3 * 3600; KF = 8
HL24 = 24 * 3600 / np.log(2.0); HL72 = 72 * 3600 / np.log(2.0)
COLS = ["it_n7", "it_trend", "it_slope7", "it_q95gap", "exhaust"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


@nb.njit(parallel=True)
def new_flags(prev, hist_off, H, h_item, h_et, out):
    for f in nb.prange(len(H)):
        p = prev[f]; a = hist_off[f]; n = H[f]
        if p < 0:
            for k in range(n):
                out[a + k] = 2
            continue
        pa = hist_off[p]; pn = H[p]
        keys = np.empty(pn, np.int64)
        for k in range(pn):
            keys[k] = np.int64(h_item[pa + k]) * 16 + h_et[pa + k]
        keys.sort()
        for k in range(n):
            key = np.int64(h_item[a + k]) * 16 + h_et[a + k]
            j = np.searchsorted(keys, key)
            out[a + k] = 0 if (j < pn and keys[j] == key) else 1


knots = np.load(ROOT / "data/feats/life_f4_age_knots.npy"); K = 64; CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = x.astype(np.float64); u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


f = pl.read_parquet(FLAT / "train/feeds.parquet")
fts = f["timestamp"].to_numpy().astype(np.int64); uid = f["user_id"].to_numpy()
M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
co = f["cand_off"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
order = np.lexsort((np.arange(len(f)), fts, uid))
prev = np.full(len(f), -1, np.int64)
same = (uid[order][1:] == uid[order][:-1]) & (uid[order][1:] != 0)
prev[order[1:][same]] = order[:-1][same]
h_item = np.load(FLAT / "train/hist_item_id.npy"); h_et = np.load(FLAT / "train/hist_event_type.npy")
h_new = np.empty(len(h_item), np.int8)
new_flags(prev, ho, H, h_item, h_et, h_new)
hp = np.load(FLAT / "train/hist_hpos.npy")
sel = np.flatnonzero((hp < KF) & (h_new != 0))
hfid = np.repeat(np.arange(len(f)), H)
EV = pl.DataFrame({"item": h_item[sel].astype(np.int64), "t": fts[hfid[sel]]})
del h_item, h_et, h_new, hp, hfid
log(f"dated interactions {EV.height} (prev-feed share {(prev >= 0).mean():.4f})")
c_item = np.load(FLAT / "train/cand_item_id.npy"); cfid = np.load(FLAT / "train/cand_fidx.npy")
CD = pl.DataFrame({"item": c_item.astype(np.int64), "day": ((fts[cfid] + MSK) // 86400).astype(np.int32)}).unique()
del c_item, cfid
log(f"candidate (item, day) pairs {CD.height}")


def query(items, A, f4):
    """items, as-of A (int64 per row), f4 -> dict of 5 columns."""
    Q = pl.DataFrame({"r": np.arange(len(items)), "item": items.astype(np.int64), "A": A.astype(np.int64)})
    outs = []
    for a in np.unique(A):
        a = int(a)
        q = Q.filter(pl.col("A") == a)
        it = q.select("item").unique()
        ev = EV.filter(pl.col("t") < a).join(it, on="item", how="semi")
        ev = ev.with_columns(((pl.lit(a) - pl.col("t")) / 86400.0).alias("age_d"))
        g = ev.group_by("item").agg(
            (pl.col("age_d") < 7).sum().alias("n7"),
            (-(pl.lit(a) - pl.col("t")) / HL24).exp().sum().alias("dec24"),
            (-(pl.lit(a) - pl.col("t")) / HL72).exp().sum().alias("dec72"),
            pl.col("t").quantile(0.95, interpolation="linear").alias("q95t"),
            *[((pl.col("age_d") >= 6 - j) & (pl.col("age_d") < 7 - j)).sum().alias(f"d{j}") for j in range(7)])
        cd = CD.filter(pl.col("day") < (a + MSK) // 86400).join(it, on="item", how="semi").group_by("item").agg(pl.len().alias("c_ndays"))
        q = q.join(g, on="item", how="left").join(cd, on="item", how="left")
        outs.append(q)
    R = pl.concat(outs).sort("r")
    n7 = R["n7"].fill_null(0).to_numpy().astype(np.float64)
    d24 = R["dec24"].fill_null(0).to_numpy(); d72 = R["dec72"].fill_null(0).to_numpy()
    D = np.stack([R[f"d{j}"].fill_null(0).to_numpy().astype(np.float64) for j in range(7)], 1)  # j = 0 oldest .. 6 newest
    x = np.arange(7) - 3.0
    slope = (D * x).sum(1) / 28.0
    mean = D.mean(1)
    out = {"it_n7": n7, "it_trend": np.log((d24 + 0.5) / (d72 + 0.5)),
           "it_slope7": np.where(mean > 0, slope / np.where(mean > 0, mean, 1), np.nan),
           "it_q95gap": (R["A"].to_numpy() - R["q95t"].fill_null(np.nan).to_numpy()) / 86400.0}
    age_d = np.maximum(age_hours(f4), 1.0) / 24.0
    out["exhaust"] = R["c_ndays"].fill_null(0).to_numpy().astype(np.float64) / age_d
    return out


from h import holdout_ids
hold = holdout_ids()[0]
hrows = np.concatenate([np.arange(co[i], co[i] + M[i]) for i in hold])
cf = np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r")
hfeed = np.repeat(hold, M[hold])
A_h = ((fts[hfeed] + MSK) // 86400) * 86400 - MSK - 5 * 86400
log(f"holdout as-of values: {len(np.unique(A_h))} distinct, min {A_h.min()} max {A_h.max()}")
jobs = [("val", np.load(FLAT / "val/cand_item_id.npy"), None, np.load(FLAT / "val/cand_f4.npy")),
        ("test", np.load(FLAT / "test/cand_item_id.npy"), None, np.load(FLAT / "test/cand_f4.npy")),
        ("hold", np.asarray(np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")[hrows]), A_h,
         np.asarray(np.load(FLAT / "train/cand_f4.npy", mmap_mode="r")[hrows]))]
for tag, items, A, f4 in jobs:
    if A is None:
        A = np.full(len(items), TR_END, np.int64)
    o = query(items, A, f4)
    for c in COLS:
        np.save(OUT / f"c07_{c}_{tag}.npy", o[c].astype(np.float32))
    log(f"{tag}: rows {len(items)} | " + " ".join(f"{c}: finite {np.isfinite(o[c]).mean():.3f} mean {np.nanmean(o[c]):.3f}" for c in COLS))
log("DONE")
