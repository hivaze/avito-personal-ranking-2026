"""life_ family, part 4: EXACT LISTING POSTING TIME recovered from f4 ("POST").

Finding: f4 is a fixed piecewise-linear transform of the listing's AGE. For 1.1M train items seen on several days,
f4 never decreases and its growth over a time gap is a deterministic function of its level (std of the 11-day
increment within an f4 bin ~0.005 at increments of ~0.35). A least-squares fit of a 64-knot piecewise-linear map
age(f4) (knots at k/64) on 3M (first, last appearance) pairs reproduces every observed time gap with a median error
of 0.4 s (q99 50 s); knots: 1/64 = 0 h, 2/64 = 0.7 h, 9/64 = 24 h, 20/64 = 164 h, 30/64 = 573 h, 63/64 = 39,900 h (cap).
Hence  t_post = t_feed - age(f4)  is the listing's creation time; it lands exactly on the dataset's 128-s timestamp
grid (t_post mod 128 = 0 for 99% of young items), i.e. it is recovered at full timestamp resolution, and is
consistent across appearances of the same item (median std 0.08 s). f4 = 63/64 (age >= 4.5 years, capped) -> NaN.

Features (prefix life_post_; all label-free, computed identically for every split from the row's own f4 and feed
time, so they work for items never seen anywhere else):
  hour         MSK hour-of-day of posting (fractional)
  min_of_day_c / _s   cos/sin of posting time-of-day (circular encoding)
  dow          MSK day of week of posting (0 = Mon) [drift-checked; test fresh items are posted on weekend days]
  slot_ratio   bulk-upload proxy: distinct items (all splits' candidates) posted in the same 128-s slot divided by the
               mean count of the 30 slots before and after (+1 smoothing). Censoring-robust: under-observation of
               recently posted items affects the slot and its neighbours alike.
  slot_n       log1p(distinct items posted in the same 128-s slot)
  hist_same_slot  number of the user's history items posted in the same 128-s slot as the candidate (posting times
               of history items come from their own candidate appearances anywhere; ~seller/batch affinity)
  feed_same_slot  number of other candidates of the same feed posted in the same 128-s slot
  age_rank     within-feed percentile rank of age (= of f4) is already in base as f4_rk -> not produced
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

MSK = 3 * 3600
GRID = 128
K = 64
CAP = 63 / 64 - 1e-6
SCR = Path("/tmp/scratchpad")
KNOTS_PATH = FEATS / "life_f4_age_knots.npy"


def fit_knots():
    f = pl.read_parquet(FLAT / "train" / "feeds.parquet")
    ts = f["timestamp"].to_numpy()
    fi = np.load(FLAT / "train/cand_fidx.npy")
    df = pl.DataFrame({"item": np.load(FLAT / "train/cand_item_id.npy"), "t": ts[fi], "f4": np.load(FLAT / "train/cand_f4.npy")})
    g = df.group_by("item").agg(pl.col("t").first().alias("t1"), pl.col("f4").first().alias("x1"),
                                pl.col("t").last().alias("t2"), pl.col("f4").last().alias("x2"))
    g = g.filter((pl.col("t2") - pl.col("t1") > 86400) & (pl.col("x2") < CAP) & (pl.col("x1") > 1 / 64 + 1e-6))
    g = g.sample(min(3_000_000, g.height), seed=0)
    x1, x2 = g["x1"].to_numpy().astype(np.float64), g["x2"].to_numpy().astype(np.float64)
    dt = (g["t2"] - g["t1"]).to_numpy() / 3600.0

    def basis(x):
        u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
        B = np.zeros((len(x), K)); B[np.arange(len(x)), k] += 1 - w; B[np.arange(len(x)), k + 1] += w
        return B
    A = (basis(x2) - basis(x1))[:, 2:]
    a = np.linalg.lstsq(A, dt, rcond=None)[0]
    r = dt - A @ a
    print("f4->age fit: pairs", len(dt), "median |err| (s)", float(np.median(np.abs(r)) * 3600), "q99 (s)", float(np.quantile(np.abs(r), 0.99) * 3600), flush=True)
    knots = np.r_[0.0, 0.0, a]
    np.save(KNOTS_PATH, knots)
    return knots


def age_hours(x, knots):
    x = x.astype(np.float64)
    u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


def post_slot(split, knots):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    fi = np.load(FLAT / split / "cand_fidx.npy")
    t = f["timestamp"].to_numpy().astype(np.float64)[fi]
    a = age_hours(np.load(FLAT / split / "cand_f4.npy"), knots)
    tp = t - a * 3600.0
    slot = np.where(np.isfinite(tp), np.round(tp / GRID), -1).astype(np.int64)
    return slot, t, fi, f


@nb.njit(parallel=True, cache=True)
def same_slot_hist(hist_off, H, h_slot, cfi, c_slot, out):
    for r in nb.prange(len(cfi)):
        s = c_slot[r]
        if s < 0:
            out[r] = np.nan
            continue
        f = cfi[r]
        n = 0
        for k in range(hist_off[f], hist_off[f] + H[f]):
            if h_slot[k] == s:
                n += 1
        out[r] = n


@nb.njit(parallel=True, cache=True)
def same_slot_feed(cand_off, M, c_slot, c_item, out):
    for f in nb.prange(len(M)):
        a = cand_off[f]
        for i in range(a, a + M[f]):
            if c_slot[i] < 0:
                out[i] = np.nan
                continue
            n = 0
            for j in range(a, a + M[f]):
                if j != i and c_slot[j] == c_slot[i] and c_item[j] != c_item[i]:
                    n += 1
            out[i] = n


def main():
    nb.set_num_threads(16)
    t0 = time.time()
    knots = np.load(KNOTS_PATH) if KNOTS_PATH.exists() else fit_knots()
    # global item -> slot map and distinct items per slot (all splits' candidates)
    parts = []
    for s in ["train", "val", "test"]:
        slot, _, _, _ = post_slot(s, knots)
        it = np.load(FLAT / s / "cand_item_id.npy")
        m = slot >= 0
        parts.append(pl.DataFrame({"item": it[m], "slot": slot[m]}))
    I = pl.concat(parts).group_by("item").agg(pl.col("slot").mode().first())
    del parts
    print("items with posting slot", I.height, f"{time.time()-t0:.0f}s", flush=True)
    sc = I.group_by("slot").len().sort("slot")
    s_slot = sc["slot"].to_numpy(); s_n = sc["len"].to_numpy().astype(np.float64)
    smin = s_slot.min(); span = s_slot.max() - smin + 1
    dense = np.zeros(span + 62); dense[s_slot - smin + 31] = s_n
    cs = np.r_[0.0, np.cumsum(dense)]
    idx = s_slot - smin + 31
    neigh_mean = (cs[idx + 31] - cs[idx - 30] - s_n) / 60.0  # mean over the 30 slots on each side (own excluded)
    ratio_tab = (s_n + 1.0) / (neigh_mean + 1.0)
    item_ids = I["item"].to_numpy(); item_slot = I["slot"].to_numpy()
    NI = 111_400_000
    item2slot = np.full(NI, -1, np.int64); item2slot[item_ids] = item_slot
    for s in sys.argv[1].split(","):
        slot, t, fi, f = post_slot(s, knots)
        pos = np.searchsorted(s_slot, slot)
        pos = np.clip(pos, 0, len(s_slot) - 1)
        ok = (slot >= 0) & (s_slot[pos] == slot)
        tp = slot.astype(np.float64) * GRID
        o = {}
        hod = np.where(slot >= 0, ((tp + MSK) % 86400) / 3600.0, np.nan)
        o["hour"] = hod
        o["min_of_day_c"] = np.cos(2 * np.pi * hod / 24.0)
        o["min_of_day_s"] = np.sin(2 * np.pi * hod / 24.0)
        o["dow"] = np.where(slot >= 0, ((tp + MSK) // 86400 + 3) % 7, np.nan)  # 1970-01-01 was a Thursday -> +3: Mon=0
        o["slot_ratio"] = np.where(ok, ratio_tab[pos], np.nan)
        o["slot_n"] = np.where(ok, np.log1p(s_n[pos]), np.nan)
        hi = np.load(FLAT / s / "hist_item_id.npy")
        h_slot = item2slot[hi]
        H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
        out = np.empty(len(slot), np.float64)
        same_slot_hist(ho, H, h_slot, fi.astype(np.int64), slot, out)
        o["hist_same_slot"] = out.copy()
        M = f["M"].to_numpy().astype(np.int64); co = f["cand_off"].to_numpy()
        same_slot_feed(co, M, slot, np.load(FLAT / s / "cand_item_id.npy"), out)
        o["feed_same_slot"] = out
        for n, v in o.items():
            np.save(FEATS / s / f"life_post_{n}.npy", np.asarray(v, np.float32))
        print(s, "saved", {n: round(float(np.nanmean(v)), 3) for n, v in o.items()}, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
