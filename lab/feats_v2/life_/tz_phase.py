"""life_ family, part 3: DIURNAL PHASE / TIME-ZONE proxy ("TZ").

Russia spans 11 time zones but every timestamp is Moscow time. The MSK hours at which an item is shown (train
production candidates: shown when its local audience is active) and freshly interacted with (history events among
the user's 8 most recent, dated by the feed time) give the item's diurnal activity profile; an item whose audience
is active at 02:00-06:00 MSK is most likely in Siberia / the Far East. Likewise the profile of the user's history
items locates the user. Item-user phase mismatch flags non-local (e.g. delivery) listings.

Item profile: 24-bin MSK-hour histogram of (a) train candidate impressions, (b) fresh history appearances (hpos < 8)
in train/val/test histories. For train rows the profile only uses train feeds strictly before the feed day
(expanding by day, so no same-day look-ahead); for val/test rows it uses all train impressions + all fresh history
appearances of train/val/test (transductive, label-free, time-symmetric: diurnal phase is a static item attribute).

Features (prefix life_tz_):
  item_night     smoothed share of the item's profile mass at 00-06 MSK
  item_rc, item_rs   item mean (cos, sin) of hour angle minus the global mean vector (phase residual), shrunk
  user_rc, user_rs   mean phase residual of the user's history items (profile-weighted)
  align          cosine between user and item residual vectors x min(1, |.|) (soft alignment)
  dist           euclidean distance between user and item residual vectors
  feed_align     cos(feed-hour angle - item mean angle) x item resultant length
  item_n         log1p(profile mass)
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

NI = 111_400_000
MSK = 3 * 3600
KF = 8
SHRINK = 5.0
ANG = 2 * np.pi * (np.arange(24) + 0.5) / 24.0
COS, SIN = np.cos(ANG), np.sin(ANG)


@nb.njit(parallel=False, cache=True)
def add_hist(prof, items, hours, w):
    for r in range(len(items)):
        prof[items[r], hours[r]] += w


@nb.njit(parallel=True, cache=True)
def item_vec(prof, items, cosv, sinv, gc, gs, out):
    """out[r]: (n, rc, rs, night, mean_c, mean_s) for item items[r] (shrunk toward global)."""
    for r in nb.prange(len(items)):
        it = items[r]
        n = 0.0; c = 0.0; s = 0.0; nt = 0.0
        for h in range(24):
            v = prof[it, h]
            n += v; c += v * cosv[h]; s += v * sinv[h]
            if h < 6:
                nt += v
        out[r, 0] = n
        mc = (c + SHRINK * gc) / (n + SHRINK)
        ms = (s + SHRINK * gs) / (n + SHRINK)
        out[r, 1] = mc - gc
        out[r, 2] = ms - gs
        out[r, 3] = nt / n if n > 0 else np.nan
        out[r, 4] = mc
        out[r, 5] = ms


@nb.njit(parallel=True, cache=True)
def user_vec(hist_off, H, hv, out):
    """profile-mass-weighted mean residual over the feed's history items; hv rows aligned with history rows."""
    for f in nb.prange(len(H)):
        wsum = 0.0; c = 0.0; s = 0.0
        for k in range(hist_off[f], hist_off[f] + H[f]):
            n = hv[k, 0]
            if n <= 0:
                continue
            w = n / (n + SHRINK)
            wsum += w; c += w * hv[k, 1]; s += w * hv[k, 2]
        if wsum > 0:
            out[f, 0] = c / wsum; out[f, 1] = s / wsum; out[f, 2] = wsum
        else:
            out[f, 0] = np.nan; out[f, 1] = np.nan; out[f, 2] = 0.0


def hours_of(split, kind):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    hr = ((f["timestamp"].to_numpy().astype(np.int64) + MSK) // 3600 % 24).astype(np.int8)
    fi = np.load(FLAT / split / f"{kind}_fidx.npy")
    return hr[fi], fi, f


def profile_static():
    """item x 24 profile from all train candidates + fresh history appearances (all splits)."""
    prof = np.zeros((NI, 24), np.float32)
    h, _, _ = hours_of("train", "cand")
    add_hist(prof, np.load(FLAT / "train/cand_item_id.npy"), h, 1.0)
    del h
    for s in ["train", "val", "test"]:
        h, _, _ = hours_of(s, "hist")
        hp = np.load(FLAT / s / "hist_hpos.npy")
        m = hp < KF
        add_hist(prof, np.load(FLAT / s / "hist_item_id.npy")[m], h[m], 1.0)
    return prof


def feats_for(split, prof, gc, gs):
    ci = np.load(FLAT / split / "cand_item_id.npy")
    hc, cfi, f = hours_of(split, "cand")
    iv = np.empty((len(ci), 6), np.float64)
    item_vec(prof, ci, COS, SIN, gc, gs, iv)
    hi = np.load(FLAT / split / "hist_item_id.npy")
    hv = np.empty((len(hi), 6), np.float64)
    item_vec(prof, hi, COS, SIN, gc, gs, hv)
    H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
    uv = np.empty((len(f), 3), np.float64)
    user_vec(ho, H, hv, uv)
    del hv
    u = uv[cfi]
    o = {}
    o["item_night"] = (iv[:, 3] * iv[:, 0] + 2 * 0.0) / (iv[:, 0] + 2)  # placeholder replaced below
    nt = np.where(np.isfinite(iv[:, 3]), iv[:, 3], 0.0)
    o["item_night"] = np.where(iv[:, 0] > 0, (nt * iv[:, 0] + SHRINK * GNIGHT) / (iv[:, 0] + SHRINK), np.nan)
    has = iv[:, 0] > 0
    o["item_rc"] = np.where(has, iv[:, 1], np.nan)
    o["item_rs"] = np.where(has, iv[:, 2], np.nan)
    o["user_rc"] = u[:, 0]
    o["user_rs"] = u[:, 1]
    ni = np.hypot(iv[:, 1], iv[:, 2]); nu = np.hypot(u[:, 0], u[:, 1])
    dot = iv[:, 1] * u[:, 0] + iv[:, 2] * u[:, 1]
    o["align"] = np.where(has, dot / (ni * nu + 1e-6) * np.minimum(1.0, ni / 0.05) * np.minimum(1.0, nu / 0.05), np.nan)
    o["dist"] = np.where(has, np.hypot(iv[:, 1] - u[:, 0], iv[:, 2] - u[:, 1]), np.nan)
    fa = 2 * np.pi * (hc.astype(np.float64) + 0.5) / 24.0
    o["feed_align"] = np.where(has, np.cos(fa) * iv[:, 4] + np.sin(fa) * iv[:, 5], np.nan)
    o["item_n"] = np.log1p(iv[:, 0])
    return o


GNIGHT = 0.0


def main():
    global GNIGHT
    nb.set_num_threads(16)
    t0 = time.time()
    prof = profile_static()
    g = prof.sum(0, dtype=np.float64)
    gp = g / g.sum()
    gc, gs = float((gp * COS).sum()), float((gp * SIN).sum())
    GNIGHT = float(gp[:6].sum())
    print("global profile", np.round(gp * 100, 1), "gc,gs", gc, gs, "night", GNIGHT, f"{time.time()-t0:.0f}s", flush=True)
    for s in sys.argv[1].split(","):
        o = feats_for(s, prof, gc, gs)
        for n, v in o.items():
            np.save(FEATS / s / f"life_tz_{n}.npy", np.asarray(v, np.float32))
        print(s, "saved", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
