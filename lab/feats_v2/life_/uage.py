"""life_ family, part 5: USER FRESHNESS PREFERENCE ("UAGE").

Uses the exact listing posting time recovered from f4 (see post_time.py). For every history item that appears as
a candidate anywhere (train/val/test; 39.4M items), its posting slot is known -> its age at the feed time t.
Recent history events happened shortly before t, so the age at t approximates the age at interaction. Users differ
strongly in how fresh the listings they engage with are (resellers hunting new listings vs. browsers of old stock);
base has the candidate's age (f4) but not the user's preference, nor the match between the two.

Features (prefix life_ua_; label-free; identical for every split):
  med      median log1p(age_h at t) over the user's 32 most recent history events with a known posting time
  fresh    share of those events whose item was < 48 h old at t
  n        number of those events (coverage)
  diff     log1p(candidate age_h) - med          (candidate fresher (<0) / older (>0) than the user's usual)
  absdiff  |diff|
  z        diff / (std of the user's log-ages + 0.25)
  med_e1 / diff_e1   same restricted to event_type 1 events
  new_since  candidate posting time - latest posting time among ALL the user's history items (h); the latter is a
             lower bound of the user's last activity, so > 0 means the listing appeared after the user's last known
             activity ("new since last visit")
  u_gap      t - latest posting time among the user's history items (h, user-level; upper bound of inactivity)
"""
import sys, time
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feats_v2/life_")
from common import *
import numba as nb
import post_time as P

NI = 111_400_000
LASTK = 32
CAPAGE = 40000.0


@nb.njit(parallel=True, cache=True)
def user_last_post(hist_off, H, h_slot, out):
    for f in nb.prange(len(H)):
        m = -1
        for k in range(hist_off[f], hist_off[f] + H[f]):
            if h_slot[k] > m:
                m = h_slot[k]
        out[f] = m


@nb.njit(parallel=True, cache=True)
def user_stats(hist_off, H, h_slot, h_et, h_hpos, fts, grid, out):
    """out[f]: med, fresh share, n, std, med_e1 (log1p age_h at feed time)."""
    for f in nb.prange(len(H)):
        a = hist_off[f]
        vals = np.empty(LASTK); ve = np.empty(LASTK)
        n = 0; ne = 0
        for k in range(a, a + H[f]):
            if h_hpos[k] >= LASTK:
                continue
            s = h_slot[k]
            if s < 0:
                continue
            age = max(fts[f] - s * grid, 0.0) / 3600.0
            v = np.log1p(age)
            if n < LASTK:
                vals[n] = v; n += 1
            if h_et[k] == 1 and ne < LASTK:
                ve[ne] = v; ne += 1
        if n == 0:
            for j in range(5):
                out[f, j] = np.nan
            out[f, 2] = 0.0
            continue
        x = np.sort(vals[:n])
        out[f, 0] = x[n // 2] if n % 2 == 1 else 0.5 * (x[n // 2 - 1] + x[n // 2])
        fr = 0.0; m = 0.0
        for j in range(n):
            if x[j] < np.log1p(48.0):
                fr += 1
            m += x[j]
        m /= n
        sd = 0.0
        for j in range(n):
            sd += (x[j] - m) ** 2
        out[f, 1] = fr / n
        out[f, 2] = n
        out[f, 3] = np.sqrt(sd / n)
        if ne > 0:
            xe = np.sort(ve[:ne])
            out[f, 4] = xe[ne // 2] if ne % 2 == 1 else 0.5 * (xe[ne // 2 - 1] + xe[ne // 2])
        else:
            out[f, 4] = np.nan


def main():
    nb.set_num_threads(16)
    t0 = time.time()
    knots = np.load(P.KNOTS_PATH)
    parts = []
    for s in ["train", "val", "test"]:
        slot, _, _, _ = P.post_slot(s, knots)
        it = np.load(FLAT / s / "cand_item_id.npy")
        m = slot >= 0
        parts.append(pl.DataFrame({"item": it[m], "slot": slot[m]}))
    I = pl.concat(parts).group_by("item").agg(pl.col("slot").mode().first())
    del parts
    item2slot = np.full(NI, -1, np.int64); item2slot[I["item"].to_numpy()] = I["slot"].to_numpy()
    del I
    print("item map", f"{time.time()-t0:.0f}s", flush=True)
    for s in sys.argv[1].split(","):
        f = pl.read_parquet(FLAT / s / "feeds.parquet")
        fts = f["timestamp"].to_numpy().astype(np.float64)
        H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy()
        h_slot = item2slot[np.load(FLAT / s / "hist_item_id.npy")]
        us = np.empty((len(f), 5))
        user_stats(ho, H, h_slot, np.load(FLAT / s / "hist_event_type.npy"), np.load(FLAT / s / "hist_hpos.npy"),
                   fts, float(P.GRID), us)
        ulp = np.empty(len(f), np.int64)
        user_last_post(ho, H, h_slot, ulp)
        del h_slot
        fi = np.load(FLAT / s / "cand_fidx.npy")
        ca = P.age_hours(np.load(FLAT / s / "cand_f4.npy"), knots)
        ca = np.log1p(np.where(np.isfinite(ca), ca, CAPAGE))
        u = us[fi]
        o = {"med": u[:, 0], "fresh": u[:, 1], "n": u[:, 2], "diff": ca - u[:, 0], "absdiff": np.abs(ca - u[:, 0]),
             "z": (ca - u[:, 0]) / (u[:, 3] + 0.25), "med_e1": u[:, 4], "diff_e1": ca - u[:, 4]}
        cslot, _, _, _ = P.post_slot(s, knots)
        ul = ulp[fi].astype(np.float64)
        has = (ul >= 0) & (cslot >= 0)
        o["new_since"] = np.where(has, (cslot - ul) * P.GRID / 3600.0, np.nan)
        o["u_gap"] = np.where(ul >= 0, (fts[fi] - ul * P.GRID) / 3600.0, np.nan)
        for n, v in o.items():
            np.save(FEATS / s / f"life_ua_{n}.npy", np.asarray(v, np.float32))
        print(s, "saved; user coverage", float(np.mean(us[:, 2] > 0)), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
