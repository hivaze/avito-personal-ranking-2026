"""life_ family, part 1: TRAIN-STREAM item lifecycle ("TS").

The dense train stream (production feeds, Sep 6-19) is replayed in time order. Per item we keep a running state built
from (a) candidate appearances and (b) history appearances; for history rows we additionally detect NEW interactions:
an event (item, event_type) in a feed's history that was not present in the same user's previous feed's history
(user_id != 0). Such an event happened in (t_prev, t], so it gives a dated interaction (dated by t, an upper bound). The history is NOT a clean sliding window (only ~9%
of consecutive same-user feeds align perfectly, and only 18% of train feeds have an earlier feed of the same user), so
a "dated interaction" is defined as a history appearance among the user's KF=8 most recent events (hpos < 8) that was
not already in the same user's previous feed (if any). This is the 'fresh interaction' stream used below (i_*).

All features are evaluated at a reference time tau:
  * train rows: tau = feed time t, state built from strictly earlier train feeds (rows with the same timestamp are
    read before they are added) -> no look-ahead, no labels used at all;
  * val/test rows: tau = end of the train stream (TR_END = Sun Sep 20 00:00 MSK). Identical definition for val and
    test ("as of the end of the dense observation window"), so the val->test time gap does not shift these values.

Outputs (float32, NaN = item never seen in the train stream before tau):
  life_ts_c_ndays     distinct MSK days the item was a train candidate
  life_ts_c_span_h    last - first candidate appearance (h)
  life_ts_c_gap_h     tau - last candidate appearance (h)        ("still being shown at the end?")
  life_ts_age_h       tau - first appearance as candidate or in any history (h; left-censored at Sep 6)
  life_ts_h_gap_h     tau - last appearance in any history (h)
  life_ts_i_n         number of new (dated) interactions, all event types
  life_ts_i_gap_h     tau - last new interaction (h)
  life_ts_i_age_h     tau - first new interaction (h)
  life_ts_i_dec24     exp-decayed new-interaction count at tau, half-life 24h
  life_ts_i_dec72     same, half-life 72h
  life_ts_c_dec24     exp-decayed candidate-exposure count at tau, half-life 24h
  life_ts_c_n         candidate appearances (exposures) in the train stream
  life_ts_i_n1        new interactions with event_type 1
  life_ts_hour_aff    hour-of-day affinity: smoothed share of the item's dated interactions in the feed's 4-hour MSK
                      bucket divided by the global share of that bucket (1 = no preference)
Derived (numpy, after the pass):
  life_ts_eng_rate    i_n / (c_n + 10)            interactions per exposure (label-free engagement rate)
  life_ts_i_trend     i_dec24 / (i_dec72 + 0.5)   short/long interaction momentum
  life_ts_c_per_day   c_n / c_ndays               exposure intensity
  life_ts_i_per_day   i_n / (age_h/24 + 1)        lifetime interaction rate
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

NI = 111_400_000
TR_END = 1789862400  # 2026-09-20 00:00 MSK
MSK = 3 * 3600
KF = 8  # 'fresh' history window: the user's 8 most recent events
HL24 = 24 * 3600 / np.log(2.0)
HL72 = 72 * 3600 / np.log(2.0)
NAMES = ["c_ndays", "c_span_h", "c_gap_h", "age_h", "h_gap_h", "i_n", "i_gap_h", "i_age_h", "i_dec24", "i_dec72",
         "c_dec24", "c_n", "i_n1", "hour_aff"]
NQ = len(NAMES)


@nb.njit(parallel=True, cache=True)
def new_flags(prev, hist_off, H, h_item, h_et, out):
    """out: 1 = new event since the user's previous feed, 0 = already there, 2 = undatable (no previous feed)."""
    for f in nb.prange(len(H)):
        p = prev[f]
        a = hist_off[f]
        n = H[f]
        if p < 0:
            for k in range(n):
                out[a + k] = 2
            continue
        pa = hist_off[p]
        pn = H[p]
        keys = np.empty(pn, np.int64)
        for k in range(pn):
            keys[k] = np.int64(h_item[pa + k]) * 16 + h_et[pa + k]
        keys.sort()
        for k in range(n):
            key = np.int64(h_item[a + k]) * 16 + h_et[a + k]
            j = np.searchsorted(keys, key)
            out[a + k] = 0 if (j < pn and keys[j] == key) else 1


@nb.njit(cache=True, inline="always")
def _query(it, tau, hb, c_first, c_last, c_n, c_nd, c_dec, h_first, h_last, i_n, i_n1, i_first, i_last,
           i_d24, i_d72, i_hb, g_hb, res):
    for q in range(NQ):
        res[q] = np.nan
    if c_n[it] > 0:
        res[0] = c_nd[it]
        res[1] = (c_last[it] - c_first[it]) / 3600.0
        res[2] = (tau - c_last[it]) / 3600.0
        res[10] = c_dec[it] * np.exp(-(tau - c_last[it]) / HL24)
        res[11] = c_n[it]
    first = 2147483647
    if c_n[it] > 0:
        first = c_first[it]
    if h_last[it] > 0:
        first = min(first, h_first[it])
        res[4] = (tau - h_last[it]) / 3600.0
    if first < 2147483647:
        res[3] = (tau - first) / 3600.0
        res[5] = i_n[it]
        res[12] = i_n1[it]
        res[8] = 0.0
        res[9] = 0.0
        if c_n[it] == 0:
            res[0] = 0.0
            res[11] = 0.0
            res[10] = 0.0
    if i_n[it] > 0:
        res[6] = (tau - i_last[it]) / 3600.0
        res[7] = (tau - i_first[it]) / 3600.0
        res[8] = i_d24[it] * np.exp(-(tau - i_last[it]) / HL24)
        res[9] = i_d72[it] * np.exp(-(tau - i_last[it]) / HL72)
    if first < 2147483647:
        tot = 0.0
        for b in range(6):
            tot += g_hb[b]
        gshare = (g_hb[hb] + 1.0) / (tot + 6.0)
        res[13] = ((i_hb[it, hb] + 2.0 * gshare) / (i_n[it] + 2.0)) / gshare


@nb.njit(cache=True)
def ts_pass(fts, cand_off, M, hist_off, H, c_item, h_item, h_new, h_et, out,
            c_first, c_last, c_n, c_nd, c_ld, c_dec, h_first, h_last, i_n, i_n1, i_first, i_last, i_d24, i_d72,
            i_hb, g_hb):
    nf = len(fts)
    res = np.empty(NQ, np.float64)
    f0 = 0
    while f0 < nf:
        t = fts[f0]
        f1 = f0
        while f1 < nf and fts[f1] == t:
            f1 += 1
        hb = ((t + MSK) // 3600 % 24) // 4
        day = (t + MSK) // 86400
        # 1) queries for all candidate rows of feeds with this timestamp (state = strictly earlier feeds)
        for f in range(f0, f1):
            for r in range(cand_off[f], cand_off[f] + M[f]):
                _query(c_item[r], t, hb, c_first, c_last, c_n, c_nd, c_dec, h_first, h_last, i_n, i_n1, i_first,
                       i_last, i_d24, i_d72, i_hb, g_hb, res)
                for q in range(NQ):
                    out[r, q] = res[q]
        # 2) updates
        for f in range(f0, f1):
            for r in range(cand_off[f], cand_off[f] + M[f]):
                it = c_item[r]
                if c_n[it] == 0:
                    c_first[it] = t
                    c_dec[it] = 1.0
                else:
                    c_dec[it] = c_dec[it] * np.exp(-(t - c_last[it]) / HL24) + 1.0
                if c_ld[it] != day:
                    c_nd[it] += 1
                    c_ld[it] = day
                c_last[it] = t
                c_n[it] += 1
            for r in range(hist_off[f], hist_off[f] + H[f]):
                it = h_item[r]
                if h_last[it] == 0:
                    h_first[it] = t
                h_last[it] = t
                if h_new[r] == 1:
                    if i_n[it] == 0:
                        i_first[it] = t
                        i_d24[it] = 1.0
                        i_d72[it] = 1.0
                    else:
                        dt = t - i_last[it]
                        i_d24[it] = i_d24[it] * np.exp(-dt / HL24) + 1.0
                        i_d72[it] = i_d72[it] * np.exp(-dt / HL72) + 1.0
                    i_last[it] = t
                    i_n[it] += 1
                    if h_et[r] == 1:
                        i_n1[it] += 1
                    i_hb[it, hb] += 1
                    g_hb[hb] += 1
        f0 = f1


@nb.njit(parallel=True, cache=True)
def ts_query_static(items, taus, hbs, out, c_first, c_last, c_n, c_nd, c_dec, h_first, h_last, i_n, i_n1, i_first,
                    i_last, i_d24, i_d72, i_hb, g_hb):
    for r in nb.prange(len(items)):
        res = np.empty(NQ, np.float64)
        _query(items[r], taus[r], hbs[r], c_first, c_last, c_n, c_nd, c_dec, h_first, h_last, i_n, i_n1, i_first,
               i_last, i_d24, i_d72, i_hb, g_hb, res)
        for q in range(NQ):
            out[r, q] = res[q]


def derived(X):
    d = {}
    g = {n: X[:, i] for i, n in enumerate(NAMES)}
    d["eng_rate"] = g["i_n"] / (g["c_n"] + 10.0)
    d["i_trend"] = g["i_dec24"] / (g["i_dec72"] + 0.5)
    d["c_per_day"] = np.where(g["c_ndays"] > 0, g["c_n"] / np.maximum(g["c_ndays"], 1), np.nan)
    d["i_per_day"] = g["i_n"] / (g["age_h"] / 24.0 + 1.0)
    return d


def save_split(split, X, prefix="life_ts_"):
    out = FEATS / split
    for i, n in enumerate(NAMES):
        np.save(out / f"{prefix}{n}.npy", X[:, i].astype(np.float32))
    for n, v in derived(X).items():
        np.save(out / f"{prefix}{n}.npy", v.astype(np.float32))


def main():
    nb.set_num_threads(16)
    t0 = time.time()
    f = pl.read_parquet(FLAT / "train" / "feeds.parquet")
    fts = f["timestamp"].to_numpy().astype(np.int64)
    uid = f["user_id"].to_numpy()
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy(); ho = f["hist_off"].to_numpy()
    # previous feed of the same (non-zero) user
    order = np.lexsort((np.arange(len(f)), fts, uid))
    prev = np.full(len(f), -1, np.int64)
    same = (uid[order][1:] == uid[order][:-1]) & (uid[order][1:] != 0)
    prev[order[1:][same]] = order[:-1][same]
    print("prev feeds", (prev >= 0).mean(), f"{time.time()-t0:.0f}s", flush=True)
    h_item = np.load(FLAT / "train/hist_item_id.npy")
    h_et = np.load(FLAT / "train/hist_event_type.npy")
    h_new = np.empty(len(h_item), np.int8)
    new_flags(prev, ho, H, h_item, h_et, h_new)
    hp = np.load(FLAT / "train/hist_hpos.npy")
    vc = np.bincount(h_new, minlength=3) / len(h_new)
    print("new flags: old/new/undatable", vc.round(4), "mean hpos of new", float(hp[h_new == 1].mean()), flush=True)
    # dated interaction = one of the user's KF most recent events, not already present in the user's previous feed
    h_new = ((hp < KF) & (h_new != 0)).astype(np.int8)
    del hp
    print("dated interactions", int(h_new.sum()), f"{time.time()-t0:.0f}s", flush=True)
    c_item = np.load(FLAT / "train/cand_item_id.npy")
    st = dict(c_first=np.zeros(NI, np.int32), c_last=np.zeros(NI, np.int32), c_n=np.zeros(NI, np.int32),
              c_nd=np.zeros(NI, np.int16), c_ld=np.zeros(NI, np.int16), c_dec=np.zeros(NI, np.float32),
              h_first=np.zeros(NI, np.int32), h_last=np.zeros(NI, np.int32), i_n=np.zeros(NI, np.int32),
              i_n1=np.zeros(NI, np.int32), i_first=np.zeros(NI, np.int32), i_last=np.zeros(NI, np.int32),
              i_d24=np.zeros(NI, np.float32), i_d72=np.zeros(NI, np.float32), i_hb=np.zeros((NI, 6), np.int32),
              g_hb=np.zeros(6, np.float64))
    out = np.empty((len(c_item), NQ), np.float32)
    ts_pass(fts, co, M, ho, H, c_item, h_item, h_new, h_et, out, st["c_first"], st["c_last"], st["c_n"], st["c_nd"],
            st["c_ld"], st["c_dec"], st["h_first"], st["h_last"], st["i_n"], st["i_n1"], st["i_first"], st["i_last"],
            st["i_d24"], st["i_d72"], st["i_hb"], st["g_hb"])
    print("pass done", f"{time.time()-t0:.0f}s", "global hour buckets", (st["g_hb"] / st["g_hb"].sum()).round(3), flush=True)
    del h_item, h_et, h_new
    save_split("train", out)
    del out
    print("train saved", f"{time.time()-t0:.0f}s", flush=True)
    for s in ["val", "test"]:
        fs = pl.read_parquet(FLAT / s / "feeds.parquet")
        fi = np.load(FLAT / s / "cand_fidx.npy")
        items = np.load(FLAT / s / "cand_item_id.npy")
        tf = fs["timestamp"].to_numpy().astype(np.int64)[fi]
        hbs = ((tf + MSK) // 3600 % 24) // 4
        taus = np.full(len(items), TR_END, np.int64)
        o = np.empty((len(items), NQ), np.float32)
        ts_query_static(items, taus, hbs, o, st["c_first"], st["c_last"], st["c_n"], st["c_nd"], st["c_dec"],
                        st["h_first"], st["h_last"], st["i_n"], st["i_n1"], st["i_first"], st["i_last"], st["i_d24"],
                        st["i_d72"], st["i_hb"], st["g_hb"])
        save_split(s, o)
        print(s, "saved; coverage", np.isfinite(o[:, 3]).mean().round(4), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
