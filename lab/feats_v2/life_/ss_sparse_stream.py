"""life_ family, part 2: SPARSE-STREAM local lifecycle ("SS") + all-data first-seen age.

Stream = the exploration feeds around a feed in time.
  * KEEP variant 'split' (prefix life_ss_): WITHIN-SPLIT stream - val rows use val feeds only, test rows use test feeds
    only. Identical definition for both, and it makes val behave like the private test w.r.t. censoring: val-Thursday
    feeds have their future truncated at the end of their split exactly like test-Tuesday feeds (val-Tuesday ~ test
    Monday). This is the 'as if val were the last split' simulation, and it is the definition actually used.
  * comparison variant 'full' (prefix life_ssF_, val only, deleted after the report): val rows see val+test feeds, i.e.
    a 5-7 day future that no private-test feed has -> optimistic.
  * train rows: a random 2.8% feed subsample of train (seeded) -> same feed density as val/test (~7k feeds/day).
For each candidate row (feed f, time t, user u, item i) we use the item's appearances in OTHER stream feeds (own feed
and same non-zero user excluded), of four kinds:
  c  as a candidate (the item is still being served = still alive)
  h  in a history (someone interacted with it before that feed), one per (feed, item)
  f  in a history among that user's 8 most recent events ('fresh' interaction, close to the feed time)
  e  in a history with event_type 1 (the event type closest to the target action)

Censoring-aware design (the stream has gaps: Sep 20-21, 23, 25 missing; it ends Sep 29 24:00 MSK):
  kernel rates   {k}_kp{T} / {k}_kn{T} = sum_e exp(-|dt_e|/T) / (sum_g exp(-|dt_g|/T) + 200) * 1e4
                 (numerator over the item's appearances before-or-at t / after t, denominator over all observed
                 stream feeds on the same side, own feed excluded). A truncated or gappy window only makes the
                 estimate noisier (shrunk to 0 by the +200 pseudo-feeds), it does not make it systematically smaller.
  window rates   {k}_r{p24,p72,n6,n24,n48} = count in window / (observed stream feeds in window + 200) * 1e4
  a_kn{T}        c_kn{T} + h_kn{T}  ('still alive' rate: served or interacted after t)
Combined with the train stream (life_ts_* must exist):
  life_age_any_h     t - first appearance anywhere (train stream, or earlier sparse feeds), hours, clipped at 360
                     (360h < earliest possible censoring point for every val/test feed); NaN = never seen before
  life_never_seen    1 if the item was never seen in any feed before t
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

NI = 111_400_000
MSK = 3 * 3600
KF = 8
TAUS = (24, 72)
KDEN = 200.0
SUB_P = 0.028
WP = (24 * 3600, 72 * 3600)
WN = (6 * 3600, 24 * 3600, 48 * 3600, 96 * 3600)
OBS_H = 96 * 3600  # availability feature horizon
KINDS = ("c", "h", "f", "e")
TR_END = 1789862400  # Sep 20 00:00 MSK


def load_feeds(split):
    return pl.read_parquet(FLAT / split / "feeds.parquet")


def stream_events(split, feed_mask=None):
    f = load_feeds(split)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    uid = f["user_id"].to_numpy().astype(np.int64)
    keep = np.ones(len(f), bool) if feed_mask is None else feed_mask
    ci = np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")
    cf = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
    hi = np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r")
    hf = np.load(FLAT / split / "hist_fidx.npy", mmap_mode="r")
    hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")
    he = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    if feed_mask is not None:
        co = f["cand_off"].to_numpy(); M = f["M"].to_numpy()
        ho = f["hist_off"].to_numpy(); H = f["H"].to_numpy()
        idx_f = np.flatnonzero(keep)
        cr = np.concatenate([np.arange(co[i], co[i] + M[i]) for i in idx_f])
        hr = np.concatenate([np.arange(ho[i], ho[i] + H[i]) for i in idx_f])
        ci, cf = np.asarray(ci[cr]), np.asarray(cf[cr])
        hi, hf, hp, he = np.asarray(hi[hr]), np.asarray(hf[hr]), np.asarray(hp[hr]), np.asarray(he[hr])
    else:
        ci, cf, hi, hf, hp, he = (np.asarray(a) for a in (ci, cf, hi, hf, hp, he))
    H = pl.DataFrame({"f": hf, "i": hi, "p": hp, "e1": he == 1}).group_by(["f", "i"]).agg(
        pl.col("p").min(), pl.col("e1").any())
    hf2, hi2, hp2, he2 = H["f"].to_numpy(), H["i"].to_numpy(), H["p"].to_numpy(), H["e1"].to_numpy()
    fr = hp2 < KF
    item = np.concatenate([ci, hi2, hi2[fr], hi2[he2]]).astype(np.int32)
    fid = np.concatenate([cf, hf2, hf2[fr], hf2[he2]]).astype(np.int64)
    kind = np.concatenate([np.zeros(len(ci), np.int8), np.ones(len(hi2), np.int8), np.full(int(fr.sum()), 2, np.int8),
                           np.full(int(he2.sum()), 3, np.int8)])
    return dict(ts=ts[keep], item=item, t=ts[fid], fid=fid, user=uid[fid], kind=kind)


def merge_streams(parts):
    out = {k: [] for k in ["item", "t", "fuid", "user", "kind"]}
    fts = []
    for code, ev in parts:
        out["item"].append(ev["item"]); out["t"].append(ev["t"]); out["fuid"].append(ev["fid"] + (code << 28))
        out["user"].append(ev["user"]); out["kind"].append(ev["kind"]); fts.append(ev["ts"])
    ev = {k: np.concatenate(v) for k, v in out.items()}
    order = np.lexsort((ev["t"], ev["item"]))
    ev = {k: v[order] for k, v in ev.items()}
    cnt = np.bincount(ev["item"], minlength=NI)
    off = np.zeros(NI + 1, np.int64); np.cumsum(cnt, out=off[1:])
    ev["off"] = off
    ev["feed_ts"] = np.sort(np.concatenate(fts))
    return ev


@nb.njit(cache=True)
def kernel_den(fts, tau):
    """dp[k] = sum_{j<=k} exp(-(fts[k]-fts[j])/tau); dn[k] = sum_{j>k} exp(-(fts[j]-fts[k])/tau)."""
    n = len(fts)
    dp = np.zeros(n); dn = np.zeros(n)
    acc = 0.0
    for k in range(n):
        if k > 0:
            acc *= np.exp(-(fts[k] - fts[k - 1]) / tau)
        acc += 1.0
        dp[k] = acc
    acc = 0.0
    for k in range(n - 1, -1, -1):
        dn[k] = acc
        if k > 0:
            acc = (acc + 1.0) * np.exp(-(fts[k] - fts[k - 1]) / tau)
    return dp, dn


@nb.njit(cache=True)
def feed_level(q_ft, own, fts, dps, dns, taus, wp, wn):
    """per query feed: observed stream feeds per window (own excluded) and kernel denominators per tau."""
    n = len(q_ft)
    nobs = np.zeros((n, len(wp) + len(wn)))
    den = np.zeros((n, len(taus), 2))
    for i in range(n):
        t = q_ft[i]
        hi = np.searchsorted(fts, t, side="right")
        for w in range(len(wp)):
            lo = np.searchsorted(fts, t - wp[w], side="left")
            nobs[i, w] = max(hi - lo - own[i], 0)
        for w in range(len(wn)):
            nobs[i, len(wp) + w] = np.searchsorted(fts, t + wn[w], side="right") - hi
        for k in range(len(taus)):
            if hi > 0:
                den[i, k, 0] = max(dps[k, hi - 1] * np.exp(-(t - fts[hi - 1]) / taus[k]) - own[i], 0.0)
            if hi < len(fts):
                den[i, k, 1] = (dns[k, hi] + 1.0) * np.exp(-(fts[hi] - t) / taus[k])
    return nobs, den


@nb.njit(parallel=True, cache=True)
def ss_query(q_item, q_t, q_fuid, q_user, ev_off, ev_t, ev_fuid, ev_user, ev_kind, wp, wn, taus, out_cnt, out_k, out_fl):
    npw = len(wp)
    nw = npw + len(wn)
    nk = out_cnt.shape[1]
    for r in nb.prange(len(q_item)):
        it = q_item[r]
        t = q_t[r]
        for k in range(nk):
            for w in range(nw):
                out_cnt[r, k, w] = 0
            for j in range(len(taus)):
                out_k[r, k, j, 0] = 0.0
                out_k[r, k, j, 1] = 0.0
        out_fl[r, 0] = -1; out_fl[r, 1] = -1
        for e in range(ev_off[it], ev_off[it + 1]):
            if ev_fuid[e] == q_fuid[r]:
                continue
            if q_user[r] != 0 and ev_user[e] == q_user[r]:
                continue
            k = ev_kind[e]
            dt = ev_t[e] - t
            if dt <= 0:
                if out_fl[r, 0] < 0:
                    out_fl[r, 0] = ev_t[e]
                for j in range(len(taus)):
                    out_k[r, k, j, 0] += np.exp(dt / taus[j])
                for w in range(npw):
                    if -dt <= wp[w]:
                        out_cnt[r, k, w] += 1
            else:
                if out_fl[r, 1] < 0:
                    out_fl[r, 1] = ev_t[e]
                for j in range(len(taus)):
                    out_k[r, k, j, 1] += np.exp(-dt / taus[j])
                for w in range(len(wn)):
                    if dt <= wn[w]:
                        out_cnt[r, k, npw + w] += 1


def build(query_split, variant, seed=0):
    t0 = time.time()
    fq = load_feeds(query_split)
    q_fts = fq["timestamp"].to_numpy().astype(np.int64)
    if query_split == "train":
        mask = np.random.default_rng(seed).random(len(fq)) < SUB_P
        ev = merge_streams([(0, stream_events("train", mask))])
        code_q, own = 0, mask.astype(np.int64)
    elif variant == "split":   # within-split stream: val rows <- val feeds, test rows <- test feeds (the KEEP variant)
        code_q = 1 if query_split == "val" else 2
        ev = merge_streams([(code_q, stream_events(query_split))])
        own = np.ones(len(fq), np.int64)
    else:                      # 'full': val+test stream (comparison only, val rows)
        ev = merge_streams([(1, stream_events("val")), (2, stream_events("test"))])
        code_q, own = (1 if query_split == "val" else 2), np.ones(len(fq), np.int64)
    print(query_split, variant, "stream events", len(ev["item"]), "feeds", len(ev["feed_ts"]), f"{time.time()-t0:.0f}s", flush=True)
    fts = ev["feed_ts"].astype(np.int64)
    taus = np.array([T * 3600.0 for T in TAUS])
    dps = np.zeros((len(taus), len(fts))); dns = np.zeros((len(taus), len(fts)))
    for k, tau in enumerate(taus):
        dps[k], dns[k] = kernel_den(fts.astype(np.float64), tau)
    wp = np.array(WP, np.int64); wn = np.array(WN, np.int64)
    nobs, den = feed_level(q_fts, own, fts, dps, dns, taus, wp, wn)
    cfi = np.load(FLAT / query_split / "cand_fidx.npy")
    q_item = np.load(FLAT / query_split / "cand_item_id.npy")
    q_t = q_fts[cfi]
    q_fuid = cfi.astype(np.int64) + (code_q << 28)
    q_user = fq["user_id"].to_numpy().astype(np.int64)[cfi]
    nw = len(WP) + len(WN)
    cnt = np.empty((len(q_item), len(KINDS), nw), np.int32)
    kk = np.empty((len(q_item), len(KINDS), len(taus), 2), np.float32)
    fl = np.empty((len(q_item), 2), np.int64)
    ss_query(q_item, q_t, q_fuid, q_user, ev["off"], ev["t"], ev["fuid"], ev["user"], ev["kind"], wp, wn, taus, cnt, kk, fl)
    print("queries done", f"{time.time()-t0:.0f}s", flush=True)
    feats = {}
    wnames = [f"p{w // 3600}" for w in WP] + [f"n{w // 3600}" for w in WN]
    for k, kn in enumerate(KINDS):
        for w, wnm in enumerate(wnames):
            feats[f"{kn}_r{wnm}"] = cnt[:, k, w] / (nobs[cfi, w] + KDEN) * 1e4
        for j, T in enumerate(TAUS):
            feats[f"{kn}_kp{T}"] = kk[:, k, j, 0] / (den[cfi, j, 0] + KDEN) * 1e4
            feats[f"{kn}_kn{T}"] = kk[:, k, j, 1] / (den[cfi, j, 1] + KDEN) * 1e4
    for T in TAUS:
        feats[f"a_kn{T}"] = feats[f"c_kn{T}"] + feats[f"h_kn{T}"]
    feats["fut_obs"] = np.log1p(nobs[cfi, len(WP) + len(WN) - 1])  # observed stream feeds in (t, t+96h]
    # diagnostics (not features): available future, used for the near-end-of-data analysis
    diag = dict(fut_den24=den[cfi, 0, 1], fut_nobs24=nobs[cfi, len(WP) + 1], fut_nobs96=nobs[cfi, len(WP) + len(WN) - 1], q_t=q_t)
    return feats, fl, dict(q_t=q_t), diag


def combined(split, fl, q):
    t = q["q_t"].astype(np.float64)
    n = len(t)
    tau = t if split == "train" else np.full(n, TR_END, np.float64)
    age_tr = np.load(FEATS / split / "life_ts_age_h.npy").astype(np.float64)
    first_tr = tau - age_tr * 3600.0
    f_ss = np.where(fl[:, 0] >= 0, fl[:, 0], np.nan)
    first_any = np.fmin(first_tr, f_ss)
    return {"age_any_h": np.minimum((t - first_any) / 3600.0, 360.0),
            "never_seen": np.isnan(first_any).astype(np.float32)}


def save(split, feats, prefix):
    for n, v in feats.items():
        np.save(FEATS / split / f"{prefix}{n}.npy", np.asarray(v, np.float32))


if __name__ == "__main__":
    nb.set_num_threads(int(__import__("os").environ.get("LIFE_THREADS", "16")))
    splits = sys.argv[1].split(",")
    variant = sys.argv[2] if len(sys.argv) > 2 else "split"
    for s in splits:
        feats, fl, q, diag = build(s, variant)
        save(s, feats, "life_ss_" if variant == "split" else "life_ssF_")
        save(s, combined(s, fl, q), "life_" if variant == "split" else "life_F")
        SCR = Path("/tmp/scratchpad")
        np.savez(SCR / f"ss_diag_{s}_{variant}.npz", **diag)
        print(s, variant, "saved", len(feats), flush=True)
