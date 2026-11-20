"""trd_ category-level population trend features (all splits).

For each candidate row (feed f at time t, candidate category c):
  pool events  = history events with hpos < K (event-type group g) of POOL feeds (see trd_lib.pool_mask), own feed excluded
  now-share    s_c(t) = (n_c(window) + A * p_c) / (N(window) + A)        (Bayesian shrinkage to the long-run share p_c)
  trd_cl_<g>_<w>   = log(s_c(t) / p_c)                                    category trend lift
  trd_ctod_<g>     = log(q_c(hour, daytype) / p_c)                        seasonal (time-of-day x weekday/weekend) lift from dense train
  trd_cres_<g>_<w> = log(s_c(t) / q_c(hour, daytype))                     "today's anomaly" vs the seasonal expectation
  trd_sup_<w>      = log(candidate-supply share of c in window / long-run supply share)
  trd_urel_<g>_<w> = log((O + .5) / (E + .5)), O = # of the user's last-20 events in c, E = 20' * s_c(t)   (user over-indexing vs population now)
Long-run baselines p_c: val/test -> pooled val+test pool events (exploration population, a constant vector);
train -> train pool long-run share. Windows: p<W> = past-only [t-W, t]; s<W> = symmetric [t-W/2, t+W/2]; pday/sday = calendar-day aligned (see WINDOWS).
"""
import sys, time
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

GROUPS = {"k10": (10, 1, 10), "k10e1": (10, 1, 1), "k10act": (10, 2, 10), "kall": (256, 1, 10), "k3": (3, 1, 10)}
WINDOWS = {"p3h": (3 * 3600, 0), "s6h": (3 * 3600, 3 * 3600), "pday": "pday", "sday": "sday"}
# pday = same MSK calendar day up to t ("today so far"); sday = whole same MSK day. Calendar-aligned windows have the
# same structure in val (isolated days) and test (contiguous days), unlike 24h windows that cross midnight.
A = 300.0


def hist_counts(split, feeds, sel):
    H = feeds["H"].to_numpy().astype(np.int64); ho = feeds["hist_off"].to_numpy()
    hc = np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r")
    he = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")
    g = list(GROUPS.values())
    km = np.array([x[0] for x in g], np.int64); lo = np.array([x[1] for x in g], np.int64); hi = np.array([x[2] for x in g], np.int64)
    out = np.zeros((len(sel), len(g), N_CAT), np.float32)
    feed_cat_counts(sel.astype(np.int64), ho, H, hc, he, hp, km, lo, hi, out)
    return out


def cand_counts(split, feeds, sel):
    M = feeds["M"].to_numpy().astype(np.int64); co = feeds["cand_off"].to_numpy()
    cc = np.load(FLAT / split / "cand_category_id.npy", mmap_mode="r")
    out = np.zeros((len(sel), N_CAT), np.float32)
    feed_cand_cat_counts(sel.astype(np.int64), co, M, cc, out)
    return out


def tod_profile():
    """q[daytype(0 weekday,1 weekend), hour, group, cat] from dense train (all feeds, recent history events)."""
    p = CACHE / "tod_profile.npy"
    if p.exists():
        return np.load(p)
    f = feeds_df("train")
    sel = np.arange(len(f))
    cnt = hist_counts("train", f, sel)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    hour = (ts + MSK) // 3600 % 24
    dow = ((ts + MSK) // 86400 + 3) % 7  # 0 = Monday (1970-01-01 was a Thursday)
    wk = (dow >= 5).astype(np.int64)
    q = np.zeros((2, 24, cnt.shape[1], N_CAT), np.float64)
    np.add.at(q, (wk, hour), cnt.astype(np.float64))
    np.save(p, q)
    return q


def base_shares(split):
    """long-run shares (group x cat) for history and candidate supply."""
    if split == "train":
        f = feeds_df("train"); pm = pool_mask("train", f); sel = np.flatnonzero(pm)
        h = hist_counts("train", f, sel).sum(0); c = cand_counts("train", f, sel).sum(0)
    else:
        h = 0; c = 0
        for s in ["val", "test"]:
            f = feeds_df(s); sel = np.arange(len(f))
            h = h + hist_counts(s, f, sel).sum(0); c = c + cand_counts(s, f, sel).sum(0)
    h = (h + 1.0) / (h + 1.0).sum(1, keepdims=True)
    c = (c + 1.0) / (c + 1.0).sum()
    return h, c


def build(split, threads=16):
    nb.set_num_threads(threads)
    t0 = time.time()
    out_dir = FEATS / split
    f = feeds_df(split)
    n = len(f)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    uid = f["user_id"].to_numpy()
    pm = pool_mask(split, f)
    psel = np.flatnonzero(pm)
    pts = ts[psel]
    hcnt = hist_counts(split, f, psel)          # (nP, G, C)
    ccnt = cand_counts(split, f, psel)          # (nP, C)
    pos_in_pool = np.full(n, -1, np.int64); pos_in_pool[psel] = np.arange(len(psel))
    hcum = np.concatenate([np.zeros((1,) + hcnt.shape[1:]), np.cumsum(hcnt, 0, dtype=np.float64)])
    ccum = np.concatenate([np.zeros((1, N_CAT)), np.cumsum(ccnt, 0, dtype=np.float64)])
    pbase, cbase = base_shares(split)
    q = tod_profile()
    q = (q + 1.0) / (q + 1.0).sum(-1, keepdims=True)           # (2, 24, G, C) shares
    hour = (ts + MSK) // 3600 % 24
    wk = ((((ts + MSK) // 86400 + 3) % 7) >= 5).astype(np.int64)
    qf = q[wk, hour]                                              # (n, G, C)
    print(split, "counts", f"{time.time()-t0:.0f}s", "pool", len(psel), flush=True)

    fidx = np.load(FLAT / split / "cand_fidx.npy")
    cc = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
    # user's last-20 category counts (for urel)
    H = f["H"].to_numpy().astype(np.int64)
    ucnt = np.zeros((n, 1, N_CAT), np.float32)
    hc = np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r"); he = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    hp = np.load(FLAT / split / "hist_hpos.npy", mmap_mode="r")
    feed_cat_counts(np.arange(n, dtype=np.int64), f["hist_off"].to_numpy(), H, hc, he, hp,
                    np.array([20], np.int64), np.array([1], np.int64), np.array([10], np.int64), ucnt)
    ucnt = ucnt[:, 0, :]
    nu = ucnt.sum(1)
    gn = list(GROUPS)

    def save(name, arr):
        np.save(out_dir / f"trd_{name}.npy", np.asarray(arr, dtype=np.float32))

    for gi, g in enumerate(gn):
        save(f"ctod_{g}", np.log(qf[fidx, gi, cc] / pbase[gi, cc]))
    for w, wv in WINDOWS.items():
        lo, hi = day_or_window_bounds(pts, ts, wv)
        own = pos_in_pool >= 0
        win = hcum[hi] - hcum[lo]                                  # (n, G, C)
        win[own] -= hcnt[pos_in_pool[own]]
        N = win.sum(2)                                             # (n, G)
        for gi, g in enumerate(gn):
            sh = (win[:, gi, :] + A * pbase[gi][None]) / (N[:, gi, None] + A)
            save(f"cl_{g}_{w}", np.log(sh / pbase[gi][None])[fidx, cc])
            if g in ("k10", "k10e1"):
                save(f"cres_{g}_{w}", np.log(sh / qf[:, gi, :])[fidx, cc])
            if g == "k10":
                E = nu[:, None] * sh
                save(f"urel_{g}_{w}", np.log((ucnt + 0.5) / (E + 0.5))[fidx, cc])
        cw = ccum[hi] - ccum[lo]
        cw[own] -= ccnt[pos_in_pool[own]]
        cN = cw.sum(1)
        AS = 3000.0
        csh = (cw + AS * cbase[None]) / (cN[:, None] + AS)
        save(f"sup_{w}", np.log(csh / cbase[None])[fidx, cc])
        print(split, w, f"{time.time()-t0:.0f}s", flush=True)
    # static population-normalized user affinity (baseline instead of now-share)
    E = nu[:, None] * pbase[0][None]
    save("urel_k10_static", np.log((ucnt + 0.5) / (E + 0.5))[fidx, cc])
    print(split, "done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        build(s, int(sys.argv[2]) if len(sys.argv) > 2 else 16)
