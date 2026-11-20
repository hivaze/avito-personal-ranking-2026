"""trd_fp_: item-attribute profiles (f1..f5 looked up for HISTORY items) — population-now and user-level.

f4/f5 are near-constant per item (within-item sd 0.011/0.022 vs 0.27/0.31 overall) -> item attributes (price-like);
f1..f3 vary per impression but ~2/3 of their variance is between items. We build an item lookup
  A_k(item) = mean of f_k over all candidate occurrences of the item in train+val+test (unlabeled features only)
and attach it to history events. Coverage: ~71% (val) / 65% (test) of hpos<20 history events.
Features per candidate (category c, observed f_k):
  trd_fp_pz{k}_<w>   (f_k - mu) / sd, mu/sd = attribute mean/std of POPULATION recent engaged items (pool events hpos<10,
                      other feeds) of category c in window w (shrunk to the long-run category moments with 30 pseudo-events)
  trd_fp_ud{k}       f_k - mean A_k over the user's own history items of category c (NaN if none)
  trd_fp_uad{k}      |trd_fp_ud{k}| / user's within-category sd (shrunk)
  trd_fp_ua{k}       f_k - mean A_k over all of the user's history items (any category)
  trd_fp_un          # of the user's history items in c with a lookup (support)
"""
import sys, time
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

NI = 111_400_000
K5 = 5
WINDOWS = {"s24h": (43200, 43200), "static": None}
PSEUDO = 30.0


def item_attr():
    p = CACHE / "item_attr.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    cnt = np.zeros(NI, np.float64)
    sums = np.zeros((K5, NI), np.float64)
    for s in ["train", "val", "test"]:
        it = np.load(FLAT / s / "cand_item_id.npy")
        cnt += np.bincount(it, minlength=NI)
        for k in range(K5):
            sums[k] += np.bincount(it, weights=np.load(FLAT / s / f"cand_f{k+1}.npy"), minlength=NI)
        print("attr", s, flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (sums / cnt[None]).astype(np.float32)   # NaN where never a candidate
    A = np.ascontiguousarray(A.T)                    # (NI, 5)
    np.save(p, A)
    return A


@nb.njit(parallel=True, cache=True)
def user_profile(cand_off, M, hist_off, H, ccat, cf, hcat, hattr, out):
    """out cols: 0..4 ud_k, 5..9 uad_k, 10..14 ua_k, 15 un"""
    n = len(M)
    for f in nb.prange(n):
        s = np.zeros((54, 5)); s2 = np.zeros((54, 5)); c = np.zeros(54)
        sa = np.zeros(5); ca = 0.0
        ho = hist_off[f]
        for j in range(H[f]):
            a0 = hattr[ho + j, 0]
            if np.isnan(a0):
                continue
            k = hcat[ho + j]
            c[k] += 1; ca += 1
            for q in range(5):
                v = hattr[ho + j, q]
                s[k, q] += v; s2[k, q] += v * v; sa[q] += v
        co = cand_off[f]
        for i in range(M[f]):
            k = ccat[co + i]
            for q in range(5):
                x = cf[co + i, q]
                if c[k] > 0:
                    mu = s[k, q] / c[k]
                    var = s2[k, q] / c[k] - mu * mu
                    sd = np.sqrt((max(var, 0.0) * c[k] + 0.04 * 3) / (c[k] + 3))   # shrink sd to 0.2
                    out[co + i, q] = x - mu
                    out[co + i, 5 + q] = abs(x - mu) / sd
                else:
                    out[co + i, q] = np.nan; out[co + i, 5 + q] = np.nan
                out[co + i, 10 + q] = x - sa[q] / ca if ca > 0 else np.nan
            out[co + i, 15] = c[k]


@nb.njit(parallel=True, cache=True)
def pool_moments(sel, hist_off, H, hcat, hattr, hpos, kmax, out):
    """out[i, c, 0]=count, [1..5]=sum A_k, [6..10]=sum A_k^2 over events hpos<kmax of pool feed sel[i]."""
    for i in nb.prange(len(sel)):
        f = sel[i]; ho = hist_off[f]
        for j in range(H[f]):
            if hpos[ho + j] >= kmax:
                continue
            a0 = hattr[ho + j, 0]
            if np.isnan(a0):
                continue
            k = hcat[ho + j]
            out[i, k, 0] += 1
            for q in range(5):
                v = hattr[ho + j, q]
                out[i, k, 1 + q] += v; out[i, k, 6 + q] += v * v


def build(split, threads=16):
    nb.set_num_threads(threads)
    t0 = time.time()
    A = item_attr()
    f = feeds_df(split); n = len(f)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy(); ho = f["hist_off"].to_numpy()
    hi = np.load(FLAT / split / "hist_item_id.npy")
    hattr = np.asarray(A[hi]); del hi
    hcat = np.load(FLAT / split / "hist_category_id.npy").astype(np.int64)
    hpos = np.load(FLAT / split / "hist_hpos.npy")
    ccat = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
    cf = np.stack([np.load(FLAT / split / f"cand_f{k+1}.npy") for k in range(K5)], 1).astype(np.float64)
    print(split, "loaded", f"{time.time()-t0:.0f}s", flush=True)
    d = FEATS / split
    out = np.empty((len(ccat), 16), np.float32)
    user_profile(co, M, ho, H, ccat, cf, hcat, hattr, out)
    for q in range(K5):
        np.save(d / f"trd_fp_ud{q+1}.npy", out[:, q]); np.save(d / f"trd_fp_uad{q+1}.npy", out[:, 5 + q])
        np.save(d / f"trd_fp_ua{q+1}.npy", out[:, 10 + q])
    np.save(d / "trd_fp_un.npy", out[:, 15])
    del out
    print(split, "user", f"{time.time()-t0:.0f}s", flush=True)
    # population moments
    pm = pool_mask(split, f); psel = np.flatnonzero(pm)
    mom = np.zeros((len(psel), 54, 11), np.float64)
    pool_moments(psel.astype(np.int64), ho, H, hcat, hattr, hpos, 10, mom)
    # long-run per-category moments (val/test: pooled val+test pools; train: own pool)
    if split == "train":
        lr = mom.sum(0)
    else:
        lr = 0
        for s2 in ["val", "test"]:
            if s2 == split:
                lr = lr + mom.sum(0); continue
            f2 = feeds_df(s2)
            hi2 = np.load(FLAT / s2 / "hist_item_id.npy")
            m2 = np.zeros((len(f2), 54, 11), np.float64)
            pool_moments(np.arange(len(f2), dtype=np.int64), f2["hist_off"].to_numpy(), f2["H"].to_numpy().astype(np.int64),
                         np.load(FLAT / s2 / "hist_category_id.npy").astype(np.int64), np.asarray(A[hi2]), np.load(FLAT / s2 / "hist_hpos.npy"), 10, m2)
            lr = lr + m2.sum(0)
    lr_mu = lr[:, 1:6] / np.maximum(lr[:, :1], 1); lr_var = lr[:, 6:11] / np.maximum(lr[:, :1], 1) - lr_mu ** 2
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    cum = np.concatenate([np.zeros((1, 54, 11)), np.cumsum(mom, 0)])
    pos_in_pool = np.full(n, -1, np.int64); pos_in_pool[psel] = np.arange(len(psel))
    for w, wv in WINDOWS.items():
        if wv is None:
            mu = lr_mu[ccat]; sd = np.sqrt(np.maximum(lr_var[ccat], 1e-4))
        else:
            lo, hi_ = window_bounds(ts[psel], ts, wv[0], wv[1])
            win = cum[hi_] - cum[lo]
            own = pos_in_pool >= 0
            win[own] -= mom[pos_in_pool[own]]
            wc = win[fidx, ccat]                                           # (Ncand, 11)
            cnt = wc[:, :1]
            mu = (wc[:, 1:6] + PSEUDO * lr_mu[ccat]) / (cnt + PSEUDO)
            ex2 = (wc[:, 6:11] + PSEUDO * (lr_var[ccat] + lr_mu[ccat] ** 2)) / (cnt + PSEUDO)
            sd = np.sqrt(np.maximum(ex2 - mu ** 2, 1e-4))
        z = (cf - mu) / sd
        for q in range(K5):
            np.save(d / f"trd_fp_pz{q+1}_{w}.npy", z[:, q].astype(np.float32))
        print(split, w, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        build(s, int(sys.argv[2]) if len(sys.argv) > 2 else 16)
