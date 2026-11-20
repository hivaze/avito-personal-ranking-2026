"""B3 block (b): TRANSITION click rates and user-cluster x item-cluster click rates (hand-made, leave-user-out, time-aware, holdout-free).

Estimation rows: every candidate row of the 3,121,208 train feeds that are NOT holdout feeds. Each feed has a user fold k (cbx.xfold_of: 5 folds
by user hash, unknown users by feed index) and a day index d (0..14). Tables are accumulated per (fold, day) and a target feed with fold k on day d
reads  sum over folds != k and days < d : so a statistic never uses the row's own feed, never a feed of the same user (leave-user-FOLD-out),
never a same-day or later feed. val / test feeds read all 15 days of the 4 other folds (their users' own train feeds are in the left-out fold).
  transition (space s in cat / c256 / c1024): P, E [history key h, candidate key c] for the first 1 / 5 / 20 history events (newest first) that
     have the key; P = clicks, E = expected clicks (position prior of block a). coec(h, c) = (P + 5 R_c) / (E + 5), R_c = candidate-key base COEC.
     columns tr_<s>_l1, _l5m, _l5p, _l20m, _l20p, _l20max, _base  (m = mean log lift over the events, p = pooled log lift, base = log R_c)
  user cluster x item cluster: ux_u256c1024, ux_u1024c1024, ux_u64c4096 = log lift of the pair COEC over R_c;  ux_c4096_base
Files: feat/b_<set>.npy, feat/b_names.json.   usage: python -u b3_build_b.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
import numba as nb

A_TR, A_BASE, L = 5.0, 20.0, 20
DAY0, ND, NFOLD = 20702, 15, 5


@nb.njit(parallel=True, cache=True)
def est_trans(part_ptr, part_feeds, coff, M, hoff, H, ckey, hkey, y, ex, T, B):
    for p in nb.prange(len(part_ptr) - 1):
        hs = np.empty(L, np.int64)
        for a in range(part_ptr[p], part_ptr[p + 1]):
            f = part_feeds[a]; n = 0; o = hoff[f]
            for j in range(H[f]):
                h = hkey[o + j]
                if h >= 0:
                    hs[n] = h; n += 1
                    if n == L:
                        break
            q = coff[f]; m = M[f]
            for i in range(m):
                c = ckey[q + i]
                if c >= 0:
                    B[p, c, 0] += y[q + i]; B[p, c, 1] += ex[q + i]
            for j in range(n):
                h = hs[j]
                for i in range(m):
                    c = ckey[q + i]
                    if c < 0:
                        continue
                    yy = y[q + i]; e = ex[q + i]
                    T[p, 2, h, c, 0] += yy; T[p, 2, h, c, 1] += e
                    if j < 5:
                        T[p, 1, h, c, 0] += yy; T[p, 1, h, c, 1] += e
                        if j < 1:
                            T[p, 0, h, c, 0] += yy; T[p, 0, h, c, 1] += e


@nb.njit(parallel=True, cache=True)
def est_pair(part_ptr, part_feeds, coff, M, ukey, ckey, y, ex, U):
    for p in nb.prange(len(part_ptr) - 1):
        for a in range(part_ptr[p], part_ptr[p + 1]):
            f = part_feeds[a]; u = ukey[f]
            if u < 0:
                continue
            q = coff[f]
            for i in range(M[f]):
                c = ckey[q + i]
                if c >= 0:
                    U[p, u, c, 0] += y[q + i]; U[p, u, c, 1] += ex[q + i]


@nb.njit(parallel=True, cache=True)
def qry_trans(coff, M, hoff, H, fold, cut, ckey, hkey, C, TOT, CB, TOTB, r0, out):
    """C [fold, cut, lag, Kh, Kc, 2] cumulative over days < cut per fold; TOT = sum over folds. out [n_rows, 7]."""
    for f in nb.prange(len(M)):
        k = fold[f]; d = cut[f]; q = coff[f]; m = M[f]
        hs = np.empty(L, np.int64); n = 0; o = hoff[f]
        for j in range(H[f]):
            h = hkey[o + j]
            if h >= 0:
                hs[n] = h; n += 1
                if n == L:
                    break
        for i in range(m):
            c = ckey[q + i]
            if c < 0 or d == 0:
                for z in range(7):
                    out[q + i, z] = np.nan
                continue
            bp = TOTB[d, c, 0] - CB[k, d, c, 0]; be = TOTB[d, c, 1] - CB[k, d, c, 1]
            R = (bp + A_BASE * r0) / (be + A_BASE)
            out[q + i, 6] = np.log(R)
            if n == 0:
                for z in range(6):
                    out[q + i, z] = np.nan
                continue
            s5 = 0.0; p5 = 0.0; e5 = 0.0; s20 = 0.0; p20 = 0.0; e20 = 0.0; mx = -1e9
            for j in range(n):
                h = hs[j]
                P = TOT[d, 2, h, c, 0] - C[k, d, 2, h, c, 0]; E = TOT[d, 2, h, c, 1] - C[k, d, 2, h, c, 1]
                ll = np.log((P + A_TR * R) / (E + A_TR) / R)
                s20 += ll; p20 += P; e20 += E
                if ll > mx:
                    mx = ll
                if j < 5:
                    P = TOT[d, 1, h, c, 0] - C[k, d, 1, h, c, 0]; E = TOT[d, 1, h, c, 1] - C[k, d, 1, h, c, 1]
                    s5 += np.log((P + A_TR * R) / (E + A_TR) / R); p5 += P; e5 += E
                    if j < 1:
                        P = TOT[d, 0, h, c, 0] - C[k, d, 0, h, c, 0]; E = TOT[d, 0, h, c, 1] - C[k, d, 0, h, c, 1]
                        out[q + i, 0] = np.log((P + A_TR * R) / (E + A_TR) / R)
            n5 = min(n, 5)
            out[q + i, 1] = s5 / n5; out[q + i, 2] = np.log((p5 + A_TR * R) / (e5 + A_TR) / R)
            out[q + i, 3] = s20 / n; out[q + i, 4] = np.log((p20 + A_TR * R) / (e20 + A_TR) / R); out[q + i, 5] = mx


@nb.njit(parallel=True, cache=True)
def qry_pair(coff, M, fold, cut, ukey, ckey, C, TOT, CB, TOTB, r0, out, col, basecol):
    for f in nb.prange(len(M)):
        k = fold[f]; d = cut[f]; q = coff[f]; u = ukey[f]
        for i in range(M[f]):
            c = ckey[q + i]
            if c < 0 or d == 0:
                out[q + i, col] = np.nan
                if basecol >= 0:
                    out[q + i, basecol] = np.nan
                continue
            bp = TOTB[d, c, 0] - CB[k, d, c, 0]; be = TOTB[d, c, 1] - CB[k, d, c, 1]
            R = (bp + A_BASE * r0) / (be + A_BASE)
            if basecol >= 0:
                out[q + i, basecol] = np.log(R)
            if u < 0:
                out[q + i, col] = np.nan
                continue
            P = TOT[d, u, c, 0] - C[k, d, u, c, 0]; E = TOT[d, u, c, 1] - C[k, d, u, c, 1]
            out[q + i, col] = np.log((P + A_TR * R) / (E + A_TR) / R)


def cumulate(T):
    """T [75, ...] per (fold, day) -> C [5, 16, ...] = sum over days < cut, TOT [16, ...] = sum over folds."""
    T = T.reshape((NFOLD, ND) + T.shape[1:])
    C = np.zeros((NFOLD, ND + 1) + T.shape[2:], np.float32)
    np.cumsum(T, axis=1, out=C[:, 1:])
    return C, C.sum(0)


def main():
    t0 = time.time()
    f = cbx.feeds("train"); uid, ts, Mtr, offtr, hofftr, Htr = f["uid"], f["ts"], f["M"], f["off"], f["hoff"], f["H"]
    S = cbx.train_sets(); n_feeds = len(Mtr)
    hold = np.zeros(n_feeds, bool); hold[S["hold"]] = True
    day = ((ts + 3 * 3600) // 86400 - DAY0).astype(np.int64); assert day.min() == 0 and day.max() == ND - 1
    fold = cbx.xfold_of(uid, np.arange(n_feeds))[0].astype(np.int64)
    part = fold * ND + day
    est = np.where(~hold)[0]
    o = np.argsort(part[est], kind="stable"); part_feeds = est[o].astype(np.int64)
    part_ptr = np.searchsorted(part[est][o], np.arange(NFOLD * ND + 1)).astype(np.int64)
    pc = np.load(FEAT / "pos_prior.npy")
    y = np.load(cbx.FLAT / "train/cand_target.npy").astype(np.float32)
    ex = pc[np.load(cbx.FLAT / "train/cand_pos.npy").astype(np.int64)].astype(np.float32)
    nh = np.repeat(~hold, Mtr); r0 = float(y[nh].sum() / ex[nh].sum()); del nh
    log(f"estimation feeds {len(est)} (holdout excluded), global COEC {r0:.4f} ({time.time() - t0:.0f}s)")
    sets = {n: cbx.load(n) for n in ("xm", "hold", "val", "test")}
    meta = {}
    for name, d in sets.items():
        ff = cbx.feeds(d.split)
        k = cbx.xfold_of(d.uid_feed, d.fidx)[0].astype(np.int64)
        cut = ((d.ts_feed + 3 * 3600) // 86400 - DAY0).astype(np.int64) if d.split == "train" else np.full(d.n_feeds, ND, np.int64)
        if d.split == "train":
            assert (k == fold[d.fidx]).all() and (cut == day[d.fidx]).all()
            if LAG:
                cut = np.maximum(cut - lag_days(d.fidx) + 1, 0)       # reads days <= own day - L (cut 0 = nothing to read -> NaN)
        meta[name] = dict(fold=k, cut=cut, hoff=ff["hoff"][d.fidx].astype(np.int64), H=ff["H"][d.fidx].astype(np.int64))
    names = []
    out = {n: [] for n in sets}
    ckeys_tr = {"cat": np.load(cbx.FLAT / "train/cand_category_id.npy").astype(np.int16)}
    for K in (256, 1024, 4096):
        ckeys_tr[f"c{K}"] = np.load(CLU / f"cand_cl_{K}_train.npy")
    ckey_set = lambda d, s: (np.asarray(d.cat).astype(np.int16) if s == "cat" else clu(d, int(s[1:])).astype(np.int16))
    base = {}
    for s, Kk in (("cat", 64), ("c256", 256), ("c1024", 1024)):
        t1 = time.time()
        hk_tr = (np.load(cbx.FLAT / "train/hist_category_id.npy").astype(np.int16) if s == "cat" else np.load(CLU / f"hist_cl_{Kk}_train.npy"))
        T = np.zeros((NFOLD * ND, 3, Kk, Kk, 2), np.float32); B = np.zeros((NFOLD * ND, Kk, 2), np.float32)
        est_trans(part_ptr, part_feeds, offtr, Mtr, hofftr, Htr, ckeys_tr[s], hk_tr, y, ex, T, B)
        log(f"[{s}] estimated ({time.time() - t1:.0f}s): clicks in base {B[..., 0].sum():.0f}, lag-20 pair mass {T[:, 2, :, :, 1].sum():.0f}")
        C, TOT = cumulate(T); del T
        CB, TOTB = cumulate(B); base[s] = (CB, TOTB)
        for name, d in sets.items():
            hk = hk_tr if d.split == "train" else (np.load(cbx.FLAT / f"{d.split}/hist_category_id.npy").astype(np.int16) if s == "cat"
                                                    else np.load(CLU / f"hist_cl_{Kk}_{d.split}.npy"))
            m = meta[name]; o_ = np.empty((d.n_rows, 7), np.float32)
            qry_trans(d.off.astype(np.int64), d.M.astype(np.int64), m["hoff"], m["H"], m["fold"], m["cut"], ckey_set(d, s), hk, C, TOT, CB, TOTB, r0, o_)
            out[name].append(o_)
        names += [f"tr_{s}_{x}" for x in ("l1", "l5m", "l5p", "l20m", "l20p", "l20max", "base")]
        del C, TOT, hk_tr
        log(f"[{s}] queried ({time.time() - t1:.0f}s) rss {cbx.rss_gb()[0]:.1f} GB")
    # c4096 base
    B = np.zeros((NFOLD * ND, 4096, 2), np.float32)
    est_pair(part_ptr, part_feeds, offtr, Mtr, np.zeros(n_feeds, np.int16), ckeys_tr["c4096"], y, ex, B.reshape(NFOLD * ND, 1, 4096, 2))
    base["c4096"] = cumulate(B)
    for j, (Ku, s) in enumerate(((256, "c1024"), (1024, "c1024"), (64, "c4096"))):
        t1 = time.time(); Kc = int(s[1:])
        uk_tr = np.load(CLU / f"user_cl_{Ku}_train.npy")
        U = np.zeros((NFOLD * ND, Ku, Kc, 2), np.float32)
        est_pair(part_ptr, part_feeds, offtr, Mtr, uk_tr, ckeys_tr[s], y, ex, U)
        C, TOT = cumulate(U); del U
        CB, TOTB = base[s]
        for name, d in sets.items():
            uk = uk_tr[d.fidx] if d.split == "train" else np.load(CLU / f"user_cl_{Ku}_{d.split}.npy")
            m = meta[name]; o_ = np.empty((d.n_rows, 2), np.float32)
            qry_pair(d.off.astype(np.int64), d.M.astype(np.int64), m["fold"], m["cut"], np.ascontiguousarray(uk), ckey_set(d, s), C, TOT, CB, TOTB, r0, o_, 0, 1 if j == 2 else -1)
            out[name].append(o_ if j == 2 else o_[:, :1])
        names += [f"ux_u{Ku}{s}"] + (["ux_c4096_base"] if j == 2 else [])
        log(f"[ux u{Ku} x {s}] done ({time.time() - t1:.0f}s)")
    json.dump(names, open(FEAT / "b_names.json", "w"))
    for name in sets:
        a = np.concatenate(out[name], 1); assert a.shape[1] == len(names)
        if LAG and sets[name].split != "train":
            continue
        np.save(FEAT / f"{PRE}b_{name}.npy", a)
        log(f"[{name}] written {a.shape}; NaN share {np.isnan(a).mean():.4f}; cut-0 feeds {(meta[name]['cut'] == 0).mean():.4f}")
    log(f"DONE block b ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
