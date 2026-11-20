"""Within-feed pair structure: are positives of the same feed similar to each other? Do near-duplicates share labels?
All statistics exclude the 117 leak feeds."""
from fs_common import *
import numba as nb, time

EDGES = np.array([-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01], np.float32)
NB = len(EDGES) - 1


def wrank(x, off, M):
    out = np.empty(len(x), np.float32)
    for f in range(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
        out[o:o + m] = r / max(m - 1, 1)
    return out


@nb.njit(parallel=True, cache=True)
def pair_stats(X, miss, off, M, t, cat, item, f1, f3, f4r, f5r, f4, edges, use):
    n = len(M); nb_ = len(edges) - 1
    # stats[f, type(pp,pn,nn), k]: 0 count,1 samecat,2 cos,3 cos>.9,4 |df4r|,5 |df5r|,6 |df1|,7 |df3|,8 sameitem,9 |df4raw|,10 cos>.95, 11 count with both emb
    S = np.zeros((n, 3, 12))
    H = np.zeros((n, 2, 2, nb_, 2))  # anchor label, samecat, cos bin, j label  (ordered pairs i != j)
    for f in nb.prange(n):
        if not use[f]:
            continue
        o = off[f]; m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                ti = t[o + i]; tj = t[o + j]
                sc = 1 if cat[o + i] == cat[o + j] else 0
                ok = (not miss[o + i]) and (not miss[o + j])
                c = C[i, j]
                if ok:
                    b = np.searchsorted(edges, c, side="right") - 1
                    if b < 0: b = 0
                    if b >= nb_: b = nb_ - 1
                    H[f, ti, sc, b, tj] += 1
                if j > i:
                    ty = 0 if (ti == 1 and tj == 1) else (1 if ti + tj == 1 else 2)
                    S[f, ty, 0] += 1
                    S[f, ty, 1] += sc
                    S[f, ty, 4] += abs(f4r[o + i] - f4r[o + j])
                    S[f, ty, 5] += abs(f5r[o + i] - f5r[o + j])
                    S[f, ty, 6] += abs(f1[o + i] - f1[o + j])
                    S[f, ty, 7] += abs(f3[o + i] - f3[o + j])
                    S[f, ty, 8] += 1 if item[o + i] == item[o + j] else 0
                    S[f, ty, 9] += abs(f4[o + i] - f4[o + j])
                    if ok:
                        S[f, ty, 11] += 1
                        S[f, ty, 2] += c
                        S[f, ty, 3] += 1 if c > 0.9 else 0
                        S[f, ty, 10] += 1 if c > 0.95 else 0
    return S, H


def main():
    t0 = time.time()
    d = load_val()
    X, miss = load_emb_norm(d["emb_idx"])
    M, off, t, fid = d["M"], d["off"], d["target"].astype(np.int64), d["fid"]
    use = ~d["leak"]
    npos = np.bincount(fid, t, len(M))
    print(f"loaded {time.time()-t0:.1f}s; feeds used {use.sum()}; multi-pos (>=2) share {np.mean(npos[use]>=2):.4f}")
    # order-leak sanity: expected first/last rate if order random = mean over feeds of npos/M
    idx = np.arange(len(t)) - off[fid]
    print("array-order check: rate at first idx %.4f, last idx %.4f, expected under random order mean(npos/M) %.4f" % (
        t[idx == 0].mean(), t[idx == M[fid] - 1].mean(), (npos / M).mean()))
    f4r = wrank(d["f4"], off, M); f5r = wrank(d["f5"], off, M)
    np.save(CACHE / "f4r.npy", f4r); np.save(CACHE / "f5r.npy", f5r)
    S, H = pair_stats(X, miss, off, M, t, d["category_id"].astype(np.int64), d["item_id"].astype(np.int64),
                      d["f1"].astype(np.float32), d["f3"].astype(np.float32), f4r, f5r, d["f4"].astype(np.float32), EDGES, use)
    np.save(CACHE / "pair_S.npy", S); np.save(CACHE / "pair_H.npy", H)
    print(f"pairs done {time.time()-t0:.1f}s")

    # ---- B. pos-pos vs pos-neg vs neg-neg (per-feed means, feeds with >=2 positives, paired over feeds)
    mp = use & (npos >= 2)
    names = {1: "same category", 2: "mean cos", 3: "P(cos>0.9)", 10: "P(cos>0.95)", 4: "|d f4 rank|", 5: "|d f5 rank|",
             6: "|d f1|", 7: "|d f3|", 8: "same item_id", 9: "|d f4 raw|"}
    res = {}
    print("\nB. feeds with >=2 positives: per-feed mean over pairs; diff pp - pn and pp - nn (paired over feeds)")
    for k, nm in names.items():
        den = 11 if k in (2, 3, 10) else 0
        with np.errstate(invalid="ignore", divide="ignore"):
            v = S[:, :, k] / S[:, :, den]
        ok = mp & np.isfinite(v).all(1)
        pp, pn, nn_ = v[ok, 0], v[ok, 1], v[ok, 2]
        d1 = pp - pn; d2 = pp - nn_
        res[nm] = dict(pp=pp.mean(), pn=pn.mean(), nn=nn_.mean(), d_pp_pn=d1.mean(), se1=d1.std(ddof=1) / np.sqrt(ok.sum()),
                       d_pp_nn=d2.mean(), se2=d2.std(ddof=1) / np.sqrt(ok.sum()), nfeeds=int(ok.sum()))
        r = res[nm]
        print(f"  {nm:16s} pp {r['pp']:.4f} pn {r['pn']:.4f} nn {r['nn']:.4f} | pp-pn {r['d_pp_pn']:+.4f} ± {r['se1']:.4f} | pp-nn {r['d_pp_nn']:+.4f} ± {r['se2']:.4f} (n={r['nfeeds']})")
    # same-category for pp restricted to feeds where category diversity exists
    # ---- C. conditional label lift  P(t_j=1 | anchor label, cos bin, same cat)
    Hs = H[use].sum(0)  # (2,2,NB,2)
    print("\nC. P(t_j = 1 | t_i, cos(i,j), same-category) over ordered pairs i != j (non-leak feeds)")
    print("   base P(t_j=1 | t_i=1) = %.4f ; P(t_j=1 | t_i=0) = %.4f" % (
        Hs[1, :, :, 1].sum() / Hs[1].sum(), Hs[0, :, :, 1].sum() / Hs[0].sum()))
    rows = []
    for sc in (0, 1):
        print(f"  same category = {sc}")
        for b in range(NB):
            n1 = Hs[1, sc, b].sum(); n0 = Hs[0, sc, b].sum()
            p1 = Hs[1, sc, b, 1] / max(n1, 1); p0 = Hs[0, sc, b, 1] / max(n0, 1)
            rows.append(dict(sc=sc, lo=float(EDGES[b]), hi=float(EDGES[b + 1]), n_i_pos=int(n1), p_given_pos=p1, n_i_neg=int(n0), p_given_neg=p0))
            print(f"    cos [{EDGES[b]:.3f},{EDGES[b+1]:.3f}) anchor+ n={int(n1):8d} P={p1:.4f} | anchor- n={int(n0):10d} P={p0:.4f} | ratio {p1/max(p0,1e-9):.2f}")
    json.dump(dict(B=res, C=rows), open(OUT / "out_01_pairs.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
