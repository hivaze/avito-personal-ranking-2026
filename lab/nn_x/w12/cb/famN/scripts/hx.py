"""N1x: the embedding calcers of CatBoost re-done BY HAND as numeric columns (the work-around of docs_feat 3.1), leak-clean by construction.
usage: CUDA_VISIBLE_DEVICES="" python -u hx.py [--nfit 30000 --nsel 5000] [--what lda knn] [--kinds pair ehm ue] [--test]

Columns (float32, written to cache/hx/n<K>k/<col>_<set>.npy, aligned to the rows of the set; NaN on xm rows outside the fit / selection feeds):
  hlda_<kind>     Fisher direction w = (Sw / n + 1e-4 I)^-1 (mu1 - mu0) of the click label on the full-dimension vector (pair 128, ehm 128, ue 256).
                  fit rows of user fold k: direction fitted on the fit rows of the OTHER user folds (cbx.xfold_of, the teacher's 5 user folds);
                  selection / val / holdout / test rows: direction fitted on ALL fit rows (none of those feeds or users is in the fit rows).
  hknn50_<kind>,  mean OUT-OF-FOLD teacher logit (centred per feed) of the 50 / 200 nearest fit rows in the 32-d PCA space of the vector (L2),
  hknn200_<kind>  neighbours taken from the fit rows of the OTHER user folds for a fit row, from all fit rows for every other row.
                  faiss IndexIVFFlat (nlist 2048, nprobe 24), fold exclusion by an id-range selector (the cloud is stored sorted by fold).
Leak statement: labels and teacher values come from train fit feeds only (never holdout / val / test / selection feeds); a fit row never sees its own
feed or any feed of its own user (leave-user-fold-out), so no later snapshot of the user either; teacher values are the cross-fitted teacher_oof.
PCA is label-free."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn  # noqa: E402
from fn import cbx, np, log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nfit", type=int, default=30000); ap.add_argument("--nsel", type=int, default=5000)
ap.add_argument("--what", nargs="+", default=["lda", "knn"]); ap.add_argument("--kinds", nargs="+", default=["pair", "ehm", "ue"])
ap.add_argument("--test", action="store_true"); ap.add_argument("--nlist", type=int, default=2048); ap.add_argument("--nprobe", type=int, default=24)
a = ap.parse_args()
tr = cbx.load("xm"); fit, sel = fn.subset(a.nfit or None, a.nsel or None); nk = int(fit.sum()) // 1000
OUT = fn.CACHE / "hx" / f"n{nk}k"; OUT.mkdir(parents=True, exist_ok=True)
bf, bs = fn.Block(tr, fit), fn.Block(tr, sel)
others = [("val", fn.Block(cbx.load("val"))), ("hold", fn.Block(cbx.load("hold")))] + ([("test", fn.Block(cbx.load("test")))] if a.test else [])
fold_row = tr.xfold[bf.frow].astype(np.int64)                 # user fold of every fit row
y = bf.label("y").astype(np.float64); t = bf.label("oof").astype(np.float32)
log(f"fit rows {bf.n} sel rows {bs.n}; fold sizes {np.bincount(fold_row, minlength=5).tolist()}; positives {int(y.sum())}")


def save_xm(col, v_fit, v_sel):
    o = np.full(tr.n_rows, np.nan, np.float32); o[bf.idx] = v_fit; o[bs.idx] = v_sel
    np.save(OUT / f"{col}_xm.npy", o)


def chunks(n, B=400_000):
    for s in range(0, n, B):
        yield slice(s, min(n, s + B))


for kind in a.kinds:
    if "lda" in a.what:
        t0 = time.time()
        V = bf.vec(kind); d = V.shape[1]
        st = []
        for k in range(5):
            m = fold_row == k; Vk = V[m].astype(np.float64); yk = y[m]
            st.append(dict(n1=float(yk.sum()), n0=float((1 - yk).sum()), s1=Vk[yk > 0].sum(0), s0=Vk[yk == 0].sum(0), S=Vk.T @ Vk))

        def direction(folds):
            n1 = sum(st[k]["n1"] for k in folds); n0 = sum(st[k]["n0"] for k in folds)
            s1 = sum(st[k]["s1"] for k in folds); s0 = sum(st[k]["s0"] for k in folds); S = sum(st[k]["S"] for k in folds)
            mu1, mu0 = s1 / n1, s0 / n0
            Sw = (S - n1 * np.outer(mu1, mu1) - n0 * np.outer(mu0, mu0)) / (n1 + n0)
            return np.linalg.solve(Sw + 1e-4 * np.eye(d), mu1 - mu0).astype(np.float32)
        w_all = direction(range(5)); v_fit = np.empty(bf.n, np.float32)
        for k in range(5):
            wk = direction([j for j in range(5) if j != k]); m = fold_row == k
            v_fit[m] = V[m] @ wk
            log(f"  hlda_{kind} fold {k}: rows {int(m.sum())}, cos(w_k, w_all) {float(wk @ w_all / np.linalg.norm(wk) / np.linalg.norm(w_all)):.4f}")
        del V
        save_xm(f"hlda_{kind}", v_fit, bs.vec(kind) @ w_all)
        for nm, b in others:
            np.save(OUT / f"hlda_{kind}_{nm}.npy", (b.vec(kind) @ w_all).astype(np.float32))
        np.save(OUT / f"hlda_{kind}_w.npy", w_all)
        log(f"hlda_{kind}: done in {time.time() - t0:.0f}s; cos(w_all, ones) {float(w_all.sum() / np.linalg.norm(w_all) / np.sqrt(d)):.3f}; "
            f"corr(col, click) on fit rows (out-of-fold) {float(np.corrcoef(v_fit, y)[0, 1]):.4f}")
    if "knn" in a.what:
        import faiss
        faiss.omp_set_num_threads(cbx.THREADS)
        t0 = time.time()
        P = bf.vec_pca(kind, 32)
        order = np.argsort(fold_row, kind="stable"); Pc = np.ascontiguousarray(P[order]); tc = t[order]; fc = fold_row[order]
        lo = np.searchsorted(fc, np.arange(5), "left"); hi = np.searchsorted(fc, np.arange(5), "right")
        q = faiss.IndexFlatL2(32); ix = faiss.IndexIVFFlat(q, 32, a.nlist)
        rg = np.random.default_rng(0); ix.train(Pc[rg.choice(len(Pc), min(len(Pc), 400_000), replace=False)]); ix.add(Pc)
        log(f"  hknn_{kind}: index built ({time.time() - t0:.0f}s), PCA explained variance {fn.pca_fit(kind, 32)[2]:.3f}")

        def search(Q, fold=None):
            o50 = np.empty(len(Q), np.float32); o200 = np.empty(len(Q), np.float32)
            if fold is None:
                sp = faiss.SearchParametersIVF(nprobe=a.nprobe)
            else:
                rsel = faiss.IDSelectorRange(int(lo[fold]), int(hi[fold]))     # keep both objects alive during the search
                selr = faiss.IDSelectorNot(rsel)
                sp = faiss.SearchParametersIVF(nprobe=a.nprobe, sel=selr)
            for s in chunks(len(Q), 200_000):
                _, I = ix.search(np.ascontiguousarray(Q[s]), 200, params=sp)
                ok = I >= 0; tt = np.where(ok, tc[np.maximum(I, 0)], 0.0)
                o50[s] = tt[:, :50].sum(1) / np.maximum(ok[:, :50].sum(1), 1); o200[s] = tt.sum(1) / np.maximum(ok.sum(1), 1)
            return o50, o200
        f50 = np.empty(bf.n, np.float32); f200 = np.empty(bf.n, np.float32)
        for k in range(5):
            m = fold_row == k; ts = time.time()
            f50[m], f200[m] = search(P[m], fold=k)
            log(f"  hknn_{kind} fold {k}: {int(m.sum())} queries in {time.time() - ts:.0f}s ({m.sum() / max(time.time() - ts, 1e-9):.0f} q/s)")
        s50, s200 = search(bs.vec_pca(kind, 32))
        save_xm(f"hknn50_{kind}", f50, s50); save_xm(f"hknn200_{kind}", f200, s200)
        for nm, b in others:
            ts = time.time(); v50, v200 = search(b.vec_pca(kind, 32))
            np.save(OUT / f"hknn50_{kind}_{nm}.npy", v50); np.save(OUT / f"hknn200_{kind}_{nm}.npy", v200)
            log(f"  hknn_{kind} {nm}: {b.n} queries in {time.time() - ts:.0f}s")
        log(f"hknn_{kind}: done in {time.time() - t0:.0f}s; corr(hknn50, teacher) on fit rows (out-of-fold) {float(np.corrcoef(f50, t)[0, 1]):.4f}, "
            f"hknn200 {float(np.corrcoef(f200, t)[0, 1]):.4f}; corr with click {float(np.corrcoef(f50, y)[0, 1]):.4f} / {float(np.corrcoef(f200, y)[0, 1]):.4f}")
        del ix, P, Pc
log("HX DONE")
