"""int_gnn s5: SIGNAL check of the inductive, content-based graph path on COLD val candidates (labels used for evaluation only).
For every cold candidate (stored LUO count == 0) of NF random val feeds: its content-nearest warm train items (exact search in
the bank of train-history items with degree >= 10 and an embedding; best item of each 100K bank chunk, top 5, cos >= 0.90).
Imputed columns = cosine-weighted mean over those neighbours of the neighbour's OWN graph scores against the feed's history
(s1 = 3-step walk, s2 = 5-step walk, as in s3_walk.py; train graph, own user node skipped), plus imputed log-degree and density.
Reads: feed-mean within-feed AUC among cold rows (cold positive vs cold negative), next to the pool and the content-only
encoder score; Spearman with those; nested residual on the W16 pool with the column z-scored among the cold rows of the feed
(warm rows 0). CPU only; peak memory kept low (no array of the train history is held while the bank is searched).
usage: python s5_coldsignal.py NF"""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numba as nb
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g_common import build_graph, log, FLAT, FEATS  # noqa: E402
from s3_walk import walk, feed_auc, feed_z  # noqa: E402
from s4_content import item_emb_row, table  # noqa: E402
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
COMP = ROOT / "data/competition"
NT = 10
torch.set_num_threads(NT)


def main():
    NF = int(sys.argv[1])
    nb.set_num_threads(NT)
    g = build_graph("train")
    G = g["G"]
    i_ptr = g["i_ptr"]
    deg_item = np.diff(i_ptr)
    it = np.load(FLAT / "train/hist_item_id.npy")
    em = np.load(FLAT / "train/hist_emb_idx.npy")
    emb_row = item_emb_row(it, em, G)
    del it, em
    bank_items = np.nonzero((deg_item >= 10) & (emb_row >= 0))[0]
    rng = np.random.default_rng(20261115)
    bank_items = rng.permutation(bank_items)
    NB = len(bank_items)
    rows = emb_row[bank_items].astype(np.int64)
    del emb_row
    bank = np.empty((NB, 128), np.int8)
    order = np.argsort(rows)
    srows = rows[order]
    tpath = COMP / "train/item_embeddings/data.memmap"
    ntab = json.loads((COMP / "train/item_embeddings/data.meta").read_text())["shape"][0]
    blk = 4_000_000
    for a in range(0, ntab, blk):
        b = min(a + blk, ntab)
        lo, hi = np.searchsorted(srows, [a, b])
        if hi > lo:
            buf = np.fromfile(tpath, dtype=np.int8, count=(b - a) * 128, offset=a * 128).reshape(b - a, 128)
            bank[order[lo:hi]] = buf[srows[lo:hi] - a]
            del buf
    bnorm = np.sqrt((bank.astype(np.float32) ** 2).sum(1))
    log(f"bank (degree >= 10, with embedding): {NB} items")

    f = g["feeds"]["val"]
    M = f["M"].to_numpy().astype(np.int64)
    Hh = f["H"].to_numpy().astype(np.int64)
    coff = f["cand_off"].to_numpy().astype(np.int64)
    hoff = f["hist_off"].to_numpy().astype(np.int64)
    own = g["node_of_feed"]["val"].astype(np.int64)
    nF = len(M)
    fid = np.repeat(np.arange(nF), M)
    cemb = np.load(FLAT / "val/cand_emb_idx.npy").astype(np.int64)
    luo = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    selF = np.sort(rng.choice(nF, min(NF, nF), replace=False))
    inF = np.zeros(nF, bool)
    inF[selF] = True
    cold = (luo == 0)
    qrows = np.nonzero(inF[fid] & cold & (cemb >= 0))[0]
    tab = table("val")
    qvec = np.asarray(tab[cemb[qrows]]).astype(np.float32)
    qvec /= np.maximum(np.sqrt((qvec ** 2).sum(1, keepdims=True)), 1e-6)
    nq = len(qrows)
    log(f"val feeds {len(selF)}, cold query rows {nq} (cold positives {int(y[qrows].sum())})")
    CH = 100_000
    nch = (NB + CH - 1) // CH
    best_v = np.full((nq, nch), -2.0, np.float32)
    best_i = np.zeros((nq, nch), np.int64)
    qt = torch.from_numpy(qvec)
    QB = 6000
    t1 = time.time()
    for ci, a in enumerate(range(0, NB, CH)):
        b = min(a + CH, NB)
        ck = torch.from_numpy(bank[a:b].astype(np.float32))
        ck /= torch.from_numpy(np.maximum(bnorm[a:b], 1e-6))[:, None]
        ckt = ck.T.contiguous()
        for q0 in range(0, nq, QB):
            v, i = (qt[q0:q0 + QB] @ ckt).max(dim=1)
            best_v[q0:q0 + QB, ci] = v.numpy()
            best_i[q0:q0 + QB, ci] = i.numpy() + a
        if ci % 15 == 0:
            log(f"chunk {ci}/{nch} ({time.time() - t1:.0f}s)")
    del bank, qt
    K = 5
    k5 = np.argsort(-best_v, axis=1)[:, :K]
    nv = np.take_along_axis(best_v, k5, 1).astype(np.float64)
    ni = bank_items[np.take_along_axis(best_i, k5, 1)]
    del best_v, best_i
    log(f"neighbours: top-1 cos >=0.90 {np.mean(nv[:, 0] >= 0.90):.4f} >=0.95 {np.mean(nv[:, 0] >= 0.95):.4f}; median top-1 {np.median(nv[:, 0]):.3f}")

    # pseudo-candidates = the K neighbours of every cold query row, grouped by feed (query rows are already in feed order)
    qf = fid[qrows]
    cntF = np.bincount(qf, minlength=nF).astype(np.int64) * K
    pcoff = np.concatenate([[0], np.cumsum(cntF)[:-1]]).astype(np.int64)
    pc = ni.reshape(-1).astype(np.int64)
    hist = np.load(FLAT / "val/hist_item_id.npy").astype(np.int64)
    hpos = np.load(FLAT / "val/hist_hpos.npy").astype(np.int64)
    out = np.zeros((len(pc), 3), np.float64)
    nb.set_num_threads(8)
    walk(nF, pcoff, cntF, hoff, Hh, own, pc, hist, hpos, g["u_ptr"], g["u_items"], i_ptr, g["i_users"], g["n_nodes"], G, 8, out)
    s1 = out[:, 0].reshape(nq, K)
    s2 = out[:, 1].reshape(nq, K)
    dg = out[:, 2].reshape(nq, K)
    wgt = np.where(nv >= 0.90, nv, 0.0)
    ws = np.maximum(wgt.sum(1), 1e-9)
    imp = {
        "imputed s1 (cos-weighted mean)": (wgt * s1).sum(1) / ws,
        "imputed s1 (max)": np.where(wgt > 0, s1, 0).max(1),
        "imputed s2/deg^.5": (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws,
        "imputed s2/deg": (wgt * s2 / np.maximum(dg, 1)).sum(1) / ws,
        "imputed log-degree": (wgt * np.log1p(dg)).sum(1) / ws,
        "number of linked neighbours": ((wgt > 0) & (s1 > 0)).sum(1).astype(np.float64),
        "density (top-1 cos)": nv[:, 0],
    }
    log(f"coverage on cold rows: >=1 neighbour at cos>=0.90 {np.mean(wgt.sum(1) > 0):.4f} | imputed s1 > 0 {np.mean(imp['imputed s1 (max)'] > 0):.4f} | imputed s2 > 0 {np.mean(imp['imputed s2/deg'] > 0):.4f}")

    z = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)  # noqa: E731
    pool = z(0.5 * z(np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * z(np.load(ROOT / "preds/ens_W16kd_to_val.npy")))
    ref = {"W16 pool": pool, "pre2_fe (content-only on cold rows)": np.asarray(np.load(FEATS / "val/pre2_fe.npy"), np.float64),
           "seq_max_all (max cos to history)": np.nan_to_num(np.asarray(np.load(FEATS / "val/seq_max_all.npy"), np.float64), nan=-9.0)}
    sub = np.zeros(len(y), bool)
    sub[qrows] = True
    full = {}
    for n, v in imp.items():
        x = np.zeros(len(y))
        x[qrows] = v
        full[n] = x
    line = "[AUC among cold rows of the sampled feeds, feed mean ± se]"
    for n, v in {**full, **ref}.items():
        au = feed_auc(np.asarray(v, np.float64), y, sub, coff, M)
        ok = ~np.isnan(au)
        line += f" | {n} {np.nanmean(au):.4f}±{np.nanstd(au) / np.sqrt(ok.sum()):.4f}"
    log(line + f" (feeds with a cold positive and a cold negative: {int(ok.sum())})")

    np.savez(HERE / "s5_cold_val.npz", selF=selF, qrows=qrows, nv=nv, ni=ni, s1=s1, s2=s2, dg=dg)

    # AUC after removing what two content-similarity inputs of the model already say (pooled OLS on within-feed centred
    # percentile ranks among the cold rows; controls: seq_max_new8 and pre2_fe_noid)
    def crank(v):
        r = np.zeros(len(y))
        for ff in selF:
            o = coff[ff]
            m = sub[o:o + M[ff]]
            k = int(m.sum())
            if k == 0:
                continue
            rr = (rankdata(v[o:o + M[ff]][m]) - 0.5) / k - 0.5
            tmp = np.zeros(M[ff])
            tmp[m] = rr
            r[o:o + M[ff]] = tmp
        return r
    ctrl = [crank(np.nan_to_num(np.asarray(np.load(FEATS / "val/seq_max_new8.npy"), np.float64), nan=-9.0)),
            crank(np.asarray(np.load(FEATS / "val/pre2_fe_noid.npy"), np.float64))]
    A = np.stack([c[qrows] for c in ctrl], 1)
    line = "[AUC among cold rows AFTER rank-residualising on seq_max_new8 + pre2_fe_noid]"
    for n, v in full.items():
        rx = crank(v)
        beta, *_ = np.linalg.lstsq(A, rx[qrows], rcond=None)
        res = np.zeros(len(y))
        res[qrows] = rx[qrows] - A @ beta
        au = feed_auc(res, y, sub, coff, M)
        ok = ~np.isnan(au)
        line += f" | {n} {np.nanmean(au):.4f}±{np.nanstd(au) / np.sqrt(ok.sum()):.4f} (beta {beta[0]:+.2f} {beta[1]:+.2f})"
    log(line)

    def spearman_feed(a, b):
        r = []
        for ff in selF:
            o = coff[ff]
            m = sub[o:o + M[ff]]
            if m.sum() < 8:
                continue
            ra = rankdata(a[o:o + M[ff]][m]); rb = rankdata(b[o:o + M[ff]][m])
            if ra.std() == 0 or rb.std() == 0:
                continue
            r.append(np.corrcoef(ra, rb)[0, 1])
        return np.mean(r)

    for n in ["imputed s1 (cos-weighted mean)", "imputed s2/deg^.5", "imputed s2/deg", "imputed log-degree"]:
        log(f"[spearman among cold rows, feed mean] {n} vs " + " ".join(f"{rn} {spearman_feed(full[n], rv):+.3f}" for rn, rv in ref.items()))

    # nested residual on the pool, sampled feeds only; column z-scored among the cold rows of the feed, warm rows 0
    Ms, offs = M[selF], np.concatenate([[0], np.cumsum(M[selF])[:-1]]).astype(np.int64)
    rsel = np.concatenate([np.arange(coff[x], coff[x] + M[x]) for x in selF])
    ys, ps, subs = y[rsel], pool[rsel], sub[rsel].astype(np.float64)
    fids = np.repeat(np.arange(len(selF)), Ms)
    user = f["user_id"].to_numpy().astype(np.int64)[selF]
    grp = np.where(user == 0, -1 - np.arange(len(selF)), user)
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(selF)), groups=grp))
    nd = lambda s: ndcg_per_feed(s, ys, offs, Ms)  # noqa: E731
    base = nd(ps)
    npos = np.bincount(fids, weights=ys, minlength=len(selF))
    cpos = np.bincount(fids, weights=ys * subs, minlength=len(selF))
    log(f"W16 pool NDCG on the sampled feeds {base.mean():.5f}; feeds with a cold positive {np.mean(cpos > 0):.4f}")
    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    for n, x0 in full.items():
        v = x0[rsel]
        if n.startswith("imputed s") or n.startswith("number"):
            med = np.median(v[v > 0]) if np.any(v > 0) else 1.0
            v = np.log1p(v / med)
        cnt = np.maximum(np.bincount(fids, weights=subs, minlength=len(selF)), 1.0)
        mu = np.bincount(fids, weights=v * subs, minlength=len(selF)) / cnt
        d = (v - mu[fids]) * subs
        sd = np.sqrt(np.bincount(fids, weights=d * d, minlength=len(selF)) / cnt)
        x = d / np.maximum(sd[fids], 1e-9)
        per = {w: nd(ps + w * x) for w in grid}
        delta = np.zeros(len(selF)); wsel = []
        for trf, vaf in folds:
            best = max(grid, key=lambda w: per[w][trf].mean())
            wsel.append(best)
            delta[vaf] = per[best][vaf] - base[vaf]
        se = delta.std() / np.sqrt(len(selF))
        log(f"[nested W16, cold-row z] {n}: {delta.mean():+.5f} ± {se:.5f} (z {delta.mean() / max(se, 1e-12):+.2f}) w {wsel} | feeds with a cold positive {delta[cpos > 0].mean():+.5f} "
            f"| in-sample " + " ".join(f"{w}:{(per[w] - base).mean():+.5f}" for w in grid[1:]))
    log("DONE")


if __name__ == "__main__":
    main()
