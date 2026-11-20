"""e05: intent strength per event type.  Does a type-t event on a history item predict the click on a candidate that is
(a) the same category, (b) close in item-embedding space to that item?
Per candidate and type t: mc_t = max cosine(candidate, history items carrying >=1 type-t event) (0 if none),
sc_t = log1p(# type-t events in the candidate's category).
Read-outs: (1) Mantel-Haenszel odds ratios within feed (flag vs not) for mc_t >= 0.8 and sc_t > 0;
(2) within-feed pairwise AUC of mc_t on feeds having type t;  (3) conditional logit (feed softmax, multi-positive)
on all 20 features jointly, with and without f1..f5 controls; coefficients per SD.
Splits: val (all) and a train sample of 40,000 exploration-like feeds (p_random > 0.5, seed 0)."""
import sys, os
import numpy as np, polars as pl, numba as nb
from scipy.optimize import minimize

nb.set_num_threads(12)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240); pl.Config.set_float_precision(4)
ROOT = '<repo>/data/flat'; COMP = '<repo>/data/competition'
OUT = '<repo>/nn_x/codes/events/out'
T = list(range(1, 11))


@nb.njit(parallel=True, cache=True)
def feat_kernel(coff, M, hoff, H, c_emb, c_cat, h_item, h_emb, h_cat, h_type, E, mc, sc):
    nf = len(coff)
    for f in nb.prange(nf):
        c0 = coff[f]; m = M[f]; h0 = hoff[f]; h = H[f]
        # unique history items with type mask
        items = np.empty(h, np.int64); masks = np.zeros(h, np.int64); embs = np.empty(h, np.int64); nu = 0
        for j in range(h):
            it = h_item[h0 + j]; found = -1
            for u in range(nu):
                if items[u] == it:
                    found = u; break
            if found < 0:
                items[nu] = it; embs[nu] = h_emb[h0 + j]; masks[nu] = 0; found = nu; nu += 1
            masks[found] |= (1 << h_type[h0 + j])
        # same-category counts per type
        cnt = np.zeros((64, 11), np.float32)
        for j in range(h):
            cnt[h_cat[h0 + j], h_type[h0 + j]] += 1.0
        for i in range(m):
            r = c0 + i
            for t in range(1, 11):
                sc[r, t - 1] = np.log1p(cnt[c_cat[r], t])
                mc[r, t - 1] = 0.0
            ce = c_emb[r]
            if ce < 0:
                continue
            for u in range(nu):
                he = embs[u]
                if he < 0:
                    continue
                s = 0.0
                for d in range(E.shape[1]):
                    s += E[ce, d] * E[he, d]
                for t in range(1, 11):
                    if (masks[u] >> t) & 1:
                        if s > mc[r, t - 1]:
                            mc[r, t - 1] = s


def build(split, fidx=None):
    F = pl.read_parquet(f'{ROOT}/{split}/feeds.parquet').sort('fidx')
    if fidx is not None:
        F = F.filter(pl.col('fidx').is_in(fidx)).sort('fidx')
    coff_o = F['cand_off'].to_numpy(); M = F['M'].to_numpy().astype(np.int64)
    hoff_o = F['hist_off'].to_numpy(); H = F['H'].to_numpy().astype(np.int64)
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff_o, M)])
    hrow = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff_o, H)])
    ld = lambda n, rows: np.asarray(np.load(f'{ROOT}/{split}/{n}.npy', mmap_mode='r')[rows])
    c_emb = ld('cand_emb_idx', crow).astype(np.int64); c_cat = ld('cand_category_id', crow).astype(np.int64)
    tgt = ld('cand_target', crow).astype(np.int8)
    fs = np.stack([ld(f'cand_f{k}', crow) for k in range(1, 6)], 1).astype(np.float32)
    h_item = ld('hist_item_id', hrow); h_emb = ld('hist_emb_idx', hrow).astype(np.int64)
    h_cat = ld('hist_category_id', hrow).astype(np.int64); h_type = ld('hist_event_type', hrow).astype(np.int64)
    # compact + L2-normalise the embeddings that are needed
    meta_shape = (sum(1 for _ in [0]),)
    mm = np.memmap(f'{COMP}/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r').reshape(-1, 128)
    need = np.unique(np.concatenate([c_emb[c_emb >= 0], h_emb[h_emb >= 0]]))
    E = np.asarray(mm[need]).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
    remap = lambda x: np.where(x >= 0, np.searchsorted(need, np.maximum(x, 0)), -1)
    c_emb2 = remap(c_emb); h_emb2 = remap(h_emb)
    coff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    N = len(crow)
    mc = np.zeros((N, 10), np.float32); sc = np.zeros((N, 10), np.float32)
    feat_kernel(coff, M, hoff, H, c_emb2, c_cat, h_item, h_emb2, h_cat, h_type, E, mc, sc)
    hasT = np.zeros((len(M), 10), bool)
    fid = np.repeat(np.arange(len(H)), H)
    for t in T:
        hasT[np.unique(fid[h_type == t]), t - 1] = True
    return dict(coff=coff, M=M, tgt=tgt, fs=fs, mc=mc, sc=sc, hasT=hasT, uid=F['user_id'].to_numpy())


def mh_or(flag, tgt, coff, M):
    fid = np.repeat(np.arange(len(M)), M)
    n = M[fid]
    a = np.bincount(fid, weights=(flag & (tgt == 1)), minlength=len(M))
    b = np.bincount(fid, weights=(flag & (tgt == 0)), minlength=len(M))
    c = np.bincount(fid, weights=(~flag & (tgt == 1)), minlength=len(M))
    d = np.bincount(fid, weights=(~flag & (tgt == 0)), minlength=len(M))
    nn = a + b + c + d
    num = (a * d / np.maximum(nn, 1)).sum(); den = (b * c / np.maximum(nn, 1)).sum()
    return num / max(den, 1e-12), int(flag.sum()), float(tgt[flag].mean()) if flag.any() else np.nan, float(tgt[~flag].mean())


@nb.njit(parallel=True, cache=True)
def auc_kernel(score, tgt, coff, M, use, out):
    for f in nb.prange(len(M)):
        out[f] = np.nan
        if not use[f]:
            continue
        c0 = coff[f]; m = M[f]; npos = 0; nneg = 0; s = 0.0
        for i in range(m):
            if tgt[c0 + i] == 1:
                npos += 1
        nneg = m - npos
        if npos == 0 or nneg == 0:
            continue
        for i in range(m):
            if tgt[c0 + i] != 1:
                continue
            for j in range(m):
                if tgt[c0 + j] == 1:
                    continue
                if score[c0 + i] > score[c0 + j]:
                    s += 1.0
                elif score[c0 + i] == score[c0 + j]:
                    s += 0.5
        out[f] = s / (npos * nneg)


def clogit(X, tgt, coff, M, l2=1e-4):
    fid = np.repeat(np.arange(len(M)), M)
    keep = np.bincount(fid, weights=tgt, minlength=len(M)) > 0
    rows = keep[fid]
    X = X[rows]; y = tgt[rows].astype(np.float64); fid2 = np.cumsum(keep)[fid[rows]] - 1
    nfe = keep.sum(); npos = np.bincount(fid2, weights=y, minlength=nfe)
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xs = (X - mu) / sd

    def fg(w):
        s = Xs @ w
        mx = np.full(nfe, -np.inf); np.maximum.at(mx, fid2, s)
        e = np.exp(s - mx[fid2]); Z = np.bincount(fid2, weights=e, minlength=nfe)
        lse = np.log(Z) + mx
        ll = (y * s).sum() - (npos * lse).sum()
        p = e / Z[fid2]
        g = Xs.T @ y - Xs.T @ (p * npos[fid2])
        return -ll / nfe + l2 * (w @ w), -g / nfe + 2 * l2 * w
    r = minimize(fg, np.zeros(Xs.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 500})
    # standard errors from the Hessian approximation is skipped; use a split-half check instead
    return r.x, r.fun


def report(name, d):
    tgt = d['tgt']; coff = d['coff']; M = d['M']; mc = d['mc']; sc = d['sc']; hasT = d['hasT']
    fid = np.repeat(np.arange(len(M)), M)
    print(f'\n######## {name}: feeds {len(M):,}, candidates {len(tgt):,}, positives {int(tgt.sum()):,}')
    rows = []
    for t in T:
        hf = hasT[fid, t - 1]
        o1 = mh_or((mc[:, t - 1] >= 0.8) & hf, tgt, coff, M)
        o2 = mh_or((mc[:, t - 1] >= 0.9) & hf, tgt, coff, M)
        o3 = mh_or((sc[:, t - 1] > 0) & hf, tgt, coff, M)
        auc = np.empty(len(M)); auc_kernel(mc[:, t - 1].astype(np.float64) + 1e-7 * np.random.default_rng(0).random(len(tgt)), tgt, coff, M, hasT[:, t - 1], auc)
        auc1 = np.empty(len(M)); auc_kernel(mc[:, 0].astype(np.float64) + 1e-7 * np.random.default_rng(0).random(len(tgt)), tgt, coff, M, hasT[:, t - 1], auc1)
        aucs = np.empty(len(M)); auc_kernel(sc[:, t - 1].astype(np.float64) + 1e-7 * np.random.default_rng(0).random(len(tgt)), tgt, coff, M, hasT[:, t - 1], aucs)
        dd = auc - auc1; ok = ~np.isnan(dd)
        rows.append({'type': t, 'feeds_with_t': int(hasT[:, t - 1].sum()),
                     'n_cos>=.8': o1[1], 'tr_cos>=.8': o1[2], 'MH_OR_cos>=.8': o1[0], 'n_cos>=.9': o2[1], 'MH_OR_cos>=.9': o2[0],
                     'n_samecat': o3[1], 'tr_samecat': o3[2], 'MH_OR_samecat': o3[0],
                     'AUC_mc_t': np.nanmean(auc), 'AUC_mc_1_samefeeds': np.nanmean(auc1),
                     'dAUC_vs_type1': np.nanmean(dd), 'se_dAUC': np.nanstd(dd[ok]) / np.sqrt(ok.sum()), 'AUC_sc_t': np.nanmean(aucs)})
    print('[A1] per type: within-feed Mantel-Haenszel odds ratios (flag vs not, feeds that have type t), within-feed AUC')
    print(pl.DataFrame(rows))
    # conditional logit
    X0 = np.concatenate([mc, sc], 1).astype(np.float64)
    names = [f'mc{t}' for t in T] + [f'sc{t}' for t in T]
    w0, l0 = clogit(X0, tgt, coff, M)
    Xf = np.concatenate([X0, d['fs'].astype(np.float64)], 1)
    w1, l1 = clogit(Xf, tgt, coff, M)
    # split-half stability (by feed parity)
    half = (np.arange(len(M)) % 2 == 0)
    sub = lambda h: (np.concatenate([np.arange(coff[f], coff[f] + M[f]) for f in np.where(h)[0]]))
    ra = sub(half); rb = sub(~half)
    MA = M[half]; MB = M[~half]
    ca = np.concatenate([[0], np.cumsum(MA)[:-1]]); cb = np.concatenate([[0], np.cumsum(MB)[:-1]])
    wa, _ = clogit(Xf[ra], tgt[ra], ca, MA); wb, _ = clogit(Xf[rb], tgt[rb], cb, MB)
    print('[A2] conditional logit coefficients (per SD of the feature; feed-softmax, multi-positive); half A / half B with f-controls')
    out = pl.DataFrame({'feature': names + ['f1', 'f2', 'f3', 'f4', 'f5'], 'no_controls': list(w0) + [None] * 5,
                        'with_f1..f5': list(w1), 'halfA': list(wa), 'halfB': list(wb)})
    print(out)
    out.write_csv(f'{OUT}/e05_clogit_{name}.csv')
    print(f'   mean nll/feed: no_controls {l0:.4f}  with f {l1:.4f}')


def main():
    dv = build('val')
    report('val', dv)
    z = np.load('<repo>/eda/labels_position/out/train_prandom.npz')
    pr = np.nan_to_num(z['p_random'], nan=0.0); fx = z['fidx']
    cand = fx[pr > 0.5]
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(cand, 40_000, replace=False))
    np.save(f'{OUT}/e05_train_explore40k_fidx.npy', samp)
    dt = build('train', samp)
    report('train_explore40k', dt)


if __name__ == '__main__':
    main()
