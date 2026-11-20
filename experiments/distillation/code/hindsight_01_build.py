# Privileged teacher, measurement only. For train feeds whose user has a later feed, it finds the events that
# are new in the later history and builds per-candidate features from them, with the feed's own candidates
# removed, plus matched controls built from past events. These features use the same user's later snapshot,
# which is banned from every model input; they were measured as a possible teacher-side signal only.
# CPU only, 12 threads, no arguments. It wrote out/F.npy, out/rows.npy, out/feeds_sel.parquet and
# out/01_build.log under lab/nn_x/w12/hindsight/.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/hindsight/01_build.py
"""Hindsight (privileged-information) features for TRAIN feeds: MEASUREMENT ONLY, never a model input for val/test.

Step 1: consecutive same-user pairs (A earlier, B = next feed of the user, user_id != 0), robust alignment shift k
        (B = k new events at the front + A's history, tolerant of expired / reordered events), stats for ALL pairs.
Step 2: per-candidate features for the evaluation feeds (all holdout pair feeds + a random sample of the others)
        from 7 event sets:
          LEAK    = B[0:k]                      (no exclusion: sanity, must show the label leak)
          FUT     = B[0:k] minus events whose item_id is a candidate of A
          FUTNS1  = FUT minus events with event_source == 1
          PAST    = A.hist[0:n(FUT)]            (control, K matched)
          PASTNS1 = first n(FUTNS1) events of A.hist with event_source != 1
          FUTC    = chain over up to 8 later feeds (B[0:k_AB] + C[0:k_BC] + ...), cap 256, minus A's candidate items
          PASTC   = A.hist[0:n(FUTC)]
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '12'); os.environ.setdefault('NUMBA_NUM_THREADS', '12')
import numpy as np, pandas as pd, numba as nb
nb.set_num_threads(12)
ROOT = '<repo>'; FL = f'{ROOT}/data/flat/train'; OUT = f'{ROOT}/nn_x/w12/hindsight/out'
N_SAMPLE = 120_000; NV = 24; NF = 16; NS = 7; CAP = 256; HOPS = 8
SETS = ['LEAK', 'FUT', 'FUTNS1', 'PAST', 'PASTNS1', 'FUTC', 'PASTC']
FEATS = ['n_ev', 'n_emb', 'n_strong', 'n_view', 'cos_max', 'cos_mean', 'cos_top3', 'cos_max_strong', 'cos_mean_strong',
         'cos_max_view', 'cos_mean_view', 'cat_cnt', 'cat_share', 'cat_cnt_strong', 'cat_cnt_view', 'item_match']
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)


@nb.njit(parallel=True, cache=True)
def align(a_idx, b_idx, hoff, H, hi, ht, out_k, out_vote, out_nv):
    """k = modal shift (q - t) over A's first NV events t and every B position q with the same (item, type).
    Ties -> the smaller shift (conservative: fewer 'new' events). H(A)==0 -> k = H(B) (everything is new)."""
    for p in nb.prange(len(a_idx)):
        a = a_idx[p]; b = b_idx[p]; ha = hoff[a]; la = H[a]; hb = hoff[b]; lb = H[b]
        if la == 0:
            out_k[p] = lb; out_vote[p] = 1.0; out_nv[p] = 0; continue
        nv = min(la, NV); votes = np.zeros(257, np.int32)
        for t in range(nv):
            for q in range(t, lb):
                if hi[hb + q] == hi[ha + t] and ht[hb + q] == ht[ha + t]:
                    votes[q - t] += 1
        best = 0; k = -1
        for s in range(257):
            if votes[s] > best:
                best = votes[s]; k = s
        out_k[p] = k; out_vote[p] = best / nv; out_nv[p] = nv


@nb.njit(cache=True)
def set_feats(ev, n, cemb, cok, ccat, citem, m, table, hemb, hcat, het, hitem, Fout, r0, c0):
    """ev[0:n] = global history-event indices of the set; writes NF features for the m candidates at rows r0.."""
    E = np.zeros((n, 128), np.float32); eok = np.zeros(n, np.bool_)
    strong = np.zeros(n, np.bool_); view = np.zeros(n, np.bool_)
    n_emb = 0; n_s = 0; n_v = 0
    for j in range(n):
        g = ev[j]; e = hemb[g]; ty = het[g]
        strong[j] = (ty == 2) or (ty == 3) or (ty == 4); view[j] = ty == 1
        n_s += strong[j]; n_v += view[j]
        if e >= 0:
            ss = 0.0
            for d in range(128):
                v = np.float32(table[e, d]); E[j, d] = v; ss += v * v
            if ss > 0:
                inv = np.float32(1.0 / np.sqrt(ss))
                for d in range(128):
                    E[j, d] *= inv
                eok[j] = True; n_emb += 1
    for i in range(m):
        r = r0 + i
        mx = -2.0; sm = 0.0; cn = 0; t1 = -2.0; t2 = -2.0; t3 = -2.0
        mxs = -2.0; sms = 0.0; cns = 0; mxv = -2.0; smv = 0.0; cnv = 0
        cc = 0; ccs = 0; ccv = 0; im = 0
        for j in range(n):
            g = ev[j]
            if hitem[g] == citem[i]:
                im += 1
            if hcat[g] == ccat[i]:
                cc += 1; ccs += strong[j]; ccv += view[j]
            if cok[i] and eok[j]:
                c = 0.0
                for d in range(128):
                    c += cemb[i, d] * E[j, d]
                sm += c; cn += 1
                if c > t1:
                    t3 = t2; t2 = t1; t1 = c
                elif c > t2:
                    t3 = t2; t2 = c
                elif c > t3:
                    t3 = c
                if c > mx: mx = c
                if strong[j]:
                    sms += c; cns += 1
                    if c > mxs: mxs = c
                if view[j]:
                    smv += c; cnv += 1
                    if c > mxv: mxv = c
        F = Fout[r]
        F[c0 + 0] = n; F[c0 + 1] = n_emb; F[c0 + 2] = n_s; F[c0 + 3] = n_v
        if cn > 0:
            F[c0 + 4] = mx; F[c0 + 5] = sm / cn
            F[c0 + 6] = t1 if cn == 1 else ((t1 + t2) / 2 if cn == 2 else (t1 + t2 + t3) / 3)
        else:
            F[c0 + 4] = np.nan; F[c0 + 5] = np.nan; F[c0 + 6] = np.nan
        if cns > 0:
            F[c0 + 7] = mxs; F[c0 + 8] = sms / cns
        else:
            F[c0 + 7] = np.nan; F[c0 + 8] = np.nan
        if cnv > 0:
            F[c0 + 9] = mxv; F[c0 + 10] = smv / cnv
        else:
            F[c0 + 9] = np.nan; F[c0 + 10] = np.nan
        F[c0 + 11] = cc; F[c0 + 12] = cc / n if n > 0 else np.nan; F[c0 + 13] = ccs; F[c0 + 14] = ccv; F[c0 + 15] = im


@nb.njit(parallel=True, cache=True)
def build(sel_a, nxt, kpair, okpair, ts, hoff, H, coff, M, row0, hi, ht, hs, hc, he, ci, cc, ce, table, Fout, meta):
    """nxt[f] = fidx of the next same-user feed (-1 none); kpair[f]/okpair[f] = alignment of the pair (f, nxt[f]).
    meta[p] = n_LEAK, n_FUT, n_FUTNS1, n_PAST, n_PASTNS1, n_FUTC, n_PASTC, hops, chain_gap, n_new_in_Ahist(item)"""
    for p in nb.prange(len(sel_a)):
        a = sel_a[p]; m = M[a]; ca = coff[a]; ha = hoff[a]; la = H[a]; r0 = row0[p]
        cemb = np.zeros((m, 128), np.float32); cok = np.zeros(m, np.bool_)
        for i in range(m):
            e = ce[ca + i]
            if e >= 0:
                ss = 0.0
                for d in range(128):
                    v = np.float32(table[e, d]); cemb[i, d] = v; ss += v * v
                if ss > 0:
                    inv = np.float32(1.0 / np.sqrt(ss))
                    for d in range(128):
                        cemb[i, d] *= inv
                    cok[i] = True
        citem = ci[ca:ca + m]; ccat = cc[ca:ca + m]
        cs = np.sort(citem)
        ev = np.empty(CAP, np.int64)
        b = nxt[a]; k = kpair[a]; hb = hoff[b]
        # LEAK
        n = 0
        for j in range(k):
            ev[n] = hb + j; n += 1
        meta[p, 0] = n
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 0 * NF)
        # how many of the new events are on items already present in A's history (re-interactions)
        rep = 0
        for j in range(k):
            for q in range(la):
                if hi[ha + q] == hi[hb + j]:
                    rep += 1; break
        meta[p, 9] = rep
        # FUT
        n = 0
        for j in range(k):
            it = hi[hb + j]; q = np.searchsorted(cs, it)
            if q < m and cs[q] == it: continue
            ev[n] = hb + j; n += 1
        nfut = n; meta[p, 1] = n
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 1 * NF)
        # FUTNS1
        n2 = 0
        for j in range(n):
            if hs[ev[j]] != 1:
                ev[n2] = ev[j]; n2 += 1
        nns = n2; meta[p, 2] = n2
        set_feats(ev, n2, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 2 * NF)
        # PAST
        n = min(nfut, la)
        for j in range(n): ev[j] = ha + j
        meta[p, 3] = n
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 3 * NF)
        # PASTNS1
        n = 0
        for j in range(la):
            if n >= nns: break
            if hs[ha + j] != 1:
                ev[n] = ha + j; n += 1
        meta[p, 4] = n
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 4 * NF)
        # FUTC: chain
        n = 0; cur = a; hops = 0; last = a
        while hops < HOPS and n < CAP:
            nx = nxt[cur]
            if nx < 0 or not okpair[cur]: break
            hx = hoff[nx]
            for j in range(kpair[cur] - 1, -1, -1):  # oldest new event first so that the cap keeps the earliest future
                if n >= CAP: break
                it = hi[hx + j]; q = np.searchsorted(cs, it)
                if q < m and cs[q] == it: continue
                ev[n] = hx + j; n += 1
            hops += 1; last = nx; cur = nx
        nch = n; meta[p, 5] = n; meta[p, 7] = hops; meta[p, 8] = ts[last] - ts[a]
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 5 * NF)
        n = min(nch, la)
        for j in range(n): ev[j] = ha + j
        meta[p, 6] = n
        set_feats(ev, n, cemb, cok, ccat, citem, m, table, he, hc, ht, hi, Fout, r0, 6 * NF)


def main():
    f = pd.read_parquet(f'{FL}/feeds.parquet')
    nF = len(f); assert (f.fidx.values == np.arange(nF)).all()
    M = f.M.values.astype(np.int64); H = f.H.values.astype(np.int64); coff = f.cand_off.values; hoff = f.hist_off.values
    assert (coff == np.concatenate([[0], np.cumsum(M)[:-1]])).all() and (hoff == np.concatenate([[0], np.cumsum(H)[:-1]])).all()
    ts = f.timestamp.values.astype(np.int64); uid = f.user_id.values
    ld = lambda n: np.load(f'{FL}/{n}.npy')
    hi = ld('hist_item_id'); ht = ld('hist_event_type'); hs = ld('hist_event_source'); hc = ld('hist_category_id'); he = ld('hist_emb_idx')
    ci = ld('cand_item_id'); cc = ld('cand_category_id'); ce = ld('cand_emb_idx'); y = ld('cand_target')
    assert len(hi) == H.sum() == len(ht) == len(hs) == len(hc) == len(he)
    assert len(ci) == M.sum() == len(cc) == len(ce) == len(y)
    hp = np.load(f'{FL}/hist_hpos.npy', mmap_mode='r'); assert hp[hoff[1]] == 0 and hp[hoff[1] + 1] == 1  # newest-first positions
    log('arrays loaded')
    # holdout exactly as nn_x/w5/holdout_eval.py
    z = np.load(f'{ROOT}/eda/labels_position/out/train_prandom.npz'); assert (z['fidx'] == np.arange(nF)).all()
    pr = np.nan_to_num(z['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0.0
    cand = np.where((pr > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    is_hold = np.zeros(nF, bool); is_hold[hold] = True
    # consecutive pairs
    o = np.lexsort((np.arange(nF), ts, uid)); o = o[uid[o] != 0]
    same = uid[o][1:] == uid[o][:-1]
    a_idx = o[:-1][same].astype(np.int64); b_idx = o[1:][same].astype(np.int64)
    gap = ts[b_idx] - ts[a_idx]; assert (gap >= 0).all()
    log('pairs', len(a_idx), 'of', nF, 'feeds =', len(a_idx) / nF, '| user0 feeds', int((uid == 0).sum()))
    k = np.empty(len(a_idx), np.int32); vote = np.empty(len(a_idx), np.float32); nv = np.empty(len(a_idx), np.int32)
    align(a_idx, b_idx, hoff, H, hi, ht, k, vote, nv)
    # aligned: at least half of A's first NV events vote for the modal shift (or A has no history); gap > 0
    ok = (k >= 0) & (vote >= 0.5) & (gap > 0)
    log('alignment: k<0', (k < 0).mean(), 'vote>=0.5', (vote >= 0.5).mean(), 'vote==1', (vote == 1).mean(), 'gap==0', (gap == 0).mean(),
        'ok', ok.mean(), 'ok & k>0', (ok & (k > 0)).mean(), 'H(A)==0 among ok', (H[a_idx][ok] == 0).mean())
    nxt = np.full(nF, -1, np.int64); nxt[a_idx] = b_idx
    kpair = np.zeros(nF, np.int32); kpair[a_idx] = np.where(ok, k, 0)
    okpair = np.zeros(nF, bool); okpair[a_idx] = ok
    pairs = pd.DataFrame(dict(a=a_idx, b=b_idx, gap=gap, k=k, vote=vote, ok=ok, hold=is_hold[a_idx], p_random=pr[a_idx],
                              M=M[a_idx], HA=H[a_idx], HB=H[b_idx], npos=np.add.reduceat(y.astype(np.int64), coff)[a_idx]))
    pairs.to_parquet(f'{OUT}/pairs_all.parquet')
    # evaluation feeds: aligned pairs with k > 0; all holdout ones + random sample of the rest
    elig = np.where(ok & (k > 0))[0]
    eh = elig[is_hold[a_idx[elig]]]; en = elig[~is_hold[a_idx[elig]]]
    rs = np.random.default_rng(20261114).choice(en, min(N_SAMPLE, len(en)), replace=False)
    sel = np.sort(np.concatenate([eh, rs])); sel_a = a_idx[sel]
    log('eligible (ok & k>0)', len(elig), 'holdout', len(eh), 'non-holdout', len(en), 'selected', len(sel))
    Ms = M[sel_a]; row0 = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); N = int(Ms.sum())
    import json as _j
    d = f'{ROOT}/data/competition/train/item_embeddings'; shp = _j.loads(open(f'{d}/data.meta').read())['shape']
    table = np.fromfile(f'{d}/data.memmap', dtype=np.int8).reshape(shp); assert table.shape[1] == 128
    assert ce.max() < table.shape[0] and he.max() < table.shape[0]
    log('item table', table.shape)
    F = np.empty((N, NS * NF), np.float32); meta = np.zeros((len(sel), 10), np.int64)
    build(sel_a, nxt, kpair, okpair, ts, hoff, H, coff, M, row0, hi, ht, hs, hc, he, ci, cc, ce, table, F, meta)
    log('features built', F.shape)
    rows = np.concatenate([np.arange(c, c + m) for c, m in zip(coff[sel_a], Ms)])
    np.save(f'{OUT}/F.npy', F); np.save(f'{OUT}/rows.npy', rows)
    fm = pairs.iloc[sel].reset_index(drop=True)
    for j, n in enumerate(['n_LEAK', 'n_FUT', 'n_FUTNS1', 'n_PAST', 'n_PASTNS1', 'n_FUTC', 'n_PASTC', 'hops', 'chain_gap', 'n_new_item_in_Ahist']):
        fm[n] = meta[:, j]
    fm['user_id'] = uid[sel_a]; fm['row0'] = row0
    fm.to_parquet(f'{OUT}/feeds_sel.parquet')
    json.dump(dict(sets=SETS, feats=FEATS), open(f'{OUT}/cols.json', 'w'))
    # sanity: exact item matches
    yy = y[rows]
    for s, nm in enumerate(SETS):
        im = F[:, s * NF + 15]
        log(f'{nm:8s} item_match>0: rows {int((im > 0).sum())} share {float((im > 0).mean()):.6f} | among positives {float((im[yy == 1] > 0).mean()):.4f} '
            f'among negatives {float((im[yy == 0] > 0).mean()):.6f} | mean n_ev {float(F[:, s * NF].mean()):.2f}')
    assert (F[:, 1 * NF + 15] == 0).all() and (F[:, 2 * NF + 15] == 0).all() and (F[:, 5 * NF + 15] == 0).all(), 'exclusion broken'
    assert (meta[:, 0] == fm.k.values).all()
    log('DONE')


if __name__ == '__main__':
    main()
