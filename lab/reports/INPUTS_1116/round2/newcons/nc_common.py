"""newcons: shared code (PRELIST.txt of 2026-11-16T04:56:50Z). CPU only, 12 threads; read-only on every existing file; writes only under
reports/INPUTS_1116/round2/newcons/.

The LAG VIEW DRIVER is nn_x/w12/nx3/nx3_common.py, imported unchanged (Data, Src, run_kernel; kernel of record c9 = c09_build.py): which sequences are members,
dense / sparse, kernel time, own-user skip, cap 500, degree = dense posting count floor 1. My code adds:
  Src2      the same index + the CELL of every event (evc), cell postings (sequence, event index), dense cell degree
  kern      one pass per candidate over (A) the postings of its ITEM and (B) the postings of its CELL, with the admission rule of the kernel of record
            (own non-zero user skipped; sequence admitted iff kernel time < own timestamp; dense members carry -1), and a causality audit of every admitted
            (row, sequence) pair against the TRUE attributes of the sequence (as nx3_common.audit).
  brute     an independent numpy re-derivation for sampled feeds.
Raw columns of kern (out[:, k]):
  0 fwd20     sum over occurrences of c in admitted sequences, d = 1..20, of w0(x) / d for x = the event d steps OLDER than c      (N4)
  1 bwd20     the same for the event d steps NEWER than c (diagnostic only)
  2 aud_item  sum cos(user, owner) / (n + 5) over the distinct admitted sequences that hold item c                                (N6)
  3 n_item    number of distinct admitted sequences that hold item c (0 = the walk is silent by construction: "cold row")
  4 cw_item   sum over distinct admitted sequences of cell g (event not item c) of sum_{x in s, x != c, cell(x) != g} wd(x) / len(s), / (n + 10)   (N1)
  5 cw_cell   the same with the seed matched through cells: sum_{x in s, x != c, cell(x) >= 0, cell(x) != g} wdc(cell(x)) / len(s), / (n + 10)      (N2)
  6 aud_cell  sum cos(user, owner) / (n + 5) over those sequences                                                                 (N3)
  7 n_cell    number of those sequences
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
NC_DIR = '<repo>/reports/INPUTS_1116/round2/newcons'
os.environ['NUMBA_CACHE_DIR'] = NC_DIR + '/out/numba_cache'      # numba reads this once, at its first import: nothing of this task is cached elsewhere
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb
assert nb.config.CACHE_DIR == NC_DIR + '/out/numba_cache', nb.config.CACHE_DIR
sys.path.insert(0, '<repo>/nn_x/w12/nx3')
from nx3_common import *          # Data, Src, run_kernel, c9, hc, md5, save_new, Log, NT, CAP, NROWS, ROOT, FLAT (numba is already imported: its cache dir stays mine)
assert nb.config.CACHE_DIR == NC_DIR + '/out/numba_cache'
NCO = Path(NC_DIR) / 'out'; CELLS = NCO / 'cells'; NRAW = NCO / 'raw'
NX3 = Path('<repo>/nn_x/w12/nx3')
RAWCOLS = ['fwd20', 'bwd20', 'aud_item', 'n_item', 'cw_item', 'cw_cell', 'aud_cell', 'n_cell', 'rp3_rec', 'rp3_l8']
W20 = 20


@nb.njit(cache=True)
def cell_postings(evc, NCELL, starts, lens):
    cnt = np.zeros(NCELL, np.int64)
    for i in range(len(evc)):
        if evc[i] >= 0:
            cnt[evc[i]] += 1
    pst = np.zeros(NCELL, np.int64); acc = 0
    for g in range(NCELL):
        pst[g] = acc; acc += cnt[g]
    fill = pst.copy(); p_seq = np.empty(acc, np.int32); p_ev = np.empty(acc, np.int32)
    for s in range(len(lens)):
        st = starts[s]
        for j in range(lens[s]):
            g = evc[st + j]
            if g >= 0:
                p_seq[fill[g]] = s; p_ev[fill[g]] = st + j; fill[g] += 1
    return cnt, pst, p_seq, p_ev


class Src2(Src):
    """nx3_common.Src + cells of the events, cell postings, dense cell degree"""

    def __init__(self, D, parts, hist_cell, NCELL):
        super().__init__(D, parts)
        idx = self.idx; assert len(idx.ev) < 2 ** 31
        evc = np.empty(len(idx.ev), np.int32); o = 0
        for sp, sel, _kt, _dn in parts:
            sel = np.asarray(sel, np.int64); H = D.A[sp]['H']; h = H[sel]
            off = (o + np.concatenate([[0], np.cumsum(h)[:-1]])).astype(np.int64)
            c9.gather(sel, D.A[sp]['ho'], H, hist_cell[sp], off, evc); o += int(h.sum())
        assert o == len(idx.ev)
        self.evc = evc
        self.ccnt, self.cpst, self.cp_seq, self.cp_ev = cell_postings(evc, NCELL, idx.starts, idx.lens)
        nde = int(idx.lens[:self.n_dense].sum()); d = evc[:nde]
        self.degc_dense = np.bincount(d[d >= 0], minlength=NCELL).astype(np.float64)


@nb.njit(parallel=True, cache=True)
def kern(Mq, offq, hoff, H, fuser, ftime, fcut, fgid, cand_d, cand_c, hist_d, hist_hpos, hist_c, UQ,
         ev, evc, sstart, slen, suser_k, stime_k, pst, pcnt, p_seq, cpst, ccnt, cp_seq, cp_ev,
         deg_i, deg_c, UE, s_gid, s_ts, s_slot, s_user, s_dense, nthr, cap, W, out, aud):
    nq = len(Mq)
    for t in nb.prange(nthr):
        ki = np.full(1024, -1, np.int64); w0 = np.zeros(1024); wdv = np.zeros(1024)
        kc = np.full(1024, -1, np.int64); wc = np.zeros(1024)
        used_i = np.empty(512, np.int64); used_c = np.empty(512, np.int64)
        for q in range(t, nq, nthr):
            u = fuser[q]; fd = ftime[q]; h = H[q]; ho = hoff[q]; ni = 0; ncl = 0
            for j in range(h):
                g = hist_d[ho + j]; w = 0.95 ** hist_hpos[ho + j]
                sl = ((g * 2654435761) >> 11) & 1023
                while ki[sl] != -1 and ki[sl] != g:
                    sl = (sl + 1) & 1023
                if ki[sl] == -1:
                    ki[sl] = g; w0[sl] = w; used_i[ni] = sl; ni += 1
                elif w > w0[sl]:
                    w0[sl] = w
                cc = hist_c[ho + j]
                if cc >= 0:
                    sl = ((cc * 2654435761) >> 11) & 1023
                    while kc[sl] != -1 and kc[sl] != cc:
                        sl = (sl + 1) & 1023
                    if kc[sl] == -1:
                        kc[sl] = cc; wc[sl] = w; used_c[ncl] = sl; ncl += 1
                    elif w > wc[sl]:
                        wc[sl] = w
            for j in range(ni):
                sl = used_i[j]; wdv[sl] = w0[sl] / deg_i[ki[sl]]
            for j in range(ncl):
                sl = used_c[j]; wc[sl] = wc[sl] / deg_c[kc[sl]]
            co = offq[q]
            for i in range(Mq[q]):
                r = co + i; c = cand_d[r]; g = cand_c[r]
                for k in range(8):
                    out[r, k] = 0.0
                # ---------------- A: postings of the ITEM
                if c >= 0:
                    n_occ = 0; last_s = -1; nseq = 0; fw = 0.0; bw = 0.0; sc = 0.0
                    e = pst[c] + pcnt[c] - 1
                    while e >= pst[c] and n_occ < cap:
                        s = p_seq[e]; e -= 1
                        if u != 0 and suser_k[s] == u:
                            continue
                        if stime_k[s] >= fd:
                            continue
                        n_occ += 1
                        if s == last_s:
                            continue
                        last_s = s; nseq += 1
                        aud[t, 0] += 1
                        if s_dense[s]:
                            aud[t, 1] += 1
                            if s_slot[s] >= fcut[q]: aud[t, 3] += 1
                        else:
                            aud[t, 2] += 1
                            if s_ts[s] >= fd: aud[t, 4] += 1
                        if u != 0 and s_user[s] == u: aud[t, 5] += 1
                        if s_ts[s] >= fd: aud[t, 6] += 1
                        gs = s_gid[s]
                        if gs == fgid[q]: aud[t, 7] += 1
                        dd = 0.0
                        for k in range(128):
                            dd += UQ[q, k] * UE[gs, k]
                        sc += dd
                        if ni > 0:
                            st = sstart[s]; L = slen[s]
                            for p in range(L):
                                if ev[st + p] == c:
                                    for d in range(1, W + 1):
                                        if p + d < L:
                                            x = ev[st + p + d]
                                            if x != c:
                                                sl = ((x * 2654435761) >> 11) & 1023
                                                while ki[sl] != -1 and ki[sl] != x:
                                                    sl = (sl + 1) & 1023
                                                if ki[sl] == x: fw += w0[sl] / d
                                        if p - d >= 0:
                                            x = ev[st + p - d]
                                            if x != c:
                                                sl = ((x * 2654435761) >> 11) & 1023
                                                while ki[sl] != -1 and ki[sl] != x:
                                                    sl = (sl + 1) & 1023
                                                if ki[sl] == x: bw += w0[sl] / d
                    out[r, 0] = fw; out[r, 1] = bw; out[r, 2] = sc / (nseq + 5.0); out[r, 3] = nseq
                # ---------------- B: postings of the CELL
                if g >= 0:
                    n_occ = 0; last_s = -1; nseq = 0; t1 = 0.0; t2 = 0.0; sc = 0.0
                    e = cpst[g] + ccnt[g] - 1
                    while e >= cpst[g] and n_occ < cap:
                        s = cp_seq[e]; pe = cp_ev[e]; e -= 1
                        if u != 0 and suser_k[s] == u:
                            continue
                        if stime_k[s] >= fd:
                            continue
                        if c >= 0 and ev[pe] == c:
                            continue
                        n_occ += 1
                        if s == last_s:
                            continue
                        last_s = s; nseq += 1
                        aud[t, 8] += 1
                        if s_dense[s]:
                            aud[t, 9] += 1
                            if s_slot[s] >= fcut[q]: aud[t, 11] += 1
                        else:
                            aud[t, 10] += 1
                            if s_ts[s] >= fd: aud[t, 12] += 1
                        if u != 0 and s_user[s] == u: aud[t, 13] += 1
                        if s_ts[s] >= fd: aud[t, 14] += 1
                        gs = s_gid[s]
                        if gs == fgid[q]: aud[t, 15] += 1
                        dd = 0.0
                        for k in range(128):
                            dd += UQ[q, k] * UE[gs, k]
                        sc += dd
                        if ni > 0:
                            st = sstart[s]; L = slen[s]; a1 = 0.0; a2 = 0.0
                            for p in range(L):
                                gc = evc[st + p]
                                if gc == g:
                                    continue
                                x = ev[st + p]
                                if x == c:
                                    continue
                                sl = ((x * 2654435761) >> 11) & 1023
                                while ki[sl] != -1 and ki[sl] != x:
                                    sl = (sl + 1) & 1023
                                if ki[sl] == x: a1 += wdv[sl]
                                if gc >= 0:
                                    sl = ((gc * 2654435761) >> 11) & 1023
                                    while kc[sl] != -1 and kc[sl] != gc:
                                        sl = (sl + 1) & 1023
                                    if kc[sl] == gc: a2 += wc[sl]
                            t1 += a1 / L; t2 += a2 / L
                    out[r, 4] = t1 / (nseq + 10.0); out[r, 5] = t2 / (nseq + 10.0); out[r, 6] = sc / (nseq + 5.0); out[r, 7] = nseq
            for j in range(ni):
                ki[used_i[j]] = -1
            for j in range(ncl):
                kc[used_c[j]] = -1


AUDN = ['admitted (row, sequence) pairs', 'dense', 'sparse', 'VIOLATION dense slot >= cut', 'VIOLATION sparse timestamp >= own', 'VIOLATION own non-zero user',
        'VIOLATION any member timestamp >= own', 'VIOLATION own feed is a member']


class Ctx:
    """everything label-free that the kernels need besides the nx3 Data object"""

    def __init__(self, D, P):
        self.hist_cell = {s: np.load(CELLS / f'hist_cell_{s}.npy') for s in SPLITS}
        self.cand_cell = {s: np.load(CELLS / f'cand_cell_{s}.npy', mmap_mode='r') for s in SPLITS}
        self.NCELL = int(np.load(CELLS / 'cell_base.npy')[-1])
        for s in SPLITS:
            assert len(self.hist_cell[s]) == len(D.hist_dense[s]) and len(self.cand_cell[s]) == NROWS[s]
        ue = {s: np.load(FLAT / s / 'user_emb.npy') for s in SPLITS}
        mu = ue['train'].astype(np.float64).mean(0)                       # mean train user vector (label-free constant)
        U = np.concatenate([ue[s].astype(np.float32) - mu.astype(np.float32) for s in SPLITS])
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-6)
        self.UE = np.ascontiguousarray(U.astype(np.float32)); self.mu = mu
        assert len(self.UE) == D.A['train']['n'] + D.A['val']['n'] + D.A['test']['n']
        import copy
        self.D8 = copy.copy(D)                                            # the kernel of record with the newest 8 history events as the only seeds
        self.D8.HP = {s: np.where(D.HP[s] < 8, D.HP[s], 30000).astype(np.int16) for s in SPLITS}
        assert 0.95 ** 30000 == 0.0
        P(f'ctx: cells {self.NCELL}; user vectors {self.UE.shape}; history events without a cell: ' + ', '.join(f'{s} {np.mean(self.hist_cell[s] < 0):.4f}' for s in SPLITS))


def run_all(D, X, split, S, sel, fcut):
    """all raw columns for the query feeds `sel` of `split` on index S. fcut = true dense cut slot of each query (for the audit only).
    Returns out [rows, 10] (RAWCOLS), rows, cd, cc, offq, Mq, audit counts [16]."""
    a = D.A[split]; sel = np.asarray(sel, np.int64); Mq = a['M'][sel]; offq = np.concatenate([[0], np.cumsum(Mq)[:-1]]).astype(np.int64)
    rows = np.repeat(a['co'][sel] - offq, Mq) + np.arange(int(Mq.sum()), dtype=np.int64)
    cd = D.cand_dense(D.cand_items(split)[rows]); cc = np.asarray(X.cand_cell[split][rows]).astype(np.int64)
    out = np.zeros((len(rows), 10), np.float64); aud = np.zeros((NT, 16), np.int64)
    deg_i = np.maximum(S.deg_dense, 1.0); deg_c = np.maximum(S.degc_dense, 1.0)
    gq = (D.goff[split] + sel).astype(np.int64)
    kern(Mq, offq, a['ho'][sel], a['H'][sel], a['u'][sel], a['ts'][sel], np.asarray(fcut, np.int64), gq, cd, cc, D.hist_dense[split], D.HP[split], X.hist_cell[split],
         np.ascontiguousarray(X.UE[gq]), S.idx.ev, S.evc, S.idx.starts, S.idx.lens, S.idx.users, S.idx.days, S.idx.pst, S.idx.cnt, S.idx.p_seq,
         S.cpst, S.ccnt, S.cp_seq, S.cp_ev, deg_i, deg_c, X.UE, S.s_gid, S.s_ts, S.s_slot, S.s_user, S.s_dense, NT, CAP, W20, out, aud)
    deg2 = S.deg_floor1()
    out[:, 8] = run_kernel(D, split, S.idx, deg2, sel, np.zeros(len(sel), np.int64), a['ts'][sel])[0][:, 0]        # the walk of record (driver check against nx3's raw arrays)
    out[:, 9] = run_kernel(X.D8, split, S.idx, deg2, sel, np.zeros(len(sel), np.int64), a['ts'][sel])[0][:, 0]     # N5: newest 8 events as seeds
    return out, rows, cd, cc, offq, Mq, aud.sum(0)


def brute2(D, X, split, S, f, cut_slot):
    """independent numpy re-derivation of the 8 kern columns for every candidate of feed f (true timestamps / slots / users; only the posting lists are shared)"""
    a = D.A[split]; idx = S.idx; u = a['u'][f]; t = a['ts'][f]
    hs = slice(a['ho'][f], a['ho'][f] + a['H'][f])
    hd = D.hist_dense[split][hs].astype(np.int64); hp = D.HP[split][hs].astype(np.int64); hcl = X.hist_cell[split][hs].astype(np.int64)
    w0 = {}; wcl = {}
    for g, p, cc in zip(hd.tolist(), hp.tolist(), hcl.tolist()):
        w = 0.95 ** p
        if w > w0.get(g, 0.0): w0[g] = w
        if cc >= 0 and w > wcl.get(cc, 0.0): wcl[cc] = w
    deg = np.maximum(S.deg_dense, 1.0); degc = np.maximum(S.degc_dense, 1.0)
    hk = np.array(sorted(w0), np.int64); hwd = np.array([w0[g] / deg[g] for g in sorted(w0)], np.float64)
    ck = np.array(sorted(wcl), np.int64); cwd = np.array([wcl[g] / degc[g] for g in sorted(wcl)], np.float64)

    def look(keys, vals, q):
        if len(keys) == 0 or len(q) == 0: return 0.0
        j = np.searchsorted(keys, q); j[j >= len(keys)] = 0
        return float(vals[j][keys[j] == q].sum())
    rows = a['co'][f] + np.arange(a['M'][f]); cds = D.cand_dense(D.cand_items(split)[rows]); ccs = np.asarray(X.cand_cell[split][rows]).astype(np.int64)
    uq = X.UE[D.goff[split] + f].astype(np.float64); res = np.zeros((len(rows), 8))

    def admit(seqs):
        adm = np.where(S.s_dense[seqs], S.s_slot[seqs] < cut_slot, S.s_ts[seqs] < t)
        if u != 0: adm &= S.s_user[seqs] != u
        return adm
    for i, (c, g) in enumerate(zip(cds.tolist(), ccs.tolist())):
        if c >= 0:
            seqs = idx.p_seq[idx.pst[c]:idx.pst[c] + idx.cnt[c]].astype(np.int64); seqs = seqs[admit(seqs)][-CAP:]; us = np.unique(seqs); fw = bw = sc = 0.0
            for s in us.tolist():
                ev = idx.ev[idx.starts[s]:idx.starts[s] + idx.lens[s]].astype(np.int64); sc += float(uq @ X.UE[S.s_gid[s]].astype(np.float64))
                for p in np.nonzero(ev == c)[0].tolist():
                    for d in range(1, W20 + 1):
                        if p + d < len(ev) and ev[p + d] != c: fw += w0.get(int(ev[p + d]), 0.0) / d
                        if p - d >= 0 and ev[p - d] != c: bw += w0.get(int(ev[p - d]), 0.0) / d
            res[i, :4] = fw, bw, sc / (len(us) + 5.0), len(us)
        if g >= 0:
            sl = slice(S.cpst[g], S.cpst[g] + S.ccnt[g]); seqs = S.cp_seq[sl].astype(np.int64); pev = S.cp_ev[sl].astype(np.int64)
            adm = admit(seqs)
            if c >= 0: adm &= idx.ev[pev] != c
            seqs = seqs[adm][-CAP:]; us = np.unique(seqs); t1 = t2 = sc = 0.0
            for s in us.tolist():
                ev = idx.ev[idx.starts[s]:idx.starts[s] + idx.lens[s]].astype(np.int64); ec = S.evc[idx.starts[s]:idx.starts[s] + idx.lens[s]].astype(np.int64)
                m = (ec != g) & (ev != c)
                t1 += look(hk, hwd, ev[m]) / len(ev)
                t2 += look(ck, cwd, ec[m][ec[m] >= 0]) / len(ev)
                sc += float(uq @ X.UE[S.s_gid[s]].astype(np.float64))
            res[i, 4:] = t1 / (len(us) + 10.0), t2 / (len(us) + 10.0), sc / (len(us) + 5.0), len(us)
    return res
