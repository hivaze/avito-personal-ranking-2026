"""nx3 STEP 0 -- INDEPENDENT VERIFIER, shared code (my own; nothing of nn_x/w12/nx3/*.py and nothing of c09_build.py is imported).
First-principles recomputation of the whole-sequence RP3beta walk straight from data/flat/<split>/{feeds.parquet,hist_item_id,hist_hpos,cand_item_id}.

Method (deliberately different from the kernel of record, which walks an inverted index candidate by candidate):
  SEQUENCE-CENTRIC SCAN. The source sequences of one index ("universe") are visited one by one in REVERSE index order; for every event of a
  sequence the query rows that hold this item as a candidate are looked up; for every (query row, sequence) pair that the source rules admit the
  contribution  sum_{events x of s, x != c, x in the query feed's history} 0.95^hpos_min(x) / deg(x)  / len(s)  is added.  No posting list is built.
  "latest 500 source occurrences" (the kernel's cap): a sequence contributes to a row iff fewer than 500 admitted occurrences of the candidate were
  seen in the sequences visited before it (reverse index order: sparse parts last-to-first, then dense part by descending fidx).
Source rules (BUILD_SPEC.txt 2, as amended; the rules under test):
  dense member   train feed with slot < cut: the LATEST feed (timestamp, then fidx) of each non-zero user + EVERY user-0 feed; always admitted
  sparse member  listed feed; admitted iff its timestamp < the query feed's timestamp (strict)
  same user      every sequence of the query's own non-zero user is skipped;  user 0 is never linked (never de-duplicated, never skipped)
  deg(x)         number of events (postings) of x in the DENSE part, floor 1
Variants accumulated in the same scan (bit v of the admission mask):
  V0  the rules above (admissible sources only)
  V1  + sequences of the query's own non-zero user (time rule kept)                     -> "same user's snapshots"
  V2  + sparse members dated AT OR AFTER the query feed (user rule kept)                -> "source at or after the row's cut / time"
  V3  user 0 LINKED: for a user-0 query every user-0 sequence is skipped                -> "user 0 linked"
CPU only, 12 threads at most (the scan itself is single-threaded)."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
VD = '<repo>/nn_x/w12/nx3/verify'
os.environ['NUMBA_CACHE_DIR'] = VD + '/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import numba as nb
import polars as pl

ROOT = '<repo>'
FLAT = ROOT + '/data/flat'
OUT = VD + '/out'
os.makedirs(OUT, exist_ok=True)
MSK = 3 * 3600
DAY0 = 20702                  # MSK day of Sep 6 = slot 0 (BUILD_SPEC 0)
PUB_TS = 1790497920           # public / private boundary of test (hz2_common.PUB_TS = src/calib.py)
CAP = 500
NV = 4
NROWS = {'train': 246745836, 'val': 1389971, 'test': 2904806}
H_SP = np.uint64(0xC2B2AE3D27D4EB4F)   # the shared 1/37 sample (constant of hz2_common.sparse_flag; cross-checked against the import in v01)
T0 = time.time()


class Log:
    def __init__(self, path):
        self.f = open(path, 'x')

    def __call__(self, *a):
        s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()


def sparse_flag(feed_id):
    h = (np.asarray(feed_id).astype(np.uint64) * H_SP) >> np.uint64(40)
    return (h % np.uint64(37)) == 0


class Split:
    def __init__(self, name, hist=True):
        f = pl.read_parquet(f'{FLAT}/{name}/feeds.parquet')
        self.name = name; self.n = len(f)
        assert (f['fidx'].to_numpy() == np.arange(self.n)).all()
        self.fid = f['feed_id'].to_numpy()
        self.ts = f['timestamp'].to_numpy().astype(np.int64)
        self.user = f['user_id'].to_numpy().astype(np.int64)
        self.M = f['M'].to_numpy().astype(np.int64); self.H = f['H'].to_numpy().astype(np.int64)
        self.co = f['cand_off'].to_numpy().astype(np.int64); self.ho = f['hist_off'].to_numpy().astype(np.int64)
        self.slot = (self.ts + MSK) // 86400 - DAY0
        assert int(self.M.sum()) == NROWS[name]
        assert (self.co == np.concatenate([[0], np.cumsum(self.M)[:-1]])).all() and (self.ho == np.concatenate([[0], np.cumsum(self.H)[:-1]])).all()
        assert (np.diff(self.ts) >= 0).all()          # fidx order is time order: "latest feed of a user" = its largest fidx
        if hist:
            self.hist = np.load(f'{FLAT}/{name}/hist_item_id.npy'); self.hpos = np.load(f'{FLAT}/{name}/hist_hpos.npy')
            assert len(self.hist) == int(self.H.sum()) and self.hpos.dtype == np.int16 and self.hist.dtype == np.int32
        self.cand = np.load(f'{FLAT}/{name}/cand_item_id.npy', mmap_mode='r')
        assert len(self.cand) == NROWS[name]


def dense_members(TR, cut_slot):
    """my own selection: train feeds with slot < cut_slot; latest feed of each non-zero user by (timestamp, fidx) + every user-0 feed. Sorted by fidx."""
    idx = np.nonzero(TR.slot < cut_slot)[0]
    u = TR.user[idx]
    nz = idx[u != 0]; un = TR.user[nz]
    o = np.lexsort((nz, TR.ts[nz], un))            # user, then timestamp, then fidx
    us = un[o]; last = np.ones(len(o), bool); last[:-1] = us[1:] != us[:-1]
    keep = nz[o][last]
    out = np.sort(np.concatenate([keep, idx[u == 0]])).astype(np.int64)
    return out


@nb.njit(cache=True)
def _gather(sel, ho, H, src, off, out):
    for q in range(len(sel)):
        f = sel[q]; o = off[q]; a = ho[f]
        for j in range(H[f]):
            out[o + j] = src[a + j]


class Universe:
    """parts: list of (Split, fidx array, dense flag) in INDEX ORDER (dense part first)."""

    def __init__(self, parts, maxid):
        evs, lens, ts, slot, user, dense, gid = [], [], [], [], [], [], []
        for S, sel, dn in parts:
            sel = np.asarray(sel, np.int64); h = S.H[sel]; off = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64)
            ev = np.empty(int(h.sum()), np.int32); _gather(sel, S.ho, S.H, S.hist, off, ev)
            evs.append(ev); lens.append(h); ts.append(S.ts[sel]); slot.append(S.slot[sel]); user.append(S.user[sel]); dense.append(np.full(len(sel), dn, np.bool_))
            gid.append({'train': 0, 'val': 10_000_000, 'test': 20_000_000}[S.name] + sel)
        self.ev = np.concatenate(evs); self.lens = np.concatenate(lens).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]]).astype(np.int64)
        self.ts = np.concatenate(ts); self.slot = np.concatenate(slot); self.user = np.concatenate(user); self.dense = np.concatenate(dense); self.gid = np.concatenate(gid)
        nd = int(self.dense.sum()); assert self.dense[:nd].all()
        self.nd = nd; self.nde = int(self.lens[:nd].sum())
        self.deg = np.bincount(self.ev[:self.nde], minlength=maxid + 1)      # events of the item in the DENSE part (int64 counts over raw item ids)


@nb.njit(cache=True)
def _pw(hp):
    out = np.empty(len(hp), np.float64)
    for i in range(len(hp)):
        out[i] = 0.95 ** hp[i]
    return out


class Queries:
    def __init__(self, S, qf, cut, maxid):
        qf = np.asarray(qf, np.int64); self.S = S; self.qf = qf; self.nq = len(qf)
        M = S.M[qf]; self.M = M; self.roff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); self.nrow = int(M.sum())
        self.rows_flat = np.repeat(S.co[qf] - self.roff, M) + np.arange(self.nrow, dtype=np.int64)      # row index in data/flat/<split>/cand_*.npy
        self.row_item = np.asarray(S.cand[self.rows_flat]).astype(np.int64)
        self.row_q = np.repeat(np.arange(self.nq, dtype=np.int64), M)
        H = S.H[qf]; hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64); nh = int(H.sum())
        hidx = np.repeat(S.ho[qf] - hoff, H) + np.arange(nh, dtype=np.int64)
        h_item = S.hist[hidx].astype(np.int64); h_pos = S.hpos[hidx]; h_q = np.repeat(np.arange(self.nq, dtype=np.int64), H)
        w = _pw(h_pos)
        self.items = np.unique(np.concatenate([self.row_item, h_item])); ns = len(self.items)
        self.look = np.full(maxid + 1, -1, np.int32); self.look[self.items] = np.arange(ns, dtype=np.int32)
        self.row_slot = self.look[self.row_item].astype(np.int64)
        self.c_row = np.argsort(self.row_slot, kind='stable').astype(np.int64)
        self.cptr = np.concatenate([[0], np.cumsum(np.bincount(self.row_slot, minlength=ns))]).astype(np.int64)
        hs = self.look[h_item].astype(np.int64); key = hs * self.nq + h_q
        o = np.argsort(key, kind='stable'); ks = key[o]
        if len(ks):
            first = np.ones(len(ks), bool); first[1:] = ks[1:] != ks[:-1]; st = np.nonzero(first)[0]
            self.h_wmax = np.maximum.reduceat(w[o], st); uk = ks[st]
        else:
            self.h_wmax = np.zeros(0); uk = np.zeros(0, np.int64)
        self.h_slot = uk // self.nq; self.h_q = (uk % self.nq).astype(np.int64)
        self.hptr = np.concatenate([[0], np.cumsum(np.bincount(self.h_slot, minlength=ns))]).astype(np.int64)
        self.q_ts = S.ts[qf]; self.q_user = S.user[qf]; self.q_cut = np.asarray(cut, np.int64); self.q_gid = {'train': 0, 'val': 10_000_000, 'test': 20_000_000}[S.name] + qf


@nb.njit(cache=True)
def _scan(ev, sstart, slen, s_ts, s_slot, s_user, s_dense, s_gid, look, cptr, c_row, hptr, h_q, h_w, row_q, row_slot, q_ts, q_cut, q_user, q_gid, cap,
          tot, nseq, nocc, newest_d, newest_s, tot_sparse, wts, cnt):
    """cnt: [0] admitted V0 pairs, [1] dense, [2] sparse, [3] V0-admitted dense with slot >= cut, [4] V0-admitted sparse with ts >= own,
            [5] V0-admitted sequence of the own non-zero user, [6] V0-admitted own feed, [7] V0-admitted any member with ts >= own (dense members included)"""
    nrow = len(row_q)
    stamp = np.full(nrow, -1, np.int64)
    mask = np.zeros(nrow, np.uint8)
    touched = np.empty(nrow, np.int64)
    hb = np.empty(4096, np.int64)
    for s in range(len(slen) - 1, -1, -1):
        st = sstart[s]; L = slen[s]; nt = 0
        dn = s_dense[s]; sts = s_ts[s]; su = s_user[s]
        for j in range(L):
            sl = look[ev[st + j]]
            if sl < 0:
                continue
            for k in range(cptr[sl], cptr[sl + 1]):
                r = c_row[k]; q = row_q[r]
                a_time = dn or (sts < q_ts[q])
                same_u = (q_user[q] != 0) and (su == q_user[q])
                b0 = a_time and (not same_u)
                b1 = a_time
                b2 = not same_u
                b3 = b0 and (not (q_user[q] == 0 and su == 0))
                if stamp[r] != s:
                    stamp[r] = s; m = 0
                    if b0 and nocc[0, r] < cap: m |= 1
                    if b1 and nocc[1, r] < cap: m |= 2
                    if b2 and nocc[2, r] < cap: m |= 4
                    if b3 and nocc[3, r] < cap: m |= 8
                    mask[r] = m
                    if m != 0:
                        touched[nt] = r; nt += 1
                if b0: nocc[0, r] += 1
                if b1: nocc[1, r] += 1
                if b2: nocc[2, r] += 1
                if b3: nocc[3, r] += 1
        if nt == 0:
            continue
        nh = 0
        for j in range(L):
            sl = look[ev[st + j]]
            if sl >= 0 and hptr[sl + 1] > hptr[sl]:
                hb[nh] = sl; nh += 1
        for i in range(nt):
            r = touched[i]; q = row_q[r]; cs = row_slot[r]; m = mask[r]
            acc = 0.0
            for e in range(nh):
                sl = hb[e]
                if sl == cs:
                    continue
                lo = hptr[sl]; hi = hptr[sl + 1]
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if h_q[mid] < q: lo = mid + 1
                    else: hi = mid
                if lo < hptr[sl + 1] and h_q[lo] == q:
                    acc += h_w[lo]
            c = acc / L
            if m & 1:
                tot[0, r] += c; nseq[0, r] += 1
                cnt[0] += 1
                if dn:
                    cnt[1] += 1
                    if s_slot[s] >= q_cut[q]: cnt[3] += 1
                else:
                    cnt[2] += 1
                    if sts >= q_ts[q]: cnt[4] += 1
                if q_user[q] != 0 and su == q_user[q]: cnt[5] += 1
                if s_gid[s] == q_gid[q]: cnt[6] += 1
                if sts >= q_ts[q]: cnt[7] += 1
                if acc > 0.0:
                    if dn:
                        if sts > newest_d[r]: newest_d[r] = sts
                    else:
                        if sts > newest_s[r]: newest_s[r] = sts
                        tot_sparse[r] += c
                    wts[r] += c * sts
            if m & 2:
                tot[1, r] += c; nseq[1, r] += 1
            if m & 4:
                tot[2, r] += c; nseq[2, r] += 1
            if m & 8:
                tot[3, r] += c; nseq[3, r] += 1


def run(U, Q):
    """returns dict of per-row arrays (row order = Q.rows_flat) and the audit counts of the V0 pairs"""
    degf = np.maximum(U.deg, 1).astype(np.float64)
    h_w = Q.h_wmax / degf[Q.items[Q.h_slot]]
    tot = np.zeros((NV, Q.nrow)); nseq = np.zeros((NV, Q.nrow), np.int64); nocc = np.zeros((NV, Q.nrow), np.int64)
    newest_d = np.full(Q.nrow, -1, np.int64); newest_s = np.full(Q.nrow, -1, np.int64); tot_sparse = np.zeros(Q.nrow); wts = np.zeros(Q.nrow); cnt = np.zeros(8, np.int64)
    assert int(U.lens.max()) <= 4096
    _scan(U.ev, U.starts, U.lens, U.ts, U.slot, U.user, U.dense, U.gid, Q.look, Q.cptr, Q.c_row, Q.hptr, Q.h_q, h_w, Q.row_q, Q.row_slot, Q.q_ts, Q.q_cut, Q.q_user, Q.q_gid, CAP,
          tot, nseq, nocc, newest_d, newest_s, tot_sparse, wts, cnt)
    rp3 = tot / np.sqrt(degf[Q.row_item])[None, :]
    return dict(rp3=rp3, p3a=tot, nseq=nseq, nocc=nocc, newest_d=newest_d, newest_s=newest_s, tot_sparse=tot_sparse, wts=wts, cnt=cnt, deg_c=U.deg[Q.row_item])


# ---------------------------------------------------------------- entered forms, my own implementation (BUILD_SPEC 2.3)
@nb.njit(cache=True)
def my_forms(x, off, M, K):
    """z = within-feed z (population sd) of log1p(K x) over all candidates, 0 on a feed without spread, clipped to +-5; rk = average-tie rank / (M - 1) - 0.5"""
    z = np.zeros(len(x), np.float64); rk = np.zeros(len(x), np.float64)
    for f in range(len(M)):
        o = off[f]; m = M[f]
        v = np.log1p(K * x[o:o + m])
        mu = v.mean(); var = ((v - mu) ** 2).mean()
        if m >= 2 and var > 1e-18:
            sd = np.sqrt(var)
            for i in range(m):
                t = (v[i] - mu) / sd
                z[o + i] = min(5.0, max(-5.0, t))
        if m >= 2:
            xv = x[o:o + m]
            for i in range(m):
                lt = 0; eq = 0
                for j in range(m):
                    if xv[j] < xv[i]: lt += 1
                    elif xv[j] == xv[i]: eq += 1
                rk[o + i] = (lt + 0.5 * (eq - 1)) / (m - 1) - 0.5
    return z, rk
