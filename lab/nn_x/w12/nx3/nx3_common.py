"""nx3 STEP 0 (rule: ledger nn_x/w5/logs/E_kd_predeclared.txt block "TRAINED-INPUT TEST nx3"; recipe reports/INPUTS_1116/design/BUILD_SPEC.txt sections 0-2).
Shared code of the builder of record. CPU only, 12 threads. Read-only on every existing file; writes only under nn_x/w12/nx3/ (exclusive create).

The KERNEL OF RECORD is imported unchanged: functions rp3, Index, source_fidx, day_of of nn_x/insights/ccb_1113/scratch/tester3/c09_build.py.
Everything else here is the DRIVER (which sequences are members, what the kernel is handed as "time" and as "degree") and independent checks.

Driver conventions of this build (BUILD_SPEC 2):
  dense members   = c9.source_fidx(train, cutoff day): latest feed of each non-zero user + every user-0 feed with day < cutoff; kernel time -1 (always admitted)
  sparse members  = listed feeds with their TIMESTAMP as kernel time; the query's kernel time is its own TIMESTAMP, so the kernel's test
                    `sday[s] >= fd -> skip` admits a sparse member iff timestamp < own timestamp (the kernel is unchanged; only the unit differs from c09's days)
  degree          = posting count of the item in the DENSE part of the index, floor 1 (sparse members are paths only), one row for every query of the index
  same user       = the kernel skips every sequence of the query's own non-zero user; user 0 is never linked (each user-0 feed is its own sequence)
"""
import os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
NX = '<repo>/nn_x/w12/nx3'
os.environ['NUMBA_CACHE_DIR'] = NX + '/out/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb            # imported first: fixes the numba cache dir to nn_x/w12/nx3/out/numba_cache for every module imported later
import numpy as np
import polars as pl
from pathlib import Path
T3 = '<repo>/nn_x/insights/ccb_1113/scratch/tester3'
sys.path.insert(0, T3)
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import c09_build as c9        # kernel of record (read-only import; main() is not run)
import hz2_common as hc

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; OUT = Path(NX) / 'out'; C9 = Path(T3) / 'cache'
SPLITS = ('train', 'val', 'test')
NT = 12
CAP = 500
NROWS = {'train': 246745836, 'val': 1389971, 'test': 2904806}
T0 = time.time()
TAG = os.environ.get('NX3_TAG', '')     # '' for the build of record; a non-empty tag gives separate output names (a re-run never overwrites)
FILES_DIR = OUT / f'files{TAG}'; RAW_DIR = OUT / f'raw{TAG}'; TWIN_DIR = Path(NX) / f'staleval{TAG}'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class Log:
    def __init__(self, path):
        self.f = open(path, 'x')          # exclusive create: a log of record is never overwritten

    def __call__(self, *a):
        s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()


def save_new(path, x):
    with open(path, 'xb') as f:           # exclusive create
        np.save(f, x)


class Data:
    """feeds tables, dense item ids over all history items of the three splits, history arrays (as critic/c01_lagwalk.py; the id map does not enter any value)"""

    def __init__(self, P):
        nb.set_num_threads(NT)
        self.F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
        H_ = {s: np.load(FLAT / s / 'hist_item_id.npy') for s in SPLITS}
        self.mx = max(int(h.max()) for h in H_.values()) + 1
        self.pres = np.zeros(self.mx, np.bool_)
        for h in H_.values():
            self.pres[h] = True
        self.lut = (np.cumsum(self.pres, dtype=np.int64) - 1).astype(np.int32); self.G = int(self.pres.sum())
        self.hist_dense = {s: self.lut[H_[s]] for s in SPLITS}
        del H_
        self.HP = {s: np.asarray(np.load(FLAT / s / 'hist_hpos.npy')).astype(np.int16) for s in SPLITS}
        self.A = {}
        for s in SPLITS:
            f = self.F[s]; ts = f['timestamp'].to_numpy().astype(np.int64)
            assert (f['fidx'].to_numpy() == np.arange(len(f))).all()
            self.A[s] = dict(M=f['M'].to_numpy().astype(np.int64), H=f['H'].to_numpy().astype(np.int64), co=f['cand_off'].to_numpy().astype(np.int64),
                             ho=f['hist_off'].to_numpy().astype(np.int64), u=f['user_id'].to_numpy().astype(np.int64), ts=ts, day=c9.day_of(ts),
                             fid=f['feed_id'].to_numpy(), n=len(f))
            self.A[s]['slot'] = (self.A[s]['day'] - hc.DAY0).astype(np.int64)
            assert int(self.A[s]['M'].sum()) == NROWS[s]
            assert (self.A[s]['co'] == np.concatenate([[0], np.cumsum(self.A[s]['M'])[:-1]])).all()
        self.goff = {'train': 0, 'val': self.A['train']['n'], 'test': self.A['train']['n'] + self.A['val']['n']}   # global feed id = goff + fidx
        self.cand = {}
        P(f'data: dense ids {self.G} items (max id {self.mx - 1}); feeds train {self.A["train"]["n"]} val {self.A["val"]["n"]} test {self.A["test"]["n"]}')

    def cand_items(self, split):
        if split not in self.cand:
            self.cand[split] = np.load(FLAT / split / 'cand_item_id.npy')
        return self.cand[split]

    def cand_dense(self, items):
        items = np.asarray(items).astype(np.int64); ok = items < self.mx
        out = np.full(len(items), -1, np.int64); out[ok] = np.where(self.pres[items[ok]], self.lut[items[ok]], -1)
        return out


class Src:
    """one index of the kernel of record + the TRUE attributes of its sequences, gathered here from the feeds tables (not from the index object)"""

    def __init__(self, D, parts):
        """parts: list of (split, fidx array, kernel time array int64 (-1 = dense, always admitted), dense flag)"""
        self.idx = c9.Index([(sp, sel, kt) for sp, sel, kt, _ in parts], D.G, D.hist_dense, D.F)
        self.s_ts = np.concatenate([D.A[sp]['ts'][sel] for sp, sel, _, _ in parts]).astype(np.int64)
        self.s_slot = np.concatenate([D.A[sp]['slot'][sel] for sp, sel, _, _ in parts]).astype(np.int64)
        self.s_user = np.concatenate([D.A[sp]['u'][sel] for sp, sel, _, _ in parts]).astype(np.int64)
        self.s_gid = np.concatenate([D.goff[sp] + np.asarray(sel, np.int64) for sp, sel, _, _ in parts]).astype(np.int64)
        self.s_dense = np.concatenate([np.full(len(sel), dn, np.bool_) for _, sel, _, dn in parts])
        self.n_dense = int(self.s_dense.sum()); self.n_sparse = int((~self.s_dense).sum())
        assert self.s_dense[:self.n_dense].all()                       # dense parts come first
        nde = int(self.idx.lens[:self.n_dense].sum())
        self.deg_dense = np.bincount(self.idx.ev[:nde], minlength=D.G).astype(np.float64)   # posting count over the dense part only
        assert len(self.s_ts) == len(self.idx.lens)

    def deg_floor1(self):
        return np.maximum(self.deg_dense, 1.0)[None, :]


def run_kernel(D, split, idx, deg2, sel, fslot, ftime):
    """the kernel of record on query feeds `sel` of `split`. Returns (out [rows, 3] = rp3b05, p3a, nseq), flat rows, dense candidate ids, local offsets, M)."""
    a = D.A[split]; sel = np.asarray(sel, np.int64); Mq = a['M'][sel]; offq = np.concatenate([[0], np.cumsum(Mq)[:-1]]).astype(np.int64)
    rows = np.repeat(a['co'][sel] - offq, Mq) + np.arange(int(Mq.sum()), dtype=np.int64)
    cd = D.cand_dense(D.cand_items(split)[rows])
    out = np.zeros((len(rows), 3), np.float64); q = np.arange(len(sel), dtype=np.int64)
    c9.rp3(q, offq, Mq, a['ho'][sel], a['H'][sel], a['u'][sel], np.asarray(fslot, np.int64), cd, D.hist_dense[split], D.HP[split], idx.ev, idx.starts, idx.lens,
           idx.users, idx.days, np.asarray(ftime, np.int64), idx.pst, idx.cnt, idx.p_seq, deg2, D.G, NT, CAP, out)
    return out, rows, cd, offq, Mq


# ---------------------------------------------------------------- C2: causality audit (my code; walks the postings with the kernel's admit rule and
# checks every ADMITTED (row, sequence) pair against the true attributes of the sequence and of the query feed)
ANAMES = ['admitted (row, sequence) pairs', 'of them dense', 'of them sparse', 'VIOLATION dense member with slot >= cut', 'VIOLATION sparse member with timestamp >= own',
          'VIOLATION sequence of the own non-zero user', 'VIOLATION any member with timestamp >= own', 'VIOLATION own feed is a member']


@nb.njit(parallel=True, cache=True)
def audit(coff, M, Hq, fuser_k, ftime_k, cand_d, suser_k, stime_k, pst, pcnt, p_seq, cap, fts, fcut, fuser, fgid, s_ts, s_slot, s_user, s_dense, s_gid, nthr, cnt):
    for t in nb.prange(nthr):
        for q in range(t, len(M), nthr):
            if Hq[q] == 0:
                continue
            u = fuser_k[q]; fd = ftime_k[q]; co = coff[q]
            for i in range(M[q]):
                c = cand_d[co + i]
                if c < 0:
                    continue
                n_occ = 0; last_s = -1
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
                    last_s = s
                    cnt[t, 0] += 1
                    if s_dense[s]:
                        cnt[t, 1] += 1
                        if s_slot[s] >= fcut[q]:
                            cnt[t, 3] += 1
                    else:
                        cnt[t, 2] += 1
                        if s_ts[s] >= fts[q]:
                            cnt[t, 4] += 1
                    if fuser[q] != 0 and s_user[s] == fuser[q]:
                        cnt[t, 5] += 1
                    if s_ts[s] >= fts[q]:
                        cnt[t, 6] += 1
                    if s_gid[s] == fgid[q]:
                        cnt[t, 7] += 1


def run_audit(D, split, S, sel, ftime_k, fcut, offq, Mq, cd):
    a = D.A[split]; sel = np.asarray(sel, np.int64); cnt = np.zeros((NT, 8), np.int64)
    audit(offq, Mq, a['H'][sel], a['u'][sel], np.asarray(ftime_k, np.int64), cd, S.idx.users, S.idx.days, S.idx.pst, S.idx.cnt, S.idx.p_seq, CAP,
          a['ts'][sel], np.asarray(fcut, np.int64), a['u'][sel], D.goff[split] + sel, S.s_ts, S.s_slot, S.s_user, S.s_dense, S.s_gid, NT, cnt)
    return cnt.sum(0)


# ---------------------------------------------------------------- independent re-derivation of the values on a sample (numpy; the SPEC's formula with
# the true timestamps / slots / users; only the posting lists of the index are shared with the kernel)
def brute(D, split, S, f, cut_slot, mode='ts'):
    """(rp3b05, p3a) of every candidate of feed f by BUILD_SPEC 2: members = dense sequences with slot < cut_slot + sparse sequences with timestamp < own
    timestamp (mode 'ts'); every sequence of the own non-zero user skipped; latest CAP valid occurrences; deg = dense posting count, floor 1."""
    a = D.A[split]; idx = S.idx; u = a['u'][f]; t = a['ts'][f]
    hd = D.hist_dense[split][a['ho'][f]:a['ho'][f] + a['H'][f]].astype(np.int64); hp = D.HP[split][a['ho'][f]:a['ho'][f] + a['H'][f]].astype(np.int64)
    deg = np.maximum(S.deg_dense, 1.0)
    wd = {}
    for g, p in zip(hd.tolist(), hp.tolist()):
        w = 0.95 ** p
        if w > wd.get(g, 0.0): wd[g] = w
    hk = np.array(sorted(wd), np.int64); hw = np.array([wd[g] / deg[g] for g in sorted(wd)], np.float64)
    cds = D.cand_dense(D.cand_items(split)[a['co'][f]:a['co'][f] + a['M'][f]])
    res = np.zeros((len(cds), 2))
    ok_seq_cache = {}
    for i, c in enumerate(cds.tolist()):
        if c < 0 or len(hk) == 0: continue
        seqs = idx.p_seq[idx.pst[c]:idx.pst[c] + idx.cnt[c]].astype(np.int64)
        if len(seqs) == 0: continue
        adm = np.where(S.s_dense[seqs], S.s_slot[seqs] < cut_slot, S.s_ts[seqs] < t)
        if u != 0: adm &= S.s_user[seqs] != u
        seqs = seqs[adm][-CAP:]
        tot = 0.0
        for s in np.unique(seqs).tolist():
            ev = idx.ev[idx.starts[s]:idx.starts[s] + idx.lens[s]].astype(np.int64)
            ev = ev[ev != c]
            j = np.searchsorted(hk, ev); j[j >= len(hk)] = 0
            tot += float(hw[j][hk[j] == ev].sum()) / float(idx.lens[s])
        res[i, 1] = tot; res[i, 0] = tot / np.sqrt(deg[c])
    return res


def forms_of(x, off, M, K, feed_z, feed_rk):
    """BUILD_SPEC 2.3: z = within-feed z of log1p(K x) over all candidates (zeros are values; constant feed -> 0), clipped to +-5; rk = within-feed mid-rank percentile - 0.5"""
    x = np.asarray(x, np.float64)
    return feed_z(np.log1p(K * x), off, M, 5.0), feed_rk(x, off, M)
