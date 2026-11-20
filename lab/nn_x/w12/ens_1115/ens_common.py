"""Shared loaders for the mega-ensemble study (nn_x/w12/ens_1115; CPU only, read-only on every prediction file).
Conventions copied from nn_x/w12/recheck/pool_composition/pc_common.py and nn_x/w12/kd/build_r4.py (neither is edited):
  member score = per-feed z = stack_val.feed_rank_z[:, 1] ((x - mean) / sd inside the feed);
  pool = member-equal mean of z; file = 0.5 z(FT pool) + 0.5 z(TO pool); per branch (1 - w) z(base pool) + w z(popularity-light pool).
Five kinds of prediction per member: to_val, to_test (train-only), ft_val, ft_test (n2cpv3s3 fine-tune; val = OOF), hold (30,000-feed in-period holdout)."""
import os, sys, json, re
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); W = ROOT / 'nn_x/w5/preds'; AB = ROOT / 'nn_x/abl/preds'; P = ROOT / 'preds'
HERE = ROOT / 'nn_x/w12/ens_1115'; CACHE = HERE / 'cache'; OUT = HERE / 'out'; STAGE = HERE / 'stage'
KINDS = ('to_val', 'to_test', 'ft_val', 'ft_test', 'hold')
SUF = dict(to_val='trainonly_val', to_test='trainonly_test', ft_val='n2cpv3s3_val_oof', ft_test='n2cpv3s3_test', hold='holdout')
SPLIT = dict(to_val='val', to_test='test', ft_val='val', ft_test='test', hold='hold')


def find(name, kind):
    """All existing copies of one prediction file, preds/ first (the directory build_r4.py reads)."""
    out = []
    for d in (P, W, AB):
        f = (d / 'holdout' / f'nn_{name}_holdout.npy') if kind == 'hold' else d / f'nn_{name}_{SUF[kind]}.npy'
        if f.exists(): out.append(f)
    return out


class Meta:
    def __init__(self):
        fv, self.Mv, self.ov = feed_meta('val'); ft, self.Mt, self.ot = feed_meta('test')
        self.fv, self.ft = fv, ft
        self.ov = np.asarray(self.ov, np.int64); self.ot = np.asarray(self.ot, np.int64)
        self.yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
        Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); self.Mh = Mh.astype(np.int64); self.oh = oh.astype(np.int64)
        self.yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
        self.n = dict(val=int(self.Mv.sum()), hold=int(self.Mh.sum()), test=int(self.Mt.sum()))
        self.M = dict(val=self.Mv, hold=self.Mh, test=self.Mt); self.off = dict(val=self.ov, hold=self.oh, test=self.ot)
        self.y = dict(val=self.yv, hold=self.yh)
        self.fid = {sp: np.repeat(np.arange(len(self.M[sp])), self.M[sp]) for sp in ('val', 'hold', 'test')}
        self.uid = fv['user_id'].to_numpy().astype(np.int64)

    def z(self, sp, x):
        return feed_rank_z(self.off[sp], self.M[sp], np.asarray(x, np.float32))[:, 1]

    def rk(self, sp, x):
        """Within-feed rank in [0, 1] (stack_val.feed_rank_z[:, 0]), then z-scored inside the feed."""
        r = feed_rank_z(self.off[sp], self.M[sp], np.asarray(x, np.float32))[:, 0]
        return feed_rank_z(self.off[sp], self.M[sp], r)[:, 1]

    def ev(self, s): return ndcg_fast(s, self.yv, self.ov, self.Mv)
    def eh(self, s): return ndcg_fast(s, self.yh, self.oh, self.Mh)

    def ft_folds(self, k=5):
        """The fine-tune's own partition: GroupKFold(5) by user, user 0 feeds as singletons (nn/finetune.py --u0_singleton). Returns fold id per val feed."""
        from sklearn.model_selection import GroupKFold
        g = self.uid.copy(); u0 = g == 0; g = np.where(u0, -1 - np.arange(len(g)), g)
        fold = np.full(len(g), -1, np.int64)
        for i, (_, ho) in enumerate(GroupKFold(n_splits=k).split(np.arange(len(g)), groups=g)): fold[ho] = i
        return fold


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:   # pessimistic ties: inside a run of equal scores the negatives come first
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_fast(s, y, off, M):
    return _ndcg(np.ascontiguousarray(s, np.float32), y, off, M)


class Store:
    """Memmap cache of member z arrays: cache/z_<kind>.f32 (rows = members that have that kind) + cache/index.json {kind: {member: row}}."""
    def __init__(self, meta):
        self.meta = meta; self.idx = json.load(open(CACHE / 'index.json'))
        self.mm = {k: np.memmap(CACHE / f'z_{k}.f32', np.float32, 'r', shape=(len(self.idx[k]), meta.n[SPLIT[k]])) for k in KINDS if self.idx.get(k)}

    def has(self, kind, name): return name in self.idx[kind]
    def get(self, kind, name): return self.mm[kind][self.idx[kind][name]]

    def pool(self, kind, names, w=None):
        acc = np.zeros(self.meta.n[SPLIT[kind]], np.float64)
        if w is None:
            for n_ in names: acc += self.get(kind, n_)
            return (acc / len(names)).astype(np.float32)
        for n_, w_ in zip(names, w): acc += w_ * np.asarray(self.get(kind, n_), np.float64)
        return (acc / np.sum(w)).astype(np.float32)

    def mat(self, kind, names):
        return np.stack([np.asarray(self.get(kind, n_)) for n_ in names])


se = lambda d: float(d.std() / np.sqrt(len(d)))
fmt = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'

S8 = tuple(range(951, 959)); S4 = (951, 952, 953, 954)
e10 = lambda arm, seeds=S4: [f'w5_e10_{arm}_s{s}' for s in seeds]


def arm_of(name):
    m = re.match(r'^(.*?)_(s|r)(\d+)(@last)?$', name)
    return (m.group(1) + (m.group(4) or '')) if m else name


# ---- named member lists (the same strings as build_r4.py)
W16 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]              # held-out wide students
W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4]
FD16 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]            # full-data wide students
FDXHZ = [f'w5_fdxhz_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]          # hz re-cools of the full-data snapshots
P1d = [f'abl_e10_pop1kd05t25_s{s}' for s in S4]; P1w = [f'abl_e10_pop1kd05w192_s{s}' for s in S4]   # held-out popularity-light students
FDP4 = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]; FDP12 = [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 963)]
FDPd = [f'abl_fd10_pop1kd05t25_s{s}' for s in S4]; FDP2 = [f'abl_fd10_pop2kd05w192_s{s}' for s in S4]
FDPF = [f'abl_fd10_pfkd05w192_s{s}' for s in S8]
D128 = e10('kd05') + e10('kd05t25') + e10('kd08')                                        # the 12 d128 students with fine-tunes
F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]; F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; POP1H = [f'abl_e10_pop1_s{s}' for s in S8]
HZL8 = [f'w5_{st}cd_hz_s{s}@last' for st in 'kl' for s in S4]                            # held-out hz re-cools with fine-tunes (@last)
rc = lambda arm, suf='@last', seeds=S8: [f'w5_{st}cd_{arm}_s{s}{suf}' for st in 'kl' for s in seeds]
