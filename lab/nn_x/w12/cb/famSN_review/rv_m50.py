"""val M >= 50 only (lead's notice of 02:19Z, point 3): NATS and member-rule statistic of every famN run and the famS arms, own code (clogit refitted on the bin)."""
import sys, glob; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv = Wit('val'); zw = w16(wv)
class Sub(Wit):
    def __init__(self, w, m):
        self.rows = np.repeat(m, w.M); self.sp = w.sp; self.M = w.M[m]; self.off = np.concatenate([[0], np.cumsum(self.M)[:-1]]); self.y = w.y[self.rows]; self.nf = len(self.M); self.fid = np.repeat(np.arange(self.nf), self.M)
        self.npos = np.bincount(self.fid, self.y, self.nf); self.rank0 = np.arange(len(self.y)) - np.repeat(self.off, self.M); self.disc = 1.0 / np.log2(self.rank0 + 2.0)
        self.idcg = np.bincount(self.fid, (self.rank0 < np.repeat(self.npos, self.M)) * self.disc, self.nf)
s = Sub(wv, wv.M >= 50); zs = zw[s.rows]; _, lw = s.clogit(zs); e0 = s.ndcg(zs)
res = {}
files = sorted(glob.glob(str(ROOT / 'nn_x/w12/cb/famN/preds/*_val.npy'))) + sorted(glob.glob(str(ROOT / 'nn_x/w12/cb/famS/preds/s*_val.npy'))) + [str(ROOT / 'nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy')]
for f in files:
    t = Path(f).name[:-8]; zx = wv.z(np.load(f))[s.rows]; _, l2 = s.clogit(np.column_stack([zs, zx])); d = lw - l2; dm = s.ndcg(0.9 * zs + 0.1 * zx) - e0
    res[t] = dict(nats=float(d.mean()), nats_se=se(d), mix=float(dm.mean()), mix_se=se(dm), level=float(s.ndcg(zx).mean()))
    print(f"{t}: val M>=50 ({s.nf} feeds) level {res[t]['level']:.5f} NATS {d.mean():+.5f}+-{se(d):.5f} member {dm.mean():+.5f}+-{se(dm):.5f} (z {dm.mean() / se(dm):+.2f})", flush=True)
print('max NATS', max((r['nats'], t) for t, r in res.items()), 'max member z', max((r['mix'] / r['mix_se'], t) for t, r in res.items()))
json.dump(res, open(HERE / 'out/m50_own.json', 'w'), indent=1)
