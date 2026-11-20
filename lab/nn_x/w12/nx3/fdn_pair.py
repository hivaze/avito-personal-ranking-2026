"""fdn_pair.py <tag A> <tag B> [rule]: PAIR DESCRIPTION "B minus A" of two built round-4 files from their STORED branch files
preds/ens_<tag>_{ft,to}_{val,test}.npy (CPU, read-only; nothing is written). Final score = 0.5 z(FT branch) + 0.5 z(train-only branch), z = per-feed
rank-z (stack_val.feed_rank_z[:, 1]) = build_r4.py's `fin` and the gate's --val / --test spec. Per-feed val NDCG (src/metrics.ndcg_per_feed), paired per feed:
  final val of each file; delta with paired SE and z; M < 50 / M >= 50; private-mix and public-mix views (src/pmix.py weights, pmix.wmean_se);
  the FT branch and the train-only branch alone; test: Pearson correlation of the final scores and the per-feed Spearman (mean, 5th percentile).
  rule : also evaluates the FILE RULE of the nx3 block for B against A (final val delta >= +1 SE AND private-mix delta >= 0 AND val M<50 >= -1 SE;
         the gate PASS and the construction review are stated by finish_fdn.sh / the lead, not here).
  .venv/bin/python nn_x/w12/nx3/fdn_pair.py FD16P20kd FDN16P20kd rule        (dry test on stored files: FD16P20kd FD32P20kd)"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from pathlib import Path
ta, tb = sys.argv[1], sys.argv[2]; rule = 'rule' in sys.argv[3:]
P = Path('<repo>/preds')
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
Z = {'val': lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1], 'test': lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]}
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
def load(tag):
    br = {sp: (np.load(P / f'ens_{tag}_ft_{sp}.npy'), np.load(P / f'ens_{tag}_to_{sp}.npy')) for sp in ('val', 'test')}
    fin = {sp: 0.5 * Z[sp](br[sp][0]) + 0.5 * Z[sp](br[sp][1]) for sp in ('val', 'test')}
    return dict(e=ev(fin['val']), eft=ev(br['val'][0]), eto=ev(br['val'][1]), test=fin['test'])
A, B = load(ta), load(tb); wts = pmix.weights(); big = Mv >= 50
def st(d, w=None):
    if w is None: m, se = d.mean(), d.std() / np.sqrt(len(d))
    else: m, se = pmix.wmean_se(d, w)
    return float(m), float(se)
f = lambda t: f'{t[0]:+.5f} +- {t[1]:.5f} (z {t[0] / t[1]:+.2f})'
d = B['e'] - A['e']; fin = st(d); small = st(d[~big]); bigs = st(d[big]); pub = st(d, wts['public']); prv = st(d, wts['private'])
print(f'PAIR DESCRIPTION {tb} minus {ta} (per-feed val NDCG, paired; {len(d)} val feeds; computed from preds/ens_<tag>_*.npy)')
print(f'  final val: {tb} {B["e"].mean():.5f} | {ta} {A["e"].mean():.5f} | delta {f(fin)}')
print(f'  M<50 (n {int((~big).sum())}): {f(small)} | M>=50 (n {int(big.sum())}): {f(bigs)}')
print(f'  private-mix view (src/pmix.py): {f(prv)} | public-mix view: {f(pub)}')
print(f'  FT branch alone: {f(st(B["eft"] - A["eft"]))} | train-only branch alone: {f(st(B["eto"] - A["eto"]))}')
fid = np.repeat(np.arange(len(Mt)), Mt)
def wrank(x):
    o = np.lexsort((x, fid)); r = np.empty(len(x), np.float64); r[o] = np.arange(len(x)) - np.repeat(ot, Mt); return r
ra, rb = wrank(A['test']), wrank(B['test'])
c = lambda x: x - (np.bincount(fid, x) / Mt)[fid]
ca, cb = c(ra), c(rb); den = np.sqrt(np.bincount(fid, ca * ca) * np.bincount(fid, cb * cb)); rho = np.where(den > 0, np.bincount(fid, ca * cb) / np.maximum(den, 1e-12), 1.0)
print(f'  test: correlation of the final scores {np.corrcoef(A["test"], B["test"])[0, 1]:.4f}; per-feed Spearman mean {rho.mean():.4f}, 5th percentile {np.percentile(rho, 5):.4f} ({len(Mt)} test feeds)')
print(f'  paired public SE of the val delta at the public feed count (sd / sqrt(8562)): {d.std() / np.sqrt(8562):.5f}')
if rule:
    r1 = fin[0] >= fin[1]; r2 = prv[0] >= 0; r3 = small[0] >= -small[1]
    print(f'FILE RULE of the nx3 block for {tb} against {ta} (val clauses only): final val delta >= +1 SE: {"yes" if r1 else "NO"} ({fin[0]:+.5f} vs SE {fin[1]:.5f}) | '
          f'private-mix delta >= 0: {"yes" if r2 else "NO"} ({prv[0]:+.5f}) | val M<50 >= -1 SE: {"yes" if r3 else "NO"} ({small[0]:+.5f} vs -{small[1]:.5f})  ->  val clauses {"HOLD" if r1 and r2 and r3 else "DO NOT HOLD"}')
