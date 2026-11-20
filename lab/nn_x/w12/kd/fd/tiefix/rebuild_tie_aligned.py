"""rebuild_tie_aligned.py <tag>: diagnose a gate "B reproduction" miss of subs/cand_<tag>_r4.parquet and, if it is a float32 TIE artefact only, write subs/cand_<tag>_r4b.parquet with the
same within-feed order except that exact ties of the spec score are broken by row index (the gate's convention) instead of make_sub's random key. Spec = the gate's own:
0.5 * z(ens_<tag>_ft_test) + 0.5 * z(ens_<tag>_to_test), z = feed_rank_z(...)[:, 1] of the float32 branch files. CPU, writes one new file, refuses to overwrite. (lead, 2026-11-16)"""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
import polars as pl
from stack_val import feed_rank_z
from make_sub import make_sub
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'; tag = sys.argv[1]
ft_, Mt, ot = feed_meta('test')
z = lambda n: feed_rank_z(ot, Mt, np.nan_to_num(np.load(P / f'{n}.npy').astype(np.float32)))[:, 1]
t = 0.5 * z(f'ens_{tag}_ft_test') + 0.5 * z(f'ens_{tag}_to_test')
fid = np.repeat(np.arange(len(Mt)), Mt)
def wrank(x):
    o = np.lexsort((x, fid)); r = np.empty(len(x), np.int64); r[o] = np.arange(len(x)) - np.repeat(ot, Mt); return r
s = pl.read_parquet(R / f'subs/cand_{tag}_r4.parquet')
flat = pl.DataFrame({'feed_id': np.repeat(ft_['feed_id'].to_numpy(), Mt), 'item_id': np.load(FLAT / 'test/cand_item_id.npy'), 'row': np.arange(len(t))})
sc = flat.join(s, on=['feed_id', 'item_id'], how='left').sort('row')['score'].to_numpy().astype(np.float64)
rs, rt = wrank(sc), wrank(t); diff = rs != rt; bad = np.unique(fid[diff])
print(f'{tag}: spec dtype {t.dtype}; rows whose within-feed rank differs between the file and the spec: {int(diff.sum())} in {len(bad)} feeds of {len(Mt)}')
only_ties = True
for f in bad[:20]:
    a, b = ot[f], ot[f] + Mt[f]; j = np.where(diff[a:b])[0]; vals = t[a:b][j]
    tie = len(np.unique(vals)) == 1; only_ties &= tie
    print(f'  feed index {f}: M {Mt[f]}, rows moved {len(j)}, rank shift max {int(np.abs(rs[a:b] - rt[a:b]).max())}, spec scores of the moved rows {[float(v) for v in vals]} -> {"EXACT TIE" if tie else "NOT a tie"}')
if not len(bad): print('nothing to fix'); sys.exit(0)
if not only_ties: print('NOT only ties: no file written'); sys.exit(2)
out = R / f'subs/cand_{tag}_r4b.parquet'; assert not out.exists(), 'exists'
make_sub(rt.astype(np.float64), str(out)); print('written', out)
