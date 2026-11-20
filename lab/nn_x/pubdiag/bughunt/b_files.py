"""(b) file-level audit: every feature file the r6 path reads (nn_x/r6/feat_cache -> nn/feat_cache) vs what nn/ reads
(nn/feat_cache, fallback data/feats) and vs data/feats: existence, inode identity, mtime/ctime, size."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from names import XT, EXTRA
R = '<repo>'
fmt = lambda t: time.strftime('%m-%d %H:%M:%S', time.gmtime(t))
rows = []
for split in ('train', 'val', 'test'):
    names = sorted(set(XT) | (set(EXTRA) if split != 'train' else set()))
    for n in names:
        r6 = f'{R}/nn_x/r6/feat_cache/{split}/{n}.npy'; nnc = f'{R}/nn/feat_cache/{split}/{n}.npy'; df = f'{R}/data/feats/{split}/{n}.npy'
        s6 = os.stat(r6) if os.path.exists(r6) else None
        sn = os.stat(nnc) if os.path.exists(nnc) else None
        sd = os.stat(df) if os.path.exists(df) else None
        same_r6_nn = s6 is not None and sn is not None and (s6.st_ino == sn.st_ino)
        same_nn_df = (sn is not None and sd is not None and sn.st_ino == sd.st_ino)
        rows.append((split, n, s6 is not None, sn is not None, sd is not None, same_r6_nn, same_nn_df,
                     fmt(sn.st_mtime) if sn else '-', fmt(sn.st_ctime) if sn else '-', sn.st_nlink if sn else 0,
                     fmt(sd.st_mtime) if sd else '-'))
bad = [r for r in rows if not (r[2] and r[3] and r[5])]
print('files checked', len(rows), '; r6 path missing or not same inode as nn/feat_cache:', len(bad))
for r in bad: print('  BAD', r)
nd = [r for r in rows if r[4] and not r[6]]
print('nn/feat_cache file differs (inode) from data/feats (data/feats regenerated after snapshot):', len(nd))
for r in nd: print('  ', r)
nodf = [r for r in rows if not r[4]]
print('not in data/feats any more (only cache):', len(nodf), [f'{r[0]}/{r[1]}' for r in nodf][:50])
print('latest mtime / ctime of any cache file read:', max(r[7] for r in rows), max(r[8] for r in rows))
for r in sorted(rows, key=lambda r: r[8])[-6:]: print('  latest ctime', r)
