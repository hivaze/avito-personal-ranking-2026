# Data-paradigm proxy: paired per-feed comparisons of CE models trained on different feed pools (same seeds).
import sys, numpy as np
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *
O = '<repo>/solvers/out/solvers'
fv, Mv, ov = feed_meta('val'); dayv = day_of(fv['timestamp'].to_numpy()); tue = dayv == dayv.min()
sl = {'M<50': Mv < 50, 'M>=50': Mv >= 50, 'Tue': tue, 'Thu': ~tue}
def load(tag, kind='ce', seeds=(1, 2, 3)):
    try: return [np.load(f'{O}/zoo_{tag}_{kind}_s{s}.npz') for s in seeds]
    except FileNotFoundError: return None
arms = [(a.split(':')[0], a.split(':')[1] if ':' in a else 'ce') for a in sys.argv[1:]]
L = {f'{t}:{k}': load(t, k) for t, k in arms}
ref = list(L)[0]
for name, r in L.items():
    if r is None: print(name, 'missing'); continue
    v = np.mean([x['val'] for x in r], 0); h = np.mean([x['hold'] for x in r], 0)
    dv = v - np.mean([x['val'] for x in L[ref]], 0); dh = h - np.mean([x['hold'] for x in L[ref]], 0)
    print('%-16s val %.5f hold %.5f | d vs %s: val %+.5f (%.5f) hold %+.5f (%.5f) | %s' % (name, v.mean(), h.mean(), ref,
          dv.mean(), dv.std() / np.sqrt(len(dv)), dh.mean(), dh.std() / np.sqrt(len(dh)), '  '.join('%s %+.4f' % (k, dv[m].mean()) for k, m in sl.items())))
