import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review'); sys.path.insert(0, '<repo>/nn_x/w12/cb/infra')
from rv_common import *
import cbx
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
H = ROOT / 'nn_x/w12/cb/famN/cache/hx/n30k'; va = cbx.load('val'); nm = va.names
for c in ('hlda_pair', 'hknn50_pair', 'hknn50_ehm', 'hknn50_ue'):
    o = read(wv, wh, zwv, zwh, np.load(H / f'{c}_val.npy'), np.load(H / f'{c}_hold.npy'), c); print(line(o), flush=True)
zc = wv.z(np.load(H / 'hlda_pair_val.npy')); cu = wv.z(np.asarray(va.X[:, nm.index('cos_u')], np.float64))
print('val, within-feed z: corr(hlda_pair, cos_u)', round(float(np.corrcoef(zc, cu)[0, 1]), 3), '| corr(hlda_pair, W16)', round(float(np.corrcoef(zc, zwv)[0, 1]), 3))
