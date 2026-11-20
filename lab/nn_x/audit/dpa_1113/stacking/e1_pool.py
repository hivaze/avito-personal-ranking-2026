"""E1 (c): pooling functions without fitted parameters. Per half aggregate the 20 member scores, then the G3 spec
0.5 z(ft) + 0.5 z(to) on val; holdout uses the TO half alone. Deltas vs G3 (val) / TO pool (holdout)."""
from lib import *
import time, json
from scipy.special import ndtri
import pmix

T0 = time.time()
V = S('val'); HO = S('hold'); TE = S('test')
log = open(OUT / 'e1_pool.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

W = pmix.weights(); wpriv = W['private']; wpub = W['public']
g3 = np.load(CACHE / 'val_g3.npy'); base_v = V.ev(g3)
hp = np.load(CACHE / 'hold_pool_to.npy'); base_h = HO.ev(hp)
g3t = np.load(CACHE / 'test_g3.npy')
pr(f'G3 val {base_v.mean():.5f}  holdout TO pool {base_h.mean():.5f}')

RAW = {}
def raw(sp, half):
    k = (sp.name, half)
    if k not in RAW:
        if sp.name == 'val':
            pat = 'nn_{}_n2cpv3_val_oof.npy' if half == 'ft' else 'nn_{}_trainonly_val.npy'; d = P
        elif sp.name == 'hold':
            pat = 'nn_{}_holdout.npy'; d = HD
        else:
            pat = 'nn_{}_n2cpv3_test.npy' if half == 'ft' else 'nn_{}_trainonly_test.npy'; d = P
        RAW[k] = np.stack([np.load(d / pat.format(m)).astype(np.float64) for m in MEMBERS])
    return RAW[k]

def pct_all(sp, Z):
    return np.stack([feed_pct(sp.off, sp.M, np.ascontiguousarray(z)) for z in Z])

def agg(name, sp, half):
    Z = np.asarray(sp.Z(half), np.float64)
    if name == 'zmean': return Z.mean(0)
    if name == 'rankmean': return pct_all(sp, Z).mean(0) + 1e-6 * Z.mean(0)
    if name == 'rankgauss':
        p = pct_all(sp, Z).astype(np.float64); Mr = np.repeat(sp.M, sp.M).astype(np.float64)
        return ndtri((p * (Mr - 1) + 0.5) / Mr).mean(0)
    if name.startswith('spow'):
        q = float(name[4:]); return (np.sign(Z) * np.abs(Z) ** q).mean(0)
    if name.startswith('rpow'):
        q = float(name[4:]); return (pct_all(sp, Z).astype(np.float64) ** q).mean(0) + 1e-6 * Z.mean(0)
    if name == 'clip2': return np.clip(Z, -2, 2).mean(0)
    if name.startswith('trim'):
        c = int(round(float(name[4:]) / 100 * Z.shape[0])); Zs = np.sort(Z, 0); return Zs[c:Z.shape[0] - c].mean(0)
    if name == 'median': return np.median(Z, 0)
    if name.startswith('lse'):
        T = float(name.split('T')[1]); mx = Z.max(0); return T * np.log(np.exp((Z - mx) / T).mean(0)) + mx
    if name in ('rawc', 'gscale'):
        R = raw(sp, half); fid = sp.fid
        out = np.zeros(R.shape[1])
        for r in R:
            c = r - (np.add.reduceat(r, sp.off) / sp.M)[fid]
            if name == 'gscale':
                c = c / np.sqrt((np.add.reduceat(c * c, sp.off) / sp.M).mean())
            out += c
        return out / len(R)
    raise KeyError(name)

VARS = ['zmean', 'rankmean', 'rankgauss', 'spow0.5', 'spow2', 'rpow0.25', 'rpow4', 'clip2', 'trim10', 'trim20', 'median',
        'lseT0.5', 'lseT1', 'rawc', 'gscale']
res = {}
for v in VARS:
    sv = feed_z(V.off, V.M, (0.5 * V.z(agg(v, V, 'ft')) + 0.5 * V.z(agg(v, V, 'to'))).astype(np.float32))
    dv = V.ev(sv) - base_v
    mu, se = se_cl(dv, V.cl)
    pm, pse = wmean_se(dv, wpriv)
    sl = []
    for nm, m in [('M<50', V.M < 50), ('M>=50', V.M >= 50), ('Tue', V.tue), ('Thu', ~V.tue)]:
        a, b = se_cl(dv, V.cl, m); sl.append(f'{nm} {a:+.5f}({a / b:+.1f})')
    sh = HO.ev(agg(v, HO, 'to')) - base_h
    hm, hse = se_cl(sh, HO.cl)
    hsl = []
    for nm, m in [('late20', HO.late), ('wkday', HO.dow < 5), ('wkend', HO.dow >= 5)]:
        a, b = se_cl(sh, HO.cl, m); hsl.append(f'{nm} {a:+.5f}({a / b:+.1f})')
    cv = float(np.corrcoef(sv, g3)[0, 1])
    st = feed_z(TE.off, TE.M, (0.5 * TE.z(agg(v, TE, 'ft')) + 0.5 * TE.z(agg(v, TE, 'to'))).astype(np.float32))
    ct = float(np.corrcoef(st, g3t)[0, 1])
    res[v] = dict(val=mu, val_se=se, hold=hm, hold_se=hse, pmix=pm, pmix_se=pse, corr_val=cv, corr_test=ct)
    pr(f'{v:10s} VAL {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) | pmix {pm:+.5f} +- {pse:.5f} | HOLD {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f}) '
       f'| corr val {cv:.5f} test {ct:.5f} | {" ".join(sl)} | H: {" ".join(hsl)}  [{time.time()-T0:.0f}s]')
json.dump(res, open(OUT / 'e1_pool.json', 'w'), indent=1)
