"""E1b (added after E1, before its numbers): per-feed consensus weighting of members (parameter-light): in each feed, member m
gets weight max(rho_m, 0)^k, rho_m = within-feed Pearson corr of member z with the leave-one-out mean of the other members'
z (same half); k in {1, 4, 16}; also the inverse ('contrarian', weight (1 - rho_m)). Same G3 spec / holdout TO read as E1.
Also prints the within-feed Spearman / top-1 agreement with G3 (val, test) for the E1 variants that were within 1 SE."""
from lib import *
import pmix

V = S('val'); HO = S('hold'); TE = S('test')
wpriv = pmix.weights()['private']
log = open(OUT / 'e1b_consensus.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

@nb.njit(parallel=True, cache=True)
def cons(off, M, Z, k, mode):
    n = len(M); K = Z.shape[0]; out = np.zeros(Z.shape[1])
    for f in nb.prange(n):
        o, m = off[f], M[f]
        tot = np.zeros(m)
        for i in range(m):
            for j in range(K):
                tot[i] += Z[j, o + i]
        wsum = 0.0
        acc = np.zeros(m)
        for j in range(K):
            a = Z[j, o:o + m]
            b = (tot - a) / (K - 1)
            am = a.mean(); bm = b.mean()
            num = ((a - am) * (b - bm)).sum(); den = np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum()) + 1e-12
            r = num / den
            if mode == 0:
                w = max(r, 0.0) ** k
            else:
                w = 1.0 - r + 1e-3
            wsum += w
            acc += w * a
        for i in range(m):
            out[o + i] = acc[i] / wsum
    return out

g3 = np.load(CACHE / 'val_g3.npy').astype(np.float64); base_v = V.ev(g3)
hp = np.load(CACHE / 'hold_pool_to.npy').astype(np.float64); base_h = HO.ev(hp)
g3t = np.load(CACHE / 'test_g3.npy').astype(np.float64)
for nm, k, mode in [('cons_k1', 1.0, 0), ('cons_k4', 4.0, 0), ('cons_k16', 16.0, 0), ('contrarian', 1.0, 1)]:
    agg = lambda sp, h: cons(sp.off, sp.M, np.asarray(sp.Z(h), np.float64), k, mode)
    sv = 0.5 * V.z(agg(V, 'ft').astype(np.float32)) + 0.5 * V.z(agg(V, 'to').astype(np.float32))
    dv = V.ev(sv.astype(np.float64)) - base_v; mu, se = se_cl(dv, V.cl); pm, pse = wmean_se(dv, wpriv)
    dh = HO.ev(agg(HO, 'to')) - base_h; hm, hse = se_cl(dh, HO.cl)
    st = 0.5 * TE.z(agg(TE, 'ft').astype(np.float32)) + 0.5 * TE.z(agg(TE, 'to').astype(np.float32))
    rv, tv = wf_agree(V.off, V.M, sv.astype(np.float64), g3); rt, tt = wf_agree(TE.off, TE.M, st.astype(np.float64), g3t)
    pr(f'{nm:10s} VAL {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) pmix {pm:+.5f} +- {pse:.5f} | HOLD {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f}) '
       f'| rho val {rv.mean():.5f} test {rt.mean():.5f} top1 val {tv.mean():.4f} test {tt.mean():.4f}')
