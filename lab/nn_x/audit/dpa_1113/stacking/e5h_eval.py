"""Read-out of E5h (holdout-trained MLP second level)."""
from lib import *
import pmix
V = S('val'); HO = S('hold')
wpriv = pmix.weights()['private']
log = open(OUT / 'e5h_eval.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
refh = HO.ev(np.load(CACHE / 'hold_pool_to.npy')); refv_to = V.ev(np.load(CACHE / 'val_pool_to.npy')); refv = V.ev(np.load(CACHE / 'val_g3.npy'))
g3 = np.load(CACHE / 'val_g3.npy').astype(np.float64)
for arm in ('B1', 'B2', 'B3'):
    tag = f'{arm}_hold'
    fs = [CACHE / f'oofh_mlp_{tag}_s{s}.npy' for s in (1, 2, 3)]
    if not all(f.exists() for f in fs): pr(arm, 'missing'); continue
    per = [np.load(f) for f in fs]
    dh = np.mean(per, 0) - refh; hm, hse = se_cl(dh, HO.cl)
    bh = HO.z(np.mean([HO.z(np.load(CACHE / f'scrh_mlp_{tag}_s{s}_hold.npy')) for s in (1, 2, 3)], 0).astype(np.float32))
    dbh = HO.ev(bh.astype(np.float64)) - refh; hbm, hbse = se_cl(dbh, HO.cl)
    hsl = ' '.join(f'{nm} {se_cl(dbh, HO.cl, m)[0]:+.5f}' for nm, m in [('late20', HO.late), ('wkday', HO.dow < 5), ('wkend', HO.dow >= 5)])
    vto = V.z(np.mean([V.z(np.load(CACHE / f'scrh_mlp_{tag}_s{s}_val_to.npy')) for s in (1, 2, 3)], 0).astype(np.float32))
    vft = V.z(np.mean([V.z(np.load(CACHE / f'scrh_mlp_{tag}_s{s}_val_ftas.npy')) for s in (1, 2, 3)], 0).astype(np.float32))
    dvt = V.ev(vto.astype(np.float64)) - refv_to; a, b = se_cl(dvt, V.cl)
    spec = (0.5 * V.z(vft) + 0.5 * V.z(vto)).astype(np.float64)
    dv = V.ev(spec) - refv; c, d = se_cl(dv, V.cl); pm, pse = wmean_se(dv, wpriv)
    sl = ' '.join(f'{nm} {se_cl(dv, V.cl, m)[0]:+.5f}({se_cl(dv, V.cl, m)[0] / se_cl(dv, V.cl, m)[1]:+.1f})'
                  for nm, m in [('M<50', V.M < 50), ('M>=50', V.M >= 50), ('Tue', V.tue), ('Thu', ~V.tue)])
    rho, top = wf_agree(V.off, V.M, spec, g3)
    pr(f'mlp-hold {arm}: HOLD-OOF seed-mean {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f}) seeds [{" ".join(f"{(p - refh).mean():+.5f}" for p in per)}] '
       f'bagged {hbm:+.5f} +- {hbse:.5f} (z {hbm / hbse:+.2f}) [{hsl}] | VAL(TO form vs TO pool) {a:+.5f} +- {b:.5f} (z {a / b:+.2f}) '
       f'| VAL(G3 spec vs G3) {c:+.5f} +- {d:.5f} (z {c / d:+.2f}) pmix {pm:+.5f} +- {pse:.5f} | {sl} | val rho {rho.mean():.4f} top1 {top.mean():.3f}')
