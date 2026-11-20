"""P8: read-out of the GBDT-student probes (final-round preds). Paired per feed, val + holdout.
 - K (teacher logit) vs H (hard lambdarank, tuned), X (OOF teacher), Y (teacher + 1.0*y), CY (Y + co-click), C (teacher + co-click), B (pointwise mix)
 - corr to the W8 train-only pool and blends (1-w) z(W8 TO) + w z(GBDT), w 0.05 / 0.1 / 0.2 (weights fixed here, not fitted)."""
from lb_common import *
import glob, json, os
O = ROOT / 'nn_x/w12/recheck/long_bets/out'
w8v = np.load(O / 'w8_to_val_z.npy'); w8h = np.load(O / 'w8_to_hold_z.npy'); e8v, e8h = ev(w8v), eh(w8h)
print(f'W8 TO pool: val {e8v.mean():.5f} holdout {e8h.mean():.5f}')
R = {}
for p in sorted(glob.glob(str(O / 'gbdt_*_val.npy'))):
    tag = os.path.basename(p)[5:-8]
    if tag.endswith('_2000'): continue
    R[tag] = (zv(np.load(p)), zh(np.load(p.replace('_val.npy', '_trH.npy'))))
print(f'{"arm_pop_n":18s} {"val":>8s} {"hold":>8s} {"rho_v":>6s} {"rho_h":>6s} | blend w=0.05 val/hold | w=0.1 val/hold | w=0.2 val/hold')
for tag, (v, h) in R.items():
    bl = ' | '.join(f'{f(ev((1-w)*w8v + w*v) - e8v)} / {f(eh((1-w)*w8h + w*h) - e8h)}' for w in (0.05, 0.1, 0.2))
    print(f'{tag:18s} {ev(v).mean():8.5f} {eh(h).mean():8.5f} {wcorr(v, w8v):6.4f} {wcorr(h, w8h):6.4f} | {bl}')
def cmp(a, b):
    if a in R and b in R: print(f' {a} - {b}: val {f(ev(R[a][0]) - ev(R[b][0]))} | holdout {f(eh(R[a][1]) - eh(R[b][1]))}')
print('paired contrasts (same feeds, same features, same rounds unless H):')
for a, b in (('K_xm_60000', 'H_xm_60000'), ('K_xm_60000', 'X_xm_60000'), ('Y_xm_60000', 'K_xm_60000'), ('CY_xm_60000', 'Y_xm_60000'), ('C_xm_60000', 'K_xm_60000'), ('B_xm_60000', 'K_xm_60000'),
             ('K_all_60000', 'K_xm_60000'), ('K_xm_134118', 'K_xm_60000'), ('K_all_240000', 'K_all_60000'), ('H_xm_134118', 'H_xm_60000'), ('K_xm_134118', 'H_xm_134118')):
    cmp(a, b)
for p in sorted(glob.glob(str(O / 'gbdt_*.json'))):
    d = json.load(open(p)); print(os.path.basename(p), {k: (round(v[0], 5), round(v[1], 5)) for k, v in d['res'].items()}, f"{d['minutes']:.1f} min")
