"""Step 1: reproduce build_r4.py FD16P20 (0.40271) and FD16P20@m0.2_0.4 (0.40298) with the own loader; bar in DECLARATION BLOCK 1."""
from lib import *
import json, time
t0 = time.time()
f, Mv, ov = meta('val'); Mrow = np.repeat(Mv, Mv)
res = {}
for comp, base, pop in (('FD16P4', MEM['FD16'], MEM['FDP']), ('W16P4', MEM['W16'], MEM['P1w'])):
    q = pools4('val', base, pop, tag=comp)
    for name, w in (('plain', 0.2), ('mrule', np.where(Mrow >= 50, 0.4, 0.2).astype(np.float32))):
        bf, bt = branches('val', q, w); e = ev(0.5 * bf + 0.5 * bt)
        res[f'{comp}_{name}'] = float(e.mean()); print(comp, name, f'final val {e.mean():.6f}  FT {ev(bf).mean():.6f} TO {ev(bt).mean():.6f}', flush=True)
        if comp == 'FD16P4':
            tag = 'FD16P20kd' if name == 'plain' else 'FD16P20_m02_04kd'
            sf, st = np.load(P / f'ens_{tag}_ft_val.npy'), np.load(P / f'ens_{tag}_to_val.npy')
            mf = (1 - w) * q['Bft'] + w * q['Pft']; mt = (1 - w) * q['Bto'] + w * q['Pto']
            res[f'{comp}_{name}_maxabs_ft'] = float(np.abs(sf - mf).max()); res[f'{comp}_{name}_maxabs_to'] = float(np.abs(st - mt).max())
            es = ev(0.5 * Z('val', sf) + 0.5 * Z('val', st)); res[f'{comp}_{name}_stored_final'] = float(es.mean())
            print('   stored arrays: max|d| ft', res[f'{comp}_{name}_maxabs_ft'], 'to', res[f'{comp}_{name}_maxabs_to'], f'stored final {es.mean():.6f}', flush=True)
logged = {'FD16P4_plain': 0.40271, 'FD16P4_mrule': 0.40298}
ok = all(abs(res[k] - v) <= 1.5e-5 for k, v in logged.items()) and all(res[f'FD16P4_{n}_maxabs_{b}'] <= 1e-5 for n in ('plain', 'mrule') for b in ('ft', 'to'))
res['PASS'] = bool(ok); print('STEP 1', 'PASS' if ok else 'FAIL', f'{time.time() - t0:.0f}s')
json.dump(res, open(HERE / 'out_step1.json', 'w'), indent=1)
# test pools (label-free) cached for step 6
for comp, base, pop in (('FD16P4', MEM['FD16'], MEM['FDP']),):
    pools4('test', base, pop, tag=comp)
print('test pools cached', f'{time.time() - t0:.0f}s')
