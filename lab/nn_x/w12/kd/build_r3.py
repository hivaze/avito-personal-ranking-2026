"""Round-3 candidates (pre-declared nn_x/w5/logs/E_kd_predeclared.txt, 2026-11-14 20:16Z block, items A / C / D).
  W16   = 16 wide students (kd05w192 + kd05t25w192, seeds 951-958), member-equal, final 0.5 z(FT) + 0.5 z(TO)
  W8P20 = per branch 0.8 z(W8 pool) + 0.2 z(pop1kd pool, abl_e10_pop1kd05t25_s951-954), final 0.5 z(FT) + 0.5 z(TO)
  W8TO  = z(W8 train-only pool) (fine-tune half removed; public probe of the FT half for students)
Usage: build_r3.py <W16|W8P20|W8TO>[,..] [build]     (CPU; without 'build' it only prints the read-out)"""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from make_sub import make_sub
import pmix
from pathlib import Path
want = sys.argv[1].split(','); build = 'build' in sys.argv[2:]
P = Path('<repo>/preds'); S = Path('<repo>/subs')
W = lambda seeds: [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
W8n, W16n = W(range(951, 955)), W(range(951, 959)); P1n = [f'abl_e10_pop1kd05t25_s{s}' for s in range(951, 955)]
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
Z = {'val': lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1], 'test': lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]}
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
def fmt(d, w=None):
    if w is None: return f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
    m, se = pmix.wmean_se(d, w); return f'{m:+.5f}+-{se:.5f}'
def pool(names, sp):
    z = Z[sp]; ftn, ton = SUF[sp]
    return (np.mean([z(np.load(P / f'nn_{m}_{ftn}.npy')) for m in names], 0), np.mean([z(np.load(P / f'nn_{m}_{ton}.npy')) for m in names], 0))
B = {}
for sp in ('val', 'test'):
    z = Z[sp]; w8 = pool(W8n, sp); B[sp] = {'W8': w8}
    if 'W16' in want: B[sp]['W16'] = pool(W16n, sp)
    if 'W8P20' in want:
        p1 = pool(P1n, sp); B[sp]['W8P20'] = (0.8 * z(w8[0]) + 0.2 * z(p1[0]), 0.8 * z(w8[1]) + 0.2 * z(p1[1])); B[sp]['P1'] = p1
    if 'W8TO' in want: B[sp]['W8TO'] = (w8[1], w8[1])
fin = lambda sp, k: 0.5 * Z[sp](B[sp][k][0]) + 0.5 * Z[sp](B[sp][k][1])
# reproduction check of the stored W8 branches
for sp in ('val', 'test'):
    for i, br in enumerate(('ft', 'to')):
        c = np.corrcoef(B[sp]['W8'][i], np.load(P / f'ens_W8kd_{br}_{sp}.npy'))[0, 1]; assert c > 0.999999, (sp, br, c)
e0 = ev(fin('val', 'W8')); t0 = fin('test', 'W8'); wts = pmix.weights(); big = Mv >= 50
print(f'W8 final val {e0.mean():.5f} | FT {ev(B["val"]["W8"][0]).mean():.5f} TO {ev(B["val"]["W8"][1]).mean():.5f}  (stored-branch reproduction OK)')
ok = {}
for k in [k for k in ('P1', 'W16', 'W8P20', 'W8TO') if k in B['val']]:
    e = ev(fin('val', k)); d = e - e0
    print(f'{k:6s} final val {e.mean():.5f} delta vs W8 {fmt(d)} | FT {ev(B["val"][k][0]).mean():.5f} TO {ev(B["val"][k][1]).mean():.5f} | M<50 {fmt(d[~big])} M>=50 {fmt(d[big])}'
          f' | pmix public {fmt(d, wts["public"])} private {fmt(d, wts["private"])} | test corr to W8 final {np.corrcoef(fin("test", k), t0)[0, 1]:.4f}')
    ok[k] = d.mean()
if build:
    for k in [k for k in want if k in B['val']]:
        if k == 'W8P20' and ok[k] < 0: print(f'{k}: final val delta vs W8 < 0 -> NOT built (pre-declared build rule)'); continue
        tag = f'{k}kd'; out = S / f'cand_{tag}_r3.parquet'; assert not out.exists() and not list(P.glob(f'ens_{tag}_*')), f'{tag} exists'
        for sp in ('val', 'test'):
            np.save(P / f'ens_{tag}_ft_{sp}.npy', B[sp][k][0].astype(np.float32)); np.save(P / f'ens_{tag}_to_{sp}.npy', B[sp][k][1].astype(np.float32))
        make_sub(fin('test', k).astype(np.float32), str(out)); print('written', out)
