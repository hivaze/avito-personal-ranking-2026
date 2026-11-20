"""Round-4 candidates (2026-11-15; rules: nn_x/w5/logs/E_kd_predeclared.txt blocks 03:36Z "W16P20x8", 04:5xZ "12-HOUR PLAN").
General form:  per branch (1 - w) z(base pool) + w z(popularity-light pool), final 0.5 z(FT n2cpv3s3) + 0.5 z(train-only); pools = member-equal mean of per-feed rank-z.
  build_r4.py <KEY>[,<KEY>..] [build]        (CPU; without 'build' it only prints the read-out; 'build' writes preds/ens_<KEY>kd_{ft,to}_{val,test}.npy + subs/cand_<KEY>kd_r4.parquet)
KEYS (member lists below): W16P20x8 | W16P20w | W16P20d | H16P20x8 | H16 | X16P20x8:<arm> (re-cool arm, 16 snapshots) | FD16P20 | FD16 | FDX16P20:<tag>
Read-out: final val, deltas vs W8 / W16 / W8P20 files (paired per feed), M<50 / M>=50, pmix-public / pmix-private, test corr to each reference."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from make_sub import make_sub
import pmix
from pathlib import Path
want = sys.argv[1].split(','); build = 'build' in sys.argv[2:]
P = Path('<repo>/preds'); S = Path('<repo>/subs')
S8 = range(951, 959); S4 = range(951, 955)
wide = lambda seeds: [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
rc = lambda arm: [f'w5_{st}cd_{arm}_s{s}' for st in 'kl' for s in S8]
P1d = [f'abl_e10_pop1kd05t25_s{s}' for s in S4]; P1w = [f'abl_e10_pop1kd05w192_s{s}' for s in S4]
FD16 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]; FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]
FDPd = [f'abl_fd10_pop1kd05t25_s{s}' for s in S4]
FD8 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4]; fdx = lambda tag, seeds: [f'w5_fdx{tag}_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
def spec(k):
    if k == 'W16P20x8': return wide(S8), P1d + P1w, 0.2
    if k == 'W16P20w': return wide(S8), P1w, 0.2
    if k == 'W16P20d': return wide(S8), P1d, 0.2
    if k == 'W8P20x8': return wide(S4), P1d + P1w, 0.2
    if k == 'H16': return rc('hz'), [], 0.0
    if k == 'H16P20x8': return rc('hz'), P1d + P1w, 0.2
    if k.startswith('X16P20x8:'): return rc(k.split(':')[1]), P1d + P1w, 0.2
    if k.startswith('X16:'): return rc(k.split(':')[1]), [], 0.0
    if k == 'W8P20w': return wide(S4), P1w, 0.2          # held-out twin of FD8P20
    if k.startswith('XL8P20w:'): return [f'w5_{st}cd_{k.split(":")[1]}_s{s}@last' for st in 'kl' for s in S4], P1w, 0.2   # held-out preview of FDX8P20:<arm>: the 8 re-cools (K + L, s951-954) at the LAST step + 4 wide pop1
    if k.startswith('XL8:'): return [f'w5_{st}cd_{k.split(":")[1]}_s{s}@last' for st in 'kl' for s in S4], [], 0.0
    if k == 'W8w': return wide(S4), [], 0.0
    if k == 'FD8': return FD8, [], 0.0
    if k == 'FD8P20': return FD8, FDP, 0.2               # full-data twin of W8P20w (8 wide + 4 wide popularity-light, all trained WITH the holdout feeds)
    if k.startswith('FDX8P20:'): return fdx(k.split(':')[1], S4), FDP, 0.2   # the same with the 8 wide members re-cooled by recipe <tag> (lane_fdx.sh)
    if k.startswith('FDX8:'): return fdx(k.split(':')[1], S4), [], 0.0
    if k == 'FD16': return FD16, [], 0.0
    if k == 'FD16P20': return FD16, FDP, 0.2
    if k == 'FD16P20x8': return FD16, FDP + FDPd, 0.2      # 16 full-data wide + 8 full-data popularity-light students (4 wide + 4 d128)
    if k == 'FD16P20w8': return FD16, [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 959)], 0.2     # 16 + 8 WIDE full-data popularity-light students (declared 12:55Z)
    if k == 'FD16P20w12': return FD16, [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 963)], 0.2    # 16 + 12 wide
    if k == 'FD16P20v2': return FD16, FDP + [f'abl_fd10_pop2kd05w192_s{s}' for s in S4], 0.2    # view diversity (declared 13:10Z): pop1 wide x4 + pop2 wide x4; control = FD16P20w8
    if k == 'FD16P20vf': return FD16, FDP + [f'abl_fd10_pfkd05w192_s{s}' for s in S4], 0.2      # pop1 wide x4 + popularity-free wide x4
    if k == 'FD16P20vfB': return FD16, FDP + [f'abl_fd10_pfkd05w192_s{s}' for s in range(955, 959)], 0.2     # replication twin of FD16P20vf: pop1 wide x4 + the 4 NEW popularity-free students (rule 2026-11-15T17:03Z)
    if k == 'FD16P20vf8': return FD16, FDP + [f'abl_fd10_pfkd05w192_s{s}' for s in S8], 0.2                  # pop1 wide x4 + popularity-free wide x8 (the pool if the replication reads REPLICATED)
    if k == 'U32P20w': return wide(S8) + FD16, P1w + FDP, 0.2      # UNION (declared 2026-11-15 21:0xZ, experimental): all 32 wide students (16 held-out + 16 all-data) + the 8 wide pop1 students (4 + 4)
    if k == 'U32': return wide(S8) + FD16, [], 0.0
    if k == 'FD32P20': return FD16 + [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(959, 967)], FDP, 0.2   # ENSEMBLE GROWTH (rule G, 2026-11-15 21:0xZ): 32 all-data wide students + 4 all-data wide pop1
    if k == 'FD24P20': return FD16 + [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(959, 963)], FDP, 0.2   # interim form of rule G with the first 8 new students
    if k == 'FDB16P20': return [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(959, 967)], [f'abl_fd10_pop1kd05w192_s{s}' for s in range(955, 959)], 0.2   # DISJOINT all-data pick-2 design (declared 2026-11-15 23:1xZ): the 16 NEW wide students + pop1 wide s955-958; shares no member with FD16P20
    if k == 'FD48P20': return FD16 + [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(959, 975)], FDP, 0.2   # growth to 48 all-data wide students (declared 2026-11-15 23:1xZ)
    if k == 'FDC16P20': return [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(967, 975)], [f'abl_fd10_pop1kd05w192_s{s}' for s in range(959, 963)], 0.2   # second disjoint all-data file: wide s967-974 + pop1 wide s959-962
    if k == 'FDBC32P20': return [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(959, 975)], [f'abl_fd10_pop1kd05w192_s{s}' for s in range(955, 963)], 0.2   # 32 wide s959-974 + pop1 wide s955-962: disjoint from FD16P20, same size class as FD32P20
    if k == 'FD16P20v2f': return FD16, FDP + [f'abl_fd10_pop2kd05w192_s{s}' for s in S4] + [f'abl_fd10_pfkd05w192_s{s}' for s in S4], 0.2
    if k == 'FD16P20w12d': return FD16, [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 963)] + FDPd, 0.2
    if k == 'FD8P20x8': return FD8, FDP + FDPd, 0.2
    if k.startswith('FDX16P20x8:'): return fdx(k.split(':')[1], S8), FDP + FDPd, 0.2
    if k.startswith('FDX16P20:'): return fdx(k.split(':')[1], S8), FDP, 0.2
    if k.startswith('FDX16:'): return fdx(k.split(':')[1], S8), [], 0.0
    raise SystemExit(f'unknown key {k}')
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
Z = {'val': lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1], 'test': lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]}
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
def fmt(d, w=None):
    if w is None: return f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
    m, se = pmix.wmean_se(d, w); return f'{m:+.5f}+-{se:.5f}'
def pool(names, sp):
    z = Z[sp]; ftn, ton = SUF[sp]
    miss = [f'nn_{m}_{s}.npy' for m in names for s in (ftn, ton) if not (P / f'nn_{m}_{s}.npy').exists()]
    assert not miss, f'missing in preds/: {miss[:6]} ... ({len(miss)} files)'
    return (np.mean([z(np.load(P / f'nn_{m}_{ftn}.npy')) for m in names], 0), np.mean([z(np.load(P / f'nn_{m}_{ton}.npy')) for m in names], 0))
fin = lambda sp, br: 0.5 * Z[sp](br[0]) + 0.5 * Z[sp](br[1])
stored = lambda tag, sp: (np.load(P / f'ens_{tag}_ft_{sp}.npy'), np.load(P / f'ens_{tag}_to_{sp}.npy'))
REF = {k: {sp: fin(sp, stored(t, sp)) for sp in ('val', 'test')} for k, t in (('W8', 'W8kd'), ('W16', 'W16kd'), ('W8P20', 'W8P20kd'))}
eR = {k: ev(REF[k]['val']) for k in REF}; wts = pmix.weights(); big = Mv >= 50
print('references (final val): ' + ' | '.join(f'{k} {eR[k].mean():.5f}' for k in REF) + '   public: W8 0.394486, W16 0.394483, W8P20 0.395009')
E_ALL = {}; T_ALL = {}
MROW = {'val': np.repeat(Mv, Mv), 'test': np.repeat(Mt, Mt)}
for k in want:
    # '<KEY>@m<ws>_<wl>' = feed-size rule: popularity-light weight ws on feeds with M < 50, wl on feeds with M >= 50 (declared forms only; e.g. FD16P20x8@m0.2_0.4)
    kk, mr = (k.split('@m') + [None])[:2]
    base, pop, w = spec(kk); B = {}
    for sp in ('val', 'test'):
        b = pool(base, sp)
        if pop:
            wr = w if mr is None else np.where(MROW[sp] >= 50, float(mr.split('_')[1]), float(mr.split('_')[0])).astype(np.float32)
            p1 = pool(pop, sp); z = Z[sp]; b = ((1 - wr) * z(b[0]) + wr * z(p1[0]), (1 - wr) * z(b[1]) + wr * z(p1[1]))
        B[sp] = b
    e = ev(fin('val', B['val'])); tfin = fin('test', B['test']); E_ALL[k] = (e, ev(B['val'][0]), ev(B['val'][1])); T_ALL[k] = tfin
    print(f'{k}: {len(base)} base + {len(pop)} pop-light members at w {w} | final val {e.mean():.5f} | FT {ev(B["val"][0]).mean():.5f} TO {ev(B["val"][1]).mean():.5f}')
    for r in REF:
        d = e - eR[r]
        print(f'   vs {r:6s}: {fmt(d)} | M<50 {fmt(d[~big])} M>=50 {fmt(d[big])} | pmix public {fmt(d, wts["public"])} private {fmt(d, wts["private"])} | test corr {np.corrcoef(tfin, REF[r]["test"])[0, 1]:.4f}'
              f' | paired public SE {d.std() / np.sqrt(8562):.5f}')
    if build:
        tag = k.replace(':', '_').replace('@m', '_m').replace('.', '') + 'kd'; out = S / f'cand_{tag}_r4.parquet'; assert not out.exists() and not list(P.glob(f'ens_{tag}_*')), f'{tag} exists'
        for sp in ('val', 'test'):
            np.save(P / f'ens_{tag}_ft_{sp}.npy', B[sp][0].astype(np.float32)); np.save(P / f'ens_{tag}_to_{sp}.npy', B[sp][1].astype(np.float32))
        make_sub(tfin.astype(np.float32), str(out)); print('   written', out, '| members:', ','.join(base + pop))
# pairwise read of the requested keys against each other (paired per feed; later key minus earlier key): FINAL, FT branch, train-only branch
ks = list(E_ALL)
for i, a_ in enumerate(ks):
    for b_ in ks[:i]:
        d = E_ALL[a_][0] - E_ALL[b_][0]; dft = E_ALL[a_][1] - E_ALL[b_][1]; dto = E_ALL[a_][2] - E_ALL[b_][2]
        print(f'PAIR {a_} - {b_}: final {fmt(d)} (z {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}) | M<50 {fmt(d[~big])} M>=50 {fmt(d[big])} | pmix public {fmt(d, wts["public"])} private {fmt(d, wts["private"])}'
              f' | FT branch {fmt(dft)} TO branch {fmt(dto)} | test corr {np.corrcoef(T_ALL[a_], T_ALL[b_])[0, 1]:.4f}')
