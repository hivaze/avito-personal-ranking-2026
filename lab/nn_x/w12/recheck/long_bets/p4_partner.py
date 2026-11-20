"""P4/P6/P7: (P4) what would a partner model have to look like to add to W8? Frontier law re-read at the W8 level with every
different-lineage score on disk; (P6) fine-tune damage on distilled students; (P7) student-vs-pool function gap by split.
Val only for P4/P6 (FT preds are val OOF); CPU, 2 threads."""
from lb_common import *
from common import feed_meta as _fm
ft_, Mt, ot = _fm('test'); zt = lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]
fin = lambda tag, sp, z: 0.5 * z(np.load(P / f'ens_{tag}_ft_{sp}.npy')) + 0.5 * z(np.load(P / f'ens_{tag}_to_{sp}.npy'))
w8 = zv(fin('W8kd', 'val', zv)); e8 = ev(w8); w8t = zt(fin('W8kd', 'test', zt))
print(f'W8 final val {e8.mean():.5f}')
print('== P4: partners on disk (val; blend = (1-w) z(W8) + w z(partner)) ==')
print(f'{"partner":28s} {"val":>8s} {"gap":>8s} {"rho_val":>7s} {"rho_tst":>7s} | {"w=0.1":>18s} {"w=0.2":>18s} {"w=0.3":>18s} {"w=0.5":>18s} | law(0.5)')
cands = {}
for tag in ('Q2a8s3', 'G3', 'Pall', 'Cd', 'S16kd'):
    cands[f'pool {tag} (final)'] = (zv(fin(tag, 'val', zv)), zt(fin(tag, 'test', zt)))
def fam(ms, suffix_val, suffix_test):
    return (np.mean([zv(np.load(P / f'nn_{m}_{suffix_val}.npy')) for m in ms], 0), np.mean([zt(np.load(P / f'nn_{m}_{suffix_test}.npy')) for m in ms], 0))
cands['F10 GRU x6 (TO, hard labels)'] = fam([f'v23_gru_pre2s_s{i}' for i in range(501, 507)], 'trainonly_val', 'trainonly_test')
cands['base x4 (TO, hard labels)'] = fam([f'w5_e10_base_s{s}' for s in S4], 'trainonly_val', 'trainonly_test')
cb = ROOT / 'nn_x/cb26/data'
cands['GBDT CatBoost YetiRank xm'] = (zv(np.load(cb / 'e3_xm_yeti_val.npy')), zt(np.load(cb / 'e3_xm_yeti_test.npy')))
A = ROOT / 'archive/preds'
for nm, fl in (('GBDT xgb_tr1m_ndcg', 'xgb_tr1m_ndcg'), ('GBDT cb_tr300k_none', 'cb_tr300k_none')):
    try: cands[nm] = (zv(np.load(A / f'{fl}_val.npy')), zt(np.load(A / f'{fl}_test.npy')))
    except Exception as e: print('  skip', nm, e)
L = ROOT / 'nn_x/llm/preds'
try: cands['LLM Qwen3-0.6B x2 (150K feeds)'] = (np.mean([zv(np.load(L / f'llm06_L1_e4_s{s}_trainonly_val.npy')) for s in (23, 24)], 0), np.mean([zt(np.load(L / f'llm06_L1_e4_s{s}_trainonly_test.npy')) for s in (23, 24)], 0))
except Exception as e: print('  skip llm', e)
rowsL = []
for nm, (pv, pt) in cands.items():
    e = ev(pv); gap = e8.mean() - e.mean(); rho = wcorr(w8, pv); rt = wcorr(w8t, pt)
    bl = [ev((1 - w) * w8 + w * zv(pv)) - e8 for w in (0.1, 0.2, 0.3, 0.5)]
    law = 0.0186 * (1 - rho) - 0.378 * abs(gap)  # gain of the equal mix over the BETTER pool (llm_large; sol_ensemble 2.5)
    print(f'{nm:28s} {e.mean():8.5f} {gap:+8.5f} {rho:7.4f} {rt:7.4f} | ' + ' '.join(f'{f(b):>18s}' for b in bl) + f' | {law:+.5f}')
    rowsL.append((rho, gap, bl[3].mean() - max(0, -gap)))
# re-fit the law's two coefficients at the W8 level (equal mix gain over the better of the two)
R = np.array(rowsL); X = np.stack([1 - R[:, 0], -np.abs(R[:, 1])], 1); coef, *_ = np.linalg.lstsq(X, R[:, 2], rcond=None)
print(f' law refit on these {len(R)} partners: gain(0.5 mix over the better) = {coef[0]:.4f} (1-rho) - {coef[1]:.3f} |gap|   (on record: 0.0186, 0.378)')
print(' contour (law on record | refit): equal-mix gain over the better pool')
for gap in (0.0, 0.0005, 0.001, 0.002, 0.004):
    print(f'   |gap| {gap:.4f}: ' + '  '.join(f'rho {r:.2f}: {0.0186*(1-r)-0.378*gap:+.4f}|{coef[0]*(1-r)-coef[1]*gap:+.4f}' for r in (0.99, 0.97, 0.95, 0.93, 0.90, 0.86)))

print('== P6: fine-tune effect per member (val OOF n2cpv3s3 - train-only), students vs hard-label members ==')
groups = {'W8 wide students': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4], 'kd05 d128 students': [f'w5_e10_kd05_s{s}' for s in S4] if (P / 'nn_w5_e10_kd05_s951_n2cpv3s3_val_oof.npy').exists() else [],
          'kd05t25 d128 students': [f'w5_e10_kd05t25_s{s}' for s in S4], 'F7x hard (r6_cd_m02 r101-107)': [f'r6_cd_m02_r{i}' for i in range(101, 108)], 'F10 GRU hard': [f'v23_gru_pre2s_s{i}' for i in range(501, 507)],
          'pop1 hard': [f'abl_e10_pop1_s{s}' for s in range(951, 959)]}
for g, ms in groups.items():
    ms = [m for m in ms if (P / f'nn_{m}_n2cpv3s3_val_oof.npy').exists() and (P / f'nn_{m}_trainonly_val.npy').exists()]
    if not ms: continue
    FT = [zv(np.load(P / f'nn_{m}_n2cpv3s3_val_oof.npy')) for m in ms]; TO = [zv(np.load(P / f'nn_{m}_trainonly_val.npy')) for m in ms]
    d = np.mean([ev(a) - ev(b) for a, b in zip(FT, TO)], 0); pf = ev(np.mean(FT, 0)); pt_ = ev(np.mean(TO, 0)); mix = ev(0.5 * zv(np.mean(FT, 0)) + 0.5 * zv(np.mean(TO, 0)))
    print(f' {g:30s} n={len(ms)} member FT-TO {f(d)} (M<50 {d[Mv<50].mean():+.5f}, M>=50 {d[Mv>=50].mean():+.5f}) | pool FT {pf.mean():.5f} TO {pt_.mean():.5f} FT-TO {f(pf-pt_)} | 0.5/0.5 {mix.mean():.5f} (vs TO {f(mix-pt_)})'
          f' | corr(FT pool, TO pool) {wcorr(np.mean(FT,0), np.mean(TO,0)):.4f}')

print('== P7: single student vs leave-one-out pool of the other 7 (train-only preds), by split ==')
ms = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4]
V = [zv(np.load(P / f'nn_{m}_trainonly_val.npy')) for m in ms]; Tt = [zt(np.load(P / f'nn_{m}_trainonly_test.npy')) for m in ms]; H = [zh(np.load(W / f'holdout/nn_{m}_holdout.npy')) for m in ms]
pub = np.repeat(np.arange(len(Mt)) < 8562, Mt)  # first ~30% of test by time = public (flat order is time order? checked below)
ts = ft_['timestamp'].to_numpy(); order_ok = bool((np.diff(ts) >= 0).all())
if not order_ok:
    thr = np.sort(ts)[8561]; pub = np.repeat(ts <= thr, Mt)
for nm, Z, mask in (('holdout', H, None), ('val', V, None), ('test public', Tt, pub), ('test private', Tt, ~pub)):
    cs = []; mm = []
    for i in range(8):
        rest = np.mean([Z[j] for j in range(8) if j != i], 0)
        a, b = (Z[i], rest) if mask is None else (Z[i][mask], rest[mask])
        cs.append(wcorr(a, b))
    iu = np.triu_indices(8, 1); C = np.corrcoef([z if mask is None else z[mask] for z in Z])[iu].mean()
    print(f' {nm:13s} corr(member, pool of other 7) {np.mean(cs):.4f} | member-member {C:.4f} | 1-rho_mm {1-C:.4f}')
