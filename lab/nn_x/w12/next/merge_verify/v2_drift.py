"""V2. Independent re-derivation of the drift table (part B of the 'merge' study): candidate vs subs/cand_W8kd_r2.parquet. Label-free.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/next/merge_verify/v2_drift.py > nn_x/w12/next/merge_verify/out/v2_drift.txt 2>&1"""
import sys, time; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
t0 = time.time(); V, T = Sp('val'), Sp('test')
cold = {S.name: (np.nan_to_num(np.asarray(np.load(FEATS / S.name / 'it_hist_all_luo.npy'), np.float64), nan=0.0) == 0).astype(np.float64) for S in (V, T)}
pub = T.ts < BOUND
GT = {'Sat': (T.day == 20722), 'Sun am (pub)': (T.day == 20723) & pub, 'Sun pm (priv)': (T.day == 20723) & ~pub, 'Mon': T.day == 20724, 'Tue 09-29': T.day == 20725, 'PUBLIC': pub, 'PRIVATE': ~pub}
GV = {'val Tue': V.day == 20718, 'val Thu': V.day == 20720}
print('feeds: ' + ', '.join(f'{k} {int(m.sum())}' for k, m in {**GV, **GT}.items()) + f' | other test days: {int((~np.isin(T.day, (20722, 20723, 20724, 20725))).sum())} | other val feeds: {int((~np.isin(V.day, (20718, 20720))).sum())}')
print('cold share of candidates: ' + ', '.join(f'{k} {np.mean((np.add.reduceat(cold[S.name], S.off) / S.M)[m]):.3f}' for S, G in ((V, GV), (T, GT)) for k, m in G.items()))


def scores(S, tag):
    if tag == 'W8TOkd': return S.z(np.load(P / f'ens_{tag}_to_{S.name}.npy'))
    return S.z(fin(S, tag))


def cw(S, z):
    c = cold[S.name]; nc = np.add.reduceat(c, S.off); nw = S.M - nc; ok = (nc > 0) & (nw > 0)
    g = np.add.reduceat(z * c, S.off) / np.maximum(nc, 1) - np.add.reduceat(z * (1 - c), S.off) / np.maximum(nw, 1)
    return np.where(ok, g, np.nan)


def stats(S, zc, zr):
    rc, rr = S.rank(zc), S.rank(zr); sp = S.rf(S.z(rc), S.z(rr))
    t1 = (np.maximum.reduceat(np.where(rc == S.Mrow - 1, np.arange(S.N), -1), S.off) == np.maximum.reduceat(np.where(rr == S.Mrow - 1, np.arange(S.N), -1), S.off)).astype(np.float64)
    pc, pr = S.Mrow - 1 - rc, S.Mrow - 1 - rr; m3 = pr < 3
    mv = np.bincount(S.fid[m3], np.abs(pc - pr)[m3], S.n) / np.maximum(np.bincount(S.fid[m3], minlength=S.n), 1)
    return dict(spearman=sp, top1=t1, top3move=mv, cw_score=cw(S, zc) - cw(S, zr), cw_rank=cw(S, S.z(rc)) - cw(S, S.z(rr)))


def ms(x, m):
    v = x[m & np.isfinite(x)]; return v.mean(), np.std(v) / np.sqrt(len(v)), len(v)


files = {'W16kd': 'cand_W16kd_r3', 'W8P20kd': 'cand_W8P20kd_r3', 'W8TOkd': 'cand_W8TOkd_r3', 'S16kd': 'cand_S16kd_r2', 'Q2a8s3': 'cand_Q2a8_ftbag3_memeq'}
refT, refV = scores(T, 'W8kd'), scores(V, 'W8kd'); sref = T.read_file(ROOT / 'subs/cand_W8kd_r2.parquet')
print(f'reference file == rank of preds/ens_W8kd blend: {np.mean(T.rank(refT) == sref):.6f}')
for tag, fn in files.items():
    zc = scores(T, tag); sfile = T.read_file(ROOT / f'subs/{fn}.parquet'); same = np.mean(T.rank(zc) == sfile)
    # test statistics are computed from the FILES (ranks) for spearman / top1 / top3move / cw_rank, and from the model scores for cw_score
    FT_ = stats(T, zc, refT); Ffile = stats(T, sfile, sref); FV_ = stats(V, scores(V, tag), refV)
    print(f'\n== {tag}  (file {fn}; file == rank of the ens blend in {same:.6f} of rows)')
    print(f'  {"group":14s} {"feeds":>6s} | {"cw_score":>18s} | {"cw_rank (files)":>18s} | {"spearman (files)":>18s} | {"same top-1":>16s} | {"top3 move":>16s}')
    for lab, m in GV.items():
        a = ms(FV_['cw_score'], m); b = ms(FV_['cw_rank'], m); c = ms(FV_['spearman'], m); d = ms(FV_['top1'], m); e = ms(FV_['top3move'], m)
        print(f'  {lab:14s} {int(m.sum()):6d} | {a[0]:+.4f} ± {a[1]:.4f} | {b[0]:+.4f} ± {b[1]:.4f} | {c[0]:.4f} ± {c[1]:.4f} | {d[0]:.3%} ± {d[1]:.2%} | {e[0]:.3f} ± {e[1]:.3f}')
    for lab, m in GT.items():
        a = ms(FT_['cw_score'], m); b = ms(Ffile['cw_rank'], m); c = ms(Ffile['spearman'], m); d = ms(Ffile['top1'], m); e = ms(Ffile['top3move'], m)
        print(f'  {lab:14s} {int(m.sum()):6d} | {a[0]:+.4f} ± {a[1]:.4f} | {b[0]:+.4f} ± {b[1]:.4f} | {c[0]:.4f} ± {c[1]:.4f} | {d[0]:.3%} ± {d[1]:.2%} | {e[0]:.3f} ± {e[1]:.3f}')
    def con(x, a, b, xb=None):
        ma, sa, _ = ms(x, a); mb, sb, _ = ms(x if xb is None else xb, b); return ma - mb, np.hypot(sa, sb), (ma - mb) / np.hypot(sa, sb)
    for k in ('cw_score', 'spearman', 'top1'):
        x = FT_[k] if k == 'cw_score' else Ffile[k]
        c1 = con(x, GT['PRIVATE'], GT['PUBLIC']); c2 = con(x, GT['Tue 09-29'], GV['val Tue'], FV_[k]); c3 = con(x, GT['Tue 09-29'], GT['Sat'])
        # paired-by-nothing alternative: cluster by user is not possible across groups; report the plain two-sample z
        print(f'  contrast {k:9s}: PRIVATE - PUBLIC {c1[0]:+.4f} ± {c1[1]:.4f} (z {c1[2]:+.1f}) | Tue 09-29 - val Tue {c2[0]:+.4f} ± {c2[1]:.4f} (z {c2[2]:+.1f}) | Tue 09-29 - Sat {c3[0]:+.4f} ± {c3[1]:.4f} (z {c3[2]:+.1f})')
    if tag == 'W8TOkd':
        # the known answer of ft_redesign finding 6: FT pool minus TO pool, cold-minus-warm, by day (z units)
        for S, G in ((V, GV), (T, GT)):
            ft = S.z(np.load(P / f'ens_W8kd_ft_{S.name}.npy')); to = S.z(np.load(P / f'ens_W8kd_to_{S.name}.npy')); x = cw(S, ft) - cw(S, to)
            print('  known answer (ft_redesign #6: +0.021 val Tue ... +0.044 last day), cw of z(FT pool) - z(TO pool): ' + ', '.join(f'{k} {ms(x, m)[0]:+.4f}' for k, m in G.items()))
# A/A null of the flag: W8 members split in two halves would be the right null, but no half-pool test files exist as files; use the by-day spread of the
# W16-vs-W8 contrast as a scale only.
print(f'\n[done in {time.time() - t0:.0f} s]')
