"""C. How far does the FT move predictions? corr(z_FT, z_TO) member / pool, val OOF vs test; member-member corr in each branch;
common-mode vs member-specific share of the FT shift; fold-to-fold agreement of the shift on test (test_folds)."""
from lib import *
def spear(a, b, sp):
    r = (rkv if sp == 'val' else rkt); M, off = (Mv, ov) if sp == 'val' else (Mt, ot)
    return feed_corr(r(a), r(b), M, off)
print('== member level: Pearson of per-feed z (global) and mean per-feed Spearman, FT vs TO')
print(f"{'group':15s} n | val: corr_z  spearman(mean over feeds)  RMS shift | test: corr_z  spearman  RMS shift | RMS test/val")
SH = {}
for g, ms in GROUPS.items():
    r = []
    for m in ms:
        o = []
        for sp in ('val', 'test'):
            a, b = z(m, 'ft', sp), z(m, 'to', sp); s = a - b; SH[(m, sp)] = s
            M, off = (Mv, ov) if sp == 'val' else (Mt, ot)
            o += [np.corrcoef(a, b)[0, 1], np.nanmean(spear(raw(m, 'ft', sp), raw(m, 'to', sp), sp)[M > 2]), np.sqrt((s ** 2).mean())]
        r.append(o)
    r = np.array(r).mean(0)
    print(f"{g:15s} {len(ms):2d} | {r[0]:.5f} {r[1]:.5f} {r[2]:.4f} | {r[3]:.5f} {r[4]:.5f} {r[5]:.4f} | {r[5]/r[2]:.3f}")
print('\n== pool level (FT pool vs TO pool)')
for k, ms in (('W8', W8), ('N8', N8), ('S16', W8 + N8), ('C3', C3M), ('m02x7', GROUPS['H:m02']), ('pop1x8', GROUPS['H:pop1'])):
    o = []
    for sp in ('val', 'test'):
        zz = zv if sp == 'val' else zt
        a, b = zz(pool(ms, 'ft', sp)), zz(pool(ms, 'to', sp)); s = a - b
        o += [np.corrcoef(a, b)[0, 1], np.sqrt((s ** 2).mean())]
    print(f" {k:7s} val corr {o[0]:.5f} RMS {o[1]:.4f} | test corr {o[2]:.5f} RMS {o[3]:.4f} | RMS test/val {o[3]/o[1]:.3f}")
print('\n== member-member corr of z inside a branch (mean off-diagonal), val | test')
for k, ms in (('S:kd05w192', GROUPS['S:kd05w192']), ('S:kd05t25w192', GROUPS['S:kd05t25w192']), ('W8', W8), ('N8', N8), ('H:m02', GROUPS['H:m02']), ('H:pop1', GROUPS['H:pop1']), ('H:gru', GROUPS['H:gru'])):
    o = []
    for sp in ('val', 'test'):
        for kind in ('to', 'ft'):
            c = np.corrcoef([z(m, kind, sp) for m in ms]); o.append(c[np.triu_indices(len(ms), 1)].mean())
    print(f" {k:15s} val TO {o[0]:.4f} FT {o[1]:.4f} | test TO {o[2]:.4f} FT {o[3]:.4f}")
print('\n== FT shift s_m = z_FT - z_TO: common-mode share = var(mean_m s_m) / mean_m var(s_m); mean pairwise corr of shifts')
for k, ms in (('W8', W8), ('N8', N8), ('H:m02', GROUPS['H:m02']), ('H:pop1', GROUPS['H:pop1']), ('C3', C3M)):
    o = []
    for sp in ('val', 'test'):
        S = np.array([SH[(m, sp)] for m in ms]); c = np.corrcoef(S)
        o += [S.mean(0).var() / S.var(1).mean(), c[np.triu_indices(len(ms), 1)].mean()]
    print(f" {k:7s} val share {o[0]:.3f} pair-corr {o[1]:.3f} | test share {o[2]:.3f} pair-corr {o[3]:.3f}")
# cross-family: is the students' common shift the same direction as the hard-label members' common shift?
for sp in ('val', 'test'):
    a = np.mean([SH[(m, sp)] for m in W8], 0); b = np.mean([SH[(m, sp)] for m in C3M], 0); c = np.mean([SH[(m, sp)] for m in N8], 0)
    print(f" {sp}: corr(common shift W8, common shift C3) {np.corrcoef(a, b)[0,1]:.3f} | W8 vs N8 {np.corrcoef(a, c)[0,1]:.3f} | RMS W8 {np.sqrt((a**2).mean()):.4f} C3 {np.sqrt((b**2).mean()):.4f} N8 {np.sqrt((c**2).mean()):.4f}")
print('\n== fold-to-fold agreement on test (test_folds = per-fold per-feed z of the EMA-final model, 3-seed mean); students W8')
for m in W8[:2] + W8[4:6]:
    tf = np.load(W / f'nn_{m}_n2cpv3s3_test_folds.npy').astype(np.float64); to = z(m, 'to', 'test')
    Z = np.array([zt(x) for x in tf]); S = Z - to
    c = np.corrcoef(S); cz = np.corrcoef(Z); iu = np.triu_indices(5, 1)
    rm = np.sqrt((S ** 2).mean(1)).mean(); rmean = np.sqrt((S.mean(0) ** 2).mean())
    print(f" {m:28s} fold-fold corr of z {cz[iu].mean():.5f}; corr of shifts {c[iu].mean():.3f}; RMS single-fold shift {rm:.4f}; RMS 5-fold-mean shift {rmean:.4f}; val OOF RMS {np.sqrt((SH[(m,'val')]**2).mean()):.4f}")
