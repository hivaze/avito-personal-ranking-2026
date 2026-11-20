"""R2b + gate recount (G1, G2 sums) + N items, FULL arrays (246,745,836 train rows), one view per process. My own code; reads the builder's FILES only. Label-free.
usage: r2b_full.py <a|b|c>"""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
FC, F, HZ2, N = f'{R}/nn/feat_cache/train', f'{R}/data/feats/train', f'{R}/nn_x/w12/stale/hz2', f'{R}/nn_x/w12/stale/hz2n'
view = sys.argv[1]
CFG = {'a': dict(base=f'{HZ2}/feats/train', keep=f'{HZ2}/feats/train/hz2_keep_rows.npy', d='feats_n', F='feats_nF', FS=None),
       'b': dict(base=f'{HZ2}/feats_L/train', keep=f'{HZ2}/feats_L/train/hz2_keep_rows.npy', d='feats_Ln', F='feats_LnF', FS=None),
       'c': dict(base=f'{R}/data/feats_new/train', keep=None, d='feats_hn', F='feats_hnF', FS='feats_hnFS')}[view]
mm = lambda p: np.load(p, mmap_mode='r')
T0 = time.time()
f = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
NF, NR = len(M), int(M.sum())
assert (off == np.concatenate([[0], np.cumsum(M)[:-1]])).all() and NR == 246745836
slot = (f['timestamp'].to_numpy().astype(np.int64) + 3 * 3600) // 86400 - 20702
win = np.zeros(NF, np.int8)
for k in (1, 2, 3):
    wf = np.load(f'{R}/data/pretrain/win{k}/window_feeds.npy'); assert (win[wf] == 0).all(); win[wf] = k
# the previous review's pool (hz2_review/pre2_tell.py): trPool private-like, days >= Sep 13, non-holdout
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
pool = (p > 0.5) & (M >= 50); pool[hold] = False
PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy'); assert PP.shape == (NF,)
pool13 = pool & ~PP & (slot >= 7)
ge13 = slot >= 7
fclass = (win.astype(np.int64) * 4 + ge13.astype(np.int64) * 2 + pool13.astype(np.int64))  # 16 feed classes
NFC, NRC = 16, 8  # row class = pop(4) * 2 + has
S_, V_ = mm(f'{FC}/it_hist_all_luo.npy'), mm(f"{CFG['base']}/it_hist_hz_all.npy")
KEEP = mm(CFG['keep']) if CFG['keep'] else None
FE, FS_, FZ, NOID, IDC = mm(f'{FC}/pre2_fe.npy'), mm(f'{FC}/pre2_fe_s.npy'), mm(f'{FC}/pre2_fe_z.npy'), mm(f'{F}/pre2_fe_noid.npy'), mm(f'{FC}/pre2_id_contrib.npy')
NEW, NEWS, NEWZ, HM = mm(f'{FC}/pre2_new.npy'), mm(f'{FC}/pre2_new_s.npy'), mm(f'{FC}/pre2_new_z.npy'), mm(f'{FC}/pre2_hmax32.npy')
SCV = {k: mm(f'{FC}/seq_cv_{k}.npy') for k in ('pop', 'invd20', 'norm')}
VSCV = {k: (mm(f"{CFG['base']}/seq_cv_hz_{k}.npy") if os.path.exists(f"{CFG['base']}/seq_cv_hz_{k}.npy") else None) for k in ('pop', 'invd20', 'norm')}
d = f"{N}/{CFG['d']}"
B_FE, B_FS, B_FZ, B_HAS, B_NC = mm(f'{d}/train/pre2_hz_fe.npy'), mm(f'{d}/train/pre2_hz_fe_s.npy'), mm(f'{d}/train/pre2_hz_fe_z.npy'), mm(f'{d}/aux/pre2_hz_fe_has.npy'), mm(f'{d}/train/hz2n_newly_cold.npy')
dF = f"{N}/{CFG['F']}"
B_NEW, B_NEWS, B_NEWZ = mm(f'{dF}/train/pre2_hz_new.npy'), mm(f'{dF}/train/pre2_hz_new_s.npy'), mm(f'{dF}/train/pre2_hz_new_z.npy')
B_SCV = {k: mm(f"{N}/{CFG['FS']}/train/seq_cv_hz_{k}.npy") for k in ('pop', 'invd20', 'norm')} if CFG['FS'] else None
for a in (S_, V_, FE, FS_, FZ, NOID, IDC, NEW, NEWS, NEWZ, HM, B_FE, B_FS, B_FZ, B_HAS, B_NC, B_NEW, B_NEWS, B_NEWZ):
    assert a.shape == (NR,), a.shape
assert B_NC.dtype == np.bool_ and all(a.dtype == np.float32 for a in (B_FE, B_FS, B_FZ, B_HAS, B_NEW, B_NEWS, B_NEWZ))
bits = lambda a: np.ascontiguousarray(a, np.float32).view(np.int32)
C = {}  # integer counters
def add(k, v): C[k] = C.get(k, 0) + int(v)
QN = ['n', 'fs_old', 'fs_new', 'fs_old2', 'fs_new2', 'z_old', 'z_new', 'dfe', 'dfs', 'absdz', 'hm', 'hm2', 'news_old', 'news_new', 'newz_old', 'newz_new',
      'new_changed', 'new_s_changed', 'scv_pop_pos', 'scv_invd_pos', 'scv_norm_pos', 'scv_pop_pos_FS', 'id_after', 'id_after2', 'idc_abs', 'S_sum', 'fe_s_changed', 'fe_z_changed']
ACC = {q: np.zeros(NFC * NRC) for q in QN}
feed_changed = np.zeros(NFC, np.int64); feed_n = np.bincount(fclass, minlength=NFC)
feed_has_nc = np.zeros(NFC, np.int64)
zmax = {'max_absdz_nonnc_rows': 0.0}
CH = 60000
for f0 in range(0, NF, CH):
    f1 = min(NF, f0 + CH); r0, r1 = int(off[f0]), int(off[f1 - 1] + M[f1 - 1])
    L = lambda a: np.asarray(a[r0:r1])
    S, V = L(S_), L(V_)
    keep = L(KEEP) if KEEP is not None else np.zeros(r1 - r0, bool)
    fe, fs, fz, noid, idc = L(FE), L(FS_), L(FZ), L(NOID), np.nan_to_num(L(IDC), nan=0.0)
    bfe, bfs, bfz, bhas, bnc = L(B_FE), L(B_FS), L(B_FZ), L(B_HAS), L(B_NC)
    Mc = M[f0:f1]; fl = np.repeat(np.arange(f1 - f0), Mc); fc = fclass[f0:f1][fl]
    add('nan_S', np.isnan(S).sum()); add('nan_V', np.isnan(V).sum())
    nc = (V == 0) & (S > 0) & ~keep
    eff = np.where(keep, S, V); cold = eff == 0
    add('rows', r1 - r0); add('nc', nc.sum()); add('keep', keep.sum()); add('cold_eff', cold.sum()); add('S0', (S == 0).sum())
    add('V_pos_S0', ((V > 0) & (S == 0)).sum()); add('V_gt_S', (V > S).sum()); add('keep_and_Vneq_S', (keep & (V != S)).sum())
    # R2b (1)..(6)
    add('r1_nc_mismatch', (bnc != nc).sum())
    fe2 = np.where(nc, noid, fe)
    add('r2_fe_bits_mismatch', (bits(bfe) != bits(fe2)).sum())
    exp_s = ((noid + np.float32(20)) / np.float32(6)).astype(np.float32)
    add('r3_fs_nonnc_bits_mismatch', ((bits(bfs) != bits(fs)) & ~nc).sum())
    dnc = bits(bfs) != bits(exp_s)
    add('r3_fs_nc_bits_mismatch', (dnc & nc).sum())
    add('r3_fs_nc_gt1e-6', (nc & ~(np.abs(bfs - exp_s) <= 1e-6) & ~(np.isnan(bfs) & np.isnan(exp_s))).sum())
    has_new = (~np.isnan(bfe)).astype(np.float32); has_old = ~np.isnan(fe)
    add('r4_has_file_vs_score', (bhas != has_new).sum()); add('r4_has_changed', ((bhas > 0) != has_old).sum()); add('r4_has_not01', ((bhas != 0) & (bhas != 1)).sum())
    kd = (bits(bfe) != bits(fe)) | (bits(bfs) != bits(fs)) | (bits(bfz) != bits(fz)) | ((bhas > 0) != has_old)
    add('r5_keep_rows_any_file_changed', (kd & keep).sum()); add('keep_rows_with_score', (keep & has_old).sum())
    fnc = np.bincount(fl, nc, f1 - f0) > 0
    zch = bits(bfz) != bits(fz)
    add('r6_z_changed_in_feeds_without_nc', (zch & ~fnc[fl]).sum())
    add('z_unchanged_rows_in_feeds_with_nc_and_score', (~zch & fnc[fl] & has_old).sum())
    add('nc_fe_finite_noid_nan', (nc & has_old & np.isnan(noid)).sum()); add('nc_fe_nan_noid_finite', (nc & ~has_old & ~np.isnan(noid)).sum())
    add('fe_nan_noid_finite_any', (~has_old & ~np.isnan(noid)).sum()); add('fe_finite_noid_nan_any', (has_old & np.isnan(noid)).sum())
    add('idc_ne0_and_S0', ((idc != 0) & (S == 0)).sum()); add('idc_ne0_no_score', ((idc != 0) & ~has_old).sum())
    add('sentinel_fe_le_-2', (fe <= -2).sum())
    # feed-level
    fch = np.bincount(fl, zch | (bits(bfs) != bits(fs)), f1 - f0) > 0
    feed_changed += np.bincount(fclass[f0:f1], fch, NFC).astype(np.int64); feed_has_nc += np.bincount(fclass[f0:f1], fnc, NFC).astype(np.int64)
    # row classes
    pop_ = np.where(nc, np.where(idc != 0, 2, 3), np.where(S == 0, 1, 0))  # 0 warm-eff (incl. keep rows with S > 0), 1 D as-built cold, 2 B, 3 E
    add('cold_eff_not_D_or_nc', (cold & (pop_ == 0)).sum()); add('D_not_cold_eff', ((S == 0) & ~cold).sum())
    cell = fc * NRC + pop_ * 2 + has_old
    bc = lambda w: np.bincount(cell, w, NFC * NRC)
    zc = lambda x: np.nan_to_num(np.clip(x, -5, 5), nan=0.0).astype(np.float64)
    n0 = lambda x: np.nan_to_num(x, nan=0.0).astype(np.float64)
    ACC['n'] += np.bincount(cell, minlength=NFC * NRC)
    a_fs, b_fs_ = n0(fs), n0(bfs)
    ACC['fs_old'] += bc(a_fs); ACC['fs_new'] += bc(b_fs_); ACC['fs_old2'] += bc(a_fs ** 2); ACC['fs_new2'] += bc(b_fs_ ** 2)
    zo, zn = zc(fz), zc(bfz)
    ACC['z_old'] += bc(zo); ACC['z_new'] += bc(zn); ACC['absdz'] += bc(np.abs(zn - zo))
    ACC['dfe'] += bc(n0(bfe) - n0(fe)); ACC['dfs'] += bc(b_fs_ - a_fs)
    hm = n0(np.clip(L(HM), -10, 10)); ACC['hm'] += bc(hm); ACC['hm2'] += bc(hm ** 2)
    new, news, newz, bnew, bnews, bnewz = L(NEW), L(NEWS), L(NEWZ), L(B_NEW), L(B_NEWS), L(B_NEWZ)
    ACC['news_old'] += bc(n0(news)); ACC['news_new'] += bc(n0(bnews)); ACC['newz_old'] += bc(zc(newz)); ACC['newz_new'] += bc(zc(bnewz))
    nch = bits(bnew) != bits(new)
    ACC['new_changed'] += bc(nch); ACC['new_s_changed'] += bc(bits(bnews) != bits(news))
    add('F_new_changed_nonnc', (nch & ~nc).sum()); add('F_new_s_changed_nonnc', ((bits(bnews) != bits(news)) & ~nc).sum())
    add('F_new_z_changed_in_feeds_without_nc', ((bits(bnewz) != bits(newz)) & ~fnc[fl]).sum())
    add('F_new_s_vs_(new+20)/6_gt1e-6_on_nc', (nc & ~(np.abs(bnews - (bnew + np.float32(20)) / np.float32(6)) <= 1e-6) & ~np.isnan(bnew)).sum())
    add('F_keep_rows_changed', (keep & (nch | (bits(bnews) != bits(news)) | (bits(bnewz) != bits(newz)))).sum())
    for k in ('pop', 'invd20', 'norm'):
        st = L(SCV[k])
        effk = np.where(keep, st, L(VSCV[k])) if VSCV[k] is not None else st
        ACC[{'pop': 'scv_pop_pos', 'invd20': 'scv_invd_pos', 'norm': 'scv_norm_pos'}[k]] += bc(effk > 0)
        if B_SCV is not None:
            bs = L(B_SCV[k])
            add(f'FS_{k}_bits_mismatch_vs_where(V==0,0,stored)', (bits(bs) != bits(np.where(V == 0, np.float32(0), st))).sum())
            add(f'FS_{k}_pos_on_cold', ((bs > 0) & cold).sum())
            if k == 'pop':
                ACC['scv_pop_pos_FS'] += bc(bs > 0)
    ida = (idc != 0) & (bits(bfe) != bits(noid))
    ACC['id_after'] += bc(ida); ACC['id_after2'] += bc((idc != 0) & ~(np.abs(bfs - exp_s) <= 1e-6))
    ACC['idc_abs'] += bc(np.abs(idc)); ACC['S_sum'] += bc(S)
    ACC['fe_s_changed'] += bc(bits(bfs) != bits(fs)); ACC['fe_z_changed'] += bc(zch)
    add('g1_before_all', (cold & (idc != 0)).sum()); add('g1_after_all', (cold & ida).sum())
    if f0 % (CH * 10) == 0:
        print(f'[{time.time() - T0:5.0f}s] feeds {f1}/{NF}', flush=True)
out = dict(view=view, cfg=CFG, counters=C, feed_n=feed_n.tolist(), feed_changed=feed_changed.tolist(), feed_has_nc=feed_has_nc.tolist(),
           acc={q: ACC[q].reshape(NFC, 4, 2).tolist() for q in QN}, sec=time.time() - T0)
json.dump(out, open(f'{R}/nn_x/w12/stale/hz2n_review/r2b_full_{view}.json', 'w'))
print(json.dumps(C, indent=1)); print('sec', time.time() - T0)
