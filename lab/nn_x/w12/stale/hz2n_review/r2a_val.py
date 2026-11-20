"""R2a: my own re-implementation of the 'pre2both' rule on VAL rows (numpy, float64 per-feed moments), against the witness files and against the builder's function.
Label-free: no cand_target is opened."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
CF = f'{R}/solvers/out/stale/cf'
FC = f'{R}/nn/feat_cache/val'
res = {}
f = pl.read_parquet(f'{R}/data/flat/val/feeds.parquet')
M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
assert (off == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
fid = np.repeat(np.arange(len(M)), M)
L = lambda n: np.load(f'{FC}/{n}.npy')
fe, fs, fz, luo, idc = L('pre2_fe'), L('pre2_fe_s'), L('pre2_fe_z'), L('it_hist_all_luo'), L('pre2_id_contrib')
noid = np.load(f'{R}/data/feats/val/pre2_fe_noid.npy')
b1 = np.load(f'{R}/nn_x/r6/stale_cf/both_it_hist_all_luo.npy')
p2 = '/tmp/scratchpad/review/shift/out/cf/both_it_hist_all_luo.npy'
res['both_counter_two_copies_equal'] = bool(os.path.exists(p2) and np.array_equal(np.load(p2), b1))
nc = (b1 == 0) & (luo > 0)
res['nc_rows'] = int(nc.sum()); res['nc_share'] = float(nc.mean())
res['nc_equals_witness_newly_cold'] = bool(np.array_equal(nc, np.load(f'{CF}/newly_cold.npy')))
bits = lambda a: np.ascontiguousarray(a, np.float32).view(np.int32)

def my_affine(x, ref):  # moment match of x to ref on rows > -2 and finite (written from the docstring of pretrain/assemble.affine)
    mx = np.isfinite(x) & (x > -2); mr = np.isfinite(ref) & (ref > -2)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    return a, b

def ap(v, a, b):
    return np.where(v > -2, a * v + b, v).astype(np.float32)

def z64(x):  # within-feed z, float64 moments, population SD + 1e-9
    x = x.astype(np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, (x - mu[fid]) ** 2, len(M)) / M, 0))
    return (x - mu[fid]) / (sd[fid] + 1e-9), sd

fes, noids = [], []
for t, c in (('v3At2', 'v3t2'), ('v4At2', 'v4t2')):
    x_fe = np.load(f'{R}/preds/pre_{t}_fe_val.npy').astype(np.float32); r_fe = np.load(f'{R}/preds/pre_{c}_fe_val.npy').astype(np.float32)
    x_no = np.load(f'{R}/preds/pre_{t}_fe_noid_val.npy').astype(np.float32); r_no = np.load(f'{R}/preds/pre_{c}_fe_noid_val.npy').astype(np.float32)
    a1, b1_ = my_affine(x_fe, r_fe); a2, b2_ = my_affine(x_no, r_no)
    res[f'calib_{t}'] = dict(fe=(a1, b1_), fe_noid=(a2, b2_))
    fes.append(ap(x_fe, a1, b1_)); noids.append(ap(x_no, a2, b2_))
    res[f'raw_{t}_sentinels_le_-2'] = int((x_fe <= -2).sum()); res[f'raw_{t}_nonfinite'] = int((~np.isfinite(x_fe)).sum())
res['repro_stored_fe_max_abs'] = float(np.abs(np.mean(fes, 0) - fe).max())
res['repro_stored_noid_max_abs'] = float(np.abs(np.mean(noids, 0) - noid).max())
# my rule
fe2 = np.where(nc, noid, fe).astype(np.float32)
fs2 = ((fe2 + 20) / 6).astype(np.float32)
zs, sds = zip(*[z64(np.where(nc, n_, f_).astype(np.float32)) for f_, n_ in zip(fes, noids)])
z2 = np.mean(zs, 0)
degen_feed = (np.minimum(sds[0], sds[1]) < 1e-3)
degen = degen_feed[fid]
W = {k: np.load(f'{CF}/pre2both_{k}.npy') for k in ('pre2_fe', 'pre2_fe_z', 'pre2_fe_s')}
res['mine_vs_witness_fe_rows_diff_bitwise'] = int((bits(fe2) != bits(W['pre2_fe'])).sum())
res['mine_vs_witness_fe_s_rows_diff_bitwise'] = int((bits(fs2) != bits(W['pre2_fe_s'])).sum())
dz = np.abs(z2 - W['pre2_fe_z'].astype(np.float64))
res['mine_vs_witness_fe_z_max_abs_nondegenerate'] = float(dz[~degen].max())
res['mine_vs_witness_fe_z_p9999'] = float(np.quantile(dz, 0.9999))
res['degenerate_feeds'] = int(degen_feed.sum()); res['degenerate_rows'] = int(degen.sum())
res['mine_vs_witness_fe_z_max_abs_degenerate'] = float(dz[degen].max()) if degen.any() else None
# power controls: wrong variants must NOT match the witness
z_unmixed = np.mean([z64(f_)[0] for f_ in fes], 0)
res['control_unmixed_z_vs_witness_max_abs'] = float(np.abs(z_unmixed - W['pre2_fe_z']).max())
z_naive = z64(fe2)[0]
res['control_naive_z_of_config_mean_vs_witness_max_abs'] = float(np.abs(z_naive - W['pre2_fe_z']).max())
res['stored_z_vs_witness_rows_changed'] = int((bits(fz) != bits(W['pre2_fe_z'])).sum())
res['witness_z_changed_rows_in_feeds_without_nc'] = int(((bits(fz) != bits(W['pre2_fe_z'])) & ~(np.bincount(fid, nc, len(M)) > 0)[fid]).sum())
# the witness applies (fe2+20)/6 on EVERY row; stored fs equal to that on non-nc rows?
res['stored_fe_s_vs_(fe+20)/6_rows_diff_bitwise'] = int((bits(fs) != bits(((fe + 20) / 6).astype(np.float32))).sum())
# has flag
res['has_changes'] = int((np.isnan(fe2) != np.isnan(fe)).sum()); res['nan_rows_val_fe'] = int(np.isnan(fe).sum())
# facts used later (label-free): ID contribution on cold rows, offset noid - fe on no-ID rows
res['val_P(idc!=0|luo==0)'] = float((idc[luo == 0] != 0).mean()); res['val_cold_rows'] = int((luo == 0).sum())
res['val_nc_share_with_id'] = float((idc[nc] != 0).mean())
res['val_offset_noid_minus_fe_on_noID_rows_mean'] = float((noid - fe)[idc == 0].mean())
res['val_offset_noid_minus_fe_on_noID_rows_absmax'] = float(np.abs(noid - fe)[idc == 0].max())
res['val_offset_on_nc_noID_rows_mean'] = float((noid - fe)[nc & (idc == 0)].mean())
res['val_shift_on_nc_ID_rows_mean'] = float((noid - fe)[nc & (idc != 0)].mean())
# builder's function on the same inputs
sys.path.insert(0, f'{R}/nn_x/w12/stale/hz2n')
import enc_common as ec
bfes, bnoids = ec.val_configs()
bfe2, bfs2, bz2, bhas, bfnc = ec.mix_rows(nc, fe, noid, fs, fz, bfes, bnoids, off, M)
res['builder_vs_witness_rows_diff_bitwise'] = [int((bits(bfe2) != bits(W['pre2_fe'])).sum()), int((bits(bz2) != bits(W['pre2_fe_z'])).sum()), int((bits(bfs2) != bits(W['pre2_fe_s'])).sum())]
res['builder_vs_mine_z_max_abs_nondegenerate'] = float(np.abs(bz2 - z2)[~degen].max())
res['builder_configs_vs_mine_max_abs'] = [float(np.abs(a - b).max()) for a, b in zip(bfes + bnoids, fes + noids)]
res['R2a_pass'] = bool(res['nc_equals_witness_newly_cold'] and res['mine_vs_witness_fe_rows_diff_bitwise'] == 0 and res['mine_vs_witness_fe_s_rows_diff_bitwise'] == 0
                       and res['mine_vs_witness_fe_z_max_abs_nondegenerate'] <= 1e-4)
json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/r2a_val.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
