"""OPTIONAL artefact of the review (label-free, val rows only, my own code; NOT part of the checks): the stale-val counterfactual files for a scenario 'both_pre2lt3' =
both_pre2full with the no-ID rule extended from 'both counter == 0' to 'both counter < 3' (the encoder's ID slot needs >= 3 occurrences: P(ID | counter in {1,2}) is exactly 0 on
val / public / private). mask3 = nc | ((both in {1,2}) & (pre2_id_contrib != 0)); nc = (both == 0) & (val counter > 0) (the witness rule).
Self-check: the SAME code with mask = nc must reproduce solvers/out/stale/cf/pre2both_* and pre2full_* (fe / new / _s bitwise; _z to float32-vs-float64 rounding)."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
CF = f'{R}/solvers/out/stale/cf'; FC = f'{R}/nn/feat_cache/val'; OUT = f'{R}/nn_x/w12/stale/hz2n_review/cf_lt3'
f = pl.read_parquet(f'{R}/data/flat/val/feeds.parquet')
M = f['M'].to_numpy().astype(np.int64); fid = np.repeat(np.arange(len(M)), M)
L = lambda n: np.load(f'{FC}/{n}.npy')
luo, idc = L('it_hist_all_luo'), np.nan_to_num(L('pre2_id_contrib'), nan=0.0)
both = np.load(f'{R}/nn_x/r6/stale_cf/both_it_hist_all_luo.npy')
nc = (both == 0) & (luo > 0)
mask3 = nc | ((both > 0) & (both < 3) & (idc != 0))
bits = lambda a: np.ascontiguousarray(a, np.float32).view(np.int32)

def aff(x, ref):
    mx = np.isfinite(x) & (x > -2); mr = np.isfinite(ref) & (ref > -2)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    return lambda v: np.where(v > -2, a * v + b, v).astype(np.float32)

def z64(x):
    x = x.astype(np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, (x - mu[fid]) ** 2, len(M)) / M, 0))
    return (x - mu[fid]) / (sd[fid] + 1e-9)

P = lambda t, k: np.load(f'{R}/preds/pre_{t}_{k}_val.npy').astype(np.float32)
cfg = {}
for t, c in (('v3At2', 'v3t2'), ('v4At2', 'v4t2')):
    f_fe, f_no, f_new = aff(P(t, 'fe'), P(c, 'fe')), aff(P(t, 'fe_noid'), P(c, 'fe_noid')), aff(P(t, 'new'), P(c, 'new'))
    cfg[t] = dict(fe=f_fe(P(t, 'fe')), noid=f_no(P(t, 'fe_noid')), new=f_new(P(t, 'new')), new0=f_new(P(t, 'new_noid')))

def build(mask):
    o = {}
    for head, a, b in (('fe', 'fe', 'noid'), ('new', 'new', 'new0')):
        mix = [np.where(mask, cfg[t][b], cfg[t][a]).astype(np.float32) for t in cfg]
        x = np.mean(mix, 0).astype(np.float32)
        o[f'pre2_{head}'] = x
        o[f'pre2_{head}_z'] = np.mean([z64(m_) for m_ in mix], 0).astype(np.float32)
        o[f'pre2_{head}_s'] = ((x + 20) / 6).astype(np.float32)
    return o

chk = {}
o0 = build(nc)
for k in ('pre2_fe', 'pre2_fe_z', 'pre2_fe_s'):
    w = np.load(f'{CF}/pre2both_{k}.npy'); chk[f'selfcheck_{k}'] = dict(rows_bits_diff=int((bits(o0[k]) != bits(w)).sum()), max_abs=float(np.abs(o0[k] - w).max()))
for k in ('pre2_new', 'pre2_new_z', 'pre2_new_s'):
    w = np.load(f'{CF}/pre2full_{k}.npy'); chk[f'selfcheck_{k}'] = dict(rows_bits_diff=int((bits(o0[k]) != bits(w)).sum()), max_abs=float(np.abs(o0[k] - w).max()))
ok = all(v['max_abs'] <= 1e-4 for v in chk.values()) and all(chk[f'selfcheck_{k}']['rows_bits_diff'] == 0 for k in ('pre2_fe', 'pre2_fe_s', 'pre2_new', 'pre2_new_s'))
chk['selfcheck_pass'] = bool(ok)
o3 = build(mask3)
chk.update(rows=len(mask3), nc=int(nc.sum()), mask3=int(mask3.sum()), added_rows=int((mask3 & ~nc).sum()), added_share=float((mask3 & ~nc).mean()),
           rows_changed_fe_s_vs_pre2both=int((bits(o3['pre2_fe_s']) != bits(o0['pre2_fe_s'])).sum()), mean_dfe_added_rows=float((o3['pre2_fe'] - o0['pre2_fe'])[mask3 & ~nc].mean()),
           mean_dz_added_rows=float((o3['pre2_fe_z'] - o0['pre2_fe_z'])[mask3 & ~nc].mean()), has_changes=int((np.isnan(o3['pre2_fe']) != np.isnan(o0['pre2_fe'])).sum()),
           remaining_rows_both_lt3_with_ID=int(((both < 3) & (idc != 0) & ~mask3).sum()))
if ok:
    for k, v in o3.items():
        np.save(f'{OUT}/lt3_{k}.npy', v)
    np.save(f'{OUT}/lt3_mask.npy', mask3)
json.dump(chk, open(f'{OUT}/build_cf_lt3.json', 'w'), indent=1)
print(json.dumps(chk, indent=1))
