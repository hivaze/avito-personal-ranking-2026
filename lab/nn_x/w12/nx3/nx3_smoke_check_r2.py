"""nx3 SMOKE read (CPU only; reads logs / checkpoints / predictions of the two smoke runs of nn_x/w12/nx3/launch_nx3_hl.sh with SMOKE=1, seed 23).
  (1) input lists: base = 34 names = the logged list of the stored control w5_e10_base_s951; nx3 = the same 34 + nx_rp3_z, nx_rp3_rk, has:nx_rp3_z (37)
  (2) command: the args of the base smoke against the logged args of w5_e10_base_s951 (only variant, seed and the five smoke keys may differ; abl-only keys off)
  (3) cfg of the checkpoints: base smoke = w5_e10_base_s951; nx3 = base with xt_names + 3, n_dense + 3
  (4) SMOKE CONTROL REPRODUCES THE PRECEDENT: the input standardisation statistics f_mean / f_std (104 dense inputs over 40 batches of the 19,827 smoke
      training feeds of seed 23; independent of lr) of the base smoke against the stored seed-23 smokes nn_x/abl/ckpt/abl_smoke_kv64_s23.pt and
      nn_x/r7/ckpt/r7_smoke-base_s23.pt (bitwise), and the data lines of the log against nn_x/abl/logs/abl_smoke_kv64_s23.log
  (5) nx3: f_mean / f_std on the 104 old positions = the base smoke's (bitwise); the 3 new positions (94, 95, 96) printed next to the file statistics
  (6) outputs exist, finite; final NDCG lines. Output out/smoke_check_r2.txt (exclusive create).
r2 = nx3_smoke_check.py with ONE corrected comparison in part (4) (see the comment there); the first read is kept in out/smoke_check.txt."""
import os, sys, ast, json, re
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
sys.dont_write_bytecode = True
import numpy as np, torch
ROOT = '<repo>'; ABL = ROOT + '/nn_x/abl'; NX = ROOT + '/nn_x/w12/nx3'
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 23
fo = open(NX + '/out/smoke_check_r2.txt', 'x')
ok_all = True
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); fo.write(s + '\n'); fo.flush()
def CHK(name, cond, detail=''):
    global ok_all; ok_all &= bool(cond); P(f'  [{"ok" if cond else "FAIL"}] {name} {detail}')
def args_of(log):
    ln = open(log).readline(); return ast.literal_eval(ln[ln.index('args: ') + 6:].strip())
def xt_line(log):
    for ln in open(log):
        m = re.match(r'train-stage extra features \((\d+)\): (.*)', ln)
        if m: return int(m.group(1)), ast.literal_eval(m.group(2))
def lines(log, keys):
    out = {}
    for ln in open(log):
        for k in keys:
            if ln.startswith(k) and k not in out: out[k] = ln.strip()
    return out
def L(p): return torch.load(p, map_location='cpu', weights_only=False)
vb, vn = f'abl_smoke_base_s{SEED}', f'abl_smoke_nx3_s{SEED}'; NEW = ['nx_rp3_z', 'nx_rp3_rk', 'has:nx_rp3_z']
ctl_log = ROOT + '/nn_x/w5/logs/w5_e10_base_s951.log'
P(f'=== (1) input lists ===')
nb_, xb = xt_line(f'{ABL}/logs/{vb}.log'); nn_, xn = xt_line(f'{ABL}/logs/{vn}.log'); nc_, xc = xt_line(ctl_log)
P(f'  base smoke log: train-stage extra features ({nb_}); nx3 smoke log: ({nn_}): last five {xn[-5:]}; stored control w5_e10_base_s951: ({nc_})')
CHK('base smoke lists 34 inputs, identical (names and order) to the stored control w5_e10_base_s951', nb_ == 34 and xb == xc)
CHK('nx3 smoke lists 34 + 3 = 37 inputs = the base list + nx_rp3_z, nx_rp3_rk, has:nx_rp3_z (appended last)', nn_ == 37 and xn == xb + NEW)
add = [ln.strip() for ln in open(f'{ABL}/logs/{vn}.log') if ln.startswith('abl xt_add')]
P(f'  nx3 log: {add}')
P('=== (2) command ===')
ab, an, ac = args_of(f'{ABL}/logs/{vb}.log'), args_of(f'{ABL}/logs/{vn}.log'), args_of(ctl_log)
SMK = {'variant', 'seed', 'train_feeds', 'epochs', 'eval_every', 'select_min_step', 'cliplog'}
d1 = {k: (ab.get(k), ac.get(k)) for k in sorted(set(ab) | set(ac)) if k in ab and k in ac and ab[k] != ac[k]}
only_ab = {k: ab[k] for k in sorted(set(ab) - set(ac))}; only_ac = {k: ac[k] for k in sorted(set(ac) - set(ab))}
P(f'  base smoke vs stored control, keys with different values (smoke, control): {d1}')
P(f'  keys only in the abl harness (must be off): {only_ab}; keys only in the stored control: {only_ac}')
OFF = {'xt_exclude': '', 'xt_rel_extra': '', 'retr_cat': 0, 'xt_add': '', 'xt_add_relonly': '', 'kv64': 0, 'hzcd': False, 'val_mix': 0.0, 'trunc_lifts': '', 'kd_file': ''}
off_ok = all((ab.get(k, v) == v or (k == 'kd_file' and not ab.get(k))) for k, v in OFF.items())
CHK('base smoke command = stored control command apart from variant, seed and the five smoke keys; every input / distillation flag of the abl harness off', set(d1) <= SMK and not only_ac and off_ok)
d2 = {k: (an.get(k), ab.get(k)) for k in sorted(set(an) | set(ab)) if an.get(k) != ab.get(k)}
P(f'  nx3 smoke vs base smoke, keys with different values: {d2}')
CHK('nx3 smoke command = base smoke command + --xt_add nx_rp3_z,nx_rp3_rk,has:nx_rp3_z', set(d2) == {'variant', 'xt_add'} and an['xt_add'] == ','.join(NEW))
P('=== (3) checkpoint cfg ===')
cb, cn, cc = L(f'{ABL}/ckpt/{vb}.pt'), L(f'{ABL}/ckpt/{vn}.pt'), L(ROOT + '/nn_x/w5/ckpt/w5_e10_base_s951.pt')
dc = {k: (cb['cfg'].get(k), cc['cfg'].get(k)) for k in sorted(set(cb['cfg']) | set(cc['cfg'])) if cb['cfg'].get(k) != cc['cfg'].get(k)}
CHK('cfg of the base smoke checkpoint = cfg of w5_e10_base_s951 (every key)', not dc, f'differences: {dc}')
dn = {k: (cn['cfg'].get(k) if k != 'xt_names' else f'list[{len(cn["cfg"][k])}]', cb['cfg'].get(k) if k != 'xt_names' else f'list[{len(cb["cfg"][k])}]') for k in sorted(set(cn['cfg']) | set(cb['cfg'])) if cn['cfg'].get(k) != cb['cfg'].get(k)}
P(f'  nx3 cfg vs base cfg, differing keys: {dn}')
CHK('nx3 cfg: xt_names = base + 3, n_dense 104 -> 107, nothing else differs (xt_rel, xt_relonly, xt_idcols unchanged)', set(dn) == {'xt_names', 'n_dense'} and cn['cfg']['xt_names'] == cb['cfg']['xt_names'] + NEW and cn['cfg']['n_dense'] == 107 and cb['cfg']['n_dense'] == 104)
pb = sum(v.numel() for v in cb['state_dict'].values()); pn = sum(v.numel() for v in cn['state_dict'].values())
P(f'  tensors in the state dict: base {pb} values, nx3 {pn} values (difference {pn - pb}; expected 3 x (256 + 1) first-layer / wide weights + 2 x 3 statistics = {3 * 257 + 6})')
CHK('parameter difference is exactly the three new input columns', pn - pb == 3 * 257 + 6, f'c_mlp.0.weight {tuple(cn["state_dict"]["c_mlp.0.weight"].shape)}, wide.weight {tuple(cn["state_dict"]["wide.weight"].shape)}')
P('=== (4) smoke control against the precedent ===')
fb_m, fb_s = cb['state_dict']['f_mean'], cb['state_dict']['f_std']
for nm, p in (('nn_x/abl/ckpt/abl_smoke_kv64_s23.pt', ABL + '/ckpt/abl_smoke_kv64_s23.pt'), ('nn_x/r7/ckpt/r7_smoke-base_s23.pt', ROOT + '/nn_x/r7/ckpt/r7_smoke-base_s23.pt')):
    c = L(p); m, s = c['state_dict']['f_mean'], c['state_dict']['f_std']
    same_in = c['cfg']['xt_names'] == cb['cfg']['xt_names'] and c['args']['seed'] == SEED and c['args']['train_feeds'] == 20000
    CHK(f'base smoke f_mean / f_std (104 inputs) bitwise equal to {nm} (seed {c["args"]["seed"]}, 20,000 feeds, same 34 inputs: {same_in})', same_in and torch.equal(fb_m, m) and torch.equal(fb_s, s),
        f'max abs diff mean {float((fb_m - m).abs().max()):.3e} std {float((fb_s - s).abs().max()):.3e}')
KEYS = ('holdout:', 'p_random:', 'params:', 'muon:', 'train feeds', 'cooldown mix:')
lb, lk, ln_ = lines(f'{ABL}/logs/{vb}.log', KEYS), lines(ABL + '/logs/abl_smoke_kv64_s23.log', KEYS), lines(f'{ABL}/logs/{vn}.log', KEYS)
lp = lines(ABL + '/logs/abl_smoke_pre2noid_s23.log', KEYS)
for k in KEYS: P(f'  base smoke: {lb.get(k)} | stored abl_smoke_pre2noid_s23: {lp.get(k)} | stored abl_smoke_kv64_s23: {lk.get(k)} | nx3 smoke: {ln_.get(k)}')
# r2 (correction of the first read, out/smoke_check.txt): the kv64 arm carries a K/V adapter (48 Muon matrices, 1.03M parameters), so its two MODEL-SIZE lines
# are not a precedent of a 34-input base model; they are compared with abl_smoke_pre2noid_s23 (same harness, seed 23, 34 input columns, n_dense 104) instead.
DATA = ('holdout:', 'p_random:', 'train feeds', 'cooldown mix:')
CHK('DATA lines of the base smoke log (holdout, p_random / training feeds, steps per epoch, cooldown start and exploration pool) equal to the stored seed-23 smokes abl_smoke_kv64_s23 AND abl_smoke_pre2noid_s23', all(lb.get(k) == lk.get(k) == lp.get(k) for k in DATA))
CHK('MODEL-SIZE lines of the base smoke log (params, muon matrices) equal to the stored seed-23 smoke abl_smoke_pre2noid_s23 (34 input columns, n_dense 104)', all(lb.get(k) == lp.get(k) for k in ('params:', 'muon:')))
P(f'  for scale only (different object): base smoke f_mean against the stored control w5_e10_base_s951 (all training feeds, seed 951): max abs diff {float((fb_m - cc["state_dict"]["f_mean"]).abs().max()):.4f}')
P('=== (5) nx3 inputs as the model standardises them ===')
fn_m, fn_s = cn['state_dict']['f_mean'], cn['state_dict']['f_std']; keep = [i for i in range(107) if i not in (94, 95, 96)]
CHK('nx3 f_mean / f_std on the 104 old positions (new columns at 94, 95, 96 removed) bitwise equal to the base smoke', torch.equal(fn_m[keep], fb_m) and torch.equal(fn_s[keep], fb_s),
    f'max abs diff mean {float((fn_m[keep] - fb_m).abs().max()):.3e} std {float((fn_s[keep] - fb_s).abs().max()):.3e}')
P(f'  new positions over the 40 statistic batches: nx_rp3_z mean {float(fn_m[94]):+.4f} sd {float(fn_s[94]):.4f}; nx_rp3_rk mean {float(fn_m[95]):+.5f} sd {float(fn_s[95]):.4f}; has:nx_rp3_z mean {float(fn_m[96]):.4f} sd {float(fn_s[96]):.4f}')
z = np.load(ROOT + '/nn/feat_cache/train/nx_rp3_z.npy', mmap_mode='r'); rk = np.load(ROOT + '/nn/feat_cache/train/nx_rp3_rk.npy', mmap_mode='r')
idx = np.random.default_rng(0).choice(len(z), 5_000_000, replace=False); idx.sort(); zz = np.asarray(z[idx]); rr = np.asarray(rk[idx]); hh = ~np.isnan(zz)
zt = np.nan_to_num(np.clip(zz, -5, 5), nan=0.0); rt = np.nan_to_num(rr, nan=0.0)
P(f'  the installed train files as the loader transforms them, 5,000,000 random rows of ALL train feeds: nx_rp3_z mean {zt.mean():+.4f} sd {zt.std():.4f}; nx_rp3_rk mean {rt.mean():+.5f} sd {rt.std():.4f}; has mean {hh.mean():.4f} sd {hh.std():.4f}')
CHK('new-column statistics of the model agree with the files (has share within 0.03, sd of z within 0.05, sd of rk within 0.02)', abs(float(fn_m[96]) - hh.mean()) < 0.03 and abs(float(fn_s[94]) - zt.std()) < 0.05 and abs(float(fn_s[95]) - rt.std()) < 0.02)
P('=== (6) outputs ===')
for v in (vb, vn):
    j = json.load(open(f'{ABL}/logs/{v}.json')); h = j['history'][-1]
    pv = np.load(f'{ABL}/preds/nn_{v}_trainonly_val.npy'); pt = np.load(f'{ABL}/preds/nn_{v}_trainonly_test.npy'); ph = np.load(f'{ABL}/preds/holdout/nn_{v}_holdout.npy')
    fin = bool(np.isfinite(pv).all() and np.isfinite(pt).all() and np.isfinite(ph).all())
    P(f'  {v}: best step {j["best_step"]}, val NDCG {j["val_ndcg"]:.5f}; last eval step {h["step"]}: val {h["val_ndcg"]:.5f} holdout {h["holdout_ndcg"]:.5f}; train {j["train_minutes"]:.1f} min; preds val {pv.shape} test {pt.shape} holdout {ph.shape}')
    CHK(f'{v}: predictions written for val (1,389,971), test (2,904,806), holdout (4,630,966), all finite', fin and pv.shape == (1389971,) and pt.shape == (2904806,) and ph.shape == (4630966,))
P(f'SMOKE READ: {"PASS" if ok_all else "FAIL"}')
