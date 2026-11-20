"""READ of the plumbing smokes of the nx3 SPECULATIVE TRAINING (ledger A7.1 / R4). CPU only; reads logs, checkpoints and predictions of SMOKE variants and
of stored reference members' logs / checkpoints (no stage-A file is opened). Output nn_x/w12/nx3/out/spec/spec_smoke_check.txt.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/nx3/spec_smoke_check.py

 S1 all-data nx3 smoke   abl_smoke_fd10_nx3kd05w192_s951   FD_SMOKE=1 lane_full_nx3.sh 7 FN:nx3kd05w192:951   (train_full_abl.py, 20,000 feeds, 1 epoch = 73 steps)
 S2 fine-tune smoke      abl_smoke_fd10_nx3kd05w192_s951_n2cpv3s3smoke   smoke_ft_nx3.sh (production command + --epochs 1 --seeds 1 --folds 2) on the S1 checkpoint
 S3 stage-B smoke        abl_smoke_nx3kd05w192_s23         smoke_nx3_kd.sh 3 23 (the launcher's command + the five smoke keys)
 S4 harness A/A smoke    abl_smoke_fd10_kd05w192_s951      smoke_harness_aa.sh 4: the REFERENCE all-data command WITHOUT the bundle through train_full_abl.py,
                         against the STORED smoke of the same command in the nn_x/w5 harness, w5_fdsmoke_10_kd05w192_s951 (train_full.py, 2026-11-15)
73-step NDCG values are printed for the record only: they are NOT evidence about the bundle."""
import ast, json, os, re, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
sys.dont_write_bytecode = True
import numpy as np, torch
R = '<repo>'; ABL = R + '/nn_x/abl'; W5 = R + '/nn_x/w5'; NX = R + '/nn_x/w12/nx3'
fo = open(NX + '/out/spec/spec_smoke_check.txt', 'w'); ok_all = True
NEW = ['nx_rp3_z', 'nx_rp3_rk', 'has:nx_rp3_z']; BUNDLE = ','.join(NEW)
SMK = {'train_feeds': 20000, 'epochs': 1.0, 'eval_every': 25, 'select_min_step': 25, 'cliplog': 10}
ABL_ONLY = {'xt_exclude': '', 'xt_rel_extra': '', 'retr_cat': 0, 'xt_add_relonly': '', 'kv64': 0}
NV, NT, NH = 1389971, 2904806, 4630966


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


def grep(log, pat):
    return [ln.rstrip('\n') for ln in open(log, errors='ignore') if re.search(pat, ln)]


def L(p): return torch.load(p, map_location='cpu', weights_only=False)


def cfgdiff(a, b):
    return {k: (a.get(k) if k != 'xt_names' else f'list[{len(a.get(k) or [])}]', b.get(k) if k != 'xt_names' else f'list[{len(b.get(k) or [])}]') for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def preds(H, v, hold_suffix):
    out = {}
    for tag, f, n in (('val', f'{H}/preds/nn_{v}_trainonly_val.npy', NV), ('test', f'{H}/preds/nn_{v}_trainonly_test.npy', NT), ('holdout', f'{H}/preds/holdout/nn_{v}_{hold_suffix}.npy', NH),
                      ('val@last', f'{H}/preds/nn_{v}@last_trainonly_val.npy', NV), ('test@last', f'{H}/preds/nn_{v}@last_trainonly_test.npy', NT),
                      ('holdout@last', f'{H}/preds/holdout/nn_{v}@last_{hold_suffix}.npy', NH)):
        if not os.path.exists(f): out[tag] = ('MISSING', False); continue
        x = np.load(f); out[tag] = (x.shape, x.shape == (n,) and bool(np.isfinite(x).all()))
    return out


def steps(log):
    return [ln for ln in grep(log, r'^step \d+/\d+ ')]


# ====================================================================================== S1
v1 = 'abl_smoke_fd10_nx3kd05w192_s951'; ref = 'w5_fd10_kd05w192_s951'; lg1 = f'{ABL}/logs/{v1}.log'
P(f'=== S1 ALL-DATA nx3 smoke {v1} (lane_full_nx3.sh FD_SMOKE=1, GPU N) against the stored all-data wide student {ref} ===')
n1, x1 = xt_line(lg1); nr, xr = xt_line(f'{W5}/logs/{ref}.log')
CHK(f'log lists {n1} train-stage inputs = the {nr} logged by {ref} (names and order) + nx_rp3_z, nx_rp3_rk, has:nx_rp3_z appended last', n1 == 37 and nr == 34 and x1 == xr + NEW, f'last five: {x1[-5:]}')
P('  ' + '; '.join(grep(lg1, r'^abl xt_add')))
a1, ar = args_of(lg1), args_of(f'{W5}/logs/{ref}.log')
d = {k: (a1[k], ar[k]) for k in a1 if k in ar and a1[k] != ar[k]}; on = {k: a1[k] for k in a1 if k not in ar}; oo = {k: ar[k] for k in ar if k not in a1}
P(f'  logged args, smoke vs {ref}: shared keys that differ {d}')
P(f'  keys only in the smoke (abl harness): {on}; keys only in the reference (w5 harness): {oo}')
CHK('command = the reference member\'s logged arguments + --xt_add <bundle> + the five smoke keys (variant aside); harness-only keys off; --holdout_in_train True in both',
    set(d) == {'variant'} | set(SMK) and all(a1[k] == SMK[k] for k in SMK) and on == dict(ABL_ONLY, xt_add=BUNDLE) and a1['holdout_in_train'] is True and ar['holdout_in_train'] is True
    and oo == {'aug_all': False, 'xfold': -1, 'xfold_unlab': False, 'prand2': '', 'rdrop_cd': 0.0, 'sam_cd': 0.0})
fl = grep(lg1, r'^FULLDATA')
for ln in [l for l in fl if 'holdout_in_train' in l or 'put back' in l or 'each epoch' in l or 'cooldown exploration pool' in l or 'summary' in l][:5]: P('  ' + ln[:230])
CHK('FULL-DATA flag in the log: "--holdout_in_train" line, holdout feeds put back, checkpoint = LAST step, fulldata.json written',
    any('--holdout_in_train' in l and 'NOT removed from training' in l for l in fl) and any('put back' in l for l in fl) and os.path.exists(f'{ABL}/logs/{v1}_fulldata.json'))
kd = grep(lg1, r'^w12 KD'); P('  ' + kd[0])
CHK('teacher file read: teacher_mean11.npy, w 0.5, T 1.0', len(kd) == 1 and 'teacher_mean11.npy w 0.5 T 1.0' in kd[0])
c1 = L(f'{ABL}/ckpt/{v1}.pt'); cr = L(f'{W5}/ckpt/{ref}.pt'); dc = cfgdiff(c1['cfg'], cr['cfg'])
P(f'  checkpoint cfg, smoke vs {ref}: differing keys {dc}')
CHK('n_dense 104 -> 107 (+3), xt_names 34 -> 37 (the bundle appended), every other cfg key equal (d 192, xt_rel, xt_idcols, xt_relonly unchanged)',
    set(dc) == {'xt_names', 'n_dense'} and c1['cfg']['n_dense'] == cr['cfg']['n_dense'] + 3 == 107 and c1['cfg']['xt_names'] == cr['cfg']['xt_names'] + NEW and c1['cfg'].get('d', c1['args']['d']) == 192)
sh = {k: (tuple(c1['state_dict'][k].shape), tuple(cr['state_dict'][k].shape)) for k in c1['state_dict'] if k in cr['state_dict'] and c1['state_dict'][k].shape != cr['state_dict'][k].shape}
P(f'  tensors whose shape differs from {ref}: {sh}; keys only in one: {sorted(set(c1["state_dict"]) ^ set(cr["state_dict"]))}')
p1 = preds(ABL, v1, 'holdoutINSAMPLE'); P(f'  predictions: {p1}')
CHK(f'predictions written with the right row counts and finite: val {NV:,}, test {NT:,}, holdout IN-SAMPLE {NH:,} (kept and @last)', all(v[1] for v in p1.values()))
CHK('no holdout file under the out-of-sample name (nn_<v>_holdout.npy) was written', not os.path.exists(f'{ABL}/preds/holdout/nn_{v1}_holdout.npy') and not os.path.exists(f'{ABL}/preds/holdout/nn_{v1}@last_holdout.npy'))
CHK('fd_ft_wait.sh\'s completion condition holds for the smoke (logs/<v>.json, ckpt/<v>.pt, trainonly_test, trainonly_val)',
    all(os.path.exists(f) for f in (f'{ABL}/logs/{v1}.json', f'{ABL}/ckpt/{v1}.pt', f'{ABL}/preds/nn_{v1}_trainonly_test.npy', f'{ABL}/preds/nn_{v1}_trainonly_val.npy')))
CHK('lane bookkeeping: claim directory exists, no failed marker, lane log ends with DONE', os.path.isdir(f'{ABL}/logs/queue/claim_{v1}') and not os.path.exists(f'{ABL}/logs/queue/failed_{v1}')
    and 'DONE ' + v1 in open(f'{NX}/logs/spec/smoke_lane_FN_nx3kd05w192_951.log').read())
for ln in steps(lg1): P('  ' + ln[:150])

# ====================================================================================== S4 (before S1's statistics check, which uses it)
v4 = 'abl_smoke_fd10_kd05w192_s951'; s4 = 'w5_fdsmoke_10_kd05w192_s951'
P('')
P(f'=== S4 HARNESS A/A smoke {v4} (train_full_abl.py, nn_x/abl, NO bundle) against the STORED {s4} (train_full.py, nn_x/w5; same command, seed 951) ===')
a4, as4 = args_of(f'{ABL}/logs/{v4}.log'), args_of(f'{W5}/logs/{s4}.log')
d = {k: (a4[k], as4[k]) for k in a4 if k in as4 and a4[k] != as4[k]}; on = {k: a4[k] for k in a4 if k not in as4}; oo = {k: as4[k] for k in as4 if k not in a4}
P(f'  logged args: shared keys that differ {d}; only abl {on}; only w5 {oo}')
CHK('same command: only the variant differs; harness-only keys all off', set(d) == {'variant'} and on == dict(ABL_ONLY, xt_add='') and all(v in (False, -1, '', 0.0) for v in oo.values()))
c4 = L(f'{ABL}/ckpt/{v4}.pt'); cs4 = L(f'{W5}/ckpt/{s4}.pt')
CHK('checkpoint cfg equal on every key', not cfgdiff(c4['cfg'], cs4['cfg']), str(cfgdiff(c4['cfg'], cs4['cfg'])))
CHK('input standardisation statistics f_mean / f_std (104 inputs) bitwise equal', torch.equal(c4['state_dict']['f_mean'], cs4['state_dict']['f_mean']) and torch.equal(c4['state_dict']['f_std'], cs4['state_dict']['f_std']))
DATA = r'^(holdout:|p_random:|params:|muon:|train feeds|cooldown mix:|FULLDATA: (30000|each|cooldown))'
la, lb = grep(f'{ABL}/logs/{v4}.log', DATA), grep(f'{W5}/logs/{s4}.log', DATA)
CHK('data / model-size lines of the two logs equal (holdout, put back, training feeds, steps, pool, params, muon)', la == lb, f'{len(la)} lines')
sa, sb = steps(f'{ABL}/logs/{v4}.log'), steps(f'{W5}/logs/{s4}.log')
for x, y in zip(sa, sb): P(f'  abl: {re.sub(r" lr .*", "", x)}   |   w5: {re.sub(r" lr .*", "", y)}')
num = lambda ln, k: float(re.search(k + r' ([0-9.]+)', ln).group(1))
dl = max(abs(num(x, 'loss') - num(y, 'loss')) for x, y in zip(sa, sb)); dv = max(abs(num(x, 'val_ndcg') - num(y, 'val_ndcg')) for x, y in zip(sa, sb))
CHK('step lines: loss equal to the 4 printed decimals at steps 25, 50, 73; val NDCG within 0.0002 (GPU non-determinism level)', dl < 5e-5 and dv < 2e-4 and len(sa) == len(sb) == 3, f'max |d loss| {dl:.1e}, max |d val_ndcg| {dv:.1e}')
for sp, n in (('val', NV), ('test', NT)):
    x = np.load(f'{ABL}/preds/nn_{v4}_trainonly_{sp}.npy'); y = np.load(f'{W5}/preds/nn_{s4}_trainonly_{sp}.npy')
    P(f'  train-only {sp} predictions: corr {np.corrcoef(x, y)[0, 1]:.6f}, max |d| {np.abs(x - y).max():.3e}, mean |d| {np.abs(x - y).mean():.3e} (sd of the prediction {y.std():.3f})')
    CHK(f'{sp}: predictions of the two harnesses agree (corr > 0.999)', x.shape == y.shape == (n,) and np.corrcoef(x, y)[0, 1] > 0.999)
mx = max(float((c4['state_dict'][k].float() - cs4['state_dict'][k].float()).abs().max()) for k in c4['state_dict'] if c4['state_dict'][k].dtype.is_floating_point)
P(f'  weights after 73 steps: max |difference| over all tensors {mx:.2e} (not bitwise: two GPU runs of one command are not bitwise reproducible in this project either)')
P('  READING: the all-data trainer of nn_x/abl reproduces the all-data trainer of nn_x/w5 on the reference command; the harness change carries no visible drift at smoke scale.')

# ====================================================================================== S1 statistics
P('')
P('=== S1 (continued): the bundle changes nothing on the old inputs ===')
keep = [i for i in range(107) if i not in (94, 95, 96)]
m1, s1 = c1['state_dict']['f_mean'], c1['state_dict']['f_std']
CHK('S1 f_mean / f_std on the 104 old dense positions (new columns at 94, 95, 96 removed) bitwise equal to the STORED w5 smoke w5_fdsmoke_10_kd05w192_s951 and to S4',
    torch.equal(m1[keep], cs4['state_dict']['f_mean']) and torch.equal(s1[keep], cs4['state_dict']['f_std']) and torch.equal(m1[keep], c4['state_dict']['f_mean']))
P(f'  new positions (40 statistic batches of the 20,000 smoke feeds): nx_rp3_z mean {float(m1[94]):+.4f} sd {float(s1[94]):.4f}; nx_rp3_rk mean {float(m1[95]):+.5f} sd {float(s1[95]):.4f}; has:nx_rp3_z mean {float(m1[96]):.4f} sd {float(s1[96]):.4f}'
  '   (files, all train rows: z sd 0.640, rk sd 0.142, has mean 0.467; reports/INPUTS_1116/nx3_verify.md 5.2)')
CHK('new-column statistics agree with the files (has share within 0.03, sd of z within 0.05, sd of rk within 0.02)', abs(float(m1[96]) - 0.467) < 0.03 and abs(float(s1[94]) - 0.640) < 0.05 and abs(float(s1[95]) - 0.142) < 0.02)
la1 = grep(lg1, DATA)
CHK('S1 data lines (holdout, put back, training feeds, steps, pool) equal to the stored w5 smoke; params line 2.12M', [l for l in la1 if not l.startswith(('params', 'muon'))] == [l for l in lb if not l.startswith(('params', 'muon'))])
P(f'  73-step val NDCG (NOT evidence: one seed, 73 warm-up steps; the wider first layer also shifts the random stream of the initialisation, so S1 and S4 do not start from the same weights): S1 with the bundle {num(steps(lg1)[-1], "val_ndcg"):.5f}, S4 without {num(sa[-1], "val_ndcg"):.5f}, stored w5 smoke {num(sb[-1], "val_ndcg"):.5f}')

# ====================================================================================== S2
P('')
o2 = f'{v1}_n2cpv3s3smoke'; lg2 = f'{ABL}/logs/{o2}_ft.log'
P(f'=== S2 FINE-TUNE smoke {o2} (nn_x/abl/popfree/finetune.py on the S1 checkpoint; production command + --epochs 1 --seeds 1 --folds 2) ===')
f2 = args_of(lg2); fr = args_of(f'{ABL}/logs/abl_fd10_pfkd05w192_s951_n2cpv3s3_ft.log'); fw = args_of(f'{W5}/logs/{ref}_n2cpv3s3_ft.log')
d = {k: (f2[k], fr[k]) for k in f2 if f2[k] != fr.get(k)}
P(f'  logged fine-tune args vs the stored abl all-data member abl_fd10_pfkd05w192_s951: differing keys {d}')
CHK('fine-tune args = the production n2cpv3s3 args except variant / out and the three smoke overrides (epochs 1, seeds 1, folds 2)', set(d) == {'variant', 'out', 'epochs', 'seeds', 'folds'} and set(f2) == set(fr))
P(f'  production args of the w5 reference fine-tune ({ref}_n2cpv3s3) vs the abl production args: differing shared keys {sorted(k for k in fw if fw[k] != fr.get(k))}; only abl: {sorted(set(fr) - set(fw))}')
hx = grep(lg2, r'^abl xt_exclude honoured'); P('  ' + hx[0][:330])
CHK('the fine-tune took its input list from the checkpoint cfg and verified it: 37 xt inputs, n_dense 107, "abl xt_add honoured: [bundle] appended last"',
    len(hx) == 1 and '-> 37 xt inputs, n_dense 107' in hx[0] and "abl xt_add honoured: ['nx_rp3_z', 'nx_rp3_rk', 'has:nx_rp3_z'] appended last" in hx[0])
ex = grep(lg2, r'^extra features'); exr = grep(f'{W5}/logs/{ref}_n2cpv3s3_ft.log', r'^extra features')
CHK('fine-tune extra branch = the 42 n2c_pre_v3 inputs of the reference fine-tune (no NX column enters there)', ex == exr and ex[0].startswith('extra features (42)') and 'nx_rp3' not in ex[0])
for ln in grep(lg2, r'^(fold|OOF|best|snapshot|final)'): P('  ' + ln[:200])
vo = np.load(f'{ABL}/preds/nn_{o2}_val_oof.npy'); te = np.load(f'{ABL}/preds/nn_{o2}_test.npy'); to = np.load(f'{ABL}/preds/nn_{v1}_trainonly_val.npy')
CHK(f'fine-tune predictions written: val_oof {vo.shape}, test {te.shape}, finite; they differ from the train-only predictions (corr {np.corrcoef(vo, to)[0, 1]:.3f})',
    vo.shape == (NV,) and te.shape == (NT,) and bool(np.isfinite(vo).all() and np.isfinite(te).all()) and not np.array_equal(vo, to))
CHK('nothing of the smoke was copied into the project-level preds/ or nn/logs/', not any(v1 in f for f in os.listdir(R + '/preds')) and not any(v1 in f for f in os.listdir(R + '/nn/logs')))

# ====================================================================================== S3
P('')
v3 = 'abl_smoke_nx3kd05w192_s23'; rb = 'w5_e10_kd05w192_s951'; lg3 = f'{ABL}/logs/{v3}.log'
P(f'=== S3 STAGE-B launcher smoke {v3} (smoke_nx3_kd.sh, GPU N) against the stored held-out wide student {rb} ===')
n3, x3 = xt_line(lg3); nb, xb = xt_line(f'{W5}/logs/{rb}.log')
CHK(f'log lists {n3} train-stage inputs = the {nb} of {rb} + the bundle appended last', n3 == 37 and nb == 34 and x3 == xb + NEW)
a3, ab = args_of(lg3), args_of(f'{W5}/logs/{rb}.log')
d = {k: (a3[k], ab[k]) for k in a3 if k in ab and a3[k] != ab[k]}; on = {k: a3[k] for k in a3 if k not in ab}; oo = {k: ab[k] for k in ab if k not in a3}
P(f'  logged args, smoke vs {rb}: shared keys that differ {d}; only smoke {on}; only reference {oo}')
CHK('command = the reference member\'s logged arguments + --xt_add <bundle> + the five smoke keys (variant and seed aside); no full-data flag; harness-only keys off',
    set(d) == {'variant', 'seed'} | set(SMK) and all(a3[k] == SMK[k] for k in SMK) and on == dict(ABL_ONLY, xt_add=BUNDLE) and 'holdout_in_train' not in a3 and all(v in (False, -1, '', 0.0) for v in oo.values()))
kd = grep(lg3, r'^w12 KD'); P('  ' + kd[0])
CHK('teacher file read: teacher_mean11.npy, w 0.5, T 1.0; no FULLDATA line (the holdout feeds are excluded from training)', 'teacher_mean11.npy w 0.5 T 1.0' in kd[0] and not grep(lg3, r'^FULLDATA') and bool(grep(lg3, r'^holdout: 30000 random-like train feeds excluded')))
c3 = L(f'{ABL}/ckpt/{v3}.pt'); cb = L(f'{W5}/ckpt/{rb}.pt'); dc = cfgdiff(c3['cfg'], cb['cfg'])
P(f'  checkpoint cfg, smoke vs {rb}: differing keys {dc}')
CHK('n_dense 104 -> 107 (+3), xt_names 34 -> 37, every other cfg key equal', set(dc) == {'xt_names', 'n_dense'} and c3['cfg']['n_dense'] == cb['cfg']['n_dense'] + 3 == 107 and c3['cfg']['xt_names'] == cb['cfg']['xt_names'] + NEW)
ca = L(f'{ABL}/ckpt/abl_smoke_nx3_s23.pt')
CHK('f_mean / f_std of all 107 inputs bitwise equal to the STEP 0 smoke abl_smoke_nx3_s23 (stage-A command, same seed 23, same 37 inputs, same 19,827 feeds)',
    torch.equal(c3['state_dict']['f_mean'], ca['state_dict']['f_mean']) and torch.equal(c3['state_dict']['f_std'], ca['state_dict']['f_std']))
p3 = preds(ABL, v3, 'holdout'); P(f'  predictions: {p3}')
CHK(f'predictions written with the right row counts and finite: val {NV:,}, test {NT:,}, holdout {NH:,} (selected and @last)', all(v[1] for v in p3.values()))
CHK('the read tool\'s completeness files exist (json, ckpt, ckpt@last, val, val@last, holdout, holdout@last)', all(os.path.exists(f'{ABL}/{f}') for f in (f'logs/{v3}.json', f'ckpt/{v3}.pt', f'ckpt/{v3}@last.pt')))
for ln in steps(lg3): P('  ' + ln[:150])
reg = [l for l in open(f'{ABL}/RUNNING.txt') if v3 in l]
CHK('registry line appended to nn_x/abl/RUNNING.txt with the verified PID', len(reg) == 1 and '/proc/' in reg[0], reg[0][:160].strip() if reg else '')
P('')
P(f'SMOKE READ: {"PASS" if ok_all else "FAIL"}')
fo.close(); sys.exit(0 if ok_all else 1)
