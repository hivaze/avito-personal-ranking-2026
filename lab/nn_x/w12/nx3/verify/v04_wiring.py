"""nx3 verifier v04 (d) WIRING, my own read (CPU only; checkpoints are opened with map_location='cpu', read-only):
 1 logged arguments: smoke base / smoke nx3 (nn_x/abl/logs) against the six stored controls w5_e10_base_s951-956 (nn_x/w5/logs) and against each other
 2 the flags the stage-A launcher issues without SMOKE, parsed from the launcher text, against the logged arguments of the stored controls
 3 logged input lists (34 / 34 + 3)
 4 checkpoint cfg and standardisation statistics of the two smoke checkpoints against a stored control checkpoint and against the installed files"""
import os, sys, ast, re, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
sys.dont_write_bytecode = True
import numpy as np
ROOT = '<repo>'; OUT = ROOT + '/nn_x/w12/nx3/verify/out'
f_ = open(OUT + '/v04_wiring.txt', 'x')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); f_.write(s + '\n'); f_.flush()
def args_of(path):
    for ln in open(path):
        if ' args: {' in ln:
            return ast.literal_eval(ln[ln.index(' args: ') + 7:].strip())
    raise RuntimeError(path)
def xt_of(path):
    for ln in open(path):
        m = re.match(r'train-stage extra features \((\d+)\): (\[.*\])', ln)
        if m: return int(m.group(1)), ast.literal_eval(m.group(2))
    return None
def diff(a, b):
    ks = sorted(set(a) | set(b)); return {k: (a.get(k, '<absent>'), b.get(k, '<absent>')) for k in ks if a.get(k, '<absent>') != b.get(k, '<absent>')}
SB = args_of(ROOT + '/nn_x/abl/logs/abl_smoke_base_s23.log'); SN = args_of(ROOT + '/nn_x/abl/logs/abl_smoke_nx3_s23.log')
C = {s: args_of(ROOT + f'/nn_x/w5/logs/w5_e10_base_s{s}.log') for s in range(951, 957)}
P('=== 1. logged arguments')
P(f'smoke nx3 vs smoke base (same harness, same seed): {diff(SN, SB)}')
for s in range(952, 957):
    d = diff(C[s], C[951]); P(f'stored control s{s} vs s951: differing keys {sorted(d)}')
d = diff(SB, C[951])
P(f'smoke base vs stored control w5_e10_base_s951: {len(SB)} / {len(C[951])} keys; differing: {d}')
SMOKE5 = {'train_feeds', 'epochs', 'eval_every', 'select_min_step', 'cliplog'}
other = {k: v for k, v in d.items() if k not in SMOKE5 | {'variant', 'seed'}}
P(f'  keys that differ beyond variant, seed and the five smoke keys: {other}')
only_abl = {k: v[0] for k, v in other.items() if v[1] == '<absent>'}
P(f'  of them present only in the nn_x/abl harness (must all be the "off" value): {only_abl}')
# 2. the launcher's non-smoke flags
txt = open(ROOT + '/nn_x/w12/nx3/launch_nx3_hl.sh').read()
m = re.search(r'train\.py --variant \$v \\\n(.*?)\$XA \$EXTRA', txt, re.S); flags = m.group(1).replace('\\\n', ' ').split()
des = open(ROOT + '/reports/INPUTS_1116/design/launch_nx_hl.sh').read(); m2 = re.search(r'train\.py --variant \$v \\\n(.*?)\$XA \$EXTRA', des, re.S)
P('=== 2. stage-A launcher')
P(f'flag text of the train.py call identical to the designer\'s launch_nx_hl.sh: {m.group(1) == m2.group(1)}')
e10 = open(ROOT + '/nn_x/abl/pop1/launch_e10.sh').read() if os.path.exists(ROOT + '/nn_x/abl/pop1/launch_e10.sh') else ''
P(f'flags: {" ".join(flags)}')
exp = dict(SB)                                  # what argparse yields for the same flags without the SMOKE extras: start from the smoke base args ...
st = {'--epochs': ('epochs', float), '--eval_every': ('eval_every', int), '--select_min_step': ('select_min_step', int), '--cliplog': ('cliplog', int)}
for i, t in enumerate(flags):
    if t in st: exp[st[t][0]] = st[t][1](flags[i + 1])     # ... and take the four keys back from the launcher's own flags
exp['train_feeds'] = C[951]['train_feeds']; P(f'  train_feeds: the launcher passes none without SMOKE; the stored control logged {C[951]["train_feeds"]!r}')
# every flag of the launcher against the logged value of the stored control
amap = {'--no_pos': ('no_pos', True), '--sim_bias': ('sim_bias', 1), '--user0': ('user0', True), '--domain': ('domain', True), '--muon': ('muon', True), '--attn_gate': ('attn_gate', True),
        '--save_last': ('save_last', True), '--save_holdout': ('save_holdout', True)}
i = 0; bad = []; seen = []
while i < len(flags):
    t = flags[i]
    if t in amap:
        k, v = amap[t]; ok = C[951].get(k) == v; seen.append(k); i += 1
    elif t == '--seed':
        i += 2; continue
    else:
        k = t[2:]; raw = flags[i + 1]; cv = C[951].get(k, '<absent>'); seen.append(k)
        try: ok = float(raw) == float(cv)
        except (TypeError, ValueError): ok = str(raw) == str(cv)
        i += 2
    if not ok: bad.append((t, C[951].get(k, '<absent>')))
P(f'  launcher flags checked against w5_e10_base_s951 logged args: {len(seen)} keys {seen}; mismatches: {bad}')
d2 = {k: v for k, v in diff(exp, C[951]).items() if k not in ('variant', 'seed')}
P(f'  expected stage-A base args (smoke base args with the 5 smoke keys set back) vs stored control s951, beyond variant / seed: {d2}')
P('=== 3. logged input lists')
nb_, xb = xt_of(ROOT + '/nn_x/abl/logs/abl_smoke_base_s23.log'); nn_, xn = xt_of(ROOT + '/nn_x/abl/logs/abl_smoke_nx3_s23.log'); nc, xc = xt_of(ROOT + '/nn_x/w5/logs/w5_e10_base_s951.log')
P(f'smoke base lists {nb_} inputs; equal (names and order) to the stored control w5_e10_base_s951 ({nc}): {xb == xc}')
P(f'smoke nx3 lists {nn_} inputs; first 34 equal to base: {xn[:34] == xb}; added (indices 34..): {xn[34:]}')
for s in range(952, 957):
    P(f'  stored control s{s}: {xt_of(ROOT + f"/nn_x/w5/logs/w5_e10_base_s{s}.log")[0]} inputs, equal to s951: {xt_of(ROOT + f"/nn_x/w5/logs/w5_e10_base_s{s}.log")[1] == xc}')
P('=== 4. checkpoints (cfg, statistics)')
import torch
def ck(p): return torch.load(p, map_location='cpu', weights_only=False)
kb = ck(ROOT + '/nn_x/abl/ckpt/abl_smoke_base_s23.pt'); kn = ck(ROOT + '/nn_x/abl/ckpt/abl_smoke_nx3_s23.pt'); kc = ck(ROOT + '/nn_x/w5/ckpt/w5_e10_base_s951.pt')
P(f'checkpoint keys: {sorted(kn.keys())}')
P(f'cfg smoke base vs stored control s951: {diff(kb["cfg"], kc["cfg"])}')
dn = diff(kn['cfg'], kb['cfg']); P(f'cfg smoke nx3 vs smoke base: differing keys {sorted(dn)}; n_dense {dn.get("n_dense")}; xt_names lengths {len(kn["cfg"]["xt_names"])} / {len(kb["cfg"]["xt_names"])}; nx3 tail {kn["cfg"]["xt_names"][34:]}')
P(f'cfg xt_rel / xt_idcols / xt_relonly nx3: {kn["cfg"].get("xt_rel")} / {kn["cfg"].get("xt_idcols")} / {kn["cfg"].get("xt_relonly")}; base: {kb["cfg"].get("xt_rel")} / {kb["cfg"].get("xt_idcols")} / {kb["cfg"].get("xt_relonly")}')
P(f'saved args of the nx3 checkpoint: xt {kn.get("args", {}).get("xt")!r}, xt_add {kn.get("args", {}).get("xt_add")!r}, xt_exclude {kn.get("args", {}).get("xt_exclude")!r}, xt_add_relonly {kn.get("args", {}).get("xt_add_relonly")!r}')
sdn, sdb = kn['state_dict'], kb['state_dict']
stat = [k for k in sdn if 'f_mean' in k or 'f_std' in k]; P(f'statistics tensors: {[(k, tuple(sdn[k].shape), tuple(sdb[k].shape)) for k in stat]}')
shp = {k: (tuple(sdn[k].shape), tuple(sdb[k].shape)) for k in sdn if k in sdb and sdn[k].shape != sdb[k].shape}
P(f'tensors whose shape differs nx3 vs base: {shp}; tensors only in one: {sorted(set(sdn) ^ set(sdb))}')
P(f'parameter count nx3 - base: {sum(v.numel() for v in sdn.values()) - sum(v.numel() for v in sdb.values())}')
km = [k for k in stat if 'mean' in k][0]; ks = [k for k in stat if 'std' in k][0]
mn, sn_, mb, sb_ = sdn[km].numpy().ravel(), sdn[ks].numpy().ravel(), sdb[km].numpy().ravel(), sdb[ks].numpy().ravel()
nd = len(mb)
P(f'n_dense base {nd}, nx3 {len(mn)}')
# where do the three new inputs sit? the entries of the nx3 statistics that have no bitwise-equal counterpart at the same index of base
same_prefix = int(np.argmax(~(mn[:nd] == mb)) if (~(mn[:nd] == mb)).any() else nd)
P(f'statistics of nx3 equal to base bitwise on the first {same_prefix} of {nd} inputs (mean), std equal on the same: {bool((sn_[:same_prefix] == sb_[:same_prefix]).all())}')
P(f'the entries after that prefix: nx3 mean {np.round(mn[same_prefix:], 4).tolist()}, std {np.round(sn_[same_prefix:], 4).tolist()}; base mean {np.round(mb[same_prefix:], 4).tolist()}, std {np.round(sb_[same_prefix:], 4).tolist()}')
z = np.load(ROOT + '/nn/feat_cache/train/nx_rp3_z.npy', mmap_mode='r'); rk = np.load(ROOT + '/nn/feat_cache/train/nx_rp3_rk.npy', mmap_mode='r')
P('(for scale) the files as the loader reads them, ALL train rows: computed in v05; smoke statistics are over 40 batches of the 19,827 smoke training feeds')
P(f'stored control checkpoint f_mean / f_std equal to the smoke base on how many of {nd} inputs (different feed sample expected, so not equal): {int((kc["state_dict"][km].numpy().ravel() == mb).sum())}')
for nm in ('abl_smoke_kv64_s23', 'abl_smoke_pre2noid_s23'):
    p = ROOT + f'/nn_x/abl/ckpt/{nm}.pt'
    if os.path.exists(p):
        k2 = ck(p); m2_ = k2['state_dict'][km].numpy().ravel(); s2_ = k2['state_dict'][ks].numpy().ravel()
        n_ = min(len(m2_), nd); P(f'precedent {nm}: n_dense {len(m2_)}; f_mean bitwise equal to the smoke base on {int((m2_[:n_] == mb[:n_]).sum())} of the first {n_}; f_std on {int((s2_[:n_] == sb_[:n_]).sum())}')
P('=== 5. smoke predictions')
for arm in ('base', 'nx3'):
    for fn in (f'nn_abl_smoke_{arm}_s23_trainonly_val.npy', f'nn_abl_smoke_{arm}_s23_trainonly_test.npy', f'holdout/nn_abl_smoke_{arm}_s23_holdout.npy'):
        x = np.load(ROOT + '/nn_x/abl/preds/' + fn); P(f'  {fn}: shape {x.shape}, finite {bool(np.isfinite(x).all())}, sd {x.std():.4f}')
j1 = json.load(open(ROOT + '/nn_x/abl/logs/abl_smoke_base_s23.json')); j2 = json.load(open(ROOT + '/nn_x/abl/logs/abl_smoke_nx3_s23.json'))
P(f'smoke json: base val_ndcg {j1["val_ndcg"]:.5f} best_step {j1["best_step"]}; nx3 val_ndcg {j2["val_ndcg"]:.5f} best_step {j2["best_step"]} (73 steps: not evidence)')
P('DONE')
