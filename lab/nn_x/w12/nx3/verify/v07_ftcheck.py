"""nx3 verifier v07: the fine-tune's input check (nn_x/abl/popfree/finetune.py honour_xt_exclude, l.119-159) EMULATED on CPU for the nx3 smoke checkpoint.
The preset tables are taken from the SOURCE of nn_x/abl/train.py by executing only its module-level assignments of XT_PRESETS / XT_REL / XT_RELONLY
(nothing of the harness is imported: common.py needs a GPU id); N_DENSE from the source of model.py; xt_add_names from abl_arms.py (pure python)."""
import os, sys, ast, re
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
sys.dont_write_bytecode = True
ROOT = '<repo>'; OUT = ROOT + '/nn_x/w12/nx3/verify/out'
f_ = open(OUT + '/v07_ftcheck_r2.txt', 'x')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); f_.write(s + '\n'); f_.flush()
src = open(ROOT + '/nn_x/abl/train.py').read(); tree = ast.parse(src); ns = {}
want = {'XT_PRESETS', 'XT_REL', 'XT_RELONLY'}
for node in tree.body:
    tg = []
    if isinstance(node, ast.Assign): tg = [t for t in node.targets]
    elif isinstance(node, ast.AugAssign): tg = [node.target]
    names = set()
    for t in tg:
        for sub in ast.walk(t):
            if isinstance(sub, ast.Name): names.add(sub.id)
    if names & want:
        exec(compile(ast.Module([node], []), 'train.py', 'exec'), ns)
XT_PRESETS, XT_REL, XT_RELONLY = ns['XT_PRESETS'], ns['XT_REL'], ns['XT_RELONLY']
msrc = open(ROOT + '/nn_x/abl/model.py').read(); mns = {}
for node in ast.parse(msrc).body:      # r2: N_DENSE is an expression in N_ET (my first read took only its first term, 15: my parsing error, out/v07_ftcheck.txt kept)
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('N_ET', 'N_DENSE') for t in node.targets):
        exec(compile(ast.Module([node], []), 'model.py', 'exec'), mns)
N_DENSE = mns['N_DENSE']
asrc = open(ROOT + '/nn_x/abl/abl_arms.py').read(); atree = ast.parse(asrc); ans = {}
for node in atree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('xt_add_names', '_names', 'xt_exclude_names', 'xt_rel_extend'):
        exec(compile(ast.Module([node], []), 'abl_arms.py', 'exec'), ans)
import torch
P(f'presets from source: cvnd_pre2s has {len(XT_PRESETS["cvnd_pre2s"])} names; XT_REL {XT_REL.get("cvnd_pre2s")}; XT_RELONLY {XT_RELONLY.get("cvnd_pre2s")}; N_DENSE {N_DENSE}')
for v in ('abl_smoke_nx3_s23', 'abl_smoke_base_s23'):
    ck = torch.load(ROOT + f'/nn_x/abl/ckpt/{v}.pt', map_location='cpu', weights_only=False); args, cfg = ck['args'], ck['cfg']
    xt, excl, rel_extra, xadd = args.get('xt', ''), args.get('xt_exclude', ''), args.get('xt_rel_extra', ''), args.get('xt_add', '') or ''
    quiet = lambda *s: None
    exp = list(XT_PRESETS[xt])
    assert not excl and not rel_extra and not args.get('init') and not args.get('xt_add_relonly')
    if xadd: exp = ans['xt_add_names'](exp, xadd, quiet, reserved=XT_REL.get(xt, []) + XT_RELONLY.get(xt, []))
    rel = [exp.index(n) for n in XT_REL.get(xt, []) if n in exp] or None
    idc = [i for i, n in enumerate(exp) if n.startswith(('seq_cv', 'it_hist'))] or None
    ronly = [exp.index(n) for n in XT_RELONLY.get(xt, []) if n in exp] or None
    n_dense = (N_DENSE + 2 if args.get('user0') else N_DENSE) + len(exp) + 2 * len(rel or []) - len(ronly or [])
    ok1 = cfg.get('xt_names') == exp; ok2 = (cfg.get('xt_rel'), cfg.get('xt_idcols'), cfg.get('xt_relonly'), cfg['n_dense']) == (rel, idc, ronly, n_dense)
    P(f'{v}: saved args xt {xt!r} xt_add {xadd!r}; expected list {len(exp)} names; cfg xt_names equal: {ok1}; (xt_rel, xt_idcols, xt_relonly, n_dense) cfg {(cfg.get("xt_rel"), cfg.get("xt_idcols"), cfg.get("xt_relonly"), cfg["n_dense"])} == recomputed {(rel, idc, ronly, n_dense)}: {ok2}')
    for sp in ('train', 'val', 'test'):
        miss = [n for n in cfg['xt_names'] if not os.path.exists(f'{ROOT}/nn_x/abl/feat_cache/{sp}/{n.removeprefix("has:")}.npy')]
        P(f'    every cfg input resolves in nn_x/abl/feat_cache/{sp} (the path attach_xt reads): {not miss} {miss if miss else ""}')
P('DONE')
