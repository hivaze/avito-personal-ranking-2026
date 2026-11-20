"""build_r4_proof.py <build_r4 file> <KEY> [<out json>]: bit-level record of what a given VERSION of nn_x/w12/kd/build_r4.py computes for an existing key
(read-only form: no 'build' argument, nothing is written by build_r4). The file is executed with runpy exactly as `python build_r4.py <KEY>` would run it;
then the md5 of the four branch arrays and of the final test score (float32, as 'build' would save them) and of every per-feed val NDCG vector are taken,
and the branch arrays are compared with the stored preds/ens_<KEY>kd_{ft,to}_{val,test}.npy when those exist. Also dumps spec(k) for a fixed list of keys
through the same head-exec that nn_x/w12/kd/fd/fd_finish.sh uses. CPU only.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/nx3/build_r4_proof.py nn_x/w12/kd/build_r4.py FD16P20 out.json"""
import hashlib, io, json, os, runpy, sys, contextlib
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
import numpy as np
path, key = sys.argv[1], sys.argv[2]; outp = sys.argv[3] if len(sys.argv) > 3 else None
assert 'build' not in sys.argv
md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
src = open(path).read()
head = src[src.index('S8 = range'):src.index('fv, Mv, ov = feed_meta')]
ns = {}; exec('from pathlib import Path\n' + head, ns)
KEYS = ['W16P20x8', 'W16P20w', 'W16P20d', 'W8P20x8', 'H16', 'H16P20x8', 'X16P20x8:hz', 'X16:hz', 'W8P20w', 'XL8P20w:hz', 'XL8:hz', 'W8w', 'FD8', 'FD8P20', 'FDX8P20:hz', 'FDX8:hz',
        'FD16', 'FD16P20', 'FD16P20x8', 'FD16P20w8', 'FD16P20w12', 'FD16P20v2', 'FD16P20vf', 'FD16P20vfB', 'FD16P20vf8', 'U32P20w', 'U32', 'FD32P20', 'FD24P20', 'FDB16P20', 'FD48P20',
        'FDC16P20', 'FDBC32P20', 'FD16P20v2f', 'FD16P20w12d', 'FD8P20x8', 'FDX16P20x8:hz', 'FDX16P20:hz', 'FDX16:hz', 'FDN16P20', 'NOSUCHKEY']
specs = {}
for k in KEYS:
    try:
        b, p, w = ns['spec'](k); specs[k] = dict(base=list(b), pop=list(p), w=w)
    except SystemExit as e:
        specs[k] = f'SystemExit: {e}'
rec = dict(file=path, file_md5=hashlib.md5(src.encode()).hexdigest(), key=key, spec_md5=hashlib.md5(json.dumps(specs, sort_keys=True).encode()).hexdigest(), specs=specs)
old_argv = sys.argv; sys.argv = [path, key]; buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(path, run_name='__main__')
sys.argv = old_argv
B, T, E = g['B'], g['T_ALL'][key], g['E_ALL'][key]
rec['stdout'] = buf.getvalue(); rec['stdout_md5'] = hashlib.md5(buf.getvalue().encode()).hexdigest()
rec['arrays'] = {f'{br}_{sp}': md5(B[sp][i].astype(np.float32)) for sp in ('val', 'test') for i, br in enumerate(('ft', 'to'))}
rec['arrays']['final_test_f32'] = md5(T.astype(np.float32)); rec['arrays']['final_test_native'] = md5(T)
rec['arrays'].update({f'ndcg_val_{n}': md5(e) for n, e in zip(('final', 'ft', 'to'), E)})
tag = key.replace(':', '_').replace('@m', '_m').replace('.', '') + 'kd'; P = g['P']; st = {}
for sp in ('val', 'test'):
    for i, br in enumerate(('ft', 'to')):
        f = P / f'ens_{tag}_{br}_{sp}.npy'
        st[f.name] = bool(np.array_equal(np.load(f), B[sp][i].astype(np.float32))) if f.exists() else 'no stored file'
rec['equal_to_stored_ens_files'] = st
print(f'{path} (md5 {rec["file_md5"]}) key {key}: stdout md5 {rec["stdout_md5"]}; spec table md5 {rec["spec_md5"]} ({len(KEYS)} keys)')
for k, v in rec['arrays'].items(): print(f'   {k:18s} {v}')
print(f'   branch arrays bitwise equal to the stored preds/ens_{tag}_*.npy: {st}')
print(f'   spec(FDN16P20): {specs["FDN16P20"] if isinstance(specs["FDN16P20"], str) else str(len(specs["FDN16P20"]["base"])) + " base + " + str(len(specs["FDN16P20"]["pop"])) + " pop, w " + str(specs["FDN16P20"]["w"])}')
if outp: json.dump(rec, open(outp, 'w'), indent=1)
