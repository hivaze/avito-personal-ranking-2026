"""u20wit: CPU fp32 scores of checkpoints on the cross-user holdout sub-split (cache/subxh), domain token as in training / holdout scoring ('tr').
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/xh_score.py <variant> [<variant> ...]   -> cache/scxh/<variant>__tr.npy
Labels are not read by this script."""
from wit_common import *
import torch
torch.set_num_threads(12)
(CACHE / 'scxh').mkdir(exist_ok=True)
sm = np.load(CACHE / 'subxh_meta.npz'); data = load_subsplit(CACHE / 'subxh'); assert (data.src_fidx == sm['src_fidx']).all()
p = torch.from_numpy(sm['pold'].astype(np.float32)); dom_tr = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
log(f'XH sub-split: {data.n} feeds, {data.n_cand} rows')
for v in sys.argv[1:]:
    path = CKPT / f'{v}.pt'; outp = CACHE / 'scxh' / f'{v}__tr.npy'
    if outp.exists(): log(f'{v}: already scored'); continue
    if not path.exists(): log(f'MISSING checkpoint {path}'); continue
    st = path.stat(); model, ck = load_model(path); assert ck['cfg'].get('xt_names') == data.xt_names and ck['cfg'].get('domain'); t0 = time.time()
    sc = score(model, data, np.arange(data.n), dom=dom_tr); assert not np.isnan(sc).any()
    tmp = outp.with_suffix('.tmp.npy'); np.save(tmp, sc); os.replace(tmp, outp)
    log(f'{v}: ckpt mtime {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime))}Z, {data.n} feeds {data.n_cand} rows in {time.time() - t0:.0f}s')
log('DONE')
