"""u20wit step 2: CPU fp32 scores of checkpoints on the witness feed sets (one process, 12 threads, sequential over checkpoints).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/wit_score.py <dom> <sets csv> <variant> [<variant> ...]
     dom  = tr (training / holdout-scoring / dump_teacher domain token) | rl (val / test convention: random-like one-hot)
     sets = any of W0S,W0L,H0,W1S,W1L
     variant = checkpoint name under nn_x/w5/ckpt without .pt (e.g. w5_e10_u20b_s951 or w5_e10_u20b_s951@last)
Output: cache/sc/<variant>__<dom>.npy = float32 logits on the sub-split's flat candidate rows (NaN = set not scored yet); a second call adds sets.
The checkpoint file is read once at start (mtime + size logged). Labels are not read by this script."""
from wit_common import *
import torch
torch.set_num_threads(12)
dom, sets, variants = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
assert dom in ('tr', 'rl') and all(s in SETS for s in sets)
(CACHE / 'sc').mkdir(exist_ok=True)
sm = np.load(CACHE / 'sub_meta.npz'); data = load_subsplit(CACHE / 'sub')
assert (data.src_fidx == sm['src_fidx']).all()
p = torch.from_numpy(sm['pold'].astype(np.float32)); dom_tr = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
fid_row = np.repeat(np.arange(data.n), data.M_np)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
log(f'sub-split: {data.n} feeds, {data.n_cand} rows, xt {tuple(data.xt.shape)}; dom {dom}; sets {sets}')
for v in variants:
    path = CKPT / f'{v}.pt'; outp = CACHE / 'sc' / f'{v}__{dom}.npy'
    if not path.exists(): log(f'MISSING checkpoint {path}'); continue
    acc = np.load(outp) if outp.exists() else np.full(data.n_cand, np.nan, np.float32)
    todo = [s for s in sets if np.isnan(acc[np.isin(fid_row, np.where(sm['set_id'] == SETS.index(s))[0])]).any()]
    if not todo: log(f'{v}: all requested sets already scored'); continue
    st = path.stat(); model, ck = load_model(path)
    assert ck['cfg'].get('xt_names') == data.xt_names, 'xt_names differ from the cached sub-split'
    assert ck['cfg'].get('domain'), 'model without a domain token: dom modes are identical'
    a = ck.get('args', {})
    log(f'{v}: ckpt mtime {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime))}Z size {st.st_size} d {ck["cfg"]["d"]} step {ck.get("step", "selected")} '
        f'xfold {a.get("xfold")} unlab {a.get("xfold_unlab")} kd_file {os.path.basename(a.get("kd_file", "") or "")} kd_T {a.get("kd_T")} kd_w {a.get("kd_w")} seed {a.get("seed")}')
    for s in todo:
        feeds = np.where(sm['set_id'] == SETS.index(s))[0]; t0 = time.time()
        sc = score(model, data, feeds, dom=dom_tr if dom == 'tr' else None)
        rows = np.isin(fid_row, feeds); assert not np.isnan(sc[rows]).any() and np.isnan(sc[~rows]).all()
        acc[rows] = sc[rows]
        log(f'  {s}: {len(feeds)} feeds {int(rows.sum())} rows in {time.time() - t0:.0f}s ({rows.sum() / (time.time() - t0):.0f} rows/s)')
        tmp = outp.with_suffix('.tmp.npy'); np.save(tmp, acc); os.replace(tmp, outp)
log('DONE')
