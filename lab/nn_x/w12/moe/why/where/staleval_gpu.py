"""why/where W2b: stale-val predictions (scenario both_pre2both of nn_x/w5/stale_pred.py: the 8 popularity / co-visitation inputs at
private-like coverage + pre2_fe* with the no-ID score on newly cold rows) for the MoE-screen arms, on the stored files' path
(nn_x/w5 common.predict, bf16 autocast, tf32 on), with the w12/moe model module. Read-only outside lab/nn_x/w12/moe/why/where/.
For every member it also re-predicts plain val and checks it against the stored train-only val preds (control).
Usage: CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/moe/why/where/staleval_gpu.py <member> ...
Writes out/stale/nn_<member>_staleval.npy and logs/staleval.log."""
import os, sys, time
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '12'
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(12)
ROOT = Path('<repo>')
sys.path.insert(0, str(ROOT / 'lab/nn_x/w5'))
sys.path.insert(0, str(ROOT / 'lab/nn_x/w12/moe'))
import data as Dm  # noqa: E402
from common import eval_ndcg, predict  # noqa: E402
from model_moe import Ranker  # noqa: E402

HERE = ROOT / 'lab/nn_x/w12/moe/why/where'
OUT = HERE / 'out/stale'
CF_BOTH = ROOT / 'nn_x/r6/stale_cf'
CF = ROOT / 'solvers/out/stale/cf'
OVR = {**{n: CF_BOTH / f'both_{n}.npy' for n in ('it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo',
                                                 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm', 'seq_cv_pop')},
       **{n: CF / f'pre2both_{n}.npy' for n in ('pre2_fe', 'pre2_fe_z', 'pre2_fe_s')}}
ONLY = os.environ.get('WHERE_ONLY', '')  # optional: 'pop' = override only the 8 popularity inputs (no pre2 part)
if ONLY == 'pop':
    OVR = {k: v for k, v in OVR.items() if not k.startswith('pre2')}
elif ONLY == 'pre2':  # only the pre2_fe* (encoder item-ID score on newly cold rows) part
    OVR = {k: v for k, v in OVR.items() if k.startswith('pre2')}
_feat_path = Dm.feat_path
ACTIVE = {'on': False}


def feat_path(split, name):
    if ACTIVE['on'] and split == 'val' and name in OVR:
        return OVR[name]
    return _feat_path(split, name)


Dm.feat_path = feat_path


def ck_path(v):
    for d in (ROOT / 'lab/nn_x/w12/moe/out/ckpt', ROOT / 'nn_x/w5/ckpt'):
        if (d / f'{v}.pt').exists():
            return d / f'{v}.pt'
    raise FileNotFoundError(v)


def stored_val(v):
    for d in (ROOT / 'lab/nn_x/w12/moe/out/preds', ROOT / 'nn_x/w5/preds'):
        if (d / f'nn_{v}_trainonly_val.npy').exists():
            return d / f'nn_{v}_trainonly_val.npy'
    return None


def main(variants):
    for p in OVR.values():
        assert p.exists(), p
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda:0'); torch.backends.cuda.matmul.allow_tf32 = True
    va = Dm.SplitData('val', dev)
    log = open(HERE / 'logs/staleval.log', 'a')
    tag = {'pop': '_stalepop', 'pre2': '_stalepre2'}.get(ONLY, '_staleval')
    for v in variants:
        out = OUT / f'nn_{v}{tag}.npy'
        if out.exists():
            print(f'{v}: exists', flush=True); continue
        t0 = time.time()
        ck = torch.load(ck_path(v), map_location=dev, weights_only=False)
        m = Ranker(**ck['cfg']).to(dev); m.load_state_dict(ck['state_dict'])
        xt = ck['cfg'].get('xt_names') or []
        used = sorted(n for n in OVR if n in xt or f'has:{n}' in xt)
        ACTIVE['on'] = False; va.xt = None; va.attach_xt(xt)
        pb = predict(m, va)
        r = np.load(stored_val(v))
        msg = f'base ndcg {eval_ndcg(pb, va):.5f} stored {eval_ndcg(r, va):.5f} corr {np.corrcoef(r, pb)[0, 1]:.7f} max|d| {np.abs(r - pb).max():.2e}'
        ACTIVE['on'] = True; va.xt = None; va.attach_xt(xt)
        ps = predict(m, va); ACTIVE['on'] = False
        np.save(out, ps.astype(np.float32))
        s = f'{time.strftime("%FT%TZ", time.gmtime())} {v}{tag}: stale ndcg {eval_ndcg(ps, va):.5f}; {msg}; overridden {used}; {time.time() - t0:.0f}s'
        print(s, flush=True); log.write(s + '\n'); log.flush()
        del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1:])
