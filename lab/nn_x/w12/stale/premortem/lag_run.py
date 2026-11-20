"""premortem lag_run: score members on the lagged-holdout subset with the witness agent's tool (imported read-only), outputs under cache/lag/<view>/.
Usage: lag_run.py <control|member,member,...> [views=lag0,priv]"""
import sys, os, time
from pm_common import *            # sets threads, CUDA off, my numba cache dir (numba is imported here, before wt_common touches the environment)
WIT = ROOT / 'nn_x/w12/stale/witness'
sys.path.insert(0, str(WIT))
import torch
torch.set_num_threads(12)
import lagged_score as LS
MY = CACHE / 'lag'; WPRED = LS.PRED
LS.PRED = MY                        # pred_path() of the tool now points into my directory; nothing is written under the witness cache
names = sys.argv[1]; views = (sys.argv[2] if len(sys.argv) > 2 else 'lag0,priv').split(',')
log = open(LOGS / 'lag_run.log', 'a')
def L(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()
members = ['abl_e10_pop1kd05t25_s951'] if names == 'control' else names.split(',')
if names == 'control': views = ['priv']
# guard: every raw lag-0 column the checkpoints need is already cached by the witness agent (so raw_col() never writes into its directory)
for v in members:
    ck = torch.load(LS.ckpt_path(v), map_location='cpu', weights_only=False)
    for n in ck['cfg']['xt_names']:
        base = n[4:] if n.startswith('has:') else n
        assert (LS.SUB / f'xt_lag0_{base}.npy').exists(), (v, n)
    del ck
sub = LS.SubSplit()
for v in members:
    if names == 'control':
        p = MY / 'control' / f'{v}.npy'; p.parent.mkdir(parents=True, exist_ok=True)
        m, xn = LS.load_model(v); t1 = time.time(); sub.attach(xn, 'priv'); x = sub.score(m); np.save(p, x)
        ref = np.load(WPRED / 'priv' / f'{v}.npy'); d = float(np.abs(x - ref).max())
        L(f'{LS.now()} CONTROL {v} priv: max |logit diff| to the witness cache = {d:.2e} ({time.time() - t1:.0f}s) -> {"VALID" if d <= 1e-4 else "FAIL"}')
    else:
        LS.score_member(sub, v, views, log=L)
L(f'{LS.now()} lag_run done: {names} {views}')
