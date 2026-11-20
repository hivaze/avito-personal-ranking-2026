"""verify_where: GPU re-scoring of the @last checkpoints (all 20: 8 refs + 12 arm runs) on the 30,000 holdout feeds under lag0 and
priv_v3, reusing the analyst's lag_infer_gpu.py unchanged (imported; only its output dirs are redirected to verify_where).
Usage: CUDA_VISIBLE_DEVICES=N PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/nn_x/w12/moe/why/verify_where/lag_last_gpu.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/where')
import lag_infer_gpu as LI
V = Path('<repo>/lab/nn_x/w12/moe/why/verify_where')
LI.HERE = V; LI.OUT = V / 'out/lag_last'
S8 = range(951, 959); S4 = range(951, 955)
mem = [f'w5_e10_kd05w192_s{s}@last' for s in S8] + [f'w5_e10_{a}_s{s}@last' for a in ('kd05w192moeffn', 'kd05w192moevo', 'kd05w192b3ff1') for s in S4]
sys.argv = ['x', 'lag0,priv_v3'] + mem
LI.main()
