"""verify_where: stale-val (both: 8 popularity inputs + pre2 at private-like coverage) for the @last checkpoints of all 20 members,
reusing the analyst's staleval_gpu.py unchanged (imported; output dirs redirected to verify_where).
Usage: CUDA_VISIBLE_DEVICES=N PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/nn_x/w12/moe/why/verify_where/stale_last_gpu.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/where')
import staleval_gpu as SG
V = Path('<repo>/lab/nn_x/w12/moe/why/verify_where')
SG.HERE = V; SG.OUT = V / 'out/stale_last'
mem = [f'w5_e10_kd05w192_s{s}@last' for s in range(951, 959)] + [f'w5_e10_{a}_s{s}@last' for a in ('kd05w192moeffn', 'kd05w192moevo', 'kd05w192b3ff1') for s in range(951, 955)]
SG.main(mem)
