"""staleft: run a list of finetune_stale.py jobs sequentially inside ONE GPU process (one launch under the per-GPU lock; the
process keeps its GPU slot between jobs). Usage: CUDA_VISIBLE_DEVICES=<g> python run_queue.py <jobs.txt>
jobs.txt: one finetune_stale.py argument line per job; a job whose nn_<out>_val_oof.npy (or partial file) exists is skipped."""
import gc
import os
import shlex
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", "<repo>/nn_x/w12/stale/staleft/out/numba_cache")  # never write a numba cache into src/__pycache__
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import torch  # noqa: E402

import finetune_stale as F  # noqa: E402
from common import PREDS  # noqa: E402

F_CKPT0 = F.CKPT
jobs = [ln.strip() for ln in open(sys.argv[1]) if ln.strip() and not ln.startswith("#")]
print(f"{time.strftime('%FT%TZ', time.gmtime())} queue {sys.argv[1]}: {len(jobs)} jobs, pid {os.getpid()}, GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
for i, ln in enumerate(jobs):
    argv = shlex.split(ln)
    F.CKPT = F_CKPT0
    if argv and argv[0].startswith("CKPT="):  # per-job checkpoint directory (e.g. CKPT=<repo>/nn_x/abl/ckpt for the pop-light students)
        from pathlib import Path
        F.CKPT = Path(argv[0][5:])
        argv = argv[1:]
    out = argv[argv.index("--out") + 1]
    part = "--only_folds" in argv
    done = list(PREDS.glob(f"nn_{out}_partial_folds*.npz")) if part else [p for p in [PREDS / f"nn_{out}_val_oof.npy"] if p.exists()]
    if done:
        print(f"{time.strftime('%FT%TZ', time.gmtime())} [{i + 1}/{len(jobs)}] {out}: exists, skipped", flush=True)
        continue
    try:  # claim (atomic mkdir): two queue processes may share one job list
        os.makedirs(PREDS.parent / "claims", exist_ok=True)
        os.mkdir(PREDS.parent / "claims" / out)
    except FileExistsError:
        print(f"{time.strftime('%FT%TZ', time.gmtime())} [{i + 1}/{len(jobs)}] {out}: claimed by another queue process, skipped", flush=True)
        continue
    print(f"{time.strftime('%FT%TZ', time.gmtime())} [{i + 1}/{len(jobs)}] START {ln}", flush=True)
    t0 = time.time()
    try:
        F.main(argv)
        print(f"{time.strftime('%FT%TZ', time.gmtime())} [{i + 1}/{len(jobs)}] DONE {out} {(time.time() - t0) / 60:.1f} min", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"{time.strftime('%FT%TZ', time.gmtime())} [{i + 1}/{len(jobs)}] FAILED {out}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
print(f"{time.strftime('%FT%TZ', time.gmtime())} queue finished", flush=True)
