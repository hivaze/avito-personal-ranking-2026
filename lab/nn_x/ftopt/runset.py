"""Post-freeze descriptive FT runs on a member set with explicit params (same GPU policy as runB.py).

Usage: python runset.py A|B <tag>=<json params> [<tag>=<json params> ...]
Outputs out/<tag>/nn_<m>_<tag>_val_oof.npy; then scores each tag with confirm-style stats (set A: TUNE / CONFIRM /
ALL; set B: ALL / halves). Descriptive only (the study is frozen).
"""
import json
import os
import subprocess
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import GPUS, MAX_TOTAL_PER_GPU, MIN_FREE_MB, PY, ft_args, gpu_state  # noqa: E402
from ftlib import FT_DIR, ROOT, SET_A, SET_B, code_of  # noqa: E402
from runB import own_ft_pids  # noqa: E402


def main():
    S = {"A": SET_A, "B": SET_B}[sys.argv[1]]
    runs = []
    for arg in sys.argv[2:]:
        tag, js = arg.split("=", 1)
        runs.append((tag, json.loads(js)))
    queue = []
    for tag, p in runs:
        od = FT_DIR / "out" / tag
        od.mkdir(parents=True, exist_ok=True)
        print(f"{tag}: {p}", flush=True)
        queue += [(m, tag, p, od) for m in sorted(S, key=lambda m: 0 if "gru" in m else 1)
                  if not (od / f"nn_{m}_{tag}_val_oof.npy").exists()]
    running = []
    while queue or running:
        for r in list(running):
            pr, fh, g, m, tag = r
            if pr.poll() is not None:
                fh.close()
                print(f"{time.strftime('%H:%M:%S')} {tag} {m} exit {pr.returncode}", flush=True)
                if pr.returncode != 0:
                    raise SystemExit(f"{tag} {m} failed")
                running.remove(r)
        if queue:
            st = gpu_state()
            mine_all = own_ft_pids()
            for g in GPUS:
                if not queue:
                    break
                pids = set(st[g]["pids"])
                invisible = [r[0].pid for r in running if r[2] == g and r[0].pid not in pids]
                own_here = len(pids & mine_all) + len(invisible)
                total = len(pids) + len(invisible)
                if own_here < 2 and total < MAX_TOTAL_PER_GPU and st[g]["free"] >= MIN_FREE_MB:
                    m, tag, p, od = queue.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                               POLARS_MAX_THREADS="8", PYTHONUNBUFFERED="1")
                    cmd = [PY, str(FT_DIR / "ftrun.py"), "--code", code_of(m), "--outdir", str(od), "--no_test",
                           "--"] + ft_args(m, tag, p)
                    fh = open(od / f"{m}.stdout", "w")
                    pr = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
                    running.append((pr, fh, g, m, tag))
                    print(f"{time.strftime('%H:%M:%S')} launch {tag} {m} gpu{g} pid {pr.pid} (total on gpu {total})",
                          flush=True)
                    break
        time.sleep(20)
    for tag, _ in runs:
        for m in S:
            for suf in ("test", "test_folds"):
                f = FT_DIR / "out" / tag / f"nn_{m}_{tag}_{suf}.npy"
                if f.exists():
                    f.unlink()
    for tag, _ in runs:
        if sys.argv[1] == "B":
            subprocess.run([PY, str(FT_DIR / "confirm.py"), "B", tag, str(FT_DIR / "out" / tag)], cwd=str(FT_DIR))
        else:
            subprocess.run([PY, str(FT_DIR / "score_tag.py"), tag], cwd=str(FT_DIR))


if __name__ == "__main__":
    main()
