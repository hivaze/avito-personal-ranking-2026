"""Second confirmation: fine-tune the set-B members with a frozen trial's params (val OOF only), then score.

Usage: python runB.py <trial_number> [tag]     (tag default: B<trial>)
GPU policy as the driver: GPUs N, <= 2 own ftrun.py processes per GPU (counting the driver's jobs too, by cmdline),
< 4 compute processes per GPU in total, >= 40 GB free. Run with the driver PAUSEd or idle.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import GPUS, MAX_TOTAL_PER_GPU, MIN_FREE_MB, PY, ft_args, gpu_state  # noqa: E402
from ftlib import FT_DIR, ROOT, SET_B, code_of  # noqa: E402


def own_ft_pids():
    out = set()
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if b"nn_x/ftopt/ftrun.py" in (d / "cmdline").read_bytes():
                    out.add(int(d.name))
            except OSError:
                pass
    return out


def main():
    tn = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else f"B{tn}"
    rows = {json.loads(l)["trial"]: json.loads(l) for l in open(FT_DIR / "trials.jsonl")}
    p = rows[tn]["params"]
    od = FT_DIR / "out" / tag
    od.mkdir(parents=True, exist_ok=True)
    print(f"set B with trial {tn} params {p} -> {od}", flush=True)
    queue = [m for m in sorted(SET_B, key=lambda m: 0 if "gru" in m else 1)
             if not (od / f"nn_{m}_{tag}_val_oof.npy").exists()]
    running = {}
    while queue or running:
        for m, (pr, fh, g) in list(running.items()):
            if pr.poll() is not None:
                fh.close()
                print(f"{time.strftime('%H:%M:%S')} {m} exit {pr.returncode}", flush=True)
                if pr.returncode != 0:
                    raise SystemExit(f"{m} failed")
                del running[m]
        if queue:
            st = gpu_state()
            mine_all = own_ft_pids()
            for g in GPUS:
                if not queue:
                    break
                pids = set(st[g]["pids"])
                launched_here = [pr.pid for (pr, _, gg) in running.values() if gg == g]
                invisible = [x for x in launched_here if x not in pids]
                own_here = len(pids & mine_all) + len(invisible)
                total = len(pids) + len(invisible)
                if own_here < 2 and total < MAX_TOTAL_PER_GPU and st[g]["free"] >= MIN_FREE_MB:
                    m = queue.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                               POLARS_MAX_THREADS="8", PYTHONUNBUFFERED="1")
                    cmd = [PY, str(FT_DIR / "ftrun.py"), "--code", code_of(m), "--outdir", str(od), "--no_test",
                           "--"] + ft_args(m, tag, p)
                    fh = open(od / f"{m}.stdout", "w")
                    pr = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
                    running[m] = (pr, fh, g)
                    print(f"{time.strftime('%H:%M:%S')} launch {m} gpu{g} pid {pr.pid} (total on gpu {total})",
                          flush=True)
                    break
        time.sleep(20)
    for m in SET_B:
        for suf in ("test", "test_folds"):
            f = od / f"nn_{m}_{tag}_{suf}.npy"
            if f.exists():
                f.unlink()
    subprocess.run([PY, str(FT_DIR / "confirm.py"), "B", tag, str(od)], cwd=str(FT_DIR), check=True)


if __name__ == "__main__":
    main()
